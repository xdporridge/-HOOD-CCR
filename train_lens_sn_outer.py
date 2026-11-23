"""Train clean critic lenses and calibrate a two-sided residual detector."""

import os
import random
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tyro

from ppo import Agent, make_env

@dataclass
class LensArgs:
    env_id: str = "BreakoutNoFrameskip-v4"
    atari: bool = True
    highway: bool = False
    safety: bool = False
    trade: bool = False

    model_path: str = ""
    num_envs: int = 8
    rollout_steps: int = 20000

    seed: int = 1
    cuda: bool = True
    lens_epochs: int = 200
    batch_size: int = 256
    lens_lr: float = 1e-3
    lens_l2: float = 1e-5

    lens_anom_resid_quantile: float = 0.9987
    lens_save_dir: str = ""


def validate_args(args: LensArgs) -> None:
    if args.safety or args.trade or not args.atari:
        raise ValueError(
            "This lens trainer currently supports Atari image agents only."
        )
    if not args.model_path:
        raise ValueError("model_path must be provided.")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if min(
        args.num_envs,
        args.rollout_steps,
        args.lens_epochs,
        args.batch_size,
    ) < 1:
        raise ValueError(
            "num_envs, rollout_steps, lens_epochs, and batch_size must be positive."
        )
    if args.lens_lr <= 0.0 or args.lens_l2 < 0.0:
        raise ValueError("lens_lr must be positive and lens_l2 non-negative.")
    if not 0.0 < args.lens_anom_resid_quantile < 1.0:
        raise ValueError("lens_anom_resid_quantile must be in (0, 1).")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_envs_and_agent(args: LensArgs):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, False, "lens_train", args.atari, args.highway)
         for i in range(args.num_envs)]
    )

    agent = Agent(envs, not (args.safety or args.trade), args.safety, args.trade).to(device)
    state_dict = torch.load(args.model_path, map_location=device)
    agent.load_state_dict(state_dict)
    agent.eval()

    return envs, agent, device


def collect_clean_obs(envs, agent, device, args: LensArgs):
    """Collect clean Atari observations in compact CPU storage."""
    obs_shape = envs.single_observation_space.shape
    total_samples = args.rollout_steps * args.num_envs

    all_obs = torch.zeros((total_samples,) + obs_shape, dtype=torch.uint8, device="cpu")

    next_obs, _ = envs.reset(seed=args.seed)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)

    idx = 0
    for _ in range(args.rollout_steps):
        all_obs[idx: idx + args.num_envs] = torch.as_tensor(
            next_obs, dtype=torch.uint8, device="cpu"
        )

        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(next_obs_t)

        next_obs_np, _, _, _, _ = envs.step(action.cpu().numpy())

        next_obs = next_obs_np
        next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
        idx += args.num_envs

    print(f"[Info] Collected clean states: {all_obs.shape}, dtype={all_obs.dtype}")
    return all_obs, total_samples


def select_feature_layers(agent: nn.Module):
    """Select the three convolutional layers and final trunk linear layer."""
    if not hasattr(agent, "network"):
        raise AttributeError("Agent must expose a 'network' module.")
    net = agent.network

    layers = {}
    layers["conv1"] = net[0]
    layers["conv2"] = net[2]
    layers["conv3"] = net[4]
    last_linear_idx = None
    for i, m in reversed(list(enumerate(net))):
        if isinstance(m, nn.Linear):
            last_linear_idx = i
            break
    if last_linear_idx is None:
        raise ValueError("Agent network does not contain a linear layer.")
    layers["fc3"] = net[last_linear_idx]

    print("[Info] Feature layers selected for lens:")
    for name, module in layers.items():
        print(f"  {name}: {module}")

    return layers


def register_feature_hooks(feature_layers):
    activations = {}
    handles = []
    for name, module in feature_layers.items():
        def hook(_module, _inputs, output, layer_name=name):
            activations[layer_name] = output

        handles.append(module.register_forward_hook(hook))
    return activations, handles


def infer_feature_dims(agent, activations, sample_obs, device, layer_names):
    activations.clear()
    with torch.no_grad():
        _ = agent.get_action_and_value(sample_obs.to(device))

    feature_dims = {}
    for name in layer_names:
        feat = activations[name]
        B = feat.shape[0]
        D = int(feat.numel() // B)
        feature_dims[name] = D
        print(f"[Info] Layer {name}: feature shape {tuple(feat.shape)} -> flat dim {D}")
    return feature_dims


class MultiLayerLens(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.heads = nn.ModuleDict({name: nn.Linear(dim, 1) for name, dim in feature_dims.items()})

    def forward_single(self, name, feat_flat):
        out = self.heads[name](feat_flat)
        return out.view(-1)

def train_lens(args: LensArgs):
    validate_args(args)
    set_seed(args.seed)

    if not args.lens_save_dir:
        run_dir = os.path.dirname(args.model_path)
        args.lens_save_dir = os.path.join(run_dir, "lens_out")
    os.makedirs(args.lens_save_dir, exist_ok=True)
    print("[Info] lens_save_dir =", args.lens_save_dir)

    envs, agent, device = build_envs_and_agent(args)
    try:
        all_obs, total_samples = collect_clean_obs(envs, agent, device, args)
    finally:
        envs.close()

    for p in agent.parameters():
        p.requires_grad = False

    feature_layers = select_feature_layers(agent)
    layer_names = list(feature_layers.keys())
    activations, handles = register_feature_hooks(feature_layers)

    sample_n = min(32, total_samples)
    sample_obs = all_obs[:sample_n].to(device=device, dtype=torch.float32)
    feature_dims = infer_feature_dims(agent, activations, sample_obs, device, layer_names)

    lens = MultiLayerLens(feature_dims).to(device)
    optimizer = torch.optim.Adam(lens.parameters(), lr=args.lens_lr)

    print("[Info] Start lens training...")
    N = total_samples
    indices = torch.arange(N)

    for epoch in range(1, args.lens_epochs + 1):
        perm = indices[torch.randperm(N)]
        loss_list, mse_list = [], []

        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_idx = perm[start:end]
            batch_obs = all_obs[batch_idx].to(device=device, dtype=torch.float32)

            activations.clear()
            _, _, _, value = agent.get_action_and_value(batch_obs)
            value = value.view(-1).detach()

            per_layer_mse = []
            for name in layer_names:
                feat = activations[name]
                B = feat.shape[0]
                feat_flat = feat.view(B, -1)
                v_l = lens.forward_single(name, feat_flat)
                per_layer_mse.append(F.mse_loss(v_l, value))

            critic_mse = torch.stack(per_layer_mse).mean()

            l2_reg = torch.tensor(0.0, device=device)
            if args.lens_l2 > 0:
                for p in lens.parameters():
                    l2_reg = l2_reg + (p ** 2).sum()
                l2_reg = args.lens_l2 * l2_reg

            total_loss = critic_mse + l2_reg

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            loss_list.append(total_loss.item())
            mse_list.append(critic_mse.item())

        print(f"[Epoch {epoch:03d}] loss={np.mean(loss_list):.6e}  "
              f"mean_critic_MSE={np.mean(mse_list):.6e}")

    lens_npz_path = os.path.join(args.lens_save_dir, "multi_layer_lens_clean_first_two_new.npz")
    save_dict = {
        "layer_names": np.array(layer_names),
        "lens_epochs": args.lens_epochs,
        "batch_size": args.batch_size,
        "lens_lr": args.lens_lr,
        "lens_l2": args.lens_l2,
        "num_samples": int(N),
    }
    for name in layer_names:
        head = lens.heads[name]
        W = head.weight.detach().cpu().numpy()
        b = head.bias.detach().cpu().numpy()
        save_dict[f"{name}_W_critic"] = W.T
        save_dict[f"{name}_b_critic"] = b.reshape(1)

    np.savez(lens_npz_path, **save_dict)
    print("[Save] multi-layer critic-lens ->", lens_npz_path)

    print("[Info] Start training residual-based anomaly detector (two-sided z-score)...")
    lens.eval()

    resid_list = []
    with torch.no_grad():
        for start in range(0, N, args.batch_size):
            end = min(start + args.batch_size, N)
            batch_obs = all_obs[start:end].to(device=device, dtype=torch.float32)
            activations.clear()
            _, _, _, value = agent.get_action_and_value(batch_obs)
            value = value.view(-1)
            B = value.shape[0]

            resid_batch = torch.zeros(B, len(layer_names), device=device)
            for j, name in enumerate(layer_names):
                feat = activations[name]
                feat_flat = feat.view(B, -1)
                v_l = lens.forward_single(name, feat_flat)
                resid_batch[:, j] = value - v_l

            resid_list.append(resid_batch.cpu().numpy())

    resid = np.concatenate(resid_list, axis=0)
    print("[Info] Multi-layer residuals shape:", resid.shape)

    mu_R = resid.mean(axis=0)
    std_R = resid.std(axis=0)
    print("[Stats] Residual mean per layer:", mu_R)
    print("[Stats] Residual std  per layer:", std_R)

    eps = 1e-8
    z = (resid - mu_R) / (std_R + eps)
    u_neg = np.maximum(0.0, -z)
    S_neg = np.sum(u_neg ** 2, axis=1)
    u_pos = np.maximum(0.0,  z)
    S_pos = np.sum(u_pos ** 2, axis=1)

    q = float(args.lens_anom_resid_quantile)

    mean_S_neg = float(S_neg.mean())
    std_S_neg = float(S_neg.std())
    thresh_neg = float(np.quantile(S_neg, q))

    mean_S_pos = float(S_pos.mean())
    std_S_pos = float(S_pos.std())
    thresh_pos = float(np.quantile(S_pos, q))

    print(f"[Stats] S_neg: mean={mean_S_neg:.6e}, std={std_S_neg:.6e}, q={q:.4f}, thresh_neg={thresh_neg:.6e}")
    print(f"[Stats] S_pos: mean={mean_S_pos:.6e}, std={std_S_pos:.6e}, q={q:.4f}, thresh_pos={thresh_pos:.6e}")

    resid_npz_path = os.path.join(args.lens_save_dir, "multi_layer_resid_stats_two_new.npz")
    np.savez(
        resid_npz_path,
        detector=np.array(["two_sided"]),
        layer_names=np.array(layer_names),
        mu_R=mu_R,
        std_R=std_R,
        quantile=q,

        thresh_neg=thresh_neg,
        thresh_pos=thresh_pos,
        mean_S_neg=mean_S_neg,
        std_S_neg=std_S_neg,
        mean_S_pos=mean_S_pos,
        std_S_pos=std_S_pos,
        num_samples=int(N),
        # Backward-compatible aliases for the original one-sided runtime.
        thresh=thresh_neg,
        mean_S=mean_S_neg,
        std_S=std_S_neg,
    )
    print("[Save] multi-layer residual stats ->", resid_npz_path)
    print("[Done] Clean-only multi-layer critic-lens & two-sided residual anomaly detector training complete.")

    for h in handles:
        h.remove()


if __name__ == "__main__":
    train_lens(tyro.cli(LensArgs))
