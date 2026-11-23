from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import tyro
from sklearn.metrics import roc_auc_score

import patterns
from Adversary import BufferMan_Simple, Discrete, ImagePoison, SingleValuePoison
from ppo import Agent, make_env


LENS_WEIGHTS_FILENAME = "multi_layer_lens_clean_first_two_new.npz"
RESIDUAL_STATS_FILENAME = "multi_layer_resid_stats_two_new.npz"


@dataclass
class RuntimeArgs:
    env_id: str = "BreakoutNoFrameskip-v4"
    atari: bool = True
    highway: bool = False
    safety: bool = False
    trade: bool = False

    sn_outer: bool = True
    trojdrl: bool = False
    badrl: bool = False
    daze: bool = False
    target_action: int = 0
    p_rate: float = 0.0003
    alpha: float = 0.5
    rew_p: float = 5.0
    simple_select: bool = False

    model_path: str = ""
    num_envs: int = 8
    eval_steps: int = 5000
    seed: int = 1
    cuda: bool = True

    poison_duration_min: int = 25
    poison_duration_max: int = 30
    cooldown_min: int = 20
    cooldown_max: int = 100

    lens_save_dir: str = ""
    do_calibration: bool = True
    calib_steps: int = 4000
    calib_quantile: float = 0.975

    w_conv1: float = 1.0
    w_conv2: float = 1.0
    w_conv3: float = 1.0
    w_fc3: float = 1.0


class MultiLayerLens(nn.Module):
    def __init__(self, feature_dims: Mapping[str, int]):
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(dim, 1) for name, dim in feature_dims.items()}
        )

    def forward_single(self, name: str, feat_flat: torch.Tensor) -> torch.Tensor:
        return self.heads[name](feat_flat).view(-1)


def enabled_attack(args: RuntimeArgs) -> str:
    attacks = {
        "sn_outer": args.sn_outer,
        "trojdrl": args.trojdrl,
        "badrl": args.badrl,
        "daze": args.daze,
    }
    enabled = [name for name, active in attacks.items() if active]
    if len(enabled) != 1:
        raise ValueError(
            "Exactly one attack must be enabled; "
            f"received {enabled if enabled else 'none'}."
        )
    return enabled[0]


def validate_args(args: RuntimeArgs) -> None:
    enabled_attack(args)
    if args.safety or args.trade or not args.atari:
        raise ValueError(
            "This residual detector currently supports Atari image agents only."
        )
    if not args.model_path:
        raise ValueError("model_path must be provided.")
    if not os.path.isfile(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")
    if args.num_envs < 1 or args.eval_steps < 1:
        raise ValueError("num_envs and eval_steps must be positive.")
    if not 0.0 < args.p_rate <= 1.0:
        raise ValueError("p_rate must be in (0, 1].")
    if not 0.0 <= args.alpha <= 1.0 or args.rew_p <= 0.0:
        raise ValueError("alpha must be in [0, 1] and rew_p must be positive.")
    if args.do_calibration and args.calib_steps < 2:
        raise ValueError("calib_steps must be at least 2.")
    if not 0.0 < args.calib_quantile < 1.0:
        raise ValueError("calib_quantile must be in (0, 1).")
    if not 0 < args.poison_duration_min <= args.poison_duration_max:
        raise ValueError(
            "poison_duration_min and poison_duration_max must define a "
            "positive interval."
        )
    if not 0 <= args.cooldown_min <= args.cooldown_max:
        raise ValueError(
            "cooldown_min and cooldown_max must define a non-negative interval."
        )


def build_poison_and_bufferman(
    args: RuntimeArgs, device: torch.device
) -> Tuple[BufferMan_Simple, None] | Tuple[None, object]:
    if args.sn_outer:
        if args.safety:
            poison = SingleValuePoison([-17, -25, -29, -21], 1)
        else:
            pattern = patterns.Single_Stacked_Img_Pattern(
                (4, 84, 84), (8, 8)
            ).to(device)
            poison = ImagePoison(pattern, 0, 255)
        bufferman = BufferMan_Simple(
            poison,
            args.target_action,
            Discrete(-args.rew_p, args.rew_p),
            p_rate=args.p_rate,
            alpha=args.alpha,
            simple=args.simple_select,
        )
        return bufferman, None

    if args.safety:
        poison_batch = SingleValuePoison([-1, -9], 1)
    elif args.trade:
        poison_batch = SingleValuePoison([-1], 1)
    else:
        pattern = patterns.Stacked_Img_Pattern(
            (1, 4, 84, 84), (8, 8)
        ).to(device)
        poison_batch = ImagePoison(pattern, 0, 255)
    return None, poison_batch


def apply_trigger_weak(
    obs: torch.Tensor,
    mask: torch.Tensor,
    args: RuntimeArgs,
    bufferman: BufferMan_Simple | None,
    poison_batch: object | None,
) -> torch.Tensor:
    if not bool(mask.any()):
        return obs

    if args.sn_outer:
        if bufferman is None:
            raise RuntimeError("The SN attack injector was not initialized.")
        poisoned = bufferman.trigger(obs)
    else:
        if poison_batch is None or not callable(poison_batch):
            raise RuntimeError("The batch attack injector was not initialized.")
        poisoned = poison_batch(obs)

    out = obs.clone()
    indices = torch.where(mask)[0]
    if obs.ndim == 4:
        out[indices, -1] = poisoned[indices, -1]
    else:
        out[indices] = poisoned[indices]
    return out


def select_feature_layers(agent: nn.Module) -> Dict[str, nn.Module]:
    if not hasattr(agent, "network"):
        raise AttributeError("Agent must expose a 'network' module.")
    network = agent.network
    if len(network) < 5:
        raise ValueError("Agent network is too short for the configured feature layers.")

    layers = {
        "conv1": network[0],
        "conv2": network[2],
        "conv3": network[4],
    }
    last_linear = next(
        (module for module in reversed(network) if isinstance(module, nn.Linear)),
        None,
    )
    if last_linear is None:
        raise ValueError("Agent network does not contain a linear feature layer.")
    layers["fc3"] = last_linear
    return layers


def register_feature_hooks(
    feature_layers: Mapping[str, nn.Module],
) -> Tuple[Dict[str, torch.Tensor], List[torch.utils.hooks.RemovableHandle]]:
    activations: Dict[str, torch.Tensor] = {}
    handles = []

    for name, module in feature_layers.items():
        def hook(_module, _inputs, output, layer_name=name):
            activations[layer_name] = output

        handles.append(module.register_forward_hook(hook))

    return activations, handles


def compute_feature_dims(
    agent: Agent,
    activations: Dict[str, torch.Tensor],
    layer_order: Sequence[str],
    sample_obs: torch.Tensor,
) -> Dict[str, int]:
    activations.clear()
    with torch.no_grad():
        agent.get_action_and_value(sample_obs)

    missing = [name for name in layer_order if name not in activations]
    if missing:
        raise RuntimeError(f"Feature hooks did not capture layers: {missing}")
    return {
        name: activations[name].reshape(activations[name].shape[0], -1).shape[1]
        for name in layer_order
    }


def compute_raw_residuals(
    value: torch.Tensor,
    activations: Mapping[str, torch.Tensor],
    lens: MultiLayerLens,
    layer_order: Sequence[str],
) -> torch.Tensor:
    batch_size = value.shape[0]
    residuals = value.new_zeros((batch_size, len(layer_order)))
    for index, name in enumerate(layer_order):
        features = activations[name].reshape(batch_size, -1)
        residuals[:, index] = value - lens.forward_single(name, features)
    return residuals


def compute_two_sided_scores(
    value: torch.Tensor,
    activations: Mapping[str, torch.Tensor],
    lens: MultiLayerLens,
    layer_order: Sequence[str],
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    residuals = compute_raw_residuals(value, activations, lens, layer_order)
    z_scores = (
        residuals - residual_mean.view(1, -1)
    ) / residual_std.view(1, -1).clamp_min(1e-8)
    negative = torch.clamp(-z_scores, min=0.0)
    positive = torch.clamp(z_scores, min=0.0)
    negative_score = torch.sum(weights.view(1, -1) * negative.square(), dim=1)
    positive_score = torch.sum(weights.view(1, -1) * positive.square(), dim=1)
    return negative_score, positive_score


def environment_action(
    agent: Agent, action: torch.Tensor, safety: bool
) -> np.ndarray:
    action_np = action.cpu().numpy()
    return agent.discretizer(action_np) if safety else action_np


def calibrate_thresholds(
    envs: gym.vector.VectorEnv,
    agent: Agent,
    lens: MultiLayerLens,
    activations: Dict[str, torch.Tensor],
    layer_order: Sequence[str],
    device: torch.device,
    args: RuntimeArgs,
    weights: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, float, float, np.ndarray]:
    print(
        f"\n[Phase 1] Calibrating on {args.calib_steps} clean observations..."
    )
    next_obs, _ = envs.reset(seed=args.seed + 999)
    next_obs_t = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
    residual_batches = []
    samples_collected = 0

    while samples_collected < args.calib_steps:
        with torch.no_grad():
            activations.clear()
            action, _, _, value = agent.get_action_and_value(next_obs_t)
            residuals = compute_raw_residuals(
                value.view(-1), activations, lens, layer_order
            )
            residual_batches.append(residuals.cpu().numpy())

        next_obs, _, _, _, _ = envs.step(
            environment_action(agent, action, args.safety)
        )
        next_obs_t = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        )
        samples_collected += envs.num_envs

    all_residuals = np.concatenate(residual_batches, axis=0)[: args.calib_steps]
    residual_mean = np.mean(all_residuals, axis=0).astype(np.float32)
    residual_std = np.std(all_residuals, axis=0).astype(np.float32)

    residuals_t = torch.as_tensor(
        all_residuals, dtype=torch.float32, device=device
    )
    mean_t = torch.as_tensor(residual_mean, device=device)
    std_t = torch.as_tensor(residual_std, device=device)
    z_scores = (
        residuals_t - mean_t.view(1, -1)
    ) / std_t.view(1, -1).clamp_min(1e-8)
    negative = torch.clamp(-z_scores, min=0.0)
    positive = torch.clamp(z_scores, min=0.0)
    negative_scores = torch.sum(
        weights.view(1, -1) * negative.square(), dim=1
    ).cpu().numpy()
    positive_scores = torch.sum(
        weights.view(1, -1) * positive.square(), dim=1
    ).cpu().numpy()

    threshold_negative = float(
        np.quantile(negative_scores, args.calib_quantile)
    )
    threshold_positive = float(
        np.quantile(positive_scores, args.calib_quantile)
    )
    print(
        f"[Calibration] q={args.calib_quantile:.6f}, "
        f"negative={threshold_negative:.4f}, "
        f"positive={threshold_positive:.4f}"
    )

    eval_obs, _ = envs.reset(seed=args.seed)
    return (
        residual_mean,
        residual_std,
        threshold_negative,
        threshold_positive,
        eval_obs,
    )


def load_lens(
    path: str,
    agent: Agent,
    activations: Dict[str, torch.Tensor],
    sample_obs: torch.Tensor,
    device: torch.device,
) -> Tuple[MultiLayerLens, List[str]]:
    with np.load(path, allow_pickle=False) as archive:
        if "layer_names" not in archive:
            raise KeyError(f"Missing 'layer_names' in lens archive: {path}")
        layer_order = [str(name) for name in archive["layer_names"]]
        if len(layer_order) != len(set(layer_order)):
            raise ValueError("Lens archive contains duplicate layer names.")

        feature_dims = compute_feature_dims(
            agent, activations, layer_order, sample_obs
        )
        lens = MultiLayerLens(feature_dims).to(device)
        with torch.no_grad():
            for name in layer_order:
                weight_key = f"{name}_W_critic"
                bias_key = f"{name}_b_critic"
                if weight_key not in archive or bias_key not in archive:
                    raise KeyError(
                        f"Missing lens parameters for layer '{name}' in {path}"
                    )
                weight = torch.as_tensor(
                    archive[weight_key].T,
                    dtype=torch.float32,
                    device=device,
                )
                bias = torch.as_tensor(
                    archive[bias_key].reshape(-1),
                    dtype=torch.float32,
                    device=device,
                )
                head = lens.heads[name]
                if weight.shape != head.weight.shape or bias.shape != head.bias.shape:
                    raise ValueError(
                        f"Lens parameter shape mismatch for layer '{name}': "
                        f"expected {tuple(head.weight.shape)}/{tuple(head.bias.shape)}, "
                        f"received {tuple(weight.shape)}/{tuple(bias.shape)}."
                    )
                head.weight.copy_(weight)
                head.bias.copy_(bias)

    lens.eval()
    return lens, layer_order


def load_residual_stats(
    path: str, layer_order: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"mu_R", "std_R", "thresh_neg", "thresh_pos"}
        missing = sorted(required.difference(archive.files))
        if missing:
            raise KeyError(f"Residual archive is missing fields: {missing}")
        if "layer_names" in archive:
            saved_order = [str(name) for name in archive["layer_names"]]
            if saved_order != list(layer_order):
                raise ValueError(
                    "Residual statistics use a different layer order: "
                    f"{saved_order} != {list(layer_order)}"
                )
        residual_mean = archive["mu_R"].astype(np.float32)
        residual_std = archive["std_R"].astype(np.float32)
        threshold_negative = float(archive["thresh_neg"])
        threshold_positive = float(archive["thresh_pos"])

    expected_shape = (len(layer_order),)
    if residual_mean.shape != expected_shape or residual_std.shape != expected_shape:
        raise ValueError(
            "Residual statistics have incompatible shapes: "
            f"{residual_mean.shape}/{residual_std.shape}, expected {expected_shape}."
        )
    if not (
        np.all(np.isfinite(residual_mean))
        and np.all(np.isfinite(residual_std))
        and np.isfinite(threshold_negative)
        and np.isfinite(threshold_positive)
    ):
        raise ValueError("Residual statistics contain non-finite values.")
    return (
        residual_mean,
        residual_std,
        threshold_negative,
        threshold_positive,
    )


def calculate_metrics(
    labels_list: Iterable[np.ndarray],
    flags_list: Iterable[np.ndarray],
    scores_list: Iterable[np.ndarray],
) -> Tuple[float, float, float, float, float, int, int]:
    labels = np.concatenate(list(labels_list))
    flags = np.concatenate(list(flags_list))
    scores = np.concatenate(list(scores_list))

    true_positive = int(np.sum((labels == 1) & (flags == 1)))
    true_negative = int(np.sum((labels == 0) & (flags == 0)))
    false_positive = int(np.sum((labels == 0) & (flags == 1)))
    false_negative = int(np.sum((labels == 1) & (flags == 0)))

    recall = true_positive / max(true_positive + false_negative, 1)
    false_positive_rate = false_positive / max(
        false_positive + true_negative, 1
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    f1_score = (
        2.0 * precision * recall / max(precision + recall, 1e-12)
    )
    auc = (
        float(roc_auc_score(labels, scores))
        if np.unique(labels).size == 2
        else float("nan")
    )
    return (
        recall,
        false_positive_rate,
        precision,
        f1_score,
        auc,
        true_positive,
        false_positive,
    )


def schedule_next_window(
    current_step: int, args: RuntimeArgs
) -> Tuple[int, int]:
    start = current_step + random.randint(args.cooldown_min, args.cooldown_max)
    duration = random.randint(
        args.poison_duration_min, args.poison_duration_max
    )
    return start, start + duration


def main(args: RuntimeArgs) -> None:
    validate_args(args)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )

    lens_dir = args.lens_save_dir or os.path.join(
        os.path.dirname(args.model_path), "lens_out"
    )
    lens_path = os.path.join(lens_dir, LENS_WEIGHTS_FILENAME)
    if not os.path.isfile(lens_path):
        raise FileNotFoundError(
            f"Lens weights not found: {lens_path}. "
            "Run train_lens_sn_outer.py first."
        )

    envs = gym.vector.SyncVectorEnv(
        [
            make_env(
                args.env_id,
                index,
                False,
                "eval",
                args.atari,
                args.highway,
            )
            for index in range(args.num_envs)
        ]
    )
    handles: List[torch.utils.hooks.RemovableHandle] = []

    try:
        agent = Agent(
            envs,
            not (args.safety or args.trade),
            args.safety,
            args.trade,
        ).to(device)
        if args.sn_outer and not 0 <= args.target_action < envs.single_action_space.n:
            raise ValueError(
                f"target_action={args.target_action} is outside "
                f"[0, {envs.single_action_space.n})."
            )
        agent.load_state_dict(torch.load(args.model_path, map_location=device))
        agent.eval()

        bufferman, poison_batch = build_poison_and_bufferman(args, device)
        feature_layers_all = select_feature_layers(agent)
        activations, handles = register_feature_hooks(feature_layers_all)

        initial_obs, _ = envs.reset(seed=args.seed)
        sample_obs = torch.as_tensor(
            initial_obs, dtype=torch.float32, device=device
        )

        with np.load(lens_path, allow_pickle=False) as archive:
            layer_order = [str(name) for name in archive["layer_names"]]
        unknown_layers = sorted(set(layer_order).difference(feature_layers_all))
        if unknown_layers:
            raise ValueError(
                f"Lens archive references unsupported layers: {unknown_layers}"
            )
        for handle in handles:
            handle.remove()
        handles.clear()
        feature_layers = {
            name: feature_layers_all[name] for name in layer_order
        }
        activations, handles = register_feature_hooks(feature_layers)
        lens, loaded_order = load_lens(
            lens_path, agent, activations, sample_obs, device
        )
        if loaded_order != layer_order:
            raise RuntimeError("Lens layer order changed while loading the archive.")

        weight_map = {
            "conv1": args.w_conv1,
            "conv2": args.w_conv2,
            "conv3": args.w_conv3,
            "fc3": args.w_fc3,
        }
        weights = torch.as_tensor(
            [weight_map[name] for name in layer_order],
            dtype=torch.float32,
            device=device,
        )
        if bool((weights < 0).any()) or not bool(torch.isfinite(weights).all()):
            raise ValueError("Layer weights must be finite and non-negative.")
        print(f"[Info] Attack: {enabled_attack(args)}")
        print(f"[Info] Layer weights: {dict(zip(layer_order, weights.tolist()))}")

        if args.do_calibration:
            (
                residual_mean,
                residual_std,
                threshold_negative,
                threshold_positive,
                next_obs,
            ) = calibrate_thresholds(
                envs,
                agent,
                lens,
                activations,
                layer_order,
                device,
                args,
                weights,
            )
        else:
            stats_path = os.path.join(lens_dir, RESIDUAL_STATS_FILENAME)
            if not os.path.isfile(stats_path):
                raise FileNotFoundError(
                    f"Cached residual statistics not found: {stats_path}"
                )
            (
                residual_mean,
                residual_std,
                threshold_negative,
                threshold_positive,
            ) = load_residual_stats(stats_path, layer_order)
            next_obs, _ = envs.reset(seed=args.seed)
            print(
                f"[Info] Loaded thresholds: negative={threshold_negative:.4f}, "
                f"positive={threshold_positive:.4f}"
            )

        residual_mean_t = torch.as_tensor(
            residual_mean, dtype=torch.float32, device=device
        )
        residual_std_t = torch.as_tensor(
            residual_std, dtype=torch.float32, device=device
        )
        results = {
            mode: {"labels": [], "flags": [], "scores": []}
            for mode in ("Positive", "Negative", "TwoSided")
        }

        env_steps = np.zeros(args.num_envs, dtype=np.int64)
        next_start = np.array(
            [random.randint(0, 200) for _ in range(args.num_envs)],
            dtype=np.int64,
        )
        next_end = next_start + np.array(
            [
                random.randint(
                    args.poison_duration_min, args.poison_duration_max
                )
                for _ in range(args.num_envs)
            ],
            dtype=np.int64,
        )
        next_obs_t = torch.as_tensor(
            next_obs, dtype=torch.float32, device=device
        )

        print(
            "\n[Phase 2] Evaluating random-duration attacks "
            f"({args.poison_duration_min}-{args.poison_duration_max} steps)..."
        )
        for step in range(args.eval_steps):
            poisoning = (env_steps >= next_start) & (env_steps < next_end)
            poison_mask = torch.as_tensor(
                poisoning, dtype=torch.bool, device=device
            )
            labels = poisoning.astype(np.int8)
            agent_obs = apply_trigger_weak(
                next_obs_t,
                poison_mask,
                args,
                bufferman,
                poison_batch,
            )

            with torch.no_grad():
                activations.clear()
                action, _, _, value = agent.get_action_and_value(agent_obs)
                negative_t, positive_t = compute_two_sided_scores(
                    value.view(-1),
                    activations,
                    lens,
                    layer_order,
                    residual_mean_t,
                    residual_std_t,
                    weights,
                )

            negative = negative_t.cpu().numpy()
            positive = positive_t.cpu().numpy()
            positive_flags = (positive > threshold_positive).astype(np.int8)
            negative_flags = (negative > threshold_negative).astype(np.int8)
            two_sided_flags = np.maximum(positive_flags, negative_flags)
            two_sided_scores = np.maximum(
                positive / max(threshold_positive, 1e-8),
                negative / max(threshold_negative, 1e-8),
            )

            for mode, flags, scores in (
                ("Positive", positive_flags, positive),
                ("Negative", negative_flags, negative),
                ("TwoSided", two_sided_flags, two_sided_scores),
            ):
                results[mode]["labels"].append(labels.copy())
                results[mode]["flags"].append(flags)
                results[mode]["scores"].append(scores)

            next_obs, _, terminations, truncations, _ = envs.step(
                environment_action(agent, action, args.safety)
            )
            env_steps += 1
            next_obs_t = torch.as_tensor(
                next_obs, dtype=torch.float32, device=device
            )

            done = np.logical_or(terminations, truncations)
            finished_window = env_steps >= next_end
            for index in np.flatnonzero(done | finished_window):
                next_start[index], next_end[index] = schedule_next_window(
                    int(env_steps[index]), args
                )

            if step % 100 == 0:
                print(f"Step {step}/{args.eval_steps}...", end="\r")

        print("\n\n" + "=" * 105)
        print(
            f"Random Attack Evaluation (calibration q={args.calib_quantile})"
        )
        print("=" * 105)
        print(
            f"{'Detector':<15} | {'Recall':<12} | {'FPR':<12} | "
            f"{'Precision':<12} | {'F1':<12} | {'AUC':<10} | "
            f"{'TP':<6} | {'FP':<6}"
        )
        print("-" * 105)
        for mode, values in results.items():
            recall, fpr, precision, f1, auc, tp, fp = calculate_metrics(
                values["labels"], values["flags"], values["scores"]
            )
            print(
                f"{mode:<15} | {recall:<12.4f} | {fpr:<12.4f} | "
                f"{precision:<12.4f} | {f1:<12.4f} | {auc:<10.4f} | "
                f"{tp:<6} | {fp:<6}"
            )
        print("=" * 105)
    finally:
        for handle in handles:
            handle.remove()
        envs.close()


if __name__ == "__main__":
    main(tyro.cli(RuntimeArgs))
