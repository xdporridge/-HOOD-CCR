import csv
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib import animation

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
from matplotlib import pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3.common.atari_wrappers import (  # isort:skip
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from Adversary import (
    BadRLMiddleMan,
    BufferMan_Simple,
    DeterministicMiddleMan,
    Discrete,
    ImagePoison,
    SingleValuePoison,
    UntargetedMiddleMan,
)
import patterns

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    console_log_interval: int = 1
    """print one training summary every N PPO updates; 0 disables update summaries"""
    episode_log_interval: int = 1
    """print one episode summary every N completed episodes; 0 disables episode summaries"""
    rolling_window: int = 20
    """number of recent episodes used for console rolling return/length statistics"""

    # Attack type arguments
    atari: bool = True
    sn_outer: bool = False
    sn_inner: bool = False
    trojdrl: bool = False
    untargeted: bool = False
    """use the TrojDRL untargeted attack objective"""
    untargeted_weak: bool = False
    """preserve policy actions instead of using the paper's default uniform action replacement"""
    badrl: bool = False
    safety: bool = False
    trade: bool = False
    highway: bool = False

    # Attack arguments
    target_action: int = 0
    p_rate: float = 0.01
    alpha: float = 0.5
    rew_p: float = 1.0
    simple_select: bool = False
    strong: bool = False
    """allow the existing BadRL attack to replace actions"""
    untargeted_reward: float = 1.0
    """positive reward assigned to every untargeted poisoned transition"""
    untargeted_adr_eval_interval: int = 4
    """PPO updates between paired clean/trigger ADR evaluations"""
    untargeted_adr_eval_batch_size: int = 256
    """clean states processed per forward pass during ADR evaluation"""
    untargeted_adr_eval_max_samples: int = 0
    """maximum clean rollout states per ADR evaluation; 0 uses every clean state"""
    untargeted_adr_last_frame_only: bool = True
    """evaluate ADR with the newest stacked frame triggered, matching defense evaluation"""

    # Algorithm specific arguments
    env_id: str = "BreakoutNoFrameskip-v4"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = .00025
    """the learning rate of the optimizer"""
    num_envs: int = 8
    """the number of parallel game environments"""
    num_steps: int = 200
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def validate_attack_args(args):
    attack_modes = {
        "sn_outer": args.sn_outer,
        "sn_inner": args.sn_inner,
        "trojdrl": args.trojdrl,
        "untargeted": args.untargeted,
        "badrl": args.badrl,
    }
    enabled_modes = [name for name, enabled in attack_modes.items() if enabled]
    if len(enabled_modes) > 1:
        raise ValueError(f"attack modes are mutually exclusive: {', '.join(enabled_modes)}")
    if min(args.num_envs, args.num_steps, args.num_minibatches, args.update_epochs) < 1:
        raise ValueError(
            "num_envs, num_steps, num_minibatches, and update_epochs must be positive"
        )
    batch_size = args.num_envs * args.num_steps
    if args.total_timesteps < batch_size:
        raise ValueError("total_timesteps must cover at least one rollout")
    if args.num_minibatches > batch_size:
        raise ValueError("num_minibatches cannot exceed the rollout batch size")
    if any(enabled_modes) and not 0 < args.p_rate <= 1:
        raise ValueError("p_rate must be in the interval (0, 1]")
    if args.untargeted and args.untargeted_reward <= 0:
        raise ValueError("untargeted_reward must be positive")
    if args.untargeted_weak and not args.untargeted:
        raise ValueError("untargeted_weak requires untargeted")
    if args.untargeted and args.total_timesteps < args.num_envs * args.num_steps:
        raise ValueError("untargeted total_timesteps must cover at least one rollout")
    if args.untargeted_adr_eval_interval < 1:
        raise ValueError("untargeted_adr_eval_interval must be positive")
    if args.untargeted_adr_eval_batch_size < 1:
        raise ValueError("untargeted_adr_eval_batch_size must be positive")
    if args.untargeted_adr_eval_max_samples < 0:
        raise ValueError("untargeted_adr_eval_max_samples cannot be negative")

def make_env(env_id, idx, capture_video, run_name, atari, highway):
    def thunk():
        if atari:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            env = gym.wrappers.RecordEpisodeStatistics(env)

            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env, skip=4)
            env = EpisodicLifeEnv(env)
            if "FIRE" in env.unwrapped.get_action_meanings():
                env = FireResetEnv(env)
            env = ClipRewardEnv(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        elif "Safe" in env_id:
            try:
                import safety_gymnasium
            except ImportError as exc:
                raise ImportError(
                    "Safety environments require the safety-gymnasium package."
                ) from exc
            env = safety_gymnasium.make(env_id, render_mode=None)
            env = gym.wrappers.RecordEpisodeStatistics(env)
        elif "CarRacing" in env_id:
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array", continuous = False)
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id, continuous = False)
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.GrayScaleObservation(env)
            env = gym.wrappers.FrameStack(env, 4)
        elif "Trading" in env_id:
            try:
                import gym_trading_env  # noqa: F401
                from gym_trading_env.downloader import download
            except ImportError as exc:
                raise ImportError(
                    "Trading environments require the gym-trading-env package."
                ) from exc

            download(exchange_names = ["bitfinex2"],
                symbols= ["BTC/USDT"],
                timeframe= "1h",
                dir = "data",
                since= datetime.datetime(year= 2020, month= 1, day=1),
                until = datetime.datetime(year = 2024, month = 1, day = 1),
            )
            df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")
            df["feature_close"] = df["close"].pct_change()
            df["feature_open"] = df["open"]/df["close"]
            df["feature_high"] = df["high"]/df["close"]
            df["feature_low"] = df["low"]/df["close"]
            df["feature_volume"] = df["volume"] / df["volume"].rolling(7*24).max()
            df.dropna(inplace=True)
            env = gym.make("TradingEnv",
                    name= "BTCUSD",
                    df = df,
                    positions = [-1 + (i*.2) for i in range(11)],
                    trading_fees = 0.01/100,
                    borrow_interest_rate= 0.0003/100,
                    max_episode_duration = 8760,
                    verbose = 0,
                    windows = 4,
                )
            env = gym.wrappers.RecordEpisodeStatistics(env)

        elif highway:
            env = gym.make(env_id, render_mode="rgb_array")
            env.configure({
                "action": {"type": "DiscreteMetaAction",
                            "longitudinal": True,
                            "lateral": False},
                "observation": {"type": "GrayscaleObservation",
                                "observation_shape": (84,84),
                                "stack_size": 4,
                                "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                                "scaling": 1.75,}
            })
            env = gym.wrappers.RecordEpisodeStatistics(env)
            env = gym.wrappers.ResizeObservation(env, (84, 84))
            env = gym.wrappers.FrameStack(env, 4)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def save_frames_as_gif(frames, path='./', filename='gym_animation.gif', dpi = 72.0):

    #Mess with this to change frame size
    plt.figure(figsize=(frames[0].shape[1] / dpi, frames[0].shape[0] / dpi), dpi=int(dpi))

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
    anim.save(path + filename, writer='imagemagick', fps=30)

def scalar_float(value):
    """Convert Gym/Tensor/numpy scalar-like values to a printable Python float."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.shape == ():
        return float(array)
    return float(array.reshape(-1)[0])

def format_recent_stats(values, name):
    if not values:
        return f"{name}_mean=nan {name}_min=nan {name}_max=nan"
    array = np.asarray(values, dtype=np.float32)
    return (
        f"{name}_mean={array.mean():.2f} "
        f"{name}_min={array.min():.2f} "
        f"{name}_max={array.max():.2f}"
    )


UNTARGETED_ADR_CSV_FIELDS = (
    "update",
    "global_step",
    "adr",
    "adr_percent",
    "eval_samples",
    "normalized_marginal_entropy",
    "normalized_conditional_entropy",
    "marginal_max_action_probability",
    "total_poisoned",
    "poisoning_rate",
    "seed",
    "p_rate",
    "untargeted_reward",
    "attack_metric",
    "trigger_protocol",
    "training_trigger_protocol",
    "metric_buffer_role",
    "eval_scope",
)


def initialize_untargeted_adr_csv(path):
    with open(path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=UNTARGETED_ADR_CSV_FIELDS)
        writer.writeheader()


def append_untargeted_adr_csv(path, row):
    with open(path, "a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=UNTARGETED_ADR_CSV_FIELDS)
        writer.writerow(row)


@torch.no_grad()
def evaluate_untargeted_adr(
    agent,
    observations,
    clean_indices,
    poison_batch,
    eval_batch_size,
    last_frame_only,
):
    """Evaluate paired greedy-action disagreement on online clean states."""
    sample_count = int(clean_indices.shape[0])
    if sample_count == 0:
        return {
            "adr": float("nan"),
            "adr_percent": float("nan"),
            "eval_samples": 0,
            "normalized_marginal_entropy": float("nan"),
            "normalized_conditional_entropy": float("nan"),
            "marginal_max_action_probability": float("nan"),
            "marginal_probabilities": None,
        }

    disagreement_count = 0
    conditional_entropy_sum = 0.0
    triggered_probability_sum = None

    for start in range(0, sample_count, eval_batch_size):
        batch_indices = clean_indices[start:start + eval_batch_size]
        clean_batch = observations[batch_indices]
        triggered_full = poison_batch(clean_batch)
        if last_frame_only:
            if clean_batch.ndim != 4:
                raise ValueError(
                    "last-frame ADR evaluation requires [B, C, H, W] observations"
                )
            triggered_batch = clean_batch.clone()
            triggered_batch[:, -1] = triggered_full[:, -1]
        else:
            triggered_batch = triggered_full

        clean_probabilities = agent.get_action_dist(clean_batch)
        triggered_probabilities = agent.get_action_dist(triggered_batch)
        disagreement_count += int(
            (
                clean_probabilities.argmax(dim=1)
                != triggered_probabilities.argmax(dim=1)
            ).sum().item()
        )
        conditional_entropy_sum += float(
            Categorical(probs=triggered_probabilities).entropy().sum().item()
        )
        probability_sum = triggered_probabilities.sum(dim=0)
        if triggered_probability_sum is None:
            triggered_probability_sum = probability_sum
        else:
            triggered_probability_sum += probability_sum

    marginal_probabilities = triggered_probability_sum / sample_count
    action_count = int(marginal_probabilities.shape[0])
    max_entropy = np.log(action_count) if action_count > 1 else 1.0
    adr = float(disagreement_count) / sample_count
    return {
        "adr": adr,
        "adr_percent": 100.0 * adr,
        "eval_samples": sample_count,
        "normalized_marginal_entropy": (
            Categorical(probs=marginal_probabilities).entropy().item()
            / max_entropy
        ),
        "normalized_conditional_entropy": (
            conditional_entropy_sum / sample_count / max_entropy
        ),
        "marginal_max_action_probability": (
            marginal_probabilities.max().item()
        ),
        "marginal_probabilities": marginal_probabilities,
    }


class Discretizer:
    def __init__(self, actions):
        self.actions = actions
    def __len__(self):
        return len(self.actions)
    def __call__(self, x, dim = False):
        return self.actions[x]

class Agent(nn.Module):
    def __init__(self, envs, image = True, safety = False, trade = False):
        super().__init__()
        if image:
            self.network = nn.Sequential(
                layer_init(nn.Conv2d(4, 32, 8, stride=4)),
                nn.ReLU(),
                layer_init(nn.Conv2d(32, 64, 4, stride=2)),
                nn.ReLU(),
                layer_init(nn.Conv2d(64, 64, 3, stride=1)),
                nn.ReLU(),
                nn.Flatten(),
                layer_init(nn.Linear(64 * 7 * 7, 512)),
                nn.ReLU(),
            )
            self.actor = layer_init(nn.Linear(512, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(512, 1), std=1)
            self.norm = 255
        elif safety:
            self.safety = True
            self.discretizer = Discretizer(torch.tensor([[0,0], [1, 0], [0, 1], [1, 1]]))
            obs_space = envs.single_observation_space.shape[0]*4
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, len(self.discretizer)), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)
        elif trade:
            obs_space = envs.single_observation_space.shape[0]
            self.network = nn.Sequential(
                layer_init(nn.Linear(obs_space, 64)),
                nn.ReLU(),
                layer_init(nn.Linear(64, 64)),
                nn.ReLU(),
            )
            self.norm = 1
            self.actor = layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01)
            self.critic = layer_init(nn.Linear(64, 1), std=1)

    def get_value(self, x):
        return self.critic(self.network(x / self.norm))

    def get_action_dist(self, x):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        return probs.probs

    def get_deterministic_action(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.network(obs / self.norm)
        return torch.argmax(self.actor(hidden), dim=-1)

    def get_policy_logits(self, obs: torch.Tensor) -> torch.Tensor:
        hidden = self.network(obs / self.norm)
        return self.actor(hidden)

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

class QNetwork(nn.Module):
    def __init__(self, env, image, safety, trade):
        super().__init__()
        if image:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, 8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, 512),
                nn.ReLU(),
                nn.Linear(512, env.single_action_space.n),
            )
            self.norm = 255
        elif safety:
            self.safety = True
            self.discretizer = Discretizer(torch.tensor([[0,0], [1, 0], [0, 1], [1, 1]]))
            obs_space = env.single_observation_space.shape[0]*4
            self.network = nn.Sequential(
                nn.Linear(obs_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, len(self.discretizer))
            )
            self.norm = 1
        elif trade:
            obs_space = env.single_observation_space.shape[0]
            self.network = nn.Sequential(
                nn.Linear(obs_space, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, env.single_action_space.n)
            )
            self.norm = 1
    def forward(self, x):
        return self.network(x / self.norm)

if __name__ == "__main__":

    args = tyro.cli(Args)
    validate_attack_args(args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    if args.track:
        import wandb

    asr = 0
    untargeted_marginal_entropy = float("nan")
    untargeted_conditional_entropy = float("nan")
    untargeted_max_action_probability = float("nan")
    untargeted_adr = float("nan")
    untargeted_adr_percent = float("nan")
    untargeted_adr_eval_samples = 0
    untargeted_metrics_update = 0
    untargeted_adr_csv_path = None
    tensorboard_dir = None
    untargeted_adr_effective_last_frame_only = (
        args.untargeted_adr_last_frame_only and args.atari
    )
    untargeted_adr_trigger_protocol = (
        "last_frame_only"
        if untargeted_adr_effective_last_frame_only
        else "full_stack"
    )
    untargeted_training_trigger_protocol = (
        "full_stack" if args.atari else "full_observation"
    )

    total_poisoned = 0
    total_perturb = 0

    #Block for all the stuff they do
    if True:

        if args.sn_outer:
            run_name = f"SN__{args.p_rate}__{args.rew_p}__{args.alpha}"
        elif args.sn_inner:
            run_name = f"SN_I__{args.p_rate}__{args.rew_p}__{args.alpha}"
        elif args.trojdrl:
            run_name = f"TrojDRL__{args.p_rate}__{args.rew_p}"
        elif args.untargeted:
            threat_model = "Weak" if args.untargeted_weak else "Strong"
            run_name = (
                f"TrojDRL_Untargeted_{threat_model}__"
                f"{args.p_rate}__{args.untargeted_reward}"
            )
        elif args.badrl:
            run_name = f"BadRL__{args.p_rate}__{args.rew_p}"
        else:
            run_name = f"Benign"

        run_dir = f"runs/{args.env_id}_{run_name}"
        os.makedirs(run_dir, exist_ok=True)
        os.makedirs("images", exist_ok=True)
        if args.capture_video:
            os.makedirs(f"videos/{run_name}", exist_ok=True)
        tensorboard_dir = run_dir
        if args.untargeted:
            run_instance_id = (
                f"seed{args.seed}_"
                f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_"
                f"pid{os.getpid()}"
            )
            tensorboard_dir = os.path.join(
                run_dir,
                "training_metrics",
                run_instance_id,
            )
            os.makedirs(tensorboard_dir, exist_ok=True)
            untargeted_adr_csv_path = os.path.join(
                tensorboard_dir,
                "untargeted_adr.csv",
            )
            initialize_untargeted_adr_csv(untargeted_adr_csv_path)

        print(
            f"[config] run_name={run_name} env={args.env_id} seed={args.seed} "
            f"device={'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'} "
            f"atari={args.atari} safety={args.safety} trade={args.trade}",
            flush=True,
        )
        print(
            f"[config] total_timesteps={args.total_timesteps} num_iterations={args.num_iterations} "
            f"num_envs={args.num_envs} num_steps={args.num_steps} batch_size={args.batch_size} "
            f"minibatch_size={args.minibatch_size} update_epochs={args.update_epochs}",
            flush=True,
        )
        print(
            f"[config] attack sn_outer={args.sn_outer} sn_inner={args.sn_inner} "
            f"trojdrl={args.trojdrl} untargeted={args.untargeted} "
            f"untargeted_weak={args.untargeted_weak} badrl={args.badrl} "
            f"strong={args.strong} target_action={args.target_action}",
            flush=True,
        )
        if args.untargeted:
            print(
                f"[config] online_adr interval={args.untargeted_adr_eval_interval} "
                f"batch_size={args.untargeted_adr_eval_batch_size} "
                f"max_samples={args.untargeted_adr_eval_max_samples} "
                f"trigger_protocol={untargeted_adr_trigger_protocol} "
                f"training_trigger_protocol={untargeted_training_trigger_protocol}",
                flush=True,
            )
            if (
                args.untargeted_adr_last_frame_only
                and not untargeted_adr_effective_last_frame_only
            ):
                print(
                    "[warning] last-frame ADR is Atari-only; using full_stack "
                    "for this non-Atari observation space.",
                    flush=True,
                )
        print(
            f"[paths] tensorboard={tensorboard_dir} periodic_model={run_dir}/{args.exp_name}.cleanrl_model "
            f"final_model={run_dir}/{args.exp_name}.cleanrl_model2",
            flush=True,
        )
        if untargeted_adr_csv_path is not None:
            print(f"[paths] untargeted_adr_csv={untargeted_adr_csv_path}", flush=True)

        if args.track:
            wandb.init(
                project=args.wandb_project_name,
                entity=args.wandb_entity,
                sync_tensorboard=True,
                config=vars(args),
                name=run_name,
                monitor_gym=True,
                save_code=True,
            )
        writer = SummaryWriter(tensorboard_dir)
        writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
        )
        if args.untargeted:
            writer.add_text(
                "untargeted/ADR_definition",
                (
                    "ADR = mean[argmax pi(a|clean_state) != "
                    "argmax pi(a|triggered_state)]; U-ADR (%) = 100 * ADR. "
                    "This is an online post-update rollout diagnostic, not a "
                    "fixed held-out gate metric. Trigger protocol: "
                    f"{untargeted_adr_trigger_protocol}. Training trigger "
                    f"protocol: {untargeted_training_trigger_protocol}."
                ),
            )

        # TRY NOT TO MODIFY: seeding
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = args.torch_deterministic

        device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

        envs = gym.vector.SyncVectorEnv(
            [
                make_env(
                    args.env_id,
                    i,
                    args.capture_video,
                    run_name,
                    args.atari,
                    args.highway,
                )
                for i in range(args.num_envs)
            ],
        )

        agent = Agent(envs, not (args.safety or args.trade), args.safety, args.trade).to(device)
        num_actions = (
            len(agent.discretizer)
            if args.safety
            else envs.single_action_space.n
        )
        if (
            args.sn_outer
            or args.sn_inner
            or args.trojdrl
            or args.badrl
        ) and not 0 <= args.target_action < num_actions:
            raise ValueError(
                f"target_action={args.target_action} is outside [0, {num_actions})"
            )
        optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
        if args.safety:
            actions = torch.zeros((args.num_steps, args.num_envs)).to(device)
        else:
            actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
        logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
        rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
        dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
        values = torch.zeros((args.num_steps, args.num_envs)).to(device)
        poison_masks = torch.zeros(
            (args.num_steps, args.num_envs), dtype=torch.bool, device=device
        )

        # TRY NOT TO MODIFY: start the game
        global_step = 0
        start_time = time.time()
        next_obs, _ = envs.reset(seed=args.seed)
        next_obs = torch.Tensor(next_obs).to(device)
        next_done = torch.zeros(args.num_envs).to(device)

    # --- Set up Outer Loop Attack --- #
    if args.sn_outer:
        if args.safety:
            poison_batch = SingleValuePoison([-1-16, -9-16, -13-16, -5-16], 1)
            poison = SingleValuePoison([-1-16, -9-16, -13-16, -5-16], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8)).to(device)
            poison = ImagePoison(pattern, 0, 255)
        bufferman = BufferMan_Simple(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p),
                                        p_rate = args.p_rate, alpha = args.alpha, simple = args.simple_select)

    # --- Set up TrojDrl Attack --- #
    if args.trojdrl or args.untargeted or args.badrl:
        if args.safety:
            poison_batch = SingleValuePoison([-1, -9], 1)
            poison = SingleValuePoison([-1, -9], 1)
        elif args.trade:
            poison_batch = SingleValuePoison([-1], 1)
            poison = SingleValuePoison([-1], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1,4, 84, 84), (8,8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)

            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8,8)).to(device)
            poison = ImagePoison(pattern, 0, 255)
        if args.untargeted:
            attack_timesteps = args.num_iterations * args.batch_size
            middleman = UntargetedMiddleMan(
                poison,
                reward=args.untargeted_reward,
                total=attack_timesteps,
                budget=attack_timesteps * args.p_rate,
                n_actions=agent.actor.out_features,
                strong=not args.untargeted_weak,
            )
        elif args.trojdrl:
            middleman = DeterministicMiddleMan(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.total_timesteps, args.total_timesteps*args.p_rate)
        else:
            q_net_adv = QNetwork(envs, not (args.safety or args.trade), args.safety, args.trade)
            q_net_adv.load_state_dict(torch.load(f"dqn_models/{args.env_id}__dqn/dqn.cleanrl_model", map_location = "cpu"))
            q_net_adv.to(device)
            middleman = BadRLMiddleMan(poison, args.target_action, Discrete(-1* args.rew_p, args.rew_p), args.p_rate, q_net_adv, args.strong)

    recorded = False
    old = 0
    tenth = max(1, args.total_timesteps // 10)
    save_interval = max(1, args.num_iterations // 10) if args.num_iterations > 0 else 1
    recent_returns = deque(maxlen=max(1, args.rolling_window))
    recent_lengths = deque(maxlen=max(1, args.rolling_window))
    episode_count = 0
    last_episode_return = None
    last_episode_length = None
    frames = []

    for iteration in range(1, args.num_iterations + 1):
        if args.save_model and iteration % save_interval == 0:
            model_path = f"{run_dir}/{args.exp_name}.cleanrl_model"
            torch.save(agent.state_dict(), model_path)
            print(f"[save] update={iteration}/{args.num_iterations} model={model_path}", flush=True)

        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        #Agent-Environment interaction loop
        recording = False
        for step in range(0, args.num_steps):
            poison_plan = []
            poison_masks[step].zero_()
            #for saving gifs
            if args.capture_video and not recorded:
                recording = True
                frames.append(envs.envs[0].render())

            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            poisoned = False

            # --- TrojDRL/Untargeted/BadRL poisoning --- #
            if args.untargeted:
                poison_plan = middleman.plan_poisoning(obs[step])
            elif (args.trojdrl or args.badrl) and asr < 1:
                poisoned, k, poison_action = middleman.time_to_poison(obs[step])
                if poisoned:
                    poison_plan = [(k, poison_action)]

            if poison_plan:
                for poison_index, _ in poison_plan:
                    poison_masks[step, poison_index] = True
                    poison_obs = middleman.obs_poison(next_obs[poison_index])
                    obs[step][poison_index] = poison_obs
                    next_obs[poison_index] = poison_obs
                total_poisoned += len(poison_plan)
                poisoned = True

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                action_overrides = [
                    (index, replacement)
                    for index, replacement in poison_plan
                    if replacement is not None
                ]
                if action_overrides:
                    override_indices = torch.tensor(
                        [item[0] for item in action_overrides],
                        dtype=torch.long,
                        device=device,
                    )
                    override_actions = torch.tensor(
                        [item[1] for item in action_overrides],
                        dtype=action.dtype,
                        device=device,
                    )
                    action[override_indices] = override_actions
                    _, override_logprob, _, _ = agent.get_action_and_value(
                        next_obs[override_indices],
                        action[override_indices],
                    )
                    logprob[override_indices] = override_logprob
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            if args.safety:
                next_obs, reward, terminations, truncations, infos = envs.step(agent.discretizer(action.cpu().numpy()))
            else:
                next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())

            # --- TrojDRL/Untargeted/BadRL poisoning --- #
            if (args.trojdrl or args.untargeted or args.badrl) and poisoned:
                for poison_index, _ in poison_plan:
                    old_reward = reward[poison_index].item()
                    reward[poison_index] = middleman.reward_poison(action[poison_index])
                    total_perturb += np.absolute(old_reward - reward[poison_index])

            next_done = np.logical_or(terminations, truncations)
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        ep_return = scalar_float(info["episode"]["r"])
                        ep_length = scalar_float(info["episode"]["l"])
                        episode_count += 1
                        last_episode_return = ep_return
                        last_episode_length = ep_length
                        recent_returns.append(ep_return)
                        recent_lengths.append(ep_length)
                        if args.episode_log_interval > 0 and episode_count % args.episode_log_interval == 0:
                            print(
                                f"[episode] count={episode_count} step={global_step} "
                                f"return={ep_return:.2f} length={ep_length:.0f}",
                                flush=True,
                            )
                        writer.add_scalar("charts/episodic_return", ep_return, global_step)
                        writer.add_scalar("charts/episodic_length", ep_length, global_step)
        if recording:
            recorded = True
            save_frames_as_gif(frames, path = f"videos/{run_name}/{old}.gif")
            frames = []
        if old%tenth > global_step%tenth:
            recorded = False
        old = global_step

        # --- Poison the Batch --- #
        with torch.no_grad():
            if args.sn_outer and asr < 1:
                for i in range(args.num_envs):
                    _, _, indices, pert = bufferman(obs[:, i], actions[:, i], rewards[:, i], values[:, i], logprobs[:, i], args.gamma, agent)
                    total_perturb += pert
                    total_poisoned += len(indices)
                    continue

        # bootstrap value if not done
        with torch.no_grad():
            bootstrap_obs = next_obs
            if args.untargeted:
                bootstrap_poison_indices = middleman.peek_poisoning(next_obs)
                if bootstrap_poison_indices:
                    bootstrap_obs = next_obs.clone()
                    for poison_index in bootstrap_poison_indices:
                        bootstrap_obs[poison_index] = middleman.obs_poison(
                            bootstrap_obs[poison_index]
                        )
            next_value = agent.get_value(bootstrap_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        if args.safety:
            b_actions = actions.reshape((-1,) + ())
        else:
            b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_poison_masks = poison_masks.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (
                        mb_advantages - mb_advantages.mean()
                    ) / (mb_advantages.std(unbiased=False) + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        clipfrac_value = float(np.mean(clipfracs)) if clipfracs else 0.0
        elapsed = max(time.time() - start_time, 1e-9)
        sps = int(global_step / elapsed)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("other/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", clipfrac_value, global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("other/SPS", sps, global_step)

        # --- Evaluate attack diagnostics --- #
        with torch.no_grad():
            if (args.sn_outer) and iteration%4 == 0:
                poisoned = bufferman.trigger(b_obs)
                probs = agent.get_action_dist(poisoned)
                asr = probs[:, args.target_action].mean().item()
                writer.add_scalar("charts/AttackSuccessRate", asr, global_step)
                if total_poisoned != 0:
                    writer.add_scalar(
                        "charts/reward_perturb_average",
                        total_perturb / (total_poisoned * 2),
                        global_step,
                    )
                writer.add_scalar(
                    "charts/reward_perturb_global",
                    total_perturb / global_step,
                    global_step,
                )
                writer.add_scalar(
                    "charts/poisoning_rate",
                    total_poisoned / global_step,
                    global_step,
                )
            if args.untargeted:
                online_attack_eval_due = (
                    iteration == 1
                    or iteration % args.untargeted_adr_eval_interval == 0
                    or iteration == args.num_iterations
                )
            else:
                online_attack_eval_due = iteration % 4 == 0
            if (args.trojdrl or args.untargeted or args.badrl) and online_attack_eval_due:
                clean_eval_indices = torch.where(~b_poison_masks)[0]
                if (
                    args.untargeted
                    and args.untargeted_adr_eval_max_samples > 0
                    and clean_eval_indices.shape[0]
                    > args.untargeted_adr_eval_max_samples
                ):
                    sample_positions = torch.linspace(
                        0,
                        clean_eval_indices.shape[0] - 1,
                        steps=args.untargeted_adr_eval_max_samples,
                        device=device,
                    ).round().long()
                    clean_eval_indices = clean_eval_indices[sample_positions]
                if args.untargeted:
                    diagnostics = evaluate_untargeted_adr(
                        agent=agent,
                        observations=b_obs,
                        clean_indices=clean_eval_indices,
                        poison_batch=poison_batch,
                        eval_batch_size=args.untargeted_adr_eval_batch_size,
                        last_frame_only=(
                            untargeted_adr_effective_last_frame_only
                        ),
                    )
                    untargeted_adr = diagnostics["adr"]
                    untargeted_adr_percent = diagnostics["adr_percent"]
                    untargeted_adr_eval_samples = diagnostics["eval_samples"]
                    untargeted_marginal_entropy = diagnostics[
                        "normalized_marginal_entropy"
                    ]
                    untargeted_conditional_entropy = diagnostics[
                        "normalized_conditional_entropy"
                    ]
                    untargeted_max_action_probability = diagnostics[
                        "marginal_max_action_probability"
                    ]
                    marginal_probs = diagnostics["marginal_probabilities"]
                    untargeted_metrics_update = iteration

                    writer.add_scalar(
                        "charts/UntargetedMarginalActionEntropy",
                        untargeted_marginal_entropy,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/UntargetedMeanConditionalActionEntropy",
                        untargeted_conditional_entropy,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/UntargetedMarginalMaxActionProbability",
                        untargeted_max_action_probability,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/UntargetedADR",
                        untargeted_adr,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/UntargetedADRPercent",
                        untargeted_adr_percent,
                        global_step,
                    )
                    writer.add_scalar(
                        "charts/UntargetedADREvalSamples",
                        untargeted_adr_eval_samples,
                        global_step,
                    )
                    # Retain the old tag for compatibility with existing runs.
                    writer.add_scalar(
                        "charts/UntargetedCleanTriggerActionDisagreement",
                        untargeted_adr,
                        global_step,
                    )
                    if marginal_probs is not None:
                        for action_id, probability in enumerate(marginal_probs):
                            writer.add_scalar(
                                f"charts/UntargetedActionProbability_{action_id}",
                                probability.item(),
                                global_step,
                            )
                    poisoning_rate = float(total_poisoned) / max(1, global_step)
                    append_untargeted_adr_csv(
                        untargeted_adr_csv_path,
                        {
                            "update": iteration,
                            "global_step": global_step,
                            "adr": untargeted_adr,
                            "adr_percent": untargeted_adr_percent,
                            "eval_samples": untargeted_adr_eval_samples,
                            "normalized_marginal_entropy": (
                                untargeted_marginal_entropy
                            ),
                            "normalized_conditional_entropy": (
                                untargeted_conditional_entropy
                            ),
                            "marginal_max_action_probability": (
                                untargeted_max_action_probability
                            ),
                            "total_poisoned": total_poisoned,
                            "poisoning_rate": poisoning_rate,
                            "seed": args.seed,
                            "p_rate": args.p_rate,
                            "untargeted_reward": args.untargeted_reward,
                            "attack_metric": "action_disagreement",
                            "trigger_protocol": (
                                untargeted_adr_trigger_protocol
                            ),
                            "training_trigger_protocol": (
                                untargeted_training_trigger_protocol
                            ),
                            "metric_buffer_role": "online_clean_rollout",
                            "eval_scope": "online_rollout_post_update",
                        },
                    )
                    print(
                        f"[untargeted-eval] update={iteration} "
                        f"step={global_step} online_adr={untargeted_adr:.6f} "
                        f"uadr_pct={untargeted_adr_percent:.3f} "
                        f"samples={untargeted_adr_eval_samples} "
                        f"protocol={untargeted_adr_trigger_protocol} "
                        f"marginal_entropy={untargeted_marginal_entropy:.4f} "
                        f"conditional_entropy={untargeted_conditional_entropy:.4f}",
                        flush=True,
                    )
                else:
                    clean_eval_obs = b_obs[clean_eval_indices]
                    if clean_eval_obs.shape[0] > 0:
                        triggered_probs = agent.get_action_dist(
                            poison_batch(clean_eval_obs)
                        )
                    else:
                        triggered_probs = agent.get_action_dist(b_obs)
                    asr = triggered_probs[:, args.target_action].mean().item()
                    writer.add_scalar("charts/AttackSuccessRate", asr, global_step)
                if total_poisoned != 0:
                    writer.add_scalar(
                        "charts/reward_perturb_average",
                        total_perturb / total_poisoned,
                        global_step,
                    )
                writer.add_scalar(
                    "charts/reward_perturb_global",
                    total_perturb / global_step,
                    global_step,
                )
                writer.add_scalar(
                    "charts/poisoning_rate",
                    total_poisoned / global_step,
                    global_step,
                )

            if online_attack_eval_due:
                plt.figure(dpi=150)
                plt.hist(b_actions.cpu().numpy())
                plt.savefig("images/" + run_name + ".png")
                plt.close()

        if (
            args.console_log_interval > 0
            and (
                iteration == 1
                or iteration % args.console_log_interval == 0
                or iteration == args.num_iterations
            )
        ):
            return_stats = format_recent_stats(recent_returns, "return")
            length_stats = format_recent_stats(recent_lengths, "length")
            last_return_text = "nan" if last_episode_return is None else f"{last_episode_return:.2f}"
            last_length_text = "nan" if last_episode_length is None else f"{last_episode_length:.0f}"
            attack_stats = ""
            if args.untargeted:
                perturb_avg = 0.0 if total_poisoned == 0 else float(total_perturb) / max(1, total_poisoned)
                attack_stats = (
                    f" marginal_entropy={untargeted_marginal_entropy:.4f} "
                    f"conditional_entropy={untargeted_conditional_entropy:.4f} "
                    f"adr={untargeted_adr:.4f} "
                    f"uadr_pct={untargeted_adr_percent:.2f} "
                    f"adr_samples={untargeted_adr_eval_samples} "
                    f"marginal_max_prob={untargeted_max_action_probability:.4f} "
                    f"attack_eval_update={untargeted_metrics_update} "
                    f"poisoned={total_poisoned} perturb_avg={perturb_avg:.4f}"
                )
            elif args.sn_outer or args.sn_inner or args.trojdrl or args.badrl:
                perturb_avg = 0.0 if total_poisoned == 0 else float(total_perturb) / max(1, total_poisoned)
                attack_stats = (
                    f" asr={asr:.4f} poisoned={total_poisoned} "
                    f"perturb_avg={perturb_avg:.4f}"
                )
            print(
                f"[train] update={iteration:05d}/{args.num_iterations:05d} "
                f"step={global_step}/{args.total_timesteps} sps={sps} "
                f"episodes={episode_count} last_return={last_return_text} "
                f"last_length={last_length_text} {return_stats} {length_stats} "
                f"lr={optimizer.param_groups[0]['lr']:.6g} "
                f"loss_pi={pg_loss.item():.4f} loss_v={v_loss.item():.4f} "
                f"entropy={entropy_loss.item():.4f} kl={approx_kl.item():.5f} "
                f"clipfrac={clipfrac_value:.3f} ev={explained_var:.3f}{attack_stats}",
                flush=True,
            )

    envs.close()
    writer.close()
    if args.track:
        wandb.finish()

    model_path = f"{run_dir}/{args.exp_name}.cleanrl_model3"
    torch.save(agent.state_dict(), model_path)
    print(f"[save] final_model={model_path}", flush=True)
