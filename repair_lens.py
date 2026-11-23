"""Three-phase detection, interception, and dynamic backdoor repair pipeline."""

import os
import random
import sys
import logging
import csv
import hashlib
from datetime import datetime
from dataclasses import dataclass, replace
from typing import Dict, List, Optional

import gymnasium as gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.categorical import Categorical
import numpy as np
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)

from Adversary import ImagePoison, SingleValuePoison, BufferMan_Simple, Discrete
import patterns


@dataclass
class PipelineArgs:
    env_id: str = "BreakoutNoFrameskip-v4"
    seed: int = 1
    cuda: bool = True
    num_envs: int = 8
    target_action: int = 2

    poisoned_model_path: str = ""
    lens_model_path: str = ""

    badrl_model_path: str = ""
    badrl_lens_model_path: str = ""
    sn_model_path: str = ""
    sn_lens_model_path: str = ""
    troj1_model_path: str = ""
    troj1_lens_model_path: str = ""
    untarget_model_path: str = ""
    untarget_lens_model_path: str = ""

    safety: bool = False
    trade: bool = False
    sn_outer: bool = True
    trojdrl: bool = False
    untargeted: bool = False
    badrl: bool = False
    p_rate: float = 0.0003
    alpha: float = 0.5
    rew_p: float = 5.0
    untargeted_reward: float = 1.0
    simple_select: bool = False

    calib_steps: int = 2000
    calib_quantile: float = 0.999


    offline_buffer_size: int = 1000
    # A disjoint held-out buffer controls the entropy gate and reports ASR/H.
    # offline_buffer_size remains the number of samples used for optimization.
    repair_validation_size: int = 200
    repair_split_gap_size: int = 200
    poison_duration_min: int = 25
    poison_duration_max: int = 30
    cooldown_min: int = 20
    cooldown_max: int = 100
    phase2_max_env_steps: int = 500000

    # The entropy filter keeps only low-entropy detector alarms.
    use_entropy_filter: bool = False
    intercept_entropy_thresh: float = 0.30

    output_root: str = "defense_outputs_dynamic_entropy4"
    save_intercept_buffers: bool = True

    lr: float = 1e-4
    total_timesteps: int = 50000
    num_steps: int = 128
    minibatch_size: int = 256
    update_epochs: int = 4
    temperature: float = 2.0

    # --- Dynamic Stage 1 -> Stage 2 transition ---
    # "entropy" uses the intercepted-buffer entropy gate below. "fixed" keeps
    # the original update-count behavior for controlled comparisons.
    stage1_switch_mode: str = "entropy"
    stage1_fixed_updates: int = 5

    # A negative threshold derives an attack-specific target:
    # H_target = H0 + ratio * max(H_clean - H0, 0).
    # A non-negative threshold is interpreted as an absolute value in nats.
    stage1_entropy_threshold: float = -1.0
    stage1_entropy_recovery_ratio: float = 0.80
    stage1_entropy_window: int = 3
    stage1_entropy_patience: int = 1
    stage1_entropy_hysteresis: float = 0.02
    stage1_min_updates: int = 2

    # 0 disables forced switching. A positive value ends Stage 1 after that
    # update and records reason=timeout_forced.
    stage1_max_updates: int = 16

    beta_lens_kl: float = 1.0
    beta_ent: float = 1.0
    beta_feat_mmd: float = 3.0

    beta_vai: float = 1.0
    vai_temp: float = 2.0
    vai_adv_coef: float = 1.0
    vai_value_coef: float = 1.0
    vai_max_weight: float = 4.0

    eval_interval: int = 1
    return_eval_interval: int = 4
    eval_batch_size: int = 256
    eval_episodes: int = 10
    eval_max_steps: int = 10000
    eval_no_reward_timeout: int = 300

    asr_good: float = 28.0
    untarget_disagreement_recovery_ratio: float = 0.50
    # Optional report-only ADR pool for untargeted attacks. This pool never
    # controls the entropy switch, beta decay, loss, or checkpoint selection.
    untarget_eval_source: str = "disabled"
    untarget_eval_size: int = 2000
    untarget_eval_seed_offset: int = 40000
    untarget_eval_action_mode: str = "stochastic"
    untarget_eval_only: bool = False
    mmd_good: float = 0.08
    good_streak: int = 5
    beta_decay: float = 0.80
    beta_mmd_min: float = 0.30
    beta_vai_min: float = 0.10
    val_w_conv2: float = 0.5
    val_w_conv3: float = 1.0
    cons_beta: float = 0.5
    ent_floor_beta: float = 0.05
    value_huber_delta: float = 1.0
    stage1_max_weight: float = 3.0

    save_model: bool = True
    # Empty means use the currently selected attack flag.
    attacks: str = ""


REPAIR_METRIC_FIELDS = [
    "Experiment", "Attack", "AttackMetric", "AttackMetricGood",
    "AttackMetricGoodSource", "TriggerProtocol",
    "BufferSHA256", "TrainBufferSHA256", "GateBufferSHA256",
    "TrainReferenceSHA256", "GateReferenceSHA256",
    "Update", "Step", "Stage", "EntropyFilterEnabled", "InterceptEntropyThresh",
    "TrainBufferSize", "GateBufferSize", "MetricBufferRole",
    "ASR_Tr", "Ent_Tr", "ASR_Gate", "Ent_Gate",
    "AttackMetricActualTrain", "EntropyActualTrain",
    "AttackMetricActualTrainNumerator",
    "AttackMetricActualTrainDenominator",
    "AttackMetricActualGate", "EntropyActualGate",
    "AttackMetricActualGateNumerator",
    "AttackMetricActualGateDenominator", "ActualMetricProtocol",
    "AttackMetricEval", "AttackMetricEvalEntropy",
    "AttackMetricEvalNumerator",
    "AttackMetricEvalDenominator", "AttackEvalBufferRole",
    "AttackEvalBufferSHA256",
    "P_Ret", "P_Ret_Std", "P_Ret_N", "C_Ret", "C_Ret_Std", "C_Ret_N",
    "ReturnEvaluated", "EvalProtocol", "EvalSeedBase", "EvalMaxSteps", "EvalNoRewardTimeout",
    "GateEvalInterval", "ReturnEvalInterval",
    "L_PPO", "Penalty", "L_ENT", "L_MMD", "L_VAI",
    "Beta", "Beta_MMD", "Beta_VAI", "Streak", "BetaDecayCount", "LastBetaDecayUpdate",
    "StageSwitchMode",
    "CleanEntropyRef", "GlobalCleanEntropyRef", "InitialTriggerEntropy",
    "EntropyTarget", "EntropyTargetSource", "EntropyRecoveryDirection",
    "EntropySmooth", "EntropyRecovery", "EntropyGateStreak",
    "Stage1EntropyWindow", "Stage1EntropyPatience", "Stage1EntropyHysteresis",
    "Stage1MinUpdates", "Stage1MaxUpdates", "Stage1RecoveryRatio",
    "Stage2StartUpdate", "StageSwitchReason", "ThresholdReached", "Censored", "Stage1Outcome",
]


def batch_zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return (x - x.mean()) / (x.std(unbiased=False) + eps)


def weighted_huber(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, delta: float = 1.0) -> torch.Tensor:
    err = pred - target
    abs_err = err.abs()
    loss = torch.where(
        abs_err < delta,
        0.5 * err.pow(2),
        delta * (abs_err - 0.5 * delta)
    )
    return (weight * loss).mean()


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
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
        self.norm = 255.0

    def get_value(self, x):
        return self.critic(self.network(x / self.norm))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x / self.norm)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)

    def get_policy_logits(self, x):
        hidden = self.network(x / self.norm)
        return self.actor(hidden)


class MultiLayerLens(nn.Module):
    def __init__(self, feature_dims):
        super().__init__()
        self.critic_heads = nn.ModuleDict({
            name: nn.Linear(dim, 1) for name, dim in feature_dims.items()
        })

    def forward_critic(self, name, feat_flat):
        return self.critic_heads[name](feat_flat).view(-1)


def infer_attack_name(args: PipelineArgs) -> str:
    enabled = []
    if args.sn_outer: enabled.append("SN")
    if args.trojdrl: enabled.append("trojdrl")
    if args.untargeted: enabled.append("untarget")
    if args.badrl: enabled.append("badrl")
    if len(enabled) != 1:
        raise ValueError(
            f"Exactly one attack flag must be enabled; received {enabled or 'none'}."
        )
    if enabled[0] == "trojdrl":
        return "troj1"
    return enabled[0]


def parse_csv_list(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def build_attack_args(args: PipelineArgs, attack_name: str) -> PipelineArgs:
    attack_name = attack_name.strip().lower()
    aliases = {
        "sn": "SN",
        "sn_outer": "SN",
        "trojdrl": "troj1",
        "trojdrl1": "troj1",
        "untargeted": "untarget",
    }
    attack_name = aliases.get(attack_name, attack_name)

    if attack_name == "badrl":
        out = replace(
            args,
            sn_outer=False,
            trojdrl=False,
            untargeted=False,
            badrl=True,
            poisoned_model_path=args.poisoned_model_path or args.badrl_model_path,
            lens_model_path=args.lens_model_path or args.badrl_lens_model_path,
        )
    elif attack_name == "SN":
        out = replace(
            args,
            sn_outer=True,
            trojdrl=False,
            untargeted=False,
            badrl=False,
            poisoned_model_path=args.poisoned_model_path or args.sn_model_path,
            lens_model_path=args.lens_model_path or args.sn_lens_model_path,
        )
    elif attack_name == "troj1":
        out = replace(
            args,
            sn_outer=False,
            trojdrl=True,
            untargeted=False,
            badrl=False,
            rew_p=1.0,
            poisoned_model_path=args.poisoned_model_path or args.troj1_model_path,
            lens_model_path=args.lens_model_path or args.troj1_lens_model_path,
        )
    elif attack_name == "untarget":
        out = replace(
            args,
            sn_outer=False,
            trojdrl=False,
            untargeted=True,
            badrl=False,
            p_rate=0.001,
            untargeted_reward=1.0,
            poisoned_model_path=args.poisoned_model_path or args.untarget_model_path,
            lens_model_path=args.lens_model_path or args.untarget_lens_model_path,
        )
    else:
        raise ValueError("Unknown attack. Use one of: badrl, SN, troj1, untarget.")

    return out


def normalize_attack_args(args: PipelineArgs) -> PipelineArgs:
    return build_attack_args(args, infer_attack_name(args))


def selected_attack_names(args: PipelineArgs) -> List[str]:
    if not args.attacks.strip():
        return [infer_attack_name(args)]
    if args.attacks.strip().lower() == "all":
        return ["badrl", "SN", "troj1", "untarget"]
    return parse_csv_list(args.attacks)


def validate_pipeline_args(args: PipelineArgs) -> None:
    if args.safety or args.trade:
        raise ValueError("This repair pipeline supports Atari image agents only.")
    if not args.poisoned_model_path:
        raise ValueError("poisoned_model_path must be provided.")
    if not os.path.isfile(args.poisoned_model_path):
        raise FileNotFoundError(
            f"Poisoned model not found: {args.poisoned_model_path}"
        )
    if not args.lens_model_path:
        raise ValueError("lens_model_path must be provided.")
    if not os.path.isfile(args.lens_model_path):
        raise FileNotFoundError(f"Lens model not found: {args.lens_model_path}")
    if not args.output_root:
        raise ValueError("output_root cannot be empty.")
    if min(
        args.num_envs,
        args.calib_steps,
        args.phase2_max_env_steps,
        args.offline_buffer_size,
        args.repair_validation_size,
        args.total_timesteps,
        args.num_steps,
        args.minibatch_size,
        args.update_epochs,
    ) < 1:
        raise ValueError("Pipeline sizes and step counts must be positive.")
    if args.repair_split_gap_size < 0:
        raise ValueError("repair_split_gap_size must be non-negative.")
    if not 0.0 < args.calib_quantile < 1.0:
        raise ValueError("calib_quantile must be in (0, 1).")
    if not 0 < args.poison_duration_min <= args.poison_duration_max:
        raise ValueError("Poison durations must define a positive interval.")
    if not 0 <= args.cooldown_min <= args.cooldown_max:
        raise ValueError("Cooldown values must define a non-negative interval.")
    if args.lr <= 0.0:
        raise ValueError("lr must be positive.")
    if not 0.0 < args.p_rate <= 1.0:
        raise ValueError("p_rate must be in (0, 1].")
    if not 0.0 <= args.alpha <= 1.0 or args.rew_p <= 0.0:
        raise ValueError("alpha must be in [0, 1] and rew_p must be positive.")
    if args.intercept_entropy_thresh < 0.0:
        raise ValueError("intercept_entropy_thresh cannot be negative.")
    rollout_size = args.num_envs * args.num_steps
    if args.total_timesteps < rollout_size:
        raise ValueError(
            "total_timesteps must provide at least one complete PPO rollout."
        )


def build_poison_and_bufferman(args: PipelineArgs, device: torch.device):
    poison = None
    bufferman = None
    poison_batch = None
    if args.sn_outer:
        if args.safety:
            poison = SingleValuePoison([-1 - 16, -9 - 16, -13 - 16, -5 - 16], 1)
        else:
            pattern = patterns.Single_Stacked_Img_Pattern((4, 84, 84), (8, 8)).to(device)
            poison = ImagePoison(pattern, 0, 255)
        bufferman = BufferMan_Simple(poison, args.target_action, Discrete(-1 * args.rew_p, args.rew_p),
                                     p_rate=args.p_rate, alpha=args.alpha, simple=args.simple_select)
    if args.trojdrl or args.untargeted or args.badrl:
        if args.safety:
            poison_batch = SingleValuePoison([-1, -9], 1)
        elif args.trade:
            poison_batch = SingleValuePoison([-1], 1)
        else:
            pattern_batch = patterns.Stacked_Img_Pattern((1, 4, 84, 84), (8, 8)).to(device)
            poison_batch = ImagePoison(pattern_batch, 0, 255)
    return bufferman, poison_batch


class AttackInjector:
    def __init__(self, args: PipelineArgs, device: torch.device):
        self.args = args
        self.device = device
        self.attack_name = infer_attack_name(args)
        self.bufferman, self.poison_batch = build_poison_and_bufferman(args, device)

    def inject_full(self, obs: torch.Tensor) -> torch.Tensor:
        if self.args.sn_outer: return self.bufferman.trigger(obs)
        if self.args.trojdrl or self.args.untargeted or self.args.badrl: return self.poison_batch(obs)
        return obs


class NoRewardTimeoutEnv(gym.Wrapper):
    def __init__(self, env, timeout_steps=300):
        super().__init__(env)
        self.timeout_steps = timeout_steps
        self.steps_since_reward = 0

    def reset(self, **kwargs):
        self.steps_since_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        if reward > 0:
            self.steps_since_reward = 0
        else:
            self.steps_since_reward += 1
        if self.steps_since_reward >= self.timeout_steps and not (terminated or truncated):
            truncated = True
            info = dict(info)
            info["no_reward_timeout"] = True
        return obs, reward, terminated, truncated, info


def make_env(env_id, seed):
    def thunk():
        env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = NoRewardTimeoutEnv(env, timeout_steps=300)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_untarget_online_matched_eval_env(env_id, seed):
    """Match ppo.py's Atari wrapper stack; notably, no idle timeout."""
    def thunk():
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
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk


def make_eval_env(env_id, seed, no_reward_timeout):
    """Raw-reward evaluation without episodic-life, with an explicit idle timeout."""
    def thunk():
        env = gym.make(env_id)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = NoRewardTimeoutEnv(env, timeout_steps=no_reward_timeout)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk



def register_feature_hooks(feature_layers: Dict[str, nn.Module]):
    activations = {}
    handles = []
    for name, module in feature_layers.items():
        def hook(m, inp, out, name=name): activations[name] = out

        handles.append(module.register_forward_hook(hook))
    return activations, handles


@torch.no_grad()
def compute_raw_residuals(value, activations, lens, layer_order, device):
    B = value.shape[0]
    L = len(layer_order)
    resid_batch = torch.zeros(B, L, device=device)
    for j, name in enumerate(layer_order):
        feat = activations[name]
        feat_flat = feat.view(B, -1)
        v_l = lens.forward_critic(name, feat_flat)
        resid_batch[:, j] = value - v_l
    return resid_batch


@torch.no_grad()
def compute_two_sided_scores(value, activations, lens, layer_order, mu_R_t, std_R_t, weights_t, device):
    resid_batch = compute_raw_residuals(value, activations, lens, layer_order, device)
    z_scores = (resid_batch - mu_R_t.view(1, -1)) / (std_R_t.view(1, -1) + 1e-8)
    u_neg = torch.clamp(-z_scores, min=0.0)
    u_pos = torch.clamp(z_scores, min=0.0)
    S_neg = torch.sum(weights_t.view(1, -1) * (u_neg ** 2), dim=1)
    S_pos = torch.sum(weights_t.view(1, -1) * (u_pos ** 2), dim=1)
    return S_neg, S_pos


def compute_mmd_loss(clean_feat_c3, poison_feat_c3):
    clean_feat_flat = clean_feat_c3.view(clean_feat_c3.size(0), -1)
    poison_feat_flat = poison_feat_c3.view(poison_feat_c3.size(0), -1)
    return F.mse_loss(poison_feat_flat.mean(dim=0), clean_feat_flat.mean(dim=0))


def compute_lens_losses_code2_style(
        agent, lens, activations, mb_obs_poison, clean_entropy_mean,
        student_value, layer_order, mu_R_t, std_R_t, weights_t,
        thresh_neg, thresh_pos, args, device,
):
    poison_feat_c2 = activations["conv2"]
    poison_feat_c3 = activations["conv3"]
    poison_feat_c2_flat = poison_feat_c2.view(poison_feat_c2.size(0), -1)
    poison_feat_c3_flat = poison_feat_c3.view(poison_feat_c3.size(0), -1)

    with torch.no_grad():
        lens_value_c2 = lens.forward_critic("conv2", poison_feat_c2_flat).detach().view(-1)
        lens_value_c3 = lens.forward_critic("conv3", poison_feat_c3_flat).detach().view(-1)

        score_neg, score_pos = compute_two_sided_scores(
            student_value.detach(), activations, lens, layer_order,
            mu_R_t, std_R_t, weights_t, device,
        )
        w_neg = torch.clamp((score_neg - thresh_neg) / (thresh_neg + 1e-8), min=0.0)
        w_pos = torch.clamp((score_pos - thresh_pos) / (thresh_pos + 1e-8), min=0.0)
        w_excess = torch.max(w_neg, w_pos)
        sample_w = 1.0 + torch.clamp(w_excess, max=args.stage1_max_weight - 1.0)

    z_student = batch_zscore(student_value)
    z_lens_c2 = batch_zscore(lens_value_c2)
    z_lens_c3 = batch_zscore(lens_value_c3)

    loss_value_c2 = weighted_huber(z_student, z_lens_c2, sample_w, delta=args.value_huber_delta)
    loss_value_c3 = weighted_huber(z_student, z_lens_c3, sample_w, delta=args.value_huber_delta)

    delta_s_c2 = z_student - z_lens_c2
    delta_s_c3 = z_student - z_lens_c3
    loss_cons = weighted_huber(delta_s_c2, delta_s_c3, sample_w, delta=args.value_huber_delta)

    hidden = agent.network(mb_obs_poison / agent.norm)
    student_logits = agent.actor(hidden)
    poison_probs = F.softmax(student_logits, dim=-1)
    poison_log_probs = F.log_softmax(student_logits, dim=-1)
    poison_entropy = -(poison_probs * poison_log_probs).sum(dim=-1)
    if args.untargeted:
        entropy_alignment = F.smooth_l1_loss(
            poison_entropy,
            clean_entropy_mean.expand_as(poison_entropy),
            reduction="none",
        )
    else:
        entropy_alignment = F.relu(clean_entropy_mean - poison_entropy)
    loss_ent_floor = (sample_w * entropy_alignment).mean()

    return {
        "loss_value_c2": loss_value_c2,
        "loss_value_c3": loss_value_c3,
        "loss_cons": loss_cons,
        "loss_ent_floor": loss_ent_floor,
    }


def weighted_distill_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor, sample_weight: torch.Tensor,
                        T: float = 2.0):
    with torch.no_grad():
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    per_sample = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1) * (T ** 2)
    weight = sample_weight.detach()
    weight = weight / (weight.mean() + 1e-8)
    loss = (weight * per_sample).mean()
    return loss, per_sample.detach(), weight.detach()


def build_vai_weights(raw_adv: torch.Tensor, teacher_value: torch.Tensor, adv_coef: float, value_coef: float,
                      max_weight: float):
    adv_pos = torch.clamp(raw_adv.detach(), min=0.0)
    if adv_pos.mean() > 0:
        adv_term = adv_pos / (adv_pos.mean() + 1e-8)
    else:
        adv_term = torch.zeros_like(adv_pos)

    value_z = batch_zscore(teacher_value.detach())
    value_pos = torch.clamp(value_z, min=0.0)
    if value_pos.mean() > 0:
        value_term = value_pos / (value_pos.mean() + 1e-8)
    else:
        value_term = torch.zeros_like(value_pos)

    weight = 1.0 + adv_coef * adv_term + value_coef * value_term
    weight = torch.clamp(weight, min=1.0, max=max_weight)
    return weight, adv_term.detach(), value_term.detach()


def compute_vai_loss(agent: Agent, anchor_model: Agent, mb_obs_clean: torch.Tensor, mb_adv_raw: torch.Tensor,
                     args: PipelineArgs):
    with torch.no_grad():
        anchor_logits_clean = anchor_model.get_policy_logits(mb_obs_clean)
        anchor_value_clean = anchor_model.get_value(mb_obs_clean).view(-1)
    student_logits_clean = agent.get_policy_logits(mb_obs_clean)
    vai_weight, adv_term, value_term = build_vai_weights(
        raw_adv=mb_adv_raw, teacher_value=anchor_value_clean,
        adv_coef=args.vai_adv_coef, value_coef=args.vai_value_coef, max_weight=args.vai_max_weight,
    )
    loss_vai, _, used_weight = weighted_distill_kl(
        student_logits=student_logits_clean, teacher_logits=anchor_logits_clean,
        sample_weight=vai_weight, T=args.vai_temp,
    )
    return loss_vai, used_weight, adv_term, value_term


def attack_metric_name(args: PipelineArgs) -> str:
    return "action_disagreement" if args.untargeted else "target_action_rate"


@torch.no_grad()
def evaluate_clean_entropy(agent, clean_buffer: torch.Tensor, args: PipelineArgs) -> float:
    was_training = agent.training
    agent.eval()
    entropy_sum = 0.0
    n = clean_buffer.shape[0]
    if n == 0:
        raise ValueError("Cannot evaluate clean entropy on an empty reference buffer.")
    for start in range(0, n, args.eval_batch_size):
        logits = agent.get_policy_logits(clean_buffer[start:start + args.eval_batch_size])
        entropy_sum += float(Categorical(logits=logits).entropy().sum().item())
    if was_training:
        agent.train()
    return entropy_sum / n


@torch.no_grad()
def evaluate_attack_metric_entropy(
        agent, clean_buffer: torch.Tensor, injector: AttackInjector,
        args: PipelineArgs, device: torch.device,
        return_counts: bool = False,
):
    """Evaluate a forced trigger on paired clean states without sampling noise."""
    was_training = agent.training
    agent.eval()
    n = clean_buffer.shape[0]
    if n == 0:
        raise ValueError("Cannot evaluate attack metrics on an empty reference buffer.")

    attack_successes = 0.0
    entropy_sum = 0.0
    for start in range(0, n, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n)
        clean_obs = clean_buffer[start:end].to(
            device=device,
            dtype=torch.float32,
        )
        mask_all = torch.ones(clean_obs.shape[0], device=device, dtype=torch.bool)
        triggered_obs = apply_trigger_weak(clean_obs, mask_all, injector)
        triggered_logits = agent.get_policy_logits(triggered_obs)
        if args.untargeted:
            clean_logits = agent.get_policy_logits(clean_obs)
            attack_successes += float(
                (clean_logits.argmax(dim=-1) != triggered_logits.argmax(dim=-1)).sum().item()
            )
        else:
            attack_successes += float(
                (triggered_logits.argmax(dim=-1) == args.target_action).sum().item()
            )
        entropy_sum += float(Categorical(logits=triggered_logits).entropy().sum().item())

    if was_training:
        agent.train()
    metric_percent = 100.0 * attack_successes / n
    mean_entropy = entropy_sum / n
    if return_counts:
        return metric_percent, mean_entropy, int(attack_successes), int(n)
    return metric_percent, mean_entropy


@torch.no_grad()
def evaluate_actual_state_metric_entropy(
        agent, actual_buffer: torch.Tensor, reference_buffer: torch.Tensor,
        args: PipelineArgs, device: torch.device,
        return_counts: bool = False,
):
    """Evaluate every captured alarm exactly as observed, without reinjection."""
    if actual_buffer.shape[0] != reference_buffer.shape[0]:
        raise ValueError(
            "Actual and reference buffers must be index-aligned."
        )
    n = actual_buffer.shape[0]
    if n == 0:
        raise ValueError(
            "Cannot evaluate detected-state metrics on an empty buffer."
        )

    was_training = agent.training
    agent.eval()
    attack_successes = 0
    entropy_sum = 0.0
    for start in range(0, n, args.eval_batch_size):
        end = min(start + args.eval_batch_size, n)
        actual_obs = actual_buffer[start:end].to(
            device=device,
            dtype=torch.float32,
        )
        actual_logits = agent.get_policy_logits(actual_obs)
        if args.untargeted:
            reference_obs = reference_buffer[start:end].to(
                device=device,
                dtype=torch.float32,
            )
            reference_logits = agent.get_policy_logits(reference_obs)
            attack_successes += int(
                (
                    reference_logits.argmax(dim=-1)
                    != actual_logits.argmax(dim=-1)
                ).sum().item()
            )
        else:
            attack_successes += int(
                (
                    actual_logits.argmax(dim=-1)
                    == args.target_action
                ).sum().item()
            )
        entropy_sum += float(
            Categorical(logits=actual_logits).entropy().sum().item()
        )

    if was_training:
        agent.train()
    metric_percent = 100.0 * attack_successes / n
    mean_entropy = entropy_sum / n
    if return_counts:
        return metric_percent, mean_entropy, int(attack_successes), int(n)
    return metric_percent, mean_entropy


@torch.no_grad()
def collect_untarget_eval_reference_buffer(
        agent, args: PipelineArgs, device: torch.device,
        logger: logging.Logger,
) -> torch.Tensor:
    """Collect fixed clean states for report-only paired ADR evaluation."""
    eval_seed = args.seed + args.untarget_eval_seed_offset
    cpu_rng_state = torch.random.get_rng_state()
    numpy_rng_state = np.random.get_state()
    python_rng_state = random.getstate()
    cuda_rng_states = (
        torch.cuda.get_rng_state_all()
        if device.type == "cuda"
        else None
    )
    was_training = agent.training
    action_mode = args.untarget_eval_action_mode.strip().lower()
    eval_envs = None
    chunks = []
    collected = 0
    try:
        eval_envs = gym.vector.SyncVectorEnv([
            make_untarget_online_matched_eval_env(
                args.env_id,
                eval_seed + i,
            )
            for i in range(args.num_envs)
        ])
        torch.manual_seed(eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(eval_seed)
        agent.eval()
        observation, _ = eval_envs.reset(seed=eval_seed)
        observation_t = torch.tensor(
            observation,
            dtype=torch.float32,
            device=device,
        )
        while collected < args.untarget_eval_size:
            take = min(
                args.untarget_eval_size - collected,
                observation_t.shape[0],
            )
            snapshot = observation_t[:take].detach()
            if (
                float(snapshot.min().item()) < 0.0
                or float(snapshot.max().item()) > 255.0
                or not torch.equal(snapshot, snapshot.round())
            ):
                raise ValueError(
                    "Untarget ADR eval expects integer-valued Atari "
                    "observations in [0, 255]."
                )
            chunks.append(
                snapshot
                .to(device="cpu", dtype=torch.uint8)
                .clone()
            )
            collected += take
            if action_mode == "greedy":
                action = agent.get_policy_logits(
                    observation_t
                ).argmax(dim=-1)
            else:
                action, _, _, _ = agent.get_action_and_value(observation_t)
            next_observation, _, _, _, _ = eval_envs.step(
                action.cpu().numpy()
            )
            observation_t = torch.tensor(
                next_observation,
                dtype=torch.float32,
                device=device,
            )
    finally:
        if eval_envs is not None:
            eval_envs.close()
        torch.random.set_rng_state(cpu_rng_state)
        np.random.set_state(numpy_rng_state)
        random.setstate(python_rng_state)
        if cuda_rng_states is not None:
            torch.cuda.set_rng_state_all(cuda_rng_states)
        if was_training:
            agent.train()

    reference_buffer = torch.cat(chunks, dim=0)
    logger.info(
        "[Untarget ADR Eval] Collected fixed independent clean states | "
        f"N={reference_buffer.shape[0]} | seed={eval_seed} | "
        f"parallel_env_seeds={args.num_envs} | policy={action_mode} | "
        "wrappers=ppo_atari_no_idle_timeout | "
        f"SHA256={tensor_sha256(reference_buffer)}"
    )
    return reference_buffer


def validate_stage_transition_args(args: PipelineArgs, num_actions: int) -> str:
    if num_actions < 1:
        raise ValueError("The environment must expose at least one action.")
    if not args.untargeted and not 0 <= args.target_action < num_actions:
        raise ValueError(
            f"target_action={args.target_action} is outside [0, {num_actions})."
        )
    mode = args.stage1_switch_mode.strip().lower()
    if mode not in {"entropy", "fixed"}:
        raise ValueError("stage1_switch_mode must be either 'entropy' or 'fixed'.")
    if (args.eval_interval < 1 or args.return_eval_interval < 1
            or args.eval_batch_size < 1 or args.eval_episodes < 1):
        raise ValueError(
            "eval_interval, return_eval_interval, eval_batch_size, and eval_episodes must be positive."
        )
    if args.eval_max_steps < 1 or args.eval_no_reward_timeout < 1:
        raise ValueError("eval_max_steps and eval_no_reward_timeout must be positive.")
    if args.repair_validation_size < 1 or args.repair_split_gap_size < 0:
        raise ValueError("repair_validation_size must be positive and repair_split_gap_size non-negative.")
    if args.good_streak < 1 or not 0.0 < args.beta_decay <= 1.0:
        raise ValueError("good_streak must be positive and beta_decay must be in (0, 1].")
    if not 0.0 <= args.beta_mmd_min <= args.beta_feat_mmd:
        raise ValueError("beta_mmd_min must be in [0, beta_feat_mmd].")
    if not 0.0 <= args.beta_vai_min <= args.beta_vai:
        raise ValueError("beta_vai_min must be in [0, beta_vai].")
    if args.stage1_fixed_updates < 1:
        raise ValueError("stage1_fixed_updates must be at least 1.")
    if args.stage1_entropy_window < 1 or args.stage1_entropy_patience < 1:
        raise ValueError("Entropy window and patience must be at least 1.")
    if args.stage1_min_updates < 1:
        raise ValueError("stage1_min_updates must be at least 1.")
    if args.stage1_max_updates < 0:
        raise ValueError("stage1_max_updates must be 0 or a positive update count.")
    if 0 < args.stage1_max_updates < args.stage1_min_updates:
        raise ValueError("stage1_max_updates must be 0 or >= stage1_min_updates.")
    if not 0.0 <= args.stage1_entropy_recovery_ratio <= 1.0:
        raise ValueError("stage1_entropy_recovery_ratio must be in [0, 1].")
    if not 0.0 <= args.untarget_disagreement_recovery_ratio <= 1.0:
        raise ValueError("untarget_disagreement_recovery_ratio must be in [0, 1].")
    untarget_eval_source = args.untarget_eval_source.strip().lower()
    if untarget_eval_source not in {"disabled", "independent_clean"}:
        raise ValueError(
            "untarget_eval_source must be 'disabled' or "
            "'independent_clean'."
        )
    if args.untarget_eval_size < 1:
        raise ValueError("untarget_eval_size must be positive.")
    if args.untarget_eval_seed_offset < 1:
        raise ValueError("untarget_eval_seed_offset must be positive.")
    if args.untarget_eval_action_mode.strip().lower() not in {
        "greedy",
        "stochastic",
    }:
        raise ValueError(
            "untarget_eval_action_mode must be 'greedy' or 'stochastic'."
        )
    if args.untarget_eval_only and (
        not args.untargeted
        or untarget_eval_source != "independent_clean"
    ):
        raise ValueError(
            "untarget_eval_only requires an untargeted attack and "
            "untarget_eval_source='independent_clean'."
        )
    if args.stage1_entropy_hysteresis < 0.0:
        raise ValueError("stage1_entropy_hysteresis cannot be negative.")

    max_entropy = float(np.log(max(1, num_actions)))
    if args.stage1_entropy_threshold > max_entropy + 1e-6:
        raise ValueError(
            f"stage1_entropy_threshold={args.stage1_entropy_threshold:.4f} exceeds "
            f"the maximum categorical entropy log(|A|)={max_entropy:.4f}."
        )
    return mode


def derive_stage1_entropy_target(
        initial_trigger_entropy: float,
        clean_entropy_reference: float,
        args: PipelineArgs,
):
    if args.stage1_entropy_threshold >= 0.0:
        return float(args.stage1_entropy_threshold), "absolute"
    target = initial_trigger_entropy + args.stage1_entropy_recovery_ratio * (
        clean_entropy_reference - initial_trigger_entropy
    )
    return float(target), "clean_interpolation"


def entropy_recovery_fraction(
        entropy_smooth: float,
        initial_trigger_entropy: float,
        clean_entropy_reference: float,
) -> float:
    entropy_shift = clean_entropy_reference - initial_trigger_entropy
    if abs(entropy_shift) <= 1e-8:
        return 1.0
    return float((entropy_smooth - initial_trigger_entropy) / entropy_shift)


def update_entropy_gate(
        update: int,
        entropy_smooth: float,
        entropy_target: float,
        recovery_direction: float,
        current_streak: int,
        args: PipelineArgs,
):
    if recovery_direction >= 0.0:
        target_margin = entropy_smooth - entropy_target
    else:
        target_margin = entropy_target - entropy_smooth
    ready = (
        update >= args.stage1_min_updates
        and target_margin + 1e-8 >= 0.0
    )
    if ready:
        next_streak = current_streak + 1
    elif target_margin < -args.stage1_entropy_hysteresis:
        next_streak = 0
    else:
        next_streak = current_streak
    return ready, next_streak, next_streak >= args.stage1_entropy_patience


@torch.no_grad()
def evaluate_fixed_seed_return(
        agent, eval_envs, injector, args: PipelineArgs, device: torch.device,
        eval_seed: int, triggered: bool,
):
    """Run exactly one raw-reward episode per vector environment."""
    was_training = agent.training
    agent.eval()
    try:
        obs_np, _ = eval_envs.reset(seed=eval_seed)
        obs = torch.tensor(obs_np, dtype=torch.float32, device=device)
        completed = np.zeros(args.eval_episodes, dtype=np.bool_)
        returns = np.zeros(args.eval_episodes, dtype=np.float64)

        for _ in range(args.eval_max_steps):
            policy_obs = obs
            if triggered:
                mask_all = torch.ones(obs.shape[0], device=device, dtype=torch.bool)
                policy_obs = apply_trigger_weak(obs, mask_all, injector)
            action = agent.get_policy_logits(policy_obs).argmax(dim=-1)
            next_obs, reward, terminations, truncations, _ = eval_envs.step(action.cpu().numpy())
            active = ~completed
            returns[active] += np.asarray(reward, dtype=np.float64)[active]
            just_finished = active & np.logical_or(terminations, truncations)
            completed[just_finished] = True
            obs = torch.tensor(next_obs, dtype=torch.float32, device=device)
            if bool(completed.all()):
                break
    finally:
        if was_training:
            agent.train()
    if not bool(completed.all()):
        missing = np.flatnonzero(~completed).tolist()
        raise RuntimeError(
            f"Evaluation exceeded eval_max_steps={args.eval_max_steps}; unfinished env indices={missing}."
        )
    return float(returns.mean()), float(returns.std()), int(returns.size)


def apply_trigger_weak(obs: torch.Tensor, mask_t: torch.Tensor, injector: AttackInjector) -> torch.Tensor:
    if not mask_t.any(): return obs
    poisoned_full = injector.inject_full(obs)
    out = obs.clone()
    idx = torch.where(mask_t)[0]
    out[idx, -1] = poisoned_full[idx, -1]
    return out


def reset_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def tensor_sha256(tensor: torch.Tensor) -> str:
    cpu_tensor = tensor.detach().cpu().contiguous()
    return hashlib.sha256(cpu_tensor.numpy().tobytes()).hexdigest()


def file_sha256(path: str, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def split_repair_buffer(
        buffer: torch.Tensor, labels: np.ndarray, train_size: int,
        validation_size: int, gap_size: int,
):
    """Use later samples for gating, separated by a temporal exclusion gap."""
    total = train_size + gap_size + validation_size
    if train_size < 1 or validation_size < 1 or gap_size < 0:
        raise ValueError("Repair train/validation sizes must be positive and gap_size non-negative.")
    if buffer.shape[0] < total or len(labels) < total:
        raise ValueError(
            f"Repair pool has {buffer.shape[0]} samples but {total} are required."
        )
    labels = np.asarray(labels[:total], dtype=np.int64)
    validation_start = train_size + gap_size
    return (
        buffer[:train_size].clone(),
        buffer[validation_start:validation_start + validation_size].clone(),
        labels[:train_size],
        labels[validation_start:validation_start + validation_size],
    )


def label_summary(labels: np.ndarray):
    labels = np.asarray(labels, dtype=np.int64)
    tp = int(labels.sum())
    fp = int(labels.size - tp)
    return tp, fp, tp / max(1, tp + fp)


def write_final_summary(rows: List[Dict[str, object]], summary_file: str):
    if not rows:
        return
    fieldnames = sorted(set().union(*[set(row.keys()) for row in rows]))
    with open(summary_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def close_logger_handlers(logger: logging.Logger):
    for handler in logger.handlers[:]:
        handler.flush()
        if isinstance(handler, logging.FileHandler):
            handler.close()
        logger.removeHandler(handler)


def run_phase3_sanitization_experiment(
        exp_name: str, train_buffer: torch.Tensor, gate_buffer: torch.Tensor,
        train_reference_buffer: torch.Tensor, gate_reference_buffer: torch.Tensor,
        lens: MultiLayerLens, layer_order, mu_R_t: torch.Tensor, std_R_t: torch.Tensor,
        weights_t: torch.Tensor, thresh_neg: float, thresh_pos: float,
        injector: AttackInjector, args: PipelineArgs, device: torch.device,
        logger: logging.Logger, csv_file: str, model_dir: str, base_seed: int,
        clean_entropy_reference: float,
        train_buffer_hash: str = "", gate_buffer_hash: str = "",
        train_reference_hash: str = "", gate_reference_hash: str = "",
        fixed_eval_seed_base: Optional[int] = None,
        report_reference_buffer: Optional[torch.Tensor] = None,
        report_reference_hash: str = "",
        report_buffer_role: str = "held_out_gate",
):
    if train_buffer.shape[0] != train_reference_buffer.shape[0]:
        raise ValueError("train_buffer and train_reference_buffer must be aligned.")
    if gate_buffer.shape[0] != gate_reference_buffer.shape[0]:
        raise ValueError("gate_buffer and gate_reference_buffer must be aligned.")
    if report_reference_buffer is None:
        report_reference_buffer = gate_reference_buffer
        report_reference_hash = report_reference_hash or gate_reference_hash
        report_buffer_role = "held_out_gate"
    if report_reference_buffer.shape[0] < 1:
        raise ValueError("report_reference_buffer cannot be empty.")
    has_independent_report = (
        args.untargeted
        and report_buffer_role != "held_out_gate"
    )
    reset_all_seeds(base_seed)

    train_envs = gym.vector.SyncVectorEnv([make_env(args.env_id, base_seed + i) for i in range(args.num_envs)])
    eval_envs = gym.vector.SyncVectorEnv(
        [make_eval_env(args.env_id, base_seed + 100 + i, args.eval_no_reward_timeout)
         for i in range(args.eval_episodes)])

    agent = Agent(train_envs).to(device)
    agent.load_state_dict(torch.load(args.poisoned_model_path, map_location=device))
    agent.train()

    anchor_model = Agent(train_envs).to(device)
    anchor_model.load_state_dict(torch.load(args.poisoned_model_path, map_location=device))
    anchor_model.eval()
    for p in anchor_model.parameters(): p.requires_grad = False

    feature_layers_dict = {"conv1": agent.network[0], "conv2": agent.network[2], "conv3": agent.network[4]}
    activations, handles = register_feature_hooks(feature_layers_dict)

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    obs_buf = torch.zeros((args.num_steps, args.num_envs) + train_envs.single_observation_space.shape, device=device)
    actions_buf = torch.zeros((args.num_steps, args.num_envs) + train_envs.single_action_space.shape, device=device)
    logprobs_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    rewards_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    dones_buf = torch.zeros((args.num_steps, args.num_envs), device=device)
    values_buf = torch.zeros((args.num_steps, args.num_envs), device=device)

    global_step = 0
    next_obs, _ = train_envs.reset(seed=base_seed)
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)
    next_done = torch.zeros(args.num_envs, device=device)

    num_updates = args.total_timesteps // (args.num_envs * args.num_steps)
    if num_updates < 1:
        raise ValueError("total_timesteps must provide at least one PPO update.")
    stage1_switch_mode = validate_stage_transition_args(
        args,
        train_envs.single_action_space.n,
    )
    current_beta = args.beta_lens_kl
    beta_mmd = args.beta_feat_mmd
    current_beta_vai = args.beta_vai
    good_counter = 0
    beta_decay_count = 0
    last_beta_decay_update = ""
    eval_seed_base = fixed_eval_seed_base if fixed_eval_seed_base is not None else base_seed + 20000

    logger.info("\n" + "=" * 145)
    logger.info(
        f"[Phase 3 - {exp_name}] Two-Stage Sanitization starts. "
        f"Train buffer={train_buffer.shape[0]}, held-out gate buffer={gate_buffer.shape[0]}"
    )
    logger.info(
        f"[Phase 3 - {exp_name}] Full repair uses Lens, Entropy, MMD, and VAI."
    )
    logger.info("=" * 145)
    if has_independent_report:
        logger.info(
            "Upd | Step     | ADR_E  | ADR_G  | Act_G  | Ent_G  | P_Ret  | "
            "C_Ret  | L_PPO   | Penalty | L_MMD   | L_VAI   | Beta    | "
            "B_MMD   | B_VAI   | Streak"
        )
    else:
        logger.info(
            "Upd | Step     | Atk_G  | Act_G  | Ent_G  | P_Ret  | C_Ret  | "
            "L_PPO   | Penalty | L_MMD   | L_VAI   | Beta    | B_MMD   | "
            "B_VAI   | Streak"
        )
    logger.info("=" * 145)

    metric_train_asr_0, metric_train_ent_0 = evaluate_attack_metric_entropy(
        agent, train_reference_buffer, injector, args, device,
    )
    train_asr_0, train_ent_0 = evaluate_attack_metric_entropy(
        agent, gate_reference_buffer, injector, args, device,
    )
    # Actual-state metrics are report-only and never control repair updates.
    (
        actual_train_metric_0,
        actual_train_entropy_0,
        actual_train_numerator_0,
        actual_train_denominator_0,
    ) = evaluate_actual_state_metric_entropy(
        agent,
        train_buffer,
        train_reference_buffer,
        args,
        device,
        return_counts=True,
    )
    (
        actual_gate_metric_0,
        actual_gate_entropy_0,
        actual_gate_numerator_0,
        actual_gate_denominator_0,
    ) = evaluate_actual_state_metric_entropy(
        agent,
        gate_buffer,
        gate_reference_buffer,
        args,
        device,
        return_counts=True,
    )
    if has_independent_report:
        (
            report_attack_metric_0,
            report_attack_entropy_0,
            report_attack_numerator_0,
            report_attack_denominator_0,
        ) = evaluate_attack_metric_entropy(
            agent,
            report_reference_buffer,
            injector,
            args,
            device,
            return_counts=True,
        )
    else:
        report_attack_metric_0 = ""
        report_attack_entropy_0 = ""
        report_attack_denominator_0 = ""
        report_attack_numerator_0 = ""
    cur_p_ret_0, cur_p_ret_std_0, cur_p_ret_n_0 = evaluate_fixed_seed_return(
        agent, eval_envs, injector, args, device, eval_seed_base, triggered=True,
    )
    avg_ret_0, avg_ret_std_0, avg_ret_n_0 = evaluate_fixed_seed_return(
        agent, eval_envs, injector, args, device, eval_seed_base, triggered=False,
    )

    global_clean_entropy_reference = clean_entropy_reference
    clean_entropy_reference = evaluate_clean_entropy(
        anchor_model, gate_reference_buffer, args,
    )
    entropy_target, entropy_target_source = derive_stage1_entropy_target(
        initial_trigger_entropy=train_ent_0,
        clean_entropy_reference=clean_entropy_reference,
        args=args,
    )
    entropy_recovery_direction = 1.0 if entropy_target >= train_ent_0 else -1.0
    entropy_recovery_direction_name = (
        "increase" if entropy_recovery_direction > 0.0 else "decrease"
    )
    if args.untargeted:
        attack_metric_good = train_asr_0 * (1.0 - args.untarget_disagreement_recovery_ratio)
        attack_metric_good_source = "initial_disagreement_reduction"
    else:
        attack_metric_good = args.asr_good
        attack_metric_good_source = "absolute"
    actual_metric_protocol = (
        "all_captured_inputs_paired_action_disagreement_no_retrigger"
        if args.untargeted
        else "all_captured_inputs_target_action_rate_no_retrigger"
    )
    entropy_history = [train_ent_0]
    entropy_smooth = train_ent_0
    entropy_recovery = entropy_recovery_fraction(
        entropy_smooth,
        train_ent_0,
        clean_entropy_reference,
    )
    entropy_gate_streak = 0
    stage1_active = True
    stage2_start_update = None
    stage_switch_reason = ""
    threshold_reached = 0
    censored = 0
    stage1_outcome = "in_progress"

    logger.info(
        f"[{exp_name}] Stage switch mode={stage1_switch_mode} | "
        f"H0(gate)={train_ent_0:.4f} | "
        f"Hclean={clean_entropy_reference:.4f} | Htarget={entropy_target:.4f} "
        f"({entropy_target_source}, {entropy_recovery_direction_name}) | "
        f"metric={attack_metric_name(args)}, good<={attack_metric_good:.2f} | "
        f"window={args.stage1_entropy_window} | "
        f"patience={args.stage1_entropy_patience} | min_updates={args.stage1_min_updates} | "
        f"max_updates={args.stage1_max_updates or 'disabled'}"
    )
    if has_independent_report:
        logger.info(
            f"[{exp_name}][ADR audit] report={report_buffer_role} | "
            f"ADR_E={report_attack_metric_0:.2f}% "
            f"({report_attack_numerator_0}/"
            f"{report_attack_denominator_0}) | "
            f"ADR_G={train_asr_0:.2f}% "
            f"(N={gate_reference_buffer.shape[0]})"
        )
    logger.info(
        f"[{exp_name}][Actual-state audit] "
        f"train={actual_train_metric_0:.2f}% "
        f"({actual_train_numerator_0}/{actual_train_denominator_0}) | "
        f"gate={actual_gate_metric_0:.2f}% "
        f"({actual_gate_numerator_0}/{actual_gate_denominator_0}) | "
        f"protocol={actual_metric_protocol}"
    )
    if has_independent_report:
        logger.info(
            f"{0:<3} | {0:<8} | {report_attack_metric_0:<6.1f} | "
            f"{train_asr_0:<6.1f} | {actual_gate_metric_0:<6.1f} | "
            f"{train_ent_0:<6.2f} | "
            f"{cur_p_ret_0:<6.1f} | {avg_ret_0:<6.1f} | "
            f"{0.0:<7.4f} | {0.0:<7.4f} | {0.0:<7.4f} | "
            f"{0.0:<7.4f} | {current_beta:<7.2f} | "
            f"{beta_mmd:<7.2f} | {current_beta_vai:<7.2f} | 0"
        )
    else:
        logger.info(
            f"{0:<3} | {0:<8} | {train_asr_0:<6.1f} | "
            f"{actual_gate_metric_0:<6.1f} | {train_ent_0:<6.2f} | "
            f"{cur_p_ret_0:<6.1f} | "
            f"{avg_ret_0:<6.1f} | {0.0:<7.4f} | {0.0:<7.4f} | "
            f"{0.0:<7.4f} | {0.0:<7.4f} | {current_beta:<7.2f} | "
            f"{beta_mmd:<7.2f} | {current_beta_vai:<7.2f} | 0"
        )

    row_0 = {
        "Experiment": exp_name,
        "Attack": injector.attack_name,
        "AttackMetric": attack_metric_name(args),
        "AttackMetricGood": attack_metric_good,
        "AttackMetricGoodSource": attack_metric_good_source,
        "TriggerProtocol": "last_frame_only",
        "BufferSHA256": train_buffer_hash,
        "TrainBufferSHA256": train_buffer_hash, "GateBufferSHA256": gate_buffer_hash,
        "TrainReferenceSHA256": train_reference_hash,
        "GateReferenceSHA256": gate_reference_hash,
        "Update": 0, "Step": 0, "Stage": "init",
        "EntropyFilterEnabled": int(args.use_entropy_filter),
        "InterceptEntropyThresh": args.intercept_entropy_thresh,
        "TrainBufferSize": int(train_buffer.shape[0]), "GateBufferSize": int(gate_buffer.shape[0]),
        "MetricBufferRole": "held_out_gate",
        "ASR_Tr": metric_train_asr_0, "Ent_Tr": metric_train_ent_0,
        "ASR_Gate": train_asr_0, "Ent_Gate": train_ent_0,
        "AttackMetricActualTrain": actual_train_metric_0,
        "EntropyActualTrain": actual_train_entropy_0,
        "AttackMetricActualTrainNumerator": actual_train_numerator_0,
        "AttackMetricActualTrainDenominator": actual_train_denominator_0,
        "AttackMetricActualGate": actual_gate_metric_0,
        "EntropyActualGate": actual_gate_entropy_0,
        "AttackMetricActualGateNumerator": actual_gate_numerator_0,
        "AttackMetricActualGateDenominator": actual_gate_denominator_0,
        "ActualMetricProtocol": actual_metric_protocol,
        "AttackMetricEval": report_attack_metric_0,
        "AttackMetricEvalEntropy": report_attack_entropy_0,
        "AttackMetricEvalNumerator": report_attack_numerator_0,
        "AttackMetricEvalDenominator": report_attack_denominator_0,
        "AttackEvalBufferRole": report_buffer_role,
        "AttackEvalBufferSHA256": report_reference_hash,
        "P_Ret": cur_p_ret_0, "P_Ret_Std": cur_p_ret_std_0, "P_Ret_N": cur_p_ret_n_0,
        "C_Ret": avg_ret_0, "C_Ret_Std": avg_ret_std_0, "C_Ret_N": avg_ret_n_0,
        "ReturnEvaluated": 1,
        "EvalProtocol": "raw_reward_no_life_idle_truncation_fixed_seed_v2", "EvalSeedBase": eval_seed_base,
        "EvalMaxSteps": args.eval_max_steps, "EvalNoRewardTimeout": args.eval_no_reward_timeout,
        "GateEvalInterval": args.eval_interval, "ReturnEvalInterval": args.return_eval_interval,
        "L_PPO": 0.0,
        "Penalty": 0.0, "L_ENT": 0.0, "L_MMD": 0.0, "L_VAI": 0.0,
        "Beta": current_beta, "Beta_MMD": beta_mmd, "Beta_VAI": current_beta_vai, "Streak": 0,
        "BetaDecayCount": beta_decay_count, "LastBetaDecayUpdate": last_beta_decay_update,
        "StageSwitchMode": stage1_switch_mode,
        "CleanEntropyRef": clean_entropy_reference,
        "GlobalCleanEntropyRef": global_clean_entropy_reference,
        "InitialTriggerEntropy": train_ent_0,
        "EntropyTarget": entropy_target,
        "EntropyTargetSource": entropy_target_source,
        "EntropyRecoveryDirection": entropy_recovery_direction_name,
        "EntropySmooth": entropy_smooth,
        "EntropyRecovery": entropy_recovery,
        "EntropyGateStreak": entropy_gate_streak,
        "Stage1EntropyWindow": args.stage1_entropy_window,
        "Stage1EntropyPatience": args.stage1_entropy_patience,
        "Stage1EntropyHysteresis": args.stage1_entropy_hysteresis,
        "Stage1MinUpdates": args.stage1_min_updates,
        "Stage1MaxUpdates": args.stage1_max_updates,
        "Stage1RecoveryRatio": args.stage1_entropy_recovery_ratio,
        "Stage2StartUpdate": "",
        "StageSwitchReason": "",
        "ThresholdReached": threshold_reached, "Censored": censored,
        "Stage1Outcome": stage1_outcome,
    }
    final_row = row_0.copy()
    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPAIR_METRIC_FIELDS)
        writer.writerow(row_0)

    for update in range(1, num_updates + 1):
        stage_used = "stage1" if stage1_active else "stage2"

        for step in range(args.num_steps):
            global_step += args.num_envs
            obs_buf[step] = next_obs_t
            dones_buf[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs_t)
                values_buf[step] = value.flatten()
            actions_buf[step] = action
            logprobs_buf[step] = logprob

            next_obs_np, reward, terminations, truncations, _ = train_envs.step(action.cpu().numpy())
            rewards_buf[step] = torch.tensor(reward, device=device).view(-1)
            next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
            next_done = torch.tensor(np.logical_or(terminations, truncations), device=device, dtype=torch.float32)

        with torch.no_grad():
            next_value = agent.get_value(next_obs_t).reshape(1, -1)
            advantages = torch.zeros_like(rewards_buf)
            lastgaelam = 0.0
            for t in reversed(range(args.num_steps)):
                nextnonterminal = 1.0 - next_done if t == args.num_steps - 1 else 1.0 - dones_buf[t + 1]
                nextvalues = next_value if t == args.num_steps - 1 else values_buf[t + 1]
                delta = rewards_buf[t] + 0.99 * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + 0.99 * 0.95 * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        b_obs = obs_buf.reshape((-1,) + train_envs.single_observation_space.shape)
        b_actions = actions_buf.reshape((-1,) + train_envs.single_action_space.shape)
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_inds = np.arange(args.num_envs * args.num_steps)

        metrics = {"l_ppo": [], "l_penalty": [], "l_ent": [], "l_mmd": [], "l_vai": []}

        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, len(b_inds), args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                mb_obs_clean = b_obs[mb_inds]
                mb_actions = b_actions.long()[mb_inds]
                mb_adv_raw = b_advantages[mb_inds].detach()

                activations.clear()
                _, newlogprob, clean_entropy, newvalue = agent.get_action_and_value(mb_obs_clean, mb_actions)
                clean_feat_c3 = activations["conv3"].clone().detach()
                clean_entropy_mean = clean_entropy.mean().detach()

                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()
                mb_adv = b_advantages[mb_inds]
                mb_adv = (
                    mb_adv - mb_adv.mean()
                ) / (mb_adv.std(unbiased=False) + 1e-8)

                pg_loss = torch.max(-mb_adv * ratio, -mb_adv * torch.clamp(ratio, 0.9, 1.1)).mean()
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()
                loss_ppo = pg_loss - 0.01 * clean_entropy.mean() + 0.5 * v_loss

                loss_vai = torch.tensor(0.0, device=device)
                if stage_used == "stage2":
                    loss_vai, _, _, _ = compute_vai_loss(
                        agent=agent, anchor_model=anchor_model,
                        mb_obs_clean=mb_obs_clean, mb_adv_raw=mb_adv_raw, args=args,
                    )

                idx = torch.randint(0, train_buffer.shape[0], (len(mb_inds),), device=device)
                mb_obs_poison = train_buffer[idx]

                activations.clear()
                hidden = agent.network(mb_obs_poison / agent.norm)
                student_logits = agent.actor(hidden)
                student_value = agent.critic(hidden).view(-1)

                poison_feat_c3 = activations["conv3"]
                loss_mmd = compute_mmd_loss(clean_feat_c3, poison_feat_c3)

                # Apply stage-specific repair penalties.
                loss_ent_floor = torch.tensor(0.0, device=device)
                if stage_used == "stage1":
                    # Stage 1 disrupts the backdoor shortcut.
                    lens_losses = compute_lens_losses_code2_style(
                        agent=agent, lens=lens, activations=activations,
                        mb_obs_poison=mb_obs_poison, clean_entropy_mean=clean_entropy_mean,
                        student_value=student_value, layer_order=layer_order,
                        mu_R_t=mu_R_t, std_R_t=std_R_t, weights_t=weights_t,
                        thresh_neg=thresh_neg, thresh_pos=thresh_pos, args=args, device=device,
                    )
                    loss_stage = (
                        args.val_w_conv2 * lens_losses["loss_value_c2"]
                        + args.val_w_conv3 * lens_losses["loss_value_c3"]
                        + args.cons_beta * lens_losses["loss_cons"]
                    )
                    loss_ent_floor = lens_losses["loss_ent_floor"]

                    loss_self = (current_beta * loss_stage) + (beta_mmd * loss_mmd) + (args.beta_ent * loss_ent_floor)
                else:
                    # Stage 2 retains MMD and clean-side VAI only.
                    loss_stage = torch.tensor(0.0, device=device)
                    loss_self = (beta_mmd * loss_mmd) + (current_beta_vai * loss_vai)

                loss = loss_ppo + loss_self

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
                optimizer.step()

                metrics["l_ppo"].append(loss_ppo.item())
                metrics["l_mmd"].append(loss_mmd.item())
                metrics["l_penalty"].append(loss_stage.item())
                metrics["l_ent"].append(loss_ent_floor.item())
                metrics["l_vai"].append(loss_vai.item())

        stage_boundary_eval = stage_used == "stage1" and (
            stage1_switch_mode == "entropy"
            or (stage1_switch_mode == "fixed" and update >= args.stage1_fixed_updates)
            or (args.stage1_max_updates > 0 and update >= args.stage1_max_updates)
        )
        should_eval = update % args.eval_interval == 0 or stage_boundary_eval or update == num_updates

        if should_eval:
            metric_train_asr, metric_train_ent = evaluate_attack_metric_entropy(
                agent, train_reference_buffer, injector, args, device,
            )
            train_asr, train_ent = evaluate_attack_metric_entropy(
                agent, gate_reference_buffer, injector, args, device,
            )
            (
                actual_train_metric,
                actual_train_entropy,
                actual_train_numerator,
                actual_train_denominator,
            ) = evaluate_actual_state_metric_entropy(
                agent,
                train_buffer,
                train_reference_buffer,
                args,
                device,
                return_counts=True,
            )
            (
                actual_gate_metric,
                actual_gate_entropy,
                actual_gate_numerator,
                actual_gate_denominator,
            ) = evaluate_actual_state_metric_entropy(
                agent,
                gate_buffer,
                gate_reference_buffer,
                args,
                device,
                return_counts=True,
            )
            if has_independent_report:
                (
                    report_attack_metric,
                    report_attack_entropy,
                    report_attack_numerator,
                    report_attack_denominator,
                ) = evaluate_attack_metric_entropy(
                    agent,
                    report_reference_buffer,
                    injector,
                    args,
                    device,
                    return_counts=True,
                )
            else:
                report_attack_metric = ""
                report_attack_entropy = ""
                report_attack_denominator = ""
                report_attack_numerator = ""
            current_l_mmd = float(np.mean(metrics["l_mmd"])) if len(metrics["l_mmd"]) > 0 else 0.0

            entropy_history.append(train_ent)
            entropy_history = entropy_history[-args.stage1_entropy_window:]
            entropy_smooth = float(np.mean(entropy_history))
            entropy_recovery = entropy_recovery_fraction(
                entropy_smooth,
                train_ent_0,
                clean_entropy_reference,
            )

            switch_triggered = False
            switch_reason_this_update = ""
            entropy_ready = False
            if stage_used == "stage1":
                if stage1_switch_mode == "entropy":
                    entropy_ready, entropy_gate_streak, entropy_gate_triggered = update_entropy_gate(
                        update=update,
                        entropy_smooth=entropy_smooth,
                        entropy_target=entropy_target,
                        recovery_direction=entropy_recovery_direction,
                        current_streak=entropy_gate_streak,
                        args=args,
                    )
                    if entropy_gate_triggered:
                        threshold_reached = 1
                    if entropy_gate_triggered:
                        switch_reason_this_update = "entropy_recovered"
                elif update >= args.stage1_fixed_updates:
                    switch_reason_this_update = "fixed_updates"

                if (
                    not switch_reason_this_update
                    and args.stage1_max_updates > 0
                    and update >= args.stage1_max_updates
                ):
                    switch_reason_this_update = "timeout_forced"

                if switch_reason_this_update:
                    switch_triggered = True
                    stage2_start_update = update + 1
                    stage_switch_reason = switch_reason_this_update
                    stage1_outcome = switch_reason_this_update
                    threshold_reached = int(switch_reason_this_update == "entropy_recovered")

                logger.info(
                    f"[{exp_name}][StageGate] update={update} | Hraw={train_ent:.4f} | "
                    f"Hsmooth={entropy_smooth:.4f} | Htarget={entropy_target:.4f} | "
                    f"recovery={entropy_recovery:.3f} | ready={int(entropy_ready)} | "
                    f"gate={entropy_gate_streak}/{args.stage1_entropy_patience}"
                )

            if update == num_updates and stage_used == "stage1":
                if switch_triggered and stage2_start_update > num_updates:
                    censored = 1
                    stage1_outcome = "censored_no_stage2_budget"
                elif not switch_triggered:
                    censored = 1
                    stage_switch_reason = "censored_budget_exhausted"
                    stage1_outcome = "censored_budget_exhausted"

            return_eval_due = (
                update % args.return_eval_interval == 0
                or update == num_updates
                or switch_triggered
            )
            if return_eval_due:
                cur_p_ret, cur_p_ret_std, cur_p_ret_n = evaluate_fixed_seed_return(
                    agent, eval_envs, injector, args, device, eval_seed_base, triggered=True,
                )
                avg_ret, avg_ret_std, avg_ret_n = evaluate_fixed_seed_return(
                    agent, eval_envs, injector, args, device, eval_seed_base, triggered=False,
                )
            else:
                cur_p_ret = cur_p_ret_std = cur_p_ret_n = ""
                avg_ret = avg_ret_std = avg_ret_n = ""

            if stage_used == "stage2" and (train_asr <= attack_metric_good) and (current_l_mmd <= args.mmd_good):
                good_counter += 1
            else:
                good_counter = 0

            if good_counter >= args.good_streak:
                beta_changed = False
                next_beta_mmd = max(args.beta_mmd_min, beta_mmd * args.beta_decay)
                beta_changed = beta_changed or next_beta_mmd < beta_mmd
                beta_mmd = next_beta_mmd
                next_beta_vai = max(args.beta_vai_min, current_beta_vai * args.beta_decay)
                beta_changed = beta_changed or next_beta_vai < current_beta_vai
                current_beta_vai = next_beta_vai
                if beta_changed:
                    beta_decay_count += 1
                    last_beta_decay_update = update
                good_counter = 0

            l_ppo_mean = float(np.mean(metrics["l_ppo"])) if len(metrics["l_ppo"]) > 0 else 0.0
            l_penalty_mean = float(np.mean(metrics["l_penalty"])) if len(metrics["l_penalty"]) > 0 else 0.0
            l_ent_mean = float(np.mean(metrics["l_ent"])) if len(metrics["l_ent"]) > 0 else 0.0
            l_vai_mean = float(np.mean(metrics["l_vai"])) if len(metrics["l_vai"]) > 0 else 0.0

            p_ret_log = f"{cur_p_ret:<6.1f}" if return_eval_due else f"{'-':<6}"
            c_ret_log = f"{avg_ret:<6.1f}" if return_eval_due else f"{'-':<6}"
            if has_independent_report:
                logger.info(
                    f"{update:<3} | {global_step:<8} | "
                    f"{report_attack_metric:<6.1f} | {train_asr:<6.1f} | "
                    f"{actual_gate_metric:<6.1f} | {train_ent:<6.2f} | "
                    f"{p_ret_log} | {c_ret_log} | "
                    f"{l_ppo_mean:<7.4f} | {l_penalty_mean:<7.4f} | "
                    f"{current_l_mmd:<7.4f} | {l_vai_mean:<7.4f} | "
                    f"{current_beta:<7.2f} | {beta_mmd:<7.2f} | "
                    f"{current_beta_vai:<7.2f} | {good_counter}"
                )
            else:
                logger.info(
                    f"{update:<3} | {global_step:<8} | "
                    f"{train_asr:<6.1f} | {actual_gate_metric:<6.1f} | "
                    f"{train_ent:<6.2f} | "
                    f"{p_ret_log} | {c_ret_log} | "
                    f"{l_ppo_mean:<7.4f} | {l_penalty_mean:<7.4f} | "
                    f"{current_l_mmd:<7.4f} | {l_vai_mean:<7.4f} | "
                    f"{current_beta:<7.2f} | {beta_mmd:<7.2f} | "
                    f"{current_beta_vai:<7.2f} | {good_counter}"
                )

            row = row_0.copy()
            row.update({
                "Update": update, "Step": global_step,
                "Stage": stage_used,
                "ASR_Tr": metric_train_asr, "Ent_Tr": metric_train_ent,
                "ASR_Gate": train_asr, "Ent_Gate": train_ent,
                "AttackMetricActualTrain": actual_train_metric,
                "EntropyActualTrain": actual_train_entropy,
                "AttackMetricActualTrainNumerator": actual_train_numerator,
                "AttackMetricActualTrainDenominator": actual_train_denominator,
                "AttackMetricActualGate": actual_gate_metric,
                "EntropyActualGate": actual_gate_entropy,
                "AttackMetricActualGateNumerator": actual_gate_numerator,
                "AttackMetricActualGateDenominator": actual_gate_denominator,
                "AttackMetricEval": report_attack_metric,
                "AttackMetricEvalEntropy": report_attack_entropy,
                "AttackMetricEvalNumerator": report_attack_numerator,
                "AttackMetricEvalDenominator": report_attack_denominator,
                "P_Ret": cur_p_ret, "P_Ret_Std": cur_p_ret_std, "P_Ret_N": cur_p_ret_n,
                "C_Ret": avg_ret, "C_Ret_Std": avg_ret_std, "C_Ret_N": avg_ret_n,
                "ReturnEvaluated": int(return_eval_due),
                "L_PPO": l_ppo_mean, "Penalty": l_penalty_mean, "L_ENT": l_ent_mean,
                "L_MMD": current_l_mmd, "L_VAI": l_vai_mean,
                "Beta": current_beta, "Beta_MMD": beta_mmd, "Beta_VAI": current_beta_vai, "Streak": good_counter,
                "BetaDecayCount": beta_decay_count, "LastBetaDecayUpdate": last_beta_decay_update,
                "EntropySmooth": entropy_smooth,
                "EntropyRecovery": entropy_recovery,
                "EntropyGateStreak": entropy_gate_streak,
                "Stage2StartUpdate": stage2_start_update or "",
                "StageSwitchReason": stage_switch_reason,
                "ThresholdReached": threshold_reached, "Censored": censored,
                "Stage1Outcome": stage1_outcome,
            })
            final_row = row.copy()
            with open(csv_file, mode="a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=REPAIR_METRIC_FIELDS)
                writer.writerow(row)

            if switch_triggered:
                stage1_active = False
                current_beta = 0.0
                logger.info("-" * 145)
                logger.info(
                    f"[{exp_name}] >>> Stage 2 starts at update {stage2_start_update}; "
                    f"reason={stage_switch_reason}, Hsmooth={entropy_smooth:.4f}, "
                    f"Htarget={entropy_target:.4f}. Entropy floor is removed; "
                    f"MMD+VAI remain. <<<"
                )
                if stage2_start_update > num_updates:
                    logger.warning(
                        f"[{exp_name}] Stage 2 was scheduled after the final update. "
                        "Increase total_timesteps to execute Stage 2."
                    )
                logger.info("-" * 145)

    stage2_executed = stage2_start_update is not None and stage2_start_update <= num_updates
    if not stage2_executed:
        logger.warning(
            f"[{exp_name}] Stage 2 did not execute within {num_updates} updates. "
            f"Final Hsmooth={entropy_smooth:.4f}, Htarget={entropy_target:.4f}. "
            "This run is marked as censored."
        )

    if args.save_model:
        out_path = os.path.join(model_dir, f"{args.env_id}_phase3_{exp_name}_final.pth")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        torch.save(agent.state_dict(), out_path)
        logger.info(f"[{exp_name}] Saved final model: {out_path}")
        final_row["SavedModel"] = out_path
        final_row["ModelStatus"] = "valid"

    for h in handles: h.remove()
    train_envs.close()
    eval_envs.close()
    return final_row



def run_attack_pipeline(args: PipelineArgs, run_timestamp=None):
    args = normalize_attack_args(args)
    validate_pipeline_args(args)
    if (args.offline_buffer_size < 1 or args.repair_validation_size < 1
            or args.repair_split_gap_size < 0):
        raise ValueError(
            "offline_buffer_size/repair_validation_size must be positive and "
            "repair_split_gap_size non-negative."
        )
    timestamp = run_timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    entropy_tag = "entfilter_on" if args.use_entropy_filter else "entfilter_off"
    attack_tag = infer_attack_name(args)
    switch_tag = f"stage_{args.stage1_switch_mode.strip().lower()}"
    run_name = (
        f"{args.env_id}_{attack_tag}_full_{switch_tag}_"
        f"seed{args.seed}_{timestamp}_{entropy_tag}"
    )
    run_dir = os.path.join(args.output_root, run_name)
    log_dir = os.path.join(run_dir, "logs")
    table_dir = os.path.join(run_dir, "tables")
    model_dir = os.path.join(run_dir, "models")
    buffer_dir = os.path.join(run_dir, "buffers")
    for d in [run_dir, log_dir, table_dir, model_dir, buffer_dir]:
        os.makedirs(d, exist_ok=True)

    log_file = os.path.join(log_dir, f"{args.env_id}_defense.log")
    csv_file = os.path.join(table_dir, "repair_metrics.csv")
    summary_file = os.path.join(table_dir, "repair_final_summary.csv")
    phase2_summary_file = os.path.join(table_dir, "phase2_collection_summary.csv")

    logger = logging.getLogger("PipelineLogger")
    close_logger_handlers(logger)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(message)s")

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    logger.info("\n" + "#" * 70)
    logger.info(" >>> End-to-End Automated Defense Pipeline (Dual Buffer DBG) <<<")
    logger.info("#" * 70)
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Logging to: {log_file}")
    logger.info(f"Repair metrics CSV: {csv_file}")
    logger.info(f"Repair final summary CSV: {summary_file}")
    logger.info(f"Phase2 summary CSV: {phase2_summary_file}")
    logger.info(f"Attack: {attack_tag} | repair=full")
    logger.info(f"Stage transition: {args.stage1_switch_mode}")
    logger.info(
        f"Evaluation: gate_interval={args.eval_interval}, "
        f"return_interval={args.return_eval_interval}, episodes={args.eval_episodes}, "
        f"idle_timeout={args.eval_no_reward_timeout}"
    )
    logger.info(f"Poisoned model: {args.poisoned_model_path}")
    checkpoint_sha256 = file_sha256(args.poisoned_model_path)
    logger.info(f"Poisoned model SHA256: {checkpoint_sha256}")
    logger.info(f"Lens model: {args.lens_model_path}")
    if args.untargeted:
        logger.info(
            "Untarget report-only ADR: "
            f"source={args.untarget_eval_source}, "
            f"N={args.untarget_eval_size}, "
            f"seed_offset={args.untarget_eval_seed_offset}, "
            f"state_policy={args.untarget_eval_action_mode}"
        )
    logger.info(f"Entropy filter enabled: {args.use_entropy_filter} | threshold={args.intercept_entropy_thresh}")

    envs = gym.vector.SyncVectorEnv([make_env(args.env_id, args.seed + i) for i in range(args.num_envs)])
    validate_stage_transition_args(args, envs.single_action_space.n)

    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(args.poisoned_model_path, map_location=device))
    agent.eval()

    anchor_model = Agent(envs).to(device)
    anchor_model.load_state_dict(torch.load(args.poisoned_model_path, map_location=device))
    anchor_model.eval()
    for p in anchor_model.parameters(): p.requires_grad = False

    feature_layers_dict = {"conv1": agent.network[0], "conv2": agent.network[2], "conv3": agent.network[4]}
    layer_order = ["conv1", "conv2", "conv3"]
    activations, handles = register_feature_hooks(feature_layers_dict)

    weights_np = np.ones(len(layer_order), dtype=np.float32)
    weights_t = torch.tensor(weights_np, device=device)

    dummy_obs = torch.zeros((args.num_envs, 4, 84, 84), device=device)
    with torch.no_grad():
        activations.clear()
        _ = agent.get_action_and_value(dummy_obs)
    feature_dims = {}
    for name in layer_order:
        feat = activations[name]
        B = feat.shape[0]
        feature_dims[name] = int(feat.numel() // B)

    lens = MultiLayerLens(feature_dims).to(device)
    lens_state_dict = lens.state_dict()
    key_mapping = {
        "conv1_W_critic": "critic_heads.conv1.weight", "conv1_b_critic": "critic_heads.conv1.bias",
        "conv2_W_critic": "critic_heads.conv2.weight", "conv2_b_critic": "critic_heads.conv2.bias",
        "conv3_W_critic": "critic_heads.conv3.weight", "conv3_b_critic": "critic_heads.conv3.bias"
    }

    with np.load(args.lens_model_path, allow_pickle=False) as npz_weights:
        missing_keys = sorted(set(key_mapping).difference(npz_weights.files))
        if missing_keys:
            raise KeyError(
                f"Lens archive is missing required parameters: {missing_keys}"
            )
        for npz_key, pt_key in key_mapping.items():
            tensor_weight = torch.as_tensor(
                npz_weights[npz_key], dtype=torch.float32, device=device
            )
            expected_shape = lens_state_dict[pt_key].shape
            if tensor_weight.shape != expected_shape:
                if tensor_weight.T.shape == expected_shape:
                    tensor_weight = tensor_weight.T
                else:
                    raise ValueError(
                        f"Lens parameter '{npz_key}' has shape "
                        f"{tuple(tensor_weight.shape)}; expected "
                        f"{tuple(expected_shape)}."
                    )
            lens_state_dict[pt_key].copy_(tensor_weight)

    lens.load_state_dict(lens_state_dict, strict=False)
    lens.eval()

    injector = AttackInjector(args, device)
    untarget_eval_reference = None
    untarget_eval_reference_hash = ""
    untarget_eval_buffer_role = "disabled"
    if (
        args.untargeted
        and args.untarget_eval_source.strip().lower()
        == "independent_clean"
    ):
        untarget_eval_reference = (
            collect_untarget_eval_reference_buffer(
                agent,
                args,
                device,
                logger,
            )
        )
        untarget_eval_reference_hash = tensor_sha256(
            untarget_eval_reference
        )
        untarget_eval_buffer_role = (
            "fixed_independent_clean_rollout_report_only"
        )
        (
            preflight_adr,
            preflight_entropy,
            preflight_changed,
            preflight_n,
        ) = evaluate_attack_metric_entropy(
            agent,
            untarget_eval_reference,
            injector,
            args,
            device,
            return_counts=True,
        )
        logger.info(
            "[Untarget ADR preflight] "
            f"ADR_E={preflight_adr:.2f}% "
            f"({preflight_changed}/{preflight_n}) | "
            f"H_trigger={preflight_entropy:.4f} | "
            "protocol=fixed_clean+last_frame+paired_greedy"
        )
        if preflight_adr < 20.0:
            logger.warning(
                "[Untarget ADR preflight] ADR_E is below 20%. Do not "
                "resample states to force a larger value; this checkpoint "
                "is weaker under the fixed protocol."
            )
        preflight_row = {
            "PoisonedModelPath": args.poisoned_model_path,
            "PoisonedModelSHA256": checkpoint_sha256,
            "EvalBufferRole": untarget_eval_buffer_role,
            "EvalBufferSHA256": untarget_eval_reference_hash,
            "EvalSeed": args.seed + args.untarget_eval_seed_offset,
            "ParallelEnvSeeds": args.num_envs,
            "StatePolicy": args.untarget_eval_action_mode,
            "EnvironmentProtocol": "ppo_atari_no_idle_timeout",
            "TriggerProtocol": "last_frame_only",
            "ActionComparison": "paired_greedy_argmax",
            "ADRPercent": preflight_adr,
            "ChangedActions": preflight_changed,
            "SampleCount": preflight_n,
            "TriggeredEntropy": preflight_entropy,
        }
        preflight_path = os.path.join(
            table_dir,
            "untarget_adr_preflight.csv",
        )
        with open(preflight_path, mode="w", newline="") as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(preflight_row.keys()),
            )
            writer.writeheader()
            writer.writerow(preflight_row)
        logger.info(
            f"[Untarget ADR preflight] Saved: {preflight_path}"
        )
        if args.untarget_eval_only:
            if args.save_intercept_buffers:
                eval_buffer_path = os.path.join(
                    buffer_dir,
                    "untarget_fixed_clean_eval_reference.pt",
                )
                torch.save(untarget_eval_reference, eval_buffer_path)
                logger.info(
                    "[Untarget ADR preflight] Saved fixed buffer: "
                    f"{eval_buffer_path}"
                )
            for handle in handles:
                handle.remove()
            envs.close()
            del agent
            del anchor_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info(
                "[Done] untarget_eval_only=True; repair phases were skipped."
            )
            close_logger_handlers(logger)
            return

    # Phase 1: clean baseline calibration.
    logger.info(f"\n[Phase 1] Calibrating clean baseline for {args.calib_steps} steps...")
    next_obs, _ = envs.reset(seed=args.seed + 999)
    next_obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

    raw_residuals_list = []
    clean_entropies = []
    steps_collected = 0
    while steps_collected < args.calib_steps:
        with torch.no_grad():
            activations.clear()
            action, _, policy_entropy, v_curr = agent.get_action_and_value(next_obs_t)
            resid = compute_raw_residuals(v_curr.view(-1), activations, lens, layer_order, device)
            raw_residuals_list.append(resid.detach().cpu().numpy())
            clean_entropies.append(policy_entropy.detach().cpu().numpy())

        next_obs_np, _, _, _, _ = envs.step(action.cpu().numpy())
        next_obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        steps_collected += args.num_envs

    all_raw_resid = np.concatenate(raw_residuals_list, axis=0)[:args.calib_steps]
    mu_R_t = torch.tensor(np.mean(all_raw_resid, axis=0), device=device)
    std_R_t = torch.tensor(np.std(all_raw_resid, axis=0), device=device)

    all_raw_resid_t = torch.tensor(all_raw_resid, device=device)
    z_scores = (all_raw_resid_t - mu_R_t.view(1, -1)) / (std_R_t.view(1, -1) + 1e-8)
    u_neg = torch.clamp(-z_scores, min=0.0)
    u_pos = torch.clamp(z_scores, min=0.0)
    S_neg_all = torch.sum(weights_t.view(1, -1) * (u_neg ** 2), dim=1).detach().cpu().numpy()
    S_pos_all = torch.sum(weights_t.view(1, -1) * (u_pos ** 2), dim=1).detach().cpu().numpy()

    thresh_neg = float(np.quantile(S_neg_all, args.calib_quantile))
    thresh_pos = float(np.quantile(S_pos_all, args.calib_quantile))
    all_clean_ent = np.concatenate(clean_entropies, axis=0)[:args.calib_steps]
    clean_entropy_reference = float(np.mean(all_clean_ent))

    logger.info(
        f"[Phase 1 Done] Calibrated at {args.calib_quantile} quantile. Thresh Neg: {thresh_neg:.4f}, Pos: {thresh_pos:.4f}")
    logger.info(f" -> Clean Env Entropy: Mean={clean_entropy_reference:.4f}, Min={np.min(all_clean_ent):.4f}")

    # Phase 2: collect detector-alarm and clean-reference buffers.
    repair_pool_size = (
        args.offline_buffer_size + args.repair_split_gap_size + args.repair_validation_size
    )
    logger.info(
        f"\n[Phase 2] Parallel collection to {repair_pool_size} samples "
        f"({args.offline_buffer_size} train + {args.repair_split_gap_size} gap + "
        f"{args.repair_validation_size} held-out gate)..."
    )

    strict_samples = []
    strict_reference_samples = []
    strict_sample_labels = []

    strict_tp = strict_fp = 0
    raw_alarm_total = raw_tp = raw_fp = 0
    gt_positive_total = gt_negative_total = 0
    entropy_rejected = 0
    strict_entropy_values = []
    raw_alarm_entropy_values = []
    steps_total = 0
    report_every = 500

    env_step_counts = np.zeros(args.num_envs, dtype=int)
    next_start_at = np.random.randint(0, 201, size=args.num_envs)
    next_end_at = next_start_at + np.random.randint(
        args.poison_duration_min,
        args.poison_duration_max + 1,
        size=args.num_envs,
    )

    next_obs, _ = envs.reset(seed=args.seed + 200)
    obs_t = torch.tensor(next_obs, dtype=torch.float32, device=device)

    while (
        len(strict_samples) < repair_pool_size
        and steps_total < args.phase2_max_env_steps
    ):
        is_poisoning = (env_step_counts >= next_start_at) & (env_step_counts < next_end_at)
        mask_t = torch.tensor(is_poisoning, device=device, dtype=torch.bool)

        obs_for_agent = apply_trigger_weak(obs_t, mask_t, injector)

        with torch.no_grad():
            activations.clear()
            action, _, policy_entropy, v_curr = agent.get_action_and_value(obs_for_agent)
            s_neg_t, s_pos_t = compute_two_sided_scores(
                v_curr.view(-1), activations, lens, layer_order, mu_R_t, std_R_t, weights_t, device,
            )

        raw_pred = ((s_neg_t > thresh_neg) | (s_pos_t > thresh_pos)).detach().cpu().numpy().astype(np.bool_)
        gt = is_poisoning.astype(np.bool_)
        gt_positive_total += int(gt.sum())
        gt_negative_total += int((~gt).sum())
        ent_np = policy_entropy.detach().cpu().numpy()
        steps_total += args.num_envs

        for i in range(args.num_envs):
            if raw_pred[i]:
                raw_alarm_total += 1
                raw_alarm_entropy_values.append(float(ent_np[i]))
                if gt[i]:
                    raw_tp += 1
                else:
                    raw_fp += 1

                pass_entropy_filter = (not args.use_entropy_filter) or (ent_np[i] <= args.intercept_entropy_thresh)
                if not pass_entropy_filter:
                    entropy_rejected += 1

                if pass_entropy_filter and len(strict_samples) < repair_pool_size:
                    strict_samples.append(obs_for_agent[i].detach().cpu().clone())
                    strict_reference_samples.append(obs_t[i].detach().cpu().clone())
                    strict_sample_labels.append(int(gt[i]))
                    strict_entropy_values.append(float(ent_np[i]))
                    if gt[i]:
                        strict_tp += 1
                    else:
                        strict_fp += 1

        if steps_total % (args.num_envs * report_every) == 0:
            logger.info(f"[Phase2-Stats] Strict Collected: {len(strict_samples)}/{repair_pool_size} | "
                        f"RawAlarm={raw_alarm_total} | EntRejected={entropy_rejected} | "
                        f"EntropyFilter={args.use_entropy_filter} | Seen: {steps_total}/{args.phase2_max_env_steps}")

        next_obs_np, _, terminations, truncations, _ = envs.step(action.cpu().numpy())
        done = np.logical_or(terminations, truncations)
        obs_t = torch.tensor(next_obs_np, dtype=torch.float32, device=device)
        env_step_counts += 1

        finished_window = env_step_counts >= next_end_at
        for idx_env in np.flatnonzero(done | finished_window):
            next_start_at[idx_env] = env_step_counts[idx_env] + random.randint(
                args.cooldown_min, args.cooldown_max
            )
            next_end_at[idx_env] = next_start_at[idx_env] + random.randint(
                args.poison_duration_min, args.poison_duration_max
            )

    if len(strict_samples) < repair_pool_size:
        raise RuntimeError(
            "Phase 2 did not fill the requested repair train/gate pools before phase2_max_env_steps: "
            f"strict={len(strict_samples)}/{repair_pool_size}."
        )

    strict_total = strict_tp + strict_fp
    raw_total = raw_tp + raw_fp
    strict_precision = strict_tp / max(1, strict_total)
    raw_precision = raw_tp / max(1, raw_total)
    raw_fn = gt_positive_total - raw_tp
    raw_tn = gt_negative_total - raw_fp
    raw_recall = raw_tp / max(1, gt_positive_total)
    raw_fpr = raw_fp / max(1, gt_negative_total)
    raw_f1 = (
        2.0 * raw_precision * raw_recall
        / max(1e-12, raw_precision + raw_recall)
    )
    strict_entropy_mean = float(np.mean(strict_entropy_values)) if len(strict_entropy_values) > 0 else float("nan")
    raw_entropy_mean = float(np.mean(raw_alarm_entropy_values)) if len(raw_alarm_entropy_values) > 0 else float("nan")

    logger.info(
        f"\n[Phase 2 Done] Strict Intercepted buffer size={len(strict_samples)} "
        f"(TP={strict_tp}, FP={strict_fp}, Precision={strict_precision:.4f})")
    logger.info(f"[Phase 2 Done] Raw alarms={raw_alarm_total} (TP={raw_tp}, FP={raw_fp}, Precision={raw_precision:.4f})")
    logger.info(
        f"[Phase 2 Done] Recall={raw_recall:.4f} | "
        f"FPR={raw_fpr:.4f} | F1={raw_f1:.4f} | "
        f"FN={raw_fn} | TN={raw_tn}"
    )
    logger.info(f"[Phase 2 Done] Entropy rejected={entropy_rejected} | Entropy filter enabled={args.use_entropy_filter}")

    strict_pool = torch.stack(strict_samples).to(device)
    strict_reference_pool = torch.stack(strict_reference_samples).to(device)
    strict_labels_np = np.asarray(strict_sample_labels, dtype=np.int64)
    train_buffer_strict, gate_buffer_strict, strict_train_labels, strict_gate_labels = split_repair_buffer(
        strict_pool, strict_labels_np, args.offline_buffer_size,
        args.repair_validation_size, args.repair_split_gap_size,
    )
    train_reference_strict, gate_reference_strict, _, _ = split_repair_buffer(
        strict_reference_pool, strict_labels_np, args.offline_buffer_size,
        args.repair_validation_size, args.repair_split_gap_size,
    )
    strict_train_tp, strict_train_fp, strict_train_precision = label_summary(strict_train_labels)
    strict_gate_tp, strict_gate_fp, strict_gate_precision = label_summary(strict_gate_labels)
    strict_pool_hash = tensor_sha256(strict_pool)
    strict_labels_hash = hashlib.sha256(strict_labels_np.tobytes()).hexdigest()
    strict_train_hash = tensor_sha256(train_buffer_strict)
    strict_gate_hash = tensor_sha256(gate_buffer_strict)
    strict_train_reference_hash = tensor_sha256(train_reference_strict)
    strict_gate_reference_hash = tensor_sha256(gate_reference_strict)
    strict_train_asr, strict_train_entropy = evaluate_attack_metric_entropy(
        agent, train_reference_strict, injector, args, device,
    )
    strict_gate_asr, strict_gate_entropy = evaluate_attack_metric_entropy(
        agent, gate_reference_strict, injector, args, device,
    )
    (
        strict_train_actual_metric,
        strict_train_actual_entropy,
        strict_train_actual_numerator,
        strict_train_actual_denominator,
    ) = evaluate_actual_state_metric_entropy(
        agent,
        train_buffer_strict,
        train_reference_strict,
        args,
        device,
        return_counts=True,
    )
    (
        strict_gate_actual_metric,
        strict_gate_actual_entropy,
        strict_gate_actual_numerator,
        strict_gate_actual_denominator,
    ) = evaluate_actual_state_metric_entropy(
        agent,
        gate_buffer_strict,
        gate_reference_strict,
        args,
        device,
        return_counts=True,
    )
    strict_actual_metric_protocol = (
        "all_captured_inputs_paired_action_disagreement_no_retrigger"
        if args.untargeted
        else "all_captured_inputs_target_action_rate_no_retrigger"
    )
    untarget_eval_metric = untarget_eval_entropy = ""
    untarget_eval_numerator = untarget_eval_denominator = ""
    if args.untargeted and untarget_eval_reference is not None:
        (
            untarget_eval_metric,
            untarget_eval_entropy,
            untarget_eval_numerator,
            untarget_eval_denominator,
        ) = evaluate_attack_metric_entropy(
            agent,
            untarget_eval_reference,
            injector,
            args,
            device,
            return_counts=True,
        )

    logger.info(
        f"[Phase 2 Split] Strict train={train_buffer_strict.shape[0]} "
        f"(precision={strict_train_precision:.4f}, H={strict_train_entropy:.4f}) | "
        f"gate={gate_buffer_strict.shape[0]} "
        f"(precision={strict_gate_precision:.4f}, H={strict_gate_entropy:.4f})"
    )
    logger.info(
        f"[Phase 2 Split] TrainSHA256={strict_train_hash} | GateSHA256={strict_gate_hash}"
    )
    if args.untargeted:
        logger.info(
            "[Untarget ADR audit @ update 0] "
            f"fixed_clean={untarget_eval_metric if untarget_eval_metric != '' else 'disabled'} "
            f"(changed={untarget_eval_numerator}, "
            f"N={untarget_eval_denominator}) | "
            f"forced_reference_train={strict_train_asr:.2f} "
            f"(N={train_reference_strict.shape[0]}) | "
            f"forced_reference_gate={strict_gate_asr:.2f} "
            f"(N={gate_reference_strict.shape[0]})"
        )
    logger.info(
        "[Actual-state metric @ update 0] "
        f"train={strict_train_actual_metric:.2f}% "
        f"({strict_train_actual_numerator}/"
        f"{strict_train_actual_denominator}) | "
        f"gate={strict_gate_actual_metric:.2f}% "
        f"({strict_gate_actual_numerator}/"
        f"{strict_gate_actual_denominator}) | "
        f"protocol={strict_actual_metric_protocol}"
    )

    phase2_summary = {
        "RunDir": run_dir,
        "PoisonedModelPath": args.poisoned_model_path,
        "PoisonedModelSHA256": checkpoint_sha256,
        "Attack": attack_tag,
        "AttackMetric": attack_metric_name(args),
        "TriggerProtocol": "last_frame_only",
        "UntargetEvalSource": args.untarget_eval_source if args.untargeted else "",
        "UntargetEvalBufferRole": untarget_eval_buffer_role if args.untargeted else "",
        "UntargetEvalBufferSize": (
            int(untarget_eval_reference.shape[0])
            if untarget_eval_reference is not None
            else 0
        ),
        "UntargetEvalBufferSHA256": untarget_eval_reference_hash,
        "UntargetEvalSeed": (
            args.seed + args.untarget_eval_seed_offset
            if args.untargeted
            else ""
        ),
        "UntargetEvalActionMode": (
            args.untarget_eval_action_mode if args.untargeted else ""
        ),
        "UntargetEvalMetric": untarget_eval_metric,
        "UntargetEvalEntropy": untarget_eval_entropy,
        "UntargetEvalNumerator": untarget_eval_numerator,
        "UntargetEvalDenominator": untarget_eval_denominator,
        "Mode": "full",
        "CalibrationQuantile": args.calib_quantile,
        "DetectorThresholdNegative": thresh_neg,
        "DetectorThresholdPositive": thresh_pos,
        "CleanEntropyReference": clean_entropy_reference,
        "EntropyFilterEnabled": int(args.use_entropy_filter),
        "InterceptEntropyThresh": args.intercept_entropy_thresh,
        "OfflineBufferSizeTarget": args.offline_buffer_size,
        "RepairValidationSizeTarget": args.repair_validation_size,
        "RepairPoolSizeTarget": repair_pool_size,
        "RepairSplitGapSize": args.repair_split_gap_size,
        "RepairSplitMethod": "deterministic_temporal_gap_v1",
        "StepsSeen": steps_total,
        "RawAlarmTotal": raw_alarm_total,
        "RawTP": raw_tp,
        "RawFP": raw_fp,
        "RawFN": raw_fn,
        "RawTN": raw_tn,
        "RawGTPositive": gt_positive_total,
        "RawGTNegative": gt_negative_total,
        "RawPrecision": raw_precision,
        "RawRecall": raw_recall,
        "RawFPR": raw_fpr,
        "RawF1": raw_f1,
        "EntropyRejected": entropy_rejected,
        # Legacy StrictBuffer* fields consistently describe the saved train buffer.
        "StrictBufferSize": int(train_buffer_strict.shape[0]),
        "StrictTP": strict_train_tp,
        "StrictFP": strict_train_fp,
        "StrictPrecision": strict_train_precision,
        "RawAlarmEntropyMean": raw_entropy_mean,
        "StrictEntropyMean": strict_train_entropy,
        "StrictPoolSize": len(strict_samples),
        "StrictPoolTP": strict_tp, "StrictPoolFP": strict_fp,
        "StrictPoolPrecision": strict_precision,
        "StrictPoolEntropyMean": strict_entropy_mean,
        "StrictPoolSHA256": strict_pool_hash, "StrictLabelsSHA256": strict_labels_hash,
        "StrictTrainSize": int(train_buffer_strict.shape[0]),
        "StrictTrainTP": strict_train_tp, "StrictTrainFP": strict_train_fp,
        "StrictTrainPrecision": strict_train_precision, "StrictTrainASR": strict_train_asr,
        "StrictTrainEntropy": strict_train_entropy, "StrictTrainSHA256": strict_train_hash,
        "StrictTrainReferenceSHA256": strict_train_reference_hash,
        "StrictTrainActualMetric": strict_train_actual_metric,
        "StrictTrainActualEntropy": strict_train_actual_entropy,
        "StrictTrainActualNumerator": strict_train_actual_numerator,
        "StrictTrainActualDenominator": strict_train_actual_denominator,
        "StrictGateSize": int(gate_buffer_strict.shape[0]),
        "StrictGateTP": strict_gate_tp, "StrictGateFP": strict_gate_fp,
        "StrictGatePrecision": strict_gate_precision, "StrictGateASR": strict_gate_asr,
        "StrictGateEntropy": strict_gate_entropy, "StrictGateSHA256": strict_gate_hash,
        "StrictGateReferenceSHA256": strict_gate_reference_hash,
        "StrictGateActualMetric": strict_gate_actual_metric,
        "StrictGateActualEntropy": strict_gate_actual_entropy,
        "StrictGateActualNumerator": strict_gate_actual_numerator,
        "StrictGateActualDenominator": strict_gate_actual_denominator,
        "StrictActualMetricProtocol": strict_actual_metric_protocol,
        "StrictBufferSHA256": strict_train_hash,
    }
    with open(phase2_summary_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(phase2_summary.keys()))
        writer.writeheader()
        writer.writerow(phase2_summary)

    if args.save_intercept_buffers:
        buffers_to_save = {
            "strict_intercept_buffer.pt": train_buffer_strict,
            "strict_intercept_pool.pt": strict_pool,
            "strict_train_buffer.pt": train_buffer_strict,
            "strict_gate_buffer.pt": gate_buffer_strict,
            "strict_clean_reference_pool.pt": strict_reference_pool,
            "strict_train_clean_reference.pt": train_reference_strict,
            "strict_gate_clean_reference.pt": gate_reference_strict,
        }
        if untarget_eval_reference is not None:
            buffers_to_save[
                "untarget_fixed_clean_eval_reference.pt"
            ] = untarget_eval_reference
        for filename, tensor in buffers_to_save.items():
            path = os.path.join(buffer_dir, filename)
            torch.save(tensor.detach().cpu(), path)
            logger.info(f"[Phase 2 Done] Saved buffer: {path}")
        gate_start = args.offline_buffer_size + args.repair_split_gap_size
        split_metadata = {
            "method": "deterministic_temporal_gap_v1",
            "collection_seed": args.seed + 200,
            "strict_labels": torch.as_tensor(strict_labels_np, dtype=torch.int8),
            "train_indices": torch.arange(args.offline_buffer_size),
            "gap_indices": torch.arange(args.offline_buffer_size, gate_start),
            "gate_indices": torch.arange(gate_start, repair_pool_size),
        }
        metadata_path = os.path.join(buffer_dir, "repair_split_metadata.pt")
        torch.save(split_metadata, metadata_path)
        logger.info(f"[Phase 2 Done] Saved split metadata: {metadata_path}")
        del buffers_to_save

    strict_samples.clear()
    strict_reference_samples.clear()
    del strict_pool
    del strict_reference_pool

    for h in handles: h.remove()
    envs.close()
    del agent
    del anchor_model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Phase 3: run the full repair configuration.
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPAIR_METRIC_FIELDS)
        writer.writeheader()

    final_row = run_phase3_sanitization_experiment(
        exp_name="full",
        train_buffer=train_buffer_strict,
        gate_buffer=gate_buffer_strict,
        train_reference_buffer=train_reference_strict,
        gate_reference_buffer=gate_reference_strict,
        lens=lens,
        layer_order=layer_order,
        mu_R_t=mu_R_t,
        std_R_t=std_R_t,
        weights_t=weights_t,
        thresh_neg=thresh_neg,
        thresh_pos=thresh_pos,
        injector=injector,
        args=args,
        device=device,
        logger=logger,
        csv_file=csv_file,
        model_dir=model_dir,
        base_seed=args.seed + 3001,
        clean_entropy_reference=clean_entropy_reference,
        train_buffer_hash=strict_train_hash,
        gate_buffer_hash=strict_gate_hash,
        train_reference_hash=strict_train_reference_hash,
        gate_reference_hash=strict_gate_reference_hash,
        fixed_eval_seed_base=args.seed + 23001,
        report_reference_buffer=untarget_eval_reference,
        report_reference_hash=untarget_eval_reference_hash,
        report_buffer_role=untarget_eval_buffer_role,
    )
    write_final_summary([final_row], summary_file)
    logger.info(f"[Done] Saved final repair summary table: {summary_file}")
    logger.info(f"[Done] All outputs are saved under: {run_dir}")

    close_logger_handlers(logger)
    return run_dir


def main(args: PipelineArgs):
    suite_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    attack_names = selected_attack_names(args)
    use_per_attack_presets = len(attack_names) > 1
    for attack_name in attack_names:
        base_args = (
            replace(args, poisoned_model_path="", lens_model_path="")
            if use_per_attack_presets
            else args
        )
        attack_args = build_attack_args(base_args, attack_name)
        run_attack_pipeline(attack_args, run_timestamp=suite_timestamp)


if __name__ == "__main__":
    args = tyro.cli(PipelineArgs)
    main(args)
