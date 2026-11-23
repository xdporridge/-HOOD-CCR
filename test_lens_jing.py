"""Compare poisoned and repaired Atari policies under the CLEAR protocol."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, Mapping, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from torch.distributions.categorical import Categorical

from repair_lens import (
    Agent,
    AttackInjector,
    PipelineArgs,
    apply_trigger_weak,
    evaluate_fixed_seed_return,
    make_eval_env,
    reset_all_seeds,
)


@dataclass
class EvalArgs:
    env_id: str = "BreakoutNoFrameskip-v4"
    num_envs: int = 8
    seed: int = 10
    cuda: bool = True
    target_action: int = 2

    sn_outer: bool = True
    trojdrl: bool = False
    untargeted: bool = False
    badrl: bool = False
    p_rate: float = 0.0003
    alpha: float = 0.5
    rew_p: float = 5.0
    untargeted_reward: float = 1.0
    simple_select: bool = False

    poisoned_model_path: str = ""
    healed_model_path: str = ""
    lens_model_path: str = ""

    eval_episodes: int = 20
    eval_max_steps: int = 10000
    eval_no_reward_timeout: int = 300
    diag_steps: int = 1000
    eval_batch_size: int = 256

    output_path: str = "healed_evaluation_result.png"
    output_csv: str = "healed_evaluation_result.csv"


def infer_attack_name(args: EvalArgs) -> str:
    enabled = []
    if args.sn_outer:
        enabled.append("SN")
    if args.trojdrl:
        enabled.append("troj1")
    if args.untargeted:
        enabled.append("untarget")
    if args.badrl:
        enabled.append("badrl")
    if len(enabled) != 1:
        raise ValueError(
            f"Exactly one attack must be enabled; received {enabled or 'none'}."
        )
    return enabled[0]


def validate_args(args: EvalArgs, require_healed: bool = True) -> None:
    infer_attack_name(args)
    if min(
        args.num_envs,
        args.eval_episodes,
        args.eval_max_steps,
        args.eval_no_reward_timeout,
        args.diag_steps,
        args.eval_batch_size,
    ) < 1:
        raise ValueError("Evaluation sizes and step counts must be positive.")
    if not 0.0 < args.p_rate <= 1.0:
        raise ValueError("p_rate must be in (0, 1].")
    if not 0.0 <= args.alpha <= 1.0 or args.rew_p <= 0.0:
        raise ValueError("alpha must be in [0, 1] and rew_p must be positive.")
    paths = [("poisoned_model_path", args.poisoned_model_path)]
    if require_healed:
        paths.append(("healed_model_path", args.healed_model_path))
    for name, path in paths:
        if not path:
            raise ValueError(f"{name} must be provided.")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{name} does not exist: {path}")


def build_pipeline_args(args: EvalArgs) -> PipelineArgs:
    return PipelineArgs(
        env_id=args.env_id,
        seed=args.seed,
        cuda=args.cuda,
        num_envs=args.num_envs,
        target_action=args.target_action,
        sn_outer=args.sn_outer,
        trojdrl=args.trojdrl,
        untargeted=args.untargeted,
        badrl=args.badrl,
        p_rate=args.p_rate,
        alpha=args.alpha,
        rew_p=args.rew_p,
        untargeted_reward=args.untargeted_reward,
        simple_select=args.simple_select,
        eval_episodes=args.eval_episodes,
        eval_max_steps=args.eval_max_steps,
        eval_no_reward_timeout=args.eval_no_reward_timeout,
        eval_batch_size=args.eval_batch_size,
    )


def load_agent(path: str, envs, device: torch.device) -> Agent:
    agent = Agent(envs).to(device)
    agent.load_state_dict(torch.load(path, map_location=device))
    agent.eval()
    return agent


@torch.no_grad()
def collect_reference_states(
    agent: Agent,
    envs,
    args: EvalArgs,
    device: torch.device,
) -> torch.Tensor:
    observation, _ = envs.reset(seed=args.seed + 200)
    observation_t = torch.as_tensor(
        observation,
        dtype=torch.float32,
        device=device,
    )
    chunks = []
    collected = 0
    while collected < args.diag_steps:
        take = min(args.diag_steps - collected, observation_t.shape[0])
        chunks.append(
            observation_t[:take]
            .to(device="cpu", dtype=torch.uint8)
            .clone()
        )
        collected += take
        action = agent.get_policy_logits(observation_t).argmax(dim=-1)
        next_observation, _, _, _, _ = envs.step(action.cpu().numpy())
        observation_t = torch.as_tensor(
            next_observation,
            dtype=torch.float32,
            device=device,
        )
    return torch.cat(chunks, dim=0)


@torch.no_grad()
def evaluate_state_diagnostics(
    agent: Agent,
    reference_states: torch.Tensor,
    injector: AttackInjector,
    pipeline_args: PipelineArgs,
    args: EvalArgs,
    device: torch.device,
) -> Dict[str, object]:
    num_actions = agent.actor.out_features
    clean_action_counts = np.zeros(num_actions, dtype=np.int64)
    triggered_action_counts = np.zeros(num_actions, dtype=np.int64)
    changed = 0
    target_hits = 0
    clean_entropy_sum = 0.0
    triggered_entropy_sum = 0.0
    sample_count = int(reference_states.shape[0])

    for start in range(0, sample_count, args.eval_batch_size):
        clean_obs = reference_states[
            start:start + args.eval_batch_size
        ].to(device=device, dtype=torch.float32)
        mask = torch.ones(
            clean_obs.shape[0],
            device=device,
            dtype=torch.bool,
        )
        triggered_obs = apply_trigger_weak(clean_obs, mask, injector)
        clean_logits = agent.get_policy_logits(clean_obs)
        triggered_logits = agent.get_policy_logits(triggered_obs)
        clean_actions = clean_logits.argmax(dim=-1)
        triggered_actions = triggered_logits.argmax(dim=-1)

        clean_np = clean_actions.cpu().numpy()
        triggered_np = triggered_actions.cpu().numpy()
        clean_action_counts += np.bincount(
            clean_np,
            minlength=num_actions,
        )
        triggered_action_counts += np.bincount(
            triggered_np,
            minlength=num_actions,
        )
        changed += int((clean_actions != triggered_actions).sum().item())
        if not pipeline_args.untargeted:
            target_hits += int(
                (triggered_actions == pipeline_args.target_action).sum().item()
            )
        clean_entropy_sum += float(
            Categorical(logits=clean_logits).entropy().sum().item()
        )
        triggered_entropy_sum += float(
            Categorical(logits=triggered_logits).entropy().sum().item()
        )

    disagreement = 100.0 * changed / sample_count
    target_rate = (
        float("nan")
        if pipeline_args.untargeted
        else 100.0 * target_hits / sample_count
    )
    primary_metric = disagreement if pipeline_args.untargeted else target_rate
    return {
        "attack_metric_percent": primary_metric,
        "action_disagreement_percent": disagreement,
        "target_action_rate_percent": target_rate,
        "clean_entropy": clean_entropy_sum / sample_count,
        "triggered_entropy": triggered_entropy_sum / sample_count,
        "clean_action_distribution": clean_action_counts / sample_count,
        "triggered_action_distribution": (
            triggered_action_counts / sample_count
        ),
        "changed_actions": changed,
        "target_actions": target_hits,
        "sample_count": sample_count,
    }


def evaluate_checkpoint(
    agent: Agent,
    eval_envs,
    reference_states: torch.Tensor,
    injector: AttackInjector,
    pipeline_args: PipelineArgs,
    args: EvalArgs,
    device: torch.device,
) -> Dict[str, object]:
    eval_seed = args.seed + 10000
    triggered_return, triggered_std, triggered_n = (
        evaluate_fixed_seed_return(
            agent,
            eval_envs,
            injector,
            pipeline_args,
            device,
            eval_seed,
            triggered=True,
        )
    )
    clean_return, clean_std, clean_n = evaluate_fixed_seed_return(
        agent,
        eval_envs,
        injector,
        pipeline_args,
        device,
        eval_seed,
        triggered=False,
    )
    metrics = evaluate_state_diagnostics(
        agent,
        reference_states,
        injector,
        pipeline_args,
        args,
        device,
    )
    metrics.update({
        "clean_return": clean_return,
        "clean_return_std": clean_std,
        "clean_return_n": clean_n,
        "triggered_return": triggered_return,
        "triggered_return_std": triggered_std,
        "triggered_return_n": triggered_n,
    })
    return metrics


def evaluate_checkpoints(
    args: EvalArgs,
    repaired_paths: Mapping[str, str],
) -> Tuple[Dict[str, object], Dict[str, Dict[str, object]]]:
    validate_args(args, require_healed=False)
    for label, path in repaired_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Repaired checkpoint {label!r} does not exist: {path}"
            )

    reset_all_seeds(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
    )
    pipeline_args = build_pipeline_args(args)
    attack = infer_attack_name(args)
    print(f"Evaluation attack={attack} | device={device}")

    diagnostic_envs = gym.vector.SyncVectorEnv([
        make_eval_env(
            args.env_id,
            args.seed + 200 + index,
            args.eval_no_reward_timeout,
        )
        for index in range(args.num_envs)
    ])
    eval_envs = gym.vector.SyncVectorEnv([
        make_eval_env(
            args.env_id,
            args.seed + 10000 + index,
            args.eval_no_reward_timeout,
        )
        for index in range(args.eval_episodes)
    ])

    try:
        injector = AttackInjector(pipeline_args, device)
        if (
            not pipeline_args.untargeted
            and not (
                0
                <= pipeline_args.target_action
                < diagnostic_envs.single_action_space.n
            )
        ):
            raise ValueError(
                f"target_action={pipeline_args.target_action} is outside "
                f"[0, {diagnostic_envs.single_action_space.n})."
            )
        poisoned_agent = load_agent(
            args.poisoned_model_path,
            diagnostic_envs,
            device,
        )
        reference_states = collect_reference_states(
            poisoned_agent,
            diagnostic_envs,
            args,
            device,
        )
        poisoned_metrics = evaluate_checkpoint(
            poisoned_agent,
            eval_envs,
            reference_states,
            injector,
            pipeline_args,
            args,
            device,
        )

        repaired_metrics = {}
        for label, path in repaired_paths.items():
            print(f"Evaluating repaired checkpoint {label}: {path}")
            repaired_agent = load_agent(path, diagnostic_envs, device)
            repaired_metrics[label] = evaluate_checkpoint(
                repaired_agent,
                eval_envs,
                reference_states,
                injector,
                pipeline_args,
                args,
                device,
            )
            del repaired_agent
    finally:
        diagnostic_envs.close()
        eval_envs.close()

    return poisoned_metrics, repaired_metrics


def comparison_row(
    attack: str,
    poisoned: Dict[str, object],
    repaired: Dict[str, object],
    repaired_path: str,
) -> Dict[str, object]:
    return {
        "Attack": attack,
        "AttackMetric": (
            "action_disagreement"
            if attack == "untarget"
            else "target_action_rate"
        ),
        "RepairedModel": repaired_path,
        "PoisonedAttackMetricPercent": poisoned[
            "attack_metric_percent"
        ],
        "RepairedAttackMetricPercent": repaired[
            "attack_metric_percent"
        ],
        "AttackMetricReductionPP": (
            float(poisoned["attack_metric_percent"])
            - float(repaired["attack_metric_percent"])
        ),
        "PoisonedActionDisagreementPercent": poisoned[
            "action_disagreement_percent"
        ],
        "RepairedActionDisagreementPercent": repaired[
            "action_disagreement_percent"
        ],
        "PoisonedTargetActionRatePercent": poisoned[
            "target_action_rate_percent"
        ],
        "RepairedTargetActionRatePercent": repaired[
            "target_action_rate_percent"
        ],
        "PoisonedTriggeredEntropy": poisoned["triggered_entropy"],
        "RepairedTriggeredEntropy": repaired["triggered_entropy"],
        "PoisonedTriggeredReturn": poisoned["triggered_return"],
        "RepairedTriggeredReturn": repaired["triggered_return"],
        "PoisonedCleanReturn": poisoned["clean_return"],
        "RepairedCleanReturn": repaired["clean_return"],
        "CleanReturnChange": (
            float(repaired["clean_return"])
            - float(poisoned["clean_return"])
        ),
        "DiagnosticSamples": repaired["sample_count"],
        "EvaluationEpisodes": repaired["clean_return_n"],
    }


def write_csv(row: Dict[str, object], output_path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    with open(output_path, mode="w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def plot_comparison(
    poisoned: Dict[str, object],
    repaired: Dict[str, object],
    args: EvalArgs,
) -> None:
    poisoned_dist = np.asarray(
        poisoned["triggered_action_distribution"],
        dtype=np.float64,
    )
    repaired_dist = np.asarray(
        repaired["triggered_action_distribution"],
        dtype=np.float64,
    )
    actions = np.arange(poisoned_dist.size)
    width = 0.35

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        actions - width / 2,
        poisoned_dist,
        width,
        label="Poisoned",
    )
    axis.bar(
        actions + width / 2,
        repaired_dist,
        width,
        label="Repaired",
    )
    axis.set_xlabel("Action")
    axis.set_ylabel("Frequency")
    axis.set_title(
        f"{infer_attack_name(args)}: action distribution under weak trigger"
    )
    axis.set_xticks(actions)
    axis.legend()
    figure.tight_layout()

    output_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(output_dir, exist_ok=True)
    figure.savefig(args.output_path, dpi=200)
    plt.close(figure)


def print_comparison(row: Dict[str, object]) -> None:
    print("\nIndependent poisoned-versus-repaired evaluation")
    print(
        f"Attack metric: "
        f"{float(row['PoisonedAttackMetricPercent']):.2f}% -> "
        f"{float(row['RepairedAttackMetricPercent']):.2f}%"
    )
    print(
        f"Triggered return: "
        f"{float(row['PoisonedTriggeredReturn']):.2f} -> "
        f"{float(row['RepairedTriggeredReturn']):.2f}"
    )
    print(
        f"Clean return: "
        f"{float(row['PoisonedCleanReturn']):.2f} -> "
        f"{float(row['RepairedCleanReturn']):.2f}"
    )


def main(args: EvalArgs) -> None:
    validate_args(args)
    poisoned, repaired_map = evaluate_checkpoints(
        args,
        {"repaired": args.healed_model_path},
    )
    repaired = repaired_map["repaired"]
    row = comparison_row(
        infer_attack_name(args),
        poisoned,
        repaired,
        args.healed_model_path,
    )
    print_comparison(row)
    if args.output_csv:
        write_csv(row, args.output_csv)
        print(f"Saved evaluation CSV: {args.output_csv}")
    if args.output_path:
        plot_comparison(poisoned, repaired, args)
        print(f"Saved action-distribution plot: {args.output_path}")


if __name__ == "__main__":
    main(tyro.cli(EvalArgs))
