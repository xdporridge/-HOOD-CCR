# CLEAR

Anonymous implementation of CLEAR for detecting and repairing backdoored Atari
policies.

## Files

| File | Purpose |
| --- | --- |
| `ppo.py` | Trains clean or backdoored PPO policies. |
| `train_lens_sn_outer.py` | Trains critic lenses from clean observations. |
| `repair_lens.py` | Runs the full CLEAR detection, interception, and repair pipeline. |
| `repair_lens_q_sweep.py` | Compares detector and repair results across calibration quantiles. |
| `test_asr_lens.py` | Evaluates the residual detector and attack success rate. |
| `test_lens_jing.py` | Compares poisoned and repaired policies. |
| `test_q_sweep_models.py` | Independently evaluates every repaired model from a q sweep. |
| `Adversary.py` | Implements the supported backdoor attacks. |
| `patterns.py` | Defines visual trigger patterns. |

## Setup

```bash
python -m pip install -r requirements.txt
AutoROM --accept-license
```

## Train a poisoned policy

```bash
# SleeperNets
python ppo.py --env-id BreakoutNoFrameskip-v4 --sn-outer \
  --p-rate 0.0003 --target-action 2 --total-timesteps 20000000

# Targeted TrojDRL
python ppo.py --env-id BreakoutNoFrameskip-v4 --trojdrl \
  --p-rate 0.0003 --target-action 2 --total-timesteps 20000000

# BadRL
python ppo.py --env-id BreakoutNoFrameskip-v4 --badrl --strong \
  --p-rate 0.0003 --target-action 2 --total-timesteps 20000000

# Untargeted TrojDRL
python ppo.py --env-id BreakoutNoFrameskip-v4 --untargeted \
  --p-rate 0.001 --untargeted-reward 1.0 --total-timesteps 20000000
```

## Train a critic lens

```bash
python train_lens_sn_outer.py \
  --model-path <path/to/poisoned_policy.pth> \
  --lens-save-dir <path/to/lens_output>
```

## Run CLEAR

Run one attack:

```bash
python repair_lens.py \
  --attacks SN \
  --poisoned-model-path <path/to/poisoned_policy.pth> \
  --lens-model-path <path/to/multi_layer_lens_clean_first_two_new.npz> \
  --output-root defense_outputs
```

Use `SN`, `troj1`, `badrl`, or `untarget` as the attack name.

Run all attacks:

```bash
python repair_lens.py \
  --attacks all \
  --sn-model-path <path/to/sn_policy.pth> \
  --sn-lens-model-path <path/to/sn_lens.npz> \
  --troj1-model-path <path/to/trojdrl_policy.pth> \
  --troj1-lens-model-path <path/to/trojdrl_lens.npz> \
  --badrl-model-path <path/to/badrl_policy.pth> \
  --badrl-lens-model-path <path/to/badrl_lens.npz> \
  --untarget-model-path <path/to/untargeted_policy.pth> \
  --untarget-lens-model-path <path/to/untargeted_lens.npz> \
  --output-root defense_outputs
```

Show all available options:

```bash
python repair_lens.py --help
```

## Run the q-value experiment

```bash
python repair_lens_q_sweep.py \
  --attacks SN \
  --poisoned-model-path <path/to/poisoned_policy.pth> \
  --lens-model-path <path/to/multi_layer_lens_clean_first_two_new.npz> \
  --output-root q_sweep_outputs
```

The default q values are `0.975,0.99,0.995,0.999`. The comparison is saved as
`q_sweep_comparison.csv`.

Evaluate all repaired q-sweep models:

```bash
python test_q_sweep_models.py \
  --comparison-csv <path/to/q_sweep_comparison.csv> \
  --poisoned-model-path <path/to/poisoned_policy.pth> \
  --target-action 2
```

## Evaluate the detector

`test_asr_lens.py` takes the policy file and the directory containing the lens
archives:

```bash
python test_asr_lens.py \
  --model-path <path/to/poisoned_policy.pth> \
  --lens-save-dir <path/to/lens_out> \
  --target-action 2
```
