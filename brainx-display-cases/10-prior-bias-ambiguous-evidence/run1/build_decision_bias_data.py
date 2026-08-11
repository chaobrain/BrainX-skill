from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import brainmass
import brainstate
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "decision-prior-bias-data.json"
TEMPLATE_PATH = ROOT / "decision-prior-bias.template.html"
OUTPUT_PATH = ROOT / "decision-prior-bias.html"

DT = 0.1 * u.ms
DURATION_MS = 800.0
N_STEPS = int(DURATION_MS / float(DT.to(u.ms).mantissa))
N_TRIALS = 256
NOISE_SIGMA = 0.02 * u.nA
PRIOR_BIAS = 0.02
EVIDENCE = np.asarray(
    [-0.50, -0.30, -0.18, -0.10, -0.04, 0.0, 0.04, 0.10, 0.18, 0.30, 0.50],
    dtype=np.float32,
)


def wilson_interval(wins: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = wins / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denominator
    spread = z * np.sqrt((p * (1.0 - p) + z * z / (4.0 * total)) / total) / denominator
    return float(center - spread), float(center + spread)


def run_condition(bias: float, seed: int = 91):
    """Run one matched-noise ensemble across all evidence levels."""
    brainstate.random.seed(seed)
    evidence_by_trial = np.repeat(EVIDENCE, N_TRIALS)
    model = brainmass.WongWangStep(
        in_size=evidence_by_trial.size,
        noise_s1=brainmass.GaussianNoise(evidence_by_trial.size, sigma=NOISE_SIGMA),
        noise_s2=brainmass.GaussianNoise(evidence_by_trial.size, sigma=NOISE_SIGMA),
    )
    brainstate.nn.init_all_states(model)
    drive = jnp.asarray(evidence_by_trial + bias)

    zero_start = int(np.where(EVIDENCE == 0.0)[0][0]) * N_TRIALS
    example_pool = jnp.arange(zero_start, zero_start + 32)

    def step(_):
        model.update(coherence=drive)
        difference = model.S1.value - model.S2.value
        return difference[example_pool]

    with brainstate.environ.context(dt=DT):
        example_trajectories = brainstate.transform.for_loop(step, np.arange(N_STEPS))
    jax.block_until_ready(example_trajectories)

    final_difference = np.asarray(model.S1.value - model.S2.value)
    choices = (final_difference > 0.0).reshape(EVIDENCE.size, N_TRIALS)
    return choices, np.asarray(example_trajectories)


def select_examples(unbiased_choices, biased_choices, pool_size: int = 32) -> list[int]:
    zero_index = int(np.where(EVIDENCE == 0.0)[0][0])
    unbiased = unbiased_choices[zero_index, :pool_size]
    biased = biased_choices[zero_index, :pool_size]

    groups = [
        np.flatnonzero((~unbiased) & biased),
        np.flatnonzero(unbiased & biased),
        np.flatnonzero((~unbiased) & (~biased)),
        np.flatnonzero(unbiased & (~biased)),
    ]
    selected: list[int] = []
    targets = [2, 2, 2, 0]
    for candidates, target in zip(groups, targets):
        selected.extend(int(index) for index in candidates[:target])
    if len(selected) < 6:
        selected.extend(index for index in range(pool_size) if index not in selected)
    return selected[:6]


def benchmark_batch(batch_size: int, repeats: int = 3) -> dict[str, float | int]:
    brainstate.random.seed(700 + batch_size)
    model = brainmass.WongWangStep(
        in_size=batch_size,
        noise_s1=brainmass.GaussianNoise(batch_size, sigma=NOISE_SIGMA),
        noise_s2=brainmass.GaussianNoise(batch_size, sigma=NOISE_SIGMA),
    )
    brainstate.nn.init_all_states(model)
    drive = jnp.zeros((batch_size,), dtype=jnp.float32)
    steps = np.arange(N_STEPS)

    def run_once():
        def step(_):
            model.update(coherence=drive)
            return model.S1.value[0] - model.S2.value[0]

        with brainstate.environ.context(dt=DT):
            output = brainstate.transform.for_loop(step, steps)
        jax.block_until_ready(output)

    start = time.perf_counter()
    run_once()
    first_seconds = time.perf_counter() - start

    steady_times = []
    for _ in range(repeats):
        start = time.perf_counter()
        run_once()
        steady_times.append(time.perf_counter() - start)

    steady_seconds = statistics.median(steady_times)
    return {
        "batch": batch_size,
        "firstSeconds": round(first_seconds, 6),
        "steadySeconds": round(steady_seconds, 6),
        "firstThroughput": round(batch_size / first_seconds, 2),
        "steadyThroughput": round(batch_size / steady_seconds, 2),
    }


def build_data() -> dict:
    brainstate.environ.set(dt=DT)
    unbiased_choices, unbiased_pool = run_condition(0.0)
    biased_choices, biased_pool = run_condition(PRIOR_BIAS)

    selected = select_examples(unbiased_choices, biased_choices)
    sample_every = 50
    times = (np.arange(0, N_STEPS, sample_every) + 1) * float(DT.to(u.ms).mantissa)

    examples = []
    zero_index = int(np.where(EVIDENCE == 0.0)[0][0])
    for trial_index in selected:
        examples.append(
            {
                "trial": trial_index + 1,
                "unbiased": np.round(unbiased_pool[::sample_every, trial_index], 5).tolist(),
                "biased": np.round(biased_pool[::sample_every, trial_index], 5).tolist(),
                "unbiasedChoice": 1 if unbiased_choices[zero_index, trial_index] else 2,
                "biasedChoice": 1 if biased_choices[zero_index, trial_index] else 2,
            }
        )

    psychometric = []
    for condition, choices in (("Unbiased", unbiased_choices), ("Prior +0.02", biased_choices)):
        for evidence, row in zip(EVIDENCE, choices):
            wins = int(row.sum())
            low, high = wilson_interval(wins, N_TRIALS)
            psychometric.append(
                {
                    "condition": condition,
                    "evidence": round(float(evidence), 2),
                    "probability": round(wins / N_TRIALS, 5),
                    "low": round(low, 5),
                    "high": round(high, 5),
                    "wins": wins,
                    "trials": N_TRIALS,
                }
            )

    p_unbiased = unbiased_choices.mean(axis=1)
    p_biased = biased_choices.mean(axis=1)
    zero_delta = float(p_biased[zero_index] - p_unbiased[zero_index])
    strong_delta = float(
        np.mean(
            np.abs(
                [
                    p_biased[0] - p_unbiased[0],
                    p_biased[-1] - p_unbiased[-1],
                ]
            )
        )
    )

    speed = [benchmark_batch(batch) for batch in (1, 16, 64, 256, 1024)]
    device = jax.devices()[0]
    device_name = getattr(device, "device_kind", device.platform)

    return {
        "model": "Wong-Wang two-choice neural mass",
        "dtMs": float(DT.to(u.ms).mantissa),
        "durationMs": DURATION_MS,
        "trialsPerPoint": N_TRIALS,
        "noiseSigmaNa": float(NOISE_SIGMA.to(u.nA).mantissa),
        "priorBias": PRIOR_BIAS,
        "timesMs": np.round(times, 1).tolist(),
        "examples": examples,
        "psychometric": psychometric,
        "zeroDelta": round(zero_delta, 5),
        "strongDelta": round(strong_delta, 5),
        "speed": speed,
        "device": str(device_name),
        "platform": device.platform,
    }


def main():
    data = build_data()
    DATA_PATH.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    if "__DECISION_DATA__" not in template:
        raise RuntimeError("Data placeholder missing from visualization template")
    OUTPUT_PATH.write_text(
        template.replace("__DECISION_DATA__", json.dumps(data, separators=(",", ":"))),
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "bytes": OUTPUT_PATH.stat().st_size,
        "zeroDelta": data["zeroDelta"],
        "strongDelta": data["strongDelta"],
        "speed": data["speed"],
        "examples": [
            [item["trial"], item["unbiasedChoice"], item["biasedChoice"]]
            for item in data["examples"]
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
