"""Fit two static NEST Simulator weights and test their identifiability."""

import importlib.metadata
import json
import platform

import brainspike
import brainstate
import braintools
import brainunit as u
import jax
import jax.numpy as jnp
import numpy as np

brainstate.environ.set(precision=64)

DT_MS = DELAY_MS = 0.1
DURATION_MS = 24.0
INITIAL_MV = -65.0
THRESHOLD_MV = -55.0  # iaf_psc_alpha default
SCALE_PA = 100.0
TRUTH_PA = (170.0, 110.0)
STARTS_PA = ((40.0, 50.0), (240.0, 65.0), (95.0, 190.0))
CONTROL_PA = ((200.0, 80.0), (120.0, 160.0))
SPIKES_MS = {
    "train": ((1.0, 2.0, 3.0), (12.0, 13.0, 14.0)),
    "validation": ((2.0, 6.0, 10.0), (3.0, 7.0, 11.0)),
    "control": ((1.0, 2.0, 3.0), (1.0, 2.0, 3.0)),
}
STEPS = 120
LR = 0.05
H = 1e-4
LIMITS = {
    "weight_relative_error": 0.01,
    "validation_rmse_mv": 0.02,
    "gradient_relative_error": 0.01,
    "gradient_near_zero": 1e-6,
    "gradient_absolute_error_near_zero": 1e-8,
    "gradient_check_min_loss_mv2": 1e-3,
    "repeat_max_difference": 0.0,
    "train_min_s2_over_s1": 1e-2,
    "control_max_s2_over_s1": 1e-6,
    "control_max_trace_difference_mv": 1e-8,
}


def rollout(theta, protocol):
    sim = brainspike.Simulator(dt=DT_MS * u.ms)
    neuron = sim.create(brainspike.iaf_psc_alpha, 1, params={
        "I_e": 0.0 * u.pA,
        "V_initializer": braintools.init.Constant(INITIAL_MV * u.mV),
    })
    for i, times in enumerate(SPIKES_MS[protocol]):
        source = sim.create(brainspike.spike_generator, spike_times=jnp.asarray(times) * u.ms)
        sim.connect(source, neuron, weight=theta[i] * SCALE_PA * u.pA,
                    delay=DELAY_MS * u.ms, comm="dense")
    meter = sim.create(brainspike.multimeter, record_from=["V_m"], interval=DT_MS * u.ms)
    recorder = sim.create(brainspike.spike_recorder)
    sim.connect(meter, neuron)
    sim.connect(neuron, recorder)
    result = sim.simulate(DURATION_MS * u.ms)
    return result.trace(meter, "V_m").to_decimal(u.mV)[:, 0], result.spikes(recorder)[:, 0]


def activity(voltage, spikes):
    return jnp.max(voltage), jnp.sum(spikes)


def activity_metrics(observations):
    peak = max(float(peak) for peak, _ in observations)
    count = sum(int(count) for _, count in observations)
    return {"peak_mv": peak, "spike_count": count,
            "subthreshold": peak < THRESHOLD_MV and count == 0}


def sensitivity(theta, protocol):
    # Differentiate the fresh rollout, with physical pA as the Jacobian coordinate.
    jacobian = jax.jacrev(lambda pa: rollout(pa / SCALE_PA, protocol)[0])(theta * SCALE_PA)
    singular = np.linalg.svd(np.asarray(jacobian), compute_uv=False)
    return {"at_weights_pa": np.asarray(theta * SCALE_PA).tolist(),
            "shape": list(jacobian.shape), "singular_values_mv_per_pa": singular.tolist(),
            "s2_over_s1": float(singular[1] / singular[0])}


def gradient_check(theta, objective, value_and_grad):
    (loss, _), gradient = value_and_grad(theta)
    rows = []
    for i, autodiff in enumerate(np.asarray(gradient)):
        step = jnp.zeros_like(theta).at[i].set(H)
        fd = float((objective(theta + step)[0] - objective(theta - step)[0]) / (2 * H))
        absolute = abs(fd - autodiff)
        relative = absolute / abs(autodiff) if autodiff != 0 else None
        near_zero = abs(autodiff) < LIMITS["gradient_near_zero"]
        passed = (absolute < LIMITS["gradient_absolute_error_near_zero"] if near_zero
                  else relative < LIMITS["gradient_relative_error"])
        rows.append({"source": i + 1, "autodiff": float(autodiff), "finite_difference": fd,
                     "absolute_error": float(absolute),
                     "relative_error": None if relative is None else float(relative),
                     "near_zero": bool(near_zero), "passed": bool(passed)})
    return {"at_weights_pa": np.asarray(theta * SCALE_PA).tolist(), "loss_mv2": float(loss),
            "step_normalized": H, "step_pa": H * SCALE_PA, "components": rows,
            "passed": float(loss) > LIMITS["gradient_check_min_loss_mv2"]
            and all(row["passed"] for row in rows)}


def fit(start_pa, value_and_grad, objective):
    candidate = brainstate.ParamState(jnp.asarray(start_pa) / SCALE_PA)
    optimizer = braintools.optim.Adam(lr=LR)
    optimizer.register_trainable_weights({"weights": candidate})
    observations = []
    for _ in range(STEPS):
        (loss, observed), grad = value_and_grad(candidate.value)
        if not observations:
            initial_loss = float(loss)
        observations.append(observed)
        optimizer.update({"weights": grad})
    loss, observed = objective(candidate.value)
    observations.append(observed)
    return candidate.value, {
        "initial_weights_pa": list(start_pa), "initial_loss_mv2": initial_loss,
        "final_loss_mv2": float(loss), "training_activity": activity_metrics(observations),
    }


def experiment():
    config = {
        "dt_ms": DT_MS, "delay_ms": DELAY_MS, "duration_ms": DURATION_MS,
        "initial_mv": INITIAL_MV, "threshold_mv": THRESHOLD_MV, "external_current_pa": 0.0,
        "neuron": "iaf_psc_alpha", "synapse": "static (default)", "comm": "dense",
        "spike_times_ms": SPIKES_MS, "weight_scale_pa": SCALE_PA, "truth_pa": TRUTH_PA,
        "initial_weights_pa": STARTS_PA, "control_weights_pa": CONTROL_PA,
        "optimizer": {"name": "Adam", "lr": LR, "steps": STEPS},
        "acceptance": LIMITS, "float_precision_bits": 64,
    }
    versions = {name: importlib.metadata.version(name) for name in (
        "brainx", "brainspikes", "brainpy", "brainstate", "brainunit", "braintools",
        "jax", "jaxlib", "numpy",
    )}
    print("CONFIG", json.dumps(config), flush=True)
    print("VERSIONS", json.dumps(versions), flush=True)
    truth = jnp.asarray(TRUTH_PA) / SCALE_PA
    target, target_spikes = rollout(truth, "train")

    def objective(theta):
        voltage, spikes = rollout(theta, "train")
        return jnp.mean((voltage - target) ** 2), activity(voltage, spikes)

    # No outer JIT: spike-time validation runs on the host during construction.
    value_and_grad = jax.value_and_grad(objective, has_aux=True)
    probe = jnp.asarray(STARTS_PA[0]) / SCALE_PA
    voltage_a, spikes_a = rollout(probe, "train")
    voltage_b, spikes_b = rollout(probe, "train")
    losses = [float(jnp.mean((v - target) ** 2)) for v in (voltage_a, voltage_b)]
    repeat = {"at_weights_pa": STARTS_PA[0], "losses_mv2": losses,
              "loss_abs_difference_mv2": abs(losses[0] - losses[1]),
              "trace_max_difference_mv": float(jnp.max(jnp.abs(voltage_a - voltage_b))),
              "activity": activity_metrics([activity(voltage_a, spikes_a), activity(voltage_b, spikes_b)])}
    gradients = gradient_check(probe, objective, value_and_grad)
    train_sensitivity = sensitivity(probe, "train")
    print("GRADIENTS", json.dumps(gradients), flush=True)
    print("TRAIN_SENSITIVITY", json.dumps(train_sensitivity), flush=True)

    control_a, control_b = (jnp.asarray(pa) / SCALE_PA for pa in CONTROL_PA)
    va, sa = rollout(control_a, "control")
    vb, sb = rollout(control_b, "control")
    control = {
        "weights_pa": CONTROL_PA, "total_weight_pa": sum(CONTROL_PA[0]),
        "trace_max_difference_mv": float(jnp.max(jnp.abs(va - vb))),
        "sensitivity": sensitivity(control_a, "control"),
        "activity": activity_metrics([activity(va, sa), activity(vb, sb)]),
    }
    print("CONTROL", json.dumps(control), flush=True)

    fits = [fit(start, value_and_grad, objective) for start in STARTS_PA]
    # Held-out target and scores are computed only after all optimizer updates.
    held_out, held_out_spikes = rollout(truth, "validation")
    runs = []
    for theta, run in fits:
        fitted = np.asarray(theta * SCALE_PA)
        errors_pa = np.abs(fitted - TRUTH_PA)
        relative = errors_pa / np.abs(TRUTH_PA)
        prediction, spikes = rollout(theta, "validation")
        rmse = float(jnp.sqrt(jnp.mean((prediction - held_out) ** 2)))
        validation_activity = activity_metrics([activity(prediction, spikes)])
        run.update(
            fitted_weights_pa=fitted.tolist(), weight_absolute_errors_pa=errors_pa.tolist(),
            weight_relative_errors=relative.tolist(), validation_rmse_mv=rmse,
            validation_activity=validation_activity,
            passed=bool(np.all(relative < LIMITS["weight_relative_error"])
                        and rmse < LIMITS["validation_rmse_mv"]
                        and run["training_activity"]["subthreshold"]
                        and validation_activity["subthreshold"]),
        )
        print("FIT", json.dumps(run), flush=True)
        runs.append(run)

    target_activity = activity_metrics([activity(target, target_spikes),
                                        activity(held_out, held_out_spikes)])
    checks = {
        "all_fits": all(run["passed"] for run in runs),
        "targets_subthreshold": target_activity["subthreshold"],
        "repeatable": repeat["trace_max_difference_mv"] <= LIMITS["repeat_max_difference"]
        and repeat["loss_abs_difference_mv2"] <= LIMITS["repeat_max_difference"]
        and repeat["activity"]["subthreshold"],
        "gradient_agreement": gradients["passed"],
        "training_distinguishable": train_sensitivity["s2_over_s1"] > LIMITS["train_min_s2_over_s1"],
        "control_indistinguishable": control["trace_max_difference_mv"] < LIMITS["control_max_trace_difference_mv"]
        and control["sensitivity"]["s2_over_s1"] < LIMITS["control_max_s2_over_s1"]
        and control["activity"]["subthreshold"],
    }
    return {"config": config, "versions": versions, "python": platform.python_version(),
            "jax_backend": jax.default_backend(), "runs": runs, "target_activity": target_activity,
            "repeat": repeat, "gradients": gradients, "training_sensitivity": train_sensitivity,
            "control": control, "checks": checks, "passed": all(checks.values())}


if __name__ == "__main__":
    metrics = experiment()
    print("METRICS", json.dumps(metrics, allow_nan=False), flush=True)
    raise SystemExit(0 if metrics["passed"] else 1)
