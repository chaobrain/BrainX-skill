import numpy as np

from criticality_scan import Experiment, avalanche_observables, summarize_scan


def test_analysis_selects_variable_stable_gain_before_runaway():
    config = Experiment(
        n_exc=8,
        n_inh=2,
        n_realizations=4,
        gains=(0.8, 1.0, 1.2),
        quiet_ms=2.0,
        late_window_ms=2.0,
        runaway_rate_hz=100.0,
        max_unstable_fraction=0.25,
    )
    # Five 1-ms bins per gain/realization lane. The middle gain is variable but
    # ends; every high-gain lane remains active into the late window.
    binned = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 2, 4, 8, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 1, 2, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    # Expand each 1-ms bin to the five DT=0.2-ms simulation steps.
    counts = np.repeat(binned, 5, axis=0)
    observables = avalanche_observables(counts, config)
    _, summary = summarize_scan(observables, np.asarray(config.gains), config)

    assert summary["selected_gain"] == 1.0


def test_analysis_rejects_an_uninformative_flat_scan():
    config = Experiment(
        n_exc=8,
        n_inh=2,
        n_realizations=4,
        gains=(0.8, 1.0),
        quiet_ms=2.0,
        late_window_ms=2.0,
    )
    counts = np.zeros((25, 8), dtype=np.int32)
    counts[0] = 1
    observables = avalanche_observables(counts, config)

    with np.testing.assert_raises_regex(RuntimeError, "no avalanche-size variability"):
        summarize_scan(observables, np.asarray(config.gains), config)
