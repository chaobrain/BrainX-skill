import unittest

import numpy as np

import celegans_muscle_inference as case


class CelegansMuscleInferenceTest(unittest.TestCase):
    def test_data_split_and_sampling(self):
        time_ms, traces = case.load_experimental_traces()
        self.assertEqual(time_ms.shape, (5000,))
        self.assertEqual(set(traces), {6, 7, 8, 9})
        self.assertAlmostEqual(time_ms[1] - time_ms[0], 0.1)
        self.assertEqual(case.CURRENT_BY_TRACE[9], 30.0)
        self.assertEqual(
            {trace: case.CURRENT_BY_TRACE[trace] for trace in (6, 7, 8)},
            {6: 15.0, 7: 20.0, 8: 25.0},
        )

        training_summary = case.summarize_traces(traces[9], time_ms)[0]
        self.assertEqual(training_summary[0], 4.0)
        self.assertEqual(training_summary[-1], 0.0)

    def test_model_has_six_currents_and_seven_states(self):
        parameters = np.array([19.8, 37.0, 0.1, 22.0, 3.6, 10.0])
        cell = case.CelegansMuscleCell(parameters)
        cell.init_state()
        components = cell.currents.current_components(cell.V.value)
        self.assertEqual(set(components), {"EGL-19", "SHK-1", "SLO-2", "Kr", "Na", "Leak"})
        self.assertEqual(
            [name for name in ("m", "h", "n", "p", "q", "ca") if hasattr(cell.currents, name)],
            ["m", "h", "n", "p", "q", "ca"],
        )
        for current in components.values():
            converted = current.to_decimal(case.u.pA)
            self.assertTrue(np.all(np.isfinite(converted)))

    def test_short_rollout_is_finite(self):
        parameters = np.array([19.8, 37.0, 0.1, 22.0, 3.6, 10.0])
        time_ms = np.arange(0.0, 2.0, 0.1)
        voltage = case.simulate(parameters, np.array([0.0]), time_ms)
        self.assertEqual(voltage.shape, (20, 1))
        self.assertTrue(np.all(np.isfinite(voltage)))

    def test_parameter_map_and_source_bounds(self):
        self.assertEqual(len(case.PARAMETER_NAMES), 6)
        self.assertEqual(len(case.PARAMETER_UNITS), 6)
        self.assertTrue(np.all(case.PRIOR_LOW < case.PRIOR_HIGH))
        self.assertEqual(case.PRIOR_LOW[4], 1.5)
        self.assertEqual(case.PRIOR_HIGH[4], 2.5)

    def test_candidate_lanes_are_independent(self):
        parameters = np.array(
            [
                [19.8, 37.0, 0.1, 22.0, 2.1, 10.0],
                [15.0, 30.0, 0.2, 18.0, 1.8, 9.0],
            ]
        )
        time_ms = np.arange(0.0, 2.0, 0.1)
        batched = case.simulate(parameters, np.array([0.0, 0.0]), time_ms)
        first = case.simulate(parameters[0], np.array([0.0]), time_ms)[:, 0]
        second = case.simulate(parameters[1], np.array([0.0]), time_ms)[:, 0]
        np.testing.assert_allclose(batched[:, 0], first, rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(batched[:, 1], second, rtol=1e-5, atol=1e-5)

    def test_held_out_traces_are_not_passed_to_inference(self):
        time_ms, traces = case.load_experimental_traces()
        observed = traces[9]
        self.assertFalse(np.shares_memory(observed, traces[6]))
        summary = case.summarize_traces(observed, time_ms)
        self.assertEqual(summary.shape, (1, len(case.SUMMARY_NAMES)))


if __name__ == "__main__":
    unittest.main()
