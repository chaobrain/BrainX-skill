import numpy as np
import unittest

import brainunit as u

import celegans_model
import fit_and_validate


class ModelTests(unittest.TestCase):
    def test_data_contract_and_split(self):
        time_ms, target = fit_and_validate.training_data("Fig4A-D.txt")
        self.assertEqual(time_ms.shape, (10000,))
        self.assertEqual(target.shape, (10000,))
        self.assertTrue(np.isclose(time_ms[1] - time_ms[0], 0.05))
        self.assertNotIn(
            fit_and_validate.TRAIN_TRACE, fit_and_validate.TEST_TRACES
        )


    def test_current_protocol_has_exact_window_and_units(self):
        _, current = celegans_model.current_protocol(25.0)
        current_pA = np.asarray(current.to_decimal(u.pA))
        self.assertEqual(np.count_nonzero(current_pA), 4000)
        self.assertEqual(current_pA[999], 0.0)
        self.assertEqual(current_pA[1000], 25.0)
        self.assertEqual(current_pA[4999], 25.0)
        self.assertEqual(current_pA[5000], 0.0)


    def test_parameter_map_round_trip(self):
        vector = np.asarray(celegans_model.initial_parameter_vector())
        mapping = celegans_model.parameter_dict(vector)
        recovered = np.asarray(
            [mapping[spec.name] for spec in celegans_model.PARAMETER_SPECS]
        )
        self.assertTrue(np.array_equal(vector, recovered))


    def test_nominal_rollout_is_finite_and_zero_current_is_quiet(self):
        voltage = celegans_model.simulate(0.0, -28.4)
        voltage_mV = np.asarray(voltage.to_decimal(u.mV)).squeeze()
        self.assertEqual(voltage_mV.shape, (10000,))
        self.assertTrue(np.isfinite(voltage_mV).all())
        self.assertEqual(fit_and_validate.detect_spikes(voltage_mV).size, 0)


    def test_reset_replay_is_deterministic(self):
        cell = celegans_model.CElegansMuscle(-28.4 * u.mV)
        cell.init_state()
        celegans_model.initialize_gates_at_steady_state(cell)
        _, current = celegans_model.current_protocol(25.0)
        first = celegans_model.rollout(cell, current)
        celegans_model.reset_runtime_state(cell)
        second = celegans_model.rollout(cell, current)
        self.assertTrue(u.math.allclose(first, second))


    def test_channel_currents_have_total_current_units_and_reverse_sign(self):
        cell = celegans_model.CElegansMuscle(-28.4 * u.mV)
        cell.init_state()
        celegans_model.initialize_gates_at_steady_state(cell)
        channels = (cell.Na, cell.Kr, cell.SHK1, cell.EGL19, cell.SLO2, cell.Leak)
        for channel in channels:
            channel.current(cell.V.value).to_decimal(u.pA)
        self.assertTrue(u.math.allclose(cell.Na.current(28.0 * u.mV), 0.0 * u.pA))
        self.assertTrue(u.math.allclose(cell.Kr.current(-41.0 * u.mV), 0.0 * u.pA))


if __name__ == "__main__":
    unittest.main()
