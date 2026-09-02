import unittest

import braincell
import brainunit as u
import jax.numpy as jnp
import numpy as np

import fit_channels as model


class ChannelFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(20260902)
        cls.shk_data = model.load_shk_data()
        cls.egl_data = model.load_egl_data()
        cls.shk_params, cls.shk_points, cls.shk_diagnostics = model.fit_shk(
            cls.shk_data, compute_structure=False, max_iterations=500
        )
        cls.egl_params, cls.egl_diagnostics = model.fit_egl(
            cls.egl_data, compute_structure=False, max_iterations=500
        )
        cls.egl_points = model.extract_egl_points(cls.egl_data, cls.egl_params)

    def test_protocols_and_wave_mapping(self):
        np.testing.assert_allclose(self.shk_data.voltages_mV, model.SHK_VOLTAGES_MV)
        np.testing.assert_allclose(self.egl_data.voltages_mV, model.EGL_VOLTAGES_MV)
        self.assertAlmostEqual(self.shk_data.step_end_ms - self.shk_data.step_start_ms, 100.0, places=5)
        self.assertAlmostEqual(self.egl_data.step_end_ms - self.egl_data.step_start_ms, 100.0, places=5)
        self.assertEqual(self.shk_data.source_waves[0], "wave93-wave119")
        self.assertEqual(self.egl_data.source_waves[0], "wave5")
        self.assertEqual(self.shk_diagnostics["selected_activation_power"], 2)
        self.assertEqual(self.egl_diagnostics["selected_activation_power"], 4)
        potassium_contract = model.verify_packed_metadata(model.K_FILE)[
            "verified_wave_contract"
        ]
        calcium_contract = model.verify_packed_metadata(model.CA_FILE)[
            "verified_wave_contract"
        ]
        self.assertEqual(potassium_contract["time"], [86])
        self.assertEqual(potassium_contract["WT"], list(range(87, 99)))
        self.assertEqual(potassium_contract["shk-1 mutant"], list(range(113, 125)))
        self.assertEqual(potassium_contract["commands"], list(range(167, 179)))
        self.assertEqual(calcium_contract["time"], [0])
        self.assertEqual(calcium_contract["WT"], list(range(1, 12)))
        self.assertEqual(calcium_contract["n582"], list(range(12, 23)))
        self.assertEqual(calcium_contract["ad1006"], list(range(23, 34)))
        self.assertEqual(calcium_contract["commands"], list(range(34, 45)))

    def test_egl_mutant_differences_are_not_valid_single_channel_targets(self):
        controls = model.load_egl_mutant_controls()
        for difference in controls.values():
            late = np.mean(difference[:, -1000:], axis=1)
            self.assertTrue(np.any(late > 0.0))
            self.assertTrue(np.any(late < 0.0))

    def test_fitted_functions_are_physical(self):
        voltage = np.linspace(-80.0, 120.0, 500)
        n_inf = model.shk_n_inf(voltage, self.shk_params)
        tau_n = model.shk_tau_n(voltage, self.shk_params)
        m_inf, tau_m = model.egl_gate_functions(voltage, self.egl_params)
        for gate in (n_inf, m_inf):
            self.assertTrue(np.all((gate >= 0.0) & (gate <= 1.0)))
        for tau in (tau_n, tau_m):
            self.assertTrue(np.all(tau > 0.0))

    def test_predictions_are_finite_and_shape_aligned(self):
        shk = model.simulate_shk(self.shk_data.step_time_ms, self.shk_data.voltages_mV, self.shk_params)
        egl = model.simulate_egl(self.egl_data.step_time_ms, self.egl_data.voltages_mV, self.egl_params)
        self.assertEqual(shk.shape, self.shk_data.currents_pA.shape)
        self.assertEqual(egl.shape, self.egl_data.currents_pA.shape)
        self.assertTrue(np.all(np.isfinite(shk)))
        self.assertTrue(np.all(np.isfinite(egl)))

    def test_optimizer_domain_and_gate_point_termination(self):
        for diagnostics, names, lower, upper in (
            (
                self.shk_diagnostics,
                model.SHK_PARAMETER_ORDER,
                model.SHK_LOWER,
                model.SHK_UPPER,
            ),
            (
                self.egl_diagnostics,
                model.EGL_PARAMETER_ORDER,
                model.EGL_LOWER,
                model.EGL_UPPER,
            ),
        ):
            self.assertEqual(
                [start["seed"] for start in diagnostics["starts"]],
                list(model.OPTIMIZER_SEEDS),
            )
            for start in diagnostics["starts"]:
                np.testing.assert_array_less(lower - 1e-9, start["initial_parameters"])
                np.testing.assert_array_less(start["initial_parameters"], upper + 1e-9)
                np.testing.assert_array_less(lower - 1e-9, start["final_parameters"])
                np.testing.assert_array_less(start["final_parameters"], upper + 1e-9)
            self.assertEqual(set(diagnostics["physical_bounds"]), set(names))
        self.assertTrue(self.egl_points["fit_diagnostics"]["success"])
        self.assertTrue(
            any(
                candidate["success"]
                for candidate in self.egl_points["fit_diagnostics"]["candidates"]
            )
        )

    def test_braincell_channel_lifecycle_and_units(self):
        potassium = braincell.IonInfo(
            Ci=jnp.array([120.0]) * u.mM,
            Co=jnp.array([5.0]) * u.mM,
            E=jnp.array([self.shk_params.e_rev_mV]) * u.mV,
            valence=1,
        )
        shk = model.SHK1Channel(1, self.shk_params)
        voltage = jnp.array([20.0]) * u.mV
        shk.init_state(voltage, potassium)
        shk.n.value = shk.f_n_inf(voltage, potassium) * 0.5
        shk.compute_derivative(voltage, potassium)
        self.assertTrue(np.all(np.asarray(shk.n.derivative.to_decimal(u.Hz)) > 0.0))
        self.assertTrue(np.all(np.isfinite(shk.current(voltage, potassium).to_decimal(u.pA))))
        hold = jnp.array([model.V_HOLD_MV]) * u.mV
        shk.n.value = jnp.array([0.9])
        shk.reset_state(hold, potassium)
        np.testing.assert_allclose(shk.n.value, shk.f_n_inf(hold, potassium))

        calcium = braincell.IonInfo(
            Ci=jnp.array([1e-4]) * u.mM,
            Co=jnp.array([5.0]) * u.mM,
            E=jnp.array([self.egl_params.e_rev_mV]) * u.mV,
            valence=2,
        )
        egl = model.EGL19Channel(1, self.egl_params)
        egl.init_state(voltage, calcium)
        egl.m.value = egl.f_m_inf(voltage, calcium)
        at_reversal = egl.current(jnp.array([self.egl_params.e_rev_mV]) * u.mV, calcium)
        np.testing.assert_allclose(at_reversal.to_decimal(u.pA), 0.0, atol=1e-7)
        egl.m.value = jnp.array([0.9])
        egl.reset_state(hold, calcium)
        np.testing.assert_allclose(egl.m.value, egl.f_m_inf(hold, calcium))

    def test_analytic_and_braincell_rollouts_match(self):
        parameter_sets = (
            (
                self.shk_params,
                model.ShkParameters(
                    **dict(
                        zip(
                            model.SHK_PARAMETER_ORDER,
                            model.SHK_LOWER + 0.05 * (model.SHK_UPPER - model.SHK_LOWER),
                            strict=True,
                        )
                    )
                ),
                self.shk_data,
                model.simulate_shk,
                model.simulate_shk_braincell,
            ),
            (
                self.egl_params,
                model.EglParameters(
                    **dict(
                        zip(
                            model.EGL_PARAMETER_ORDER,
                            model.EGL_LOWER + 0.05 * (model.EGL_UPPER - model.EGL_LOWER),
                            strict=True,
                        )
                    )
                ),
                self.egl_data,
                model.simulate_egl,
                model.simulate_egl_braincell,
            ),
        )
        for nominal, boundary, data, analytic, braincell_rollout in parameter_sets:
            for parameters in (nominal, boundary):
                expected = analytic(data.step_time_ms, data.voltages_mV, parameters)
                actual = np.asarray(
                    braincell_rollout(data.step_time_ms, data.voltages_mV, parameters)
                )
                np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-3)


if __name__ == "__main__":
    unittest.main()
