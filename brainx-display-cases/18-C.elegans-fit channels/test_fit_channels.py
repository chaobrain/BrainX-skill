import unittest

import braincell
import brainunit as u
import jax.numpy as jnp
import numpy as np

import fit_channels as model


class ChannelFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shk_data = model.load_shk_data()
        cls.egl_data = model.load_egl_data()
        cls.shk_params, cls.shk_points = model.fit_shk(cls.shk_data)
        cls.egl_params = model.fit_egl(cls.egl_data)

    def test_protocols_and_wave_mapping(self):
        np.testing.assert_allclose(self.shk_data.voltages_mV, model.SHK_VOLTAGES_MV)
        np.testing.assert_allclose(self.egl_data.voltages_mV, model.EGL_VOLTAGES_MV)
        self.assertAlmostEqual(self.shk_data.step_end_ms - self.shk_data.step_start_ms, 100.0, places=5)
        self.assertAlmostEqual(self.egl_data.step_end_ms - self.egl_data.step_start_ms, 100.0, places=5)
        self.assertEqual(self.shk_data.source_waves[0], "wave160")
        self.assertEqual(self.egl_data.source_waves[0], "wave16")

    def test_fitted_functions_are_physical(self):
        voltage = np.linspace(-80.0, 120.0, 500)
        n_inf = model.shk_n_inf(voltage, self.shk_params)
        tau_n = model.shk_tau_n(voltage, self.shk_params)
        m_inf, tau_m, h_inf, tau_h = model.egl_gate_functions(voltage, self.egl_params)
        for gate in (n_inf, m_inf, h_inf):
            self.assertTrue(np.all((gate >= 0.0) & (gate <= 1.0)))
        for tau in (tau_n, tau_m, tau_h):
            self.assertTrue(np.all(tau > 0.0))

    def test_predictions_are_finite_and_shape_aligned(self):
        shk = model.simulate_shk(self.shk_data.step_time_ms, self.shk_data.voltages_mV, self.shk_params)
        egl = model.simulate_egl(self.egl_data.step_time_ms, self.egl_data.voltages_mV, self.egl_params)
        self.assertEqual(shk.shape, self.shk_data.currents_pA.shape)
        self.assertEqual(egl.shape, self.egl_data.currents_pA.shape)
        self.assertTrue(np.all(np.isfinite(shk)))
        self.assertTrue(np.all(np.isfinite(egl)))

    def test_braincell_channel_lifecycle_and_units(self):
        potassium = braincell.IonInfo(
            Ci=jnp.array([120.0]) * u.mM,
            Co=jnp.array([5.0]) * u.mM,
            E=jnp.array([model.E_K_MV]) * u.mV,
            valence=1,
        )
        shk = model.SHK1Channel(1, self.shk_params)
        voltage = jnp.array([20.0]) * u.mV
        shk.init_state(voltage, potassium)
        shk.n.value = shk.f_n_inf(voltage, potassium) * 0.5
        shk.compute_derivative(voltage, potassium)
        self.assertTrue(np.all(np.asarray(shk.n.derivative.to_decimal(u.Hz)) > 0.0))
        self.assertTrue(np.all(np.isfinite(shk.current(voltage, potassium).to_decimal(u.pA))))

        calcium = braincell.IonInfo(
            Ci=jnp.array([1e-4]) * u.mM,
            Co=jnp.array([5.0]) * u.mM,
            E=jnp.array([model.E_CA_MV]) * u.mV,
            valence=2,
        )
        egl = model.EGL19Channel(1, self.egl_params)
        egl.init_state(voltage, calcium)
        egl.m.value = egl.f_m_inf(voltage, calcium)
        egl.h.value = egl.f_h_inf(voltage, calcium)
        at_reversal = egl.current(jnp.array([model.E_CA_MV]) * u.mV, calcium)
        np.testing.assert_allclose(at_reversal.to_decimal(u.pA), 0.0, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
