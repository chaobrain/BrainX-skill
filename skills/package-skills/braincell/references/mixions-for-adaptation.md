# MixIons for adaptation

Use this reference when adding a calcium-activated potassium current, calcium-dependent afterhyperpolarization, spike-frequency adaptation, or a thalamic/rebound variant that needs dynamic calcium plus a K/Ca mixed current.

In single-compartment models, use the class pattern below directly. In multicompartment `Cell` work, use this reference for ion/channel composition choices, but keep morphology, targeting, `paint`, `place`, CV policy, clamps, and probes in `references/multicompartment/multicompartment-cell-workflow.md`.

## Rule

Use `braincell.MixIons(k, ca)` only when a channel depends on multiple ions. For calcium-activated potassium currents, potassium supplies the reversal potential and calcium supplies intracellular concentration dynamics. Keep ion order consistent with the channel's `root_type`.

## Pattern

```python
import braincell
import braintools
import brainunit as u


class AdaptingCell(braincell.SingleCompartment):
    def __init__(
        self,
        size,
        g_ahp=0.3 * u.mS / u.cm**2,
        solver="ind_exp_euler",
    ):
        super().__init__(
            size,
            V_initializer=braintools.init.Constant(-65. * u.mV),
            V_th=20. * u.mV,
            solver=solver,
        )

        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30. * u.mV))

        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=0.01 * (u.mS / u.cm ** 2)))
        self.k.add(
            IDR=braincell.channel.KDR_Ba2002(
                size,
                V_sh=-30. * u.mV,
                q10=2.0,
                temp=u.celsius2kelvin(16.),
            )
        )

        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5e-5 * u.mM,
            tau=10. * u.ms,
            d=0.5 * u.um,
        )
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.5 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm ** 2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm ** 2)))

        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=g_ahp))

        self.IL = braincell.channel.IL(
            size,
            E=-70. * u.mV,
            g_max=0.0075 * (u.mS / u.cm ** 2),
        )
```

## Test adaptation by controlled ablation

Expose only AHP `g_max`, set it to zero for the ablated condition, and keep the cell template, initialization, solver, stimulus, and every other conductance identical. Use `SingleCompartment.size` for independent conditions; it already owns the batch axis.

```python
g_ahp = u.math.asarray([0.0, 0.3]) * u.mS / u.cm**2
cell = AdaptingCell(size=g_ahp.shape, g_ahp=g_ahp)
current = u.math.asarray([2.0, 2.0]) * u.uA / u.cm**2
```

Treat the ISI trajectory as the primary adaptation check: the present-AHP lane should lengthen materially while the zero-AHP lane remains more tonic or fires faster late in the stimulus. Report the early/late windows beside rate summaries. Call a condition "intact" only when its complete model and parameter values come from a sourced template; otherwise call it "AHP present" and disclose any calibration.

Verify that stimulus onset starts from the intended baseline. If a calibrated hybrid spikes without applied current, use the same disclosed holding current and initialization in every lane, and assert that no spikes occur before the stimulus. Do not interpret an already-spiking baseline as onset adaptation.

Do not combine channel families or tune calcium influx, calcium removal, and AHP conductance solely to force the requested contrast. Preserve a coherent template when making biological claims, or label a hybrid explicitly as a teaching model and check that the conclusion survives a small parameter sweep.

Open `references/scripts/spike_frequency_adaptation.py` for a runnable calibrated teaching model with a quiet baseline, strength sweep, and zero-conductance comparison. Open `references/scripts/channel_ablation.py` only when the task needs the general one-parameter channel-ablation pattern without calcium-dependent adaptation.

## Rebound input variant

For post-inhibitory rebound, add or retain a low-threshold T-type calcium current such as `CaT_HM1992` and drive a time-dependent hyperpolarizing input.

```python
def I_of_t(t):
    return u.math.where(
        t < 200. * u.ms,
        -2. * u.uA / u.cm ** 2,
        0. * u.uA / u.cm ** 2,
    )
```

## Checks

- Add AHP/KCa channels to the `MixIons` owner, not directly to `self.k` or `self.ca`.
- Use dynamic calcium such as `CalciumDetailed` when intracellular calcium should feed back into adaptation.
- Preserve a potassium ion container even when the adaptation current is calcium-dependent; the K reversal potential still matters.
- Inspect `channel.root_type` before changing ion order.
