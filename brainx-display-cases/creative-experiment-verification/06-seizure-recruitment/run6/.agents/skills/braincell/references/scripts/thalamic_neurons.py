"""
Reference script mirrored from:
https://brainx.chaobrain.com/braincell/examples/thalamic_neurons.html

Purpose:
Advanced phenotype-comparison script. Use when the task asks for richer thalamic point-neuron variants, multiple channel compositions, calcium dynamics, HCN/AHP/T-type mechanisms, or comparing several single-compartment neuron types.

Use this when:
- The task asks for advanced thalamic point-neuron variants or phenotype comparison.
- The agent needs multiple ion/channel compositions with calcium dynamics, HCN, AHP, and T-type mechanisms.

Do not use this when:
- The task needs the minimal HH default or any morphology/cable/CV workflow.
"""

import time  # for timing the simulation

import brainstate
import braintools
import brainunit as u
import matplotlib.pyplot as plt
import braincell


# Advanced point-neuron base class; still subclasses `SingleCompartment`.
class ThalamusNeuron(braincell.SingleCompartment):
    def compute_derivative(self, I_ext=0. * u.nA):
        I_ext = self.sum_current_inputs(I_ext, self.V.value) * self.area_factor
        for key, ch in self.nodes(braincell.IonChannel, allowed_hierarchy=(1, 1)).items():
            I_ext = I_ext + ch.current(self.V.value)
        self.V.derivative = I_ext / self.C
        for key, node in self.nodes(braincell.IonChannel, allowed_hierarchy=(1, 1)).items():
            node.compute_derivative(self.V.value)

    def step_run(self, t, inp):
        # Define update rule at each time step
        with brainstate.environ.context(t=t):
            self.update(inp)
            return self.V.value


class HTC(ThalamusNeuron):
    def __init__(
        self,
        size,
        gKL=0.01 * (u.mS / u.cm **2),  # Potassium leak channel conductance
        V_initializer=braintools.init.Constant(-65. * u.mV),  # Initial membrane potential
        solver: str = 'ind_exp_euler'  # Integration method
    ):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV, solver=solver)

        # Membrane area parameter
        self.area_factor = 1e-3 / (2.9e-4 * u.cm** 2)

        # Sodium channel
        self.na = braincell.ion.SodiumFixed(size, E=50. * u.mV)  # Sodium reversal potential 50 mV
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))

        # Potassium channel
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)  # Potassium reversal potential -90 mV
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=gKL))  # Potassium leak current
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30. * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))  # Delayed rectifier potassium current

        # Calcium channel
        self.ca = braincell.ion.CalciumDetailed(
            size,
            C_rest=5e-5 * u.mM,  # Resting calcium concentration
            tau=10. * u.ms,  # Calcium decay time constant
            d=0.5 * u.um  # Calcium diffusion distance
        )
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.5 * (u.mS / u.cm **2)))  # L-type calcium channel
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.5 * (u.mS / u.cm** 2)))  # N-type calcium channel
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm **2)))  # T-type calcium channel (low-threshold)
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=3.0 * (u.mS / u.cm** 2)))  # High-threshold calcium channel

        # Calcium-activated potassium channel (IAHP)
        self.kca = braincell.MixIons(self.k, self.ca)  # Mix potassium and calcium ions
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.3 * (u.mS / u.cm **2)))

        # Hyperpolarization-activated cation current (Ih) and leak current (IL)
        self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.01 * (u.mS / u.cm** 2), E=-43 * u.mV)  # Regulates resting potential and rhythm
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm **2), E=-70 * u.mV)  # Background leak current


class RTC(ThalamusNeuron):
    def __init__(
        self,
        size,
        gKL=0.01 * (u.mS / u.cm** 2),
        V_initializer=braintools.init.Constant(-65. * u.mV),
        solver: str = 'ind_exp_euler'
    ):
        super().__init__(size, V_initializer=V_initializer, V_th=20 * u.mV, solver=solver)

        self.area_factor = 1e-3 / (2.9e-4 * u.cm **2)  # Membrane area parameter

        # Sodium channel
        self.na = braincell.ion.SodiumFixed(size)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-40 * u.mV))

        # Potassium channel
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-40 * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))  # Gating shift
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=gKL))

        # Calcium channel
        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaL=braincell.channel.CaL_IS2008(size, g_max=0.3 * (u.mS / u.cm** 2)))
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.6 * (u.mS / u.cm **2)))
        self.ca.add(ICaT=braincell.channel.CaT_HM1992(size, g_max=2.1 * (u.mS / u.cm** 2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=0.6 * (u.mS / u.cm **2)))

        # Calcium-activated potassium channel (IAHP)
        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.1 * (u.mS / u.cm** 2)))

        # Ih and IL currents
        self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.01 * (u.mS / u.cm **2), E=-43 * u.mV)
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm** 2), E=-70 * u.mV)


class IN(ThalamusNeuron):
    def __init__(
        self,
        size,
        gKL=0.01 * (u.mS / u.cm **2),
        V_initializer=braintools.init.Constant(-70. * u.mV),  # More hyperpolarized initial membrane potential
        solver: str = 'ind_exp_euler'
    ):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV, solver=solver)

        self.area_factor = 1e-3 / (1.7e-4 * u.cm** 2)

        # Sodium channel
        self.na = braincell.ion.SodiumFixed(size)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-30 * u.mV))

        # Potassium channel
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-30 * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=gKL))

        # Calcium channel
        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=10. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.1 * (u.mS / u.cm **2)))
        self.ca.add(ICaHT=braincell.channel.CaHT_HM1992(size, g_max=2.5 * (u.mS / u.cm** 2)))

        # Calcium-activated potassium channel (IAHP)
        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.2 * (u.mS / u.cm **2)))

        # Leak (IL) and hyperpolarization-activated current (Ih)
        self.IL = braincell.channel.IL(size, g_max=0.0075 * (u.mS / u.cm** 2), E=-60 * u.mV)
        self.Ih = braincell.channel.HCN_HM1992(size, g_max=0.05 * (u.mS / u.cm **2), E=-43 * u.mV)


class TRN(ThalamusNeuron):
    def __init__(
        self,
        size,
        gKL=0.01 * (u.mS / u.cm** 2),
        V_initializer=braintools.init.Constant(-70. * u.mV),
        gl=0.0075,  # Leak conductance coefficient
        solver: str = 'ind_exp_euler'
    ):
        super().__init__(size, V_initializer=V_initializer, V_th=20. * u.mV, solver=solver)

        self.area_factor = 1e-3 / (1.43e-4 * u.cm **2)

        # Sodium channel
        self.na = braincell.ion.SodiumFixed(size)
        self.na.add(INa=braincell.channel.Na_Ba2002(size, V_sh=-40 * u.mV))

        # Potassium channel
        self.k = braincell.ion.PotassiumFixed(size, E=-90. * u.mV)
        self.k.add(IDR=braincell.channel.KDR_Ba2002(size, V_sh=-40 * u.mV, q10=2.0, temp=u.celsius2kelvin(16.)))
        self.k.add(IKL=braincell.channel.K_Leak(size, g_max=gKL))

        # Calcium channel
        self.ca = braincell.ion.CalciumDetailed(size, C_rest=5e-5 * u.mM, tau=100. * u.ms, d=0.5 * u.um)
        self.ca.add(ICaN=braincell.channel.CaN_IS2008(size, g_max=0.2 * (u.mS / u.cm** 2)))
        self.ca.add(ICaT=braincell.channel.CaT_HP1992(size, g_max=1.3 * (u.mS / u.cm **2)))

        # Calcium-activated potassium channel (IAHP)
        self.kca = braincell.MixIons(self.k, self.ca)
        self.kca.add(IAHP=braincell.channel.AHP_De1994(size, g_max=0.2 * (u.mS / u.cm** 2)))

        # Leak current (IL)
        self.IL = braincell.channel.IL(size, g_max=gl * (u.mS / u.cm **2), E=-60 * u.mV)


def try_trn_neuron():
    # Set the simulation time step
    brainstate.environ.set(dt=0.02 * u.ms)

    # Generate stepwise input current
    I = braintools.input.section(
        values=[0, 0.05, 0],  # Current amplitudes
        durations=[50 * u.ms, 200 * u.ms, 100 * u.ms]  # Duration of each segment
    ) * u.uA

    # Generate simulation time series
    times = u.math.arange(I.shape[0]) * brainstate.environ.get_dt()

    # Select the neuron type to simulate (HTC here; can be replaced with RTC, IN, TRN)
    neu = HTC(1, solver='ind_exp_euler')  # Use independent exponential Euler method
    neu.init_state()  # Initialize neuron state

    # Run the simulation and record elapsed time
    t0 = time.time()
    vs = brainstate.transform.for_loop(neu.step_run, times, I)  # Loop to update neuron state
    t1 = time.time()
    print(f"Simulation elapsed time: {t1 - t0:.4f} seconds")  # Output computation time

    # Visualize membrane potential changes
    plt.plot(
        times.to_decimal(u.ms),  # x-axis: time
        u.math.squeeze(vs.to_decimal(u.mV))  # y-axis: membrane potential
    )
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.title('HTC neuron response to step current')
    plt.show()


# Execute the simulation
if __name__ == '__main__':
    try_trn_neuron()
