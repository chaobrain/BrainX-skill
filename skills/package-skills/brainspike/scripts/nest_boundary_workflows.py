"""Release-scoped NEST delay paths, custom scheduling, and hard-spike fitting."""
import brainspike
import brainstate
import braintools
import brainunit as u
import jax.numpy as jnp
import numpy as np
from native_workflows import LIF
from train_nest_simulator import rollout


class AutonomousLIF(LIF):
    """Minimal zero-argument Simulator update; no incoming synapse contract."""
    def update(self):
        return super().update(.5*u.nA)


def delay_arrivals(delay,via="direct",n_source=1):
    sim=brainspike.Simulator(dt=.1*u.ms)
    source=sim.create(brainspike.spike_generator,n_source,spike_times=jnp.asarray([1.])*u.ms)
    target=sim.create(brainspike.iaf_psc_delta,2,E_L=0*u.mV,V_reset=0*u.mV,
                      V_th=20*u.mV,t_ref=0*u.ms,V_initializer=braintools.init.Constant(0*u.mV))
    if via=="direct":
        sim.connect(source,target,weight=30*u.mV,delay=delay)
    elif via=="spec":
        sim.connect(source,target,synapse=brainspike.static_synapse(weight=30*u.mV,delay=delay))
    elif via=="continuous":
        sim.connect(source,target,synapse=brainspike.cont_delay_synapse(weight=30*u.mV,delay=delay))
    elif via=="set":
        sim.connect(source,target,synapse=brainspike.static_synapse(weight=30*u.mV,delay=.2*u.ms))
        sim.get_connections().set("delay",delay)
    else:
        raise ValueError("unknown delay path")
    rec=sim.create(brainspike.spike_recorder)
    sim.connect(target,rec)
    spikes=np.asarray(sim.simulate(3*u.ms).spikes(rec))
    return [np.flatnonzero(spikes[:,i]).tolist() for i in range(2)]


def neuron_delay_matrix(mode="grouped"):
    sim=brainspike.Simulator(dt=.1*u.ms)
    driver=sim.create(brainspike.spike_generator,spike_times=jnp.asarray([1.])*u.ms)
    params=dict(E_L=0*u.mV,V_reset=0*u.mV,V_th=20*u.mV,t_ref=0*u.ms,
                V_initializer=braintools.init.Constant(0*u.mV))
    pre=sim.create(brainspike.iaf_psc_delta,2,params=params)
    post=sim.create(brainspike.iaf_psc_delta,2,params=params)
    sim.connect(driver,pre,weight=30*u.mV,delay=.1*u.ms)
    delays=jnp.asarray([[.1,.2],[.3,.4]])*u.ms
    if mode=="grouped":
        for i in range(2):
            for j in range(2):
                sim.connect(pre[i:i+1],post[j:j+1],weight=30*u.mV,delay=delays[i,j])
    else:
        delay=delays if mode=="matrix" else .2*u.ms
        sim.connect(pre,post,weight=30*u.mV,delay=delay)
    recorder=sim.create(brainspike.spike_recorder);sim.connect(post,recorder)
    spikes=np.asarray(sim.simulate(4*u.ms).spikes(recorder))
    return [np.flatnonzero(spikes[:,i]).tolist() for i in range(2)]


def voltage_trace(delay_ms,path="direct"):
    sim=brainspike.Simulator(dt=.1*u.ms)
    source=sim.create(brainspike.spike_generator,spike_times=jnp.asarray([1.])*u.ms)
    target=sim.create(brainspike.iaf_psc_exp,1,E_L=0*u.mV,V_reset=0*u.mV,
                      V_th=10000*u.mV,V_initializer=braintools.init.Constant(0*u.mV))
    if path=="direct":
        sim.connect(source,target,weight=250*u.pA,delay=delay_ms*u.ms)
    else:
        spec={"spec":brainspike.static_synapse,"continuous":brainspike.cont_delay_synapse}[path]
        sim.connect(source,target,synapse=spec(weight=250*u.pA,delay=delay_ms*u.ms))
    meter=sim.create(brainspike.multimeter,record_from=["V_m"])
    sim.connect(meter,target)
    return np.asarray(sim.simulate(5*u.ms).trace(meter,"V_m").to_decimal(u.mV))[:,0]


def window_voltage(weight,continued=False):
    sim=brainspike.Simulator(dt=.1*u.ms)
    source=sim.create(brainspike.spike_generator,spike_times=jnp.asarray([1.,2.,3.])*u.ms)
    target=sim.create(brainspike.iaf_psc_alpha,1,
                      V_initializer=braintools.init.Constant(-65*u.mV))
    sim.connect(source,target,weight=weight*100*u.pA,delay=.1*u.ms)
    meter=sim.create(brainspike.multimeter,record_from=["V_m"]);sim.connect(meter,target)
    if continued:
        sim.cont(5*u.ms)
        result=sim.cont(5*u.ms)
    else:
        result=sim.simulate(10*u.ms)
    return result.trace(meter,"V_m").to_decimal(u.mV)[-1,0]


def fit_hard_count():
    brainstate.environ.set(precision=64)
    brainstate.random.seed(7)
    def count(current):
        return jnp.sum(rollout(jnp.asarray(.5),current=current,duration=20)[1])
    optimizer=braintools.optim.ScipyOptimizer(
        loss_fun=lambda current:(count(current)-5.)**2,
        bounds=[(jnp.asarray(5.),jnp.asarray(20.))],method="Powell",
    )
    result=optimizer.minimize(n_iter=1)
    current=float(np.asarray(result.x[0]))
    return {"current_multiplier":current,"count":float(count(current)),"loss":float(result.fun)}


if __name__=="__main__":
    brainstate.environ.set(precision=64)
    print("edge delays",delay_arrivals(jnp.asarray([.2,.5])*u.ms))
    print("hard-count bounded fit",fit_hard_count())
