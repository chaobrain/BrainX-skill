"""Focused checks of equations, integration boundary, and condition independence."""

import json
from pathlib import Path
import time

import brainmass
import brainstate
import brainunit as u
import numpy as np

from models import EICircuit, condition, simulate


def main():
    brainstate.environ.set(precision=64)
    c=condition(wEE=14.,soma_E=.25,soma_I=.6,release_E=.1,release_I=.5)
    node=EICircuit([c])
    brainstate.nn.init_all_states(node)
    node.activation.value=u.math.asarray([.7])
    e,i=.12,.09
    xe=14*(1-.7*.1)*e-15*(1-.7*.5)*i+1-.7*.25
    xi=12*(1-.7*.1)*e-3*(1-.7*.5)*i+1-.7*.6
    fe=1/(1+np.exp(-1.2*(xe-2.8)))-1/(1+np.exp(1.2*2.8))
    fi=1/(1+np.exp(-(xi-4)))-1/(1+np.exp(4))
    expected=np.array([(-e+(1-e)*fe)/10,(-i+(1-i)*fi)/20])
    actual=np.array([np.asarray(node.drE(e,i,1-.7*.25).to_decimal(u.kHz))[0],
                     np.asarray(node.drI(i,e,1-.7*.6).to_decimal(u.kHz))[0]])
    np.testing.assert_allclose(actual,expected,atol=1e-12)
    settings=dict(duration_ms=50.,onset_ms=10.,ramp_ms=5.,sample_every=1)
    conditions=[condition(wEE=12.),c]
    start=time.perf_counter(); a=simulate(conditions,**settings)
    a={k:np.asarray(v.to_decimal(u.ms) if k=="ts" else v) for k,v in a.items()}
    cold=time.perf_counter()-start
    b=simulate(conditions,**settings)
    for k in ["E","I","activation"]: np.testing.assert_array_equal(a[k],np.asarray(b[k]))
    single=simulate([c],**settings)
    for k in ["E","I"]: np.testing.assert_allclose(a[k][:,1],np.asarray(single[k])[:,0],atol=1e-12)
    no_jit=simulate(conditions,jit=False,**settings)
    for k in ["E","I"]: np.testing.assert_allclose(a[k],np.asarray(no_jit[k]),atol=1e-12)
    stock=brainmass.WilsonCowanStep(in_size=1,tau_E=10*u.ms,tau_I=20*u.ms,
                                  wEE=12.,wEI=12.,wIE=15.,wII=3.,method="rk4")
    zero=simulate([condition(wEE=12.)],initial=0.,**settings)
    reference=brainmass.Simulator(stock,dt=.2*u.ms).run(50*u.ms,inputs=lambda _i,_t:(1.,1.),monitors=["rE","rI"])
    for name,key in [("E","rE"),("I","rI")]:
        np.testing.assert_allclose(np.asarray(zero[name]),np.asarray(reference[key]),atol=1e-12)
    assert np.isclose(a["ts"][0],.2) and np.isclose(a["ts"][-1],50.)
    assert np.all(a["activation"][a["ts"]<=10.]==0)
    report=dict(passed=True,derivative_max_abs_error=float(np.abs(actual-expected).max()),
                reset_exact=True,batch_single_parity=True,jit_parity=True,stock_parity=True,
                post_update_time_verified=True,dtype=str(a["E"].dtype),cold_small_run_s=cold,
                performance_decision="Unchanged: Simulator owns compiled time loop and native condition axis. No speedup claimed.")
    Path("validation.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
