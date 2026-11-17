"""Conditional nonlinear State sharing and conservative two-clock exchange."""
import brainspike
import brainstate
import brainmass
import brainunit as u
from native_workflows import LIF


class SaturatingSynapse(brainspike.Synapse):
    """Exact constant-drive step for dr/dt=alpha*s*(1-r)-beta*r."""
    def __init__(self,shape,alpha=1/u.ms,beta=.2/u.ms):
        super().__init__(shape)
        self.state_shape=(shape,) if isinstance(shape,int) else tuple(shape)
        self.alpha,self.beta=alpha,beta

    def init_state(self,batch_size=None,**kwargs):
        shape=self.state_shape if batch_size is None else (batch_size,*self.state_shape)
        self.r=brainstate.HiddenState(u.math.zeros(shape))

    def reset_state(self,batch_size=None,**kwargs):
        shape=self.state_shape if batch_size is None else (batch_size,*self.state_shape)
        self.r.value=u.math.zeros(shape)

    def update(self,events):
        rate=self.alpha*events+self.beta
        equilibrium=self.alpha*events/rate
        self.r.value=equilibrium+(self.r.value-equilibrium)*u.math.exp(-brainstate.environ.get_dt()*rate)
        return self.r.value


class MultirateCoupling(brainstate.nn.Module):
    def __init__(self,fine_dt=.1*u.ms,ratio=5,gain=.2):
        super().__init__()
        if not isinstance(ratio,int) or ratio<1:
            raise ValueError("ratio must be a positive integer")
        self.fine_dt,self.ratio,self.gain=fine_dt,ratio,gain
        self.coarse_dt=ratio*fine_dt
        self.point=LIF(1)
        self.mass=brainmass.HopfStep(in_size=1,a=-.25,w=.3)

    def init_state(self,**kwargs):
        self.rate=brainstate.ShortTermState(u.math.zeros(1)*u.Hz)
        self.held=brainstate.ShortTermState(u.math.zeros(1))
        self.events=brainstate.ShortTermState(u.math.zeros((self.ratio,1)))
        self.end_time=brainstate.ShortTermState(0.*u.ms)

    def update(self,start):
        held=self.mass.x.value
        self.held.value=held
        def fine_step(i):
            with brainstate.environ.context(dt=self.fine_dt,t=start+i*self.fine_dt):
                return self.point(.5*u.nA+self.gain*held*.05*u.nA)
        events=brainstate.transform.for_loop(fine_step,u.math.arange(self.ratio))
        self.events.value=events
        self.rate.value=(events.sum(axis=0)/self.coarse_dt).in_unit(u.Hz)
        self.end_time.value=(start+self.coarse_dt).in_unit(u.ms)
        with brainstate.environ.context(dt=self.coarse_dt,t=self.end_time.value):
            self.mass.update(x_inp=self.gain*self.rate.value/(1000*u.Hz),y_inp=0.)


def run_multirate(ratio=5,gain=.2,duration=20*u.ms,fine_dt=.1*u.ms):
    brainstate.environ.set(precision=64)
    brainstate.random.seed(7)
    model=MultirateCoupling(fine_dt=fine_dt,ratio=ratio,gain=gain)
    result=brainmass.Simulator(model,dt=model.coarse_dt).run(
        duration,inputs=lambda i,t:t,monitors={
            "point":lambda m:m.point.V.value,"mass":lambda m:m.mass.x.value,
            "held":lambda m:m.held.value,"events":lambda m:m.events.value,
            "rate":lambda m:m.rate.value,"end_time":lambda m:m.end_time.value,
        })
    return model,result


if __name__=="__main__":
    _,result=run_multirate()
    print("fine events",result["events"].shape,"final mass",result["mass"][-1])
