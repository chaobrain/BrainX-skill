"""Batched cue -> silent delay -> masked report, with checkpointed BPTT."""
import json
import brainstate
import braintrace
import braintools
import brainunit as u
from native_workflows import LIF


class DelayedCueSNN(brainstate.nn.Module):
    def __init__(self,size=8):
        super().__init__()
        self.drive = braintrace.nn.Linear(1,size,w_init=braintools.init.Constant(0.6),b_init=braintools.init.ZeroInit())
        self.neuron = LIF(size)
        self.readout = braintrace.nn.LeakyRateReadout(size,1,tau=10*u.ms,w_init=braintools.init.XavierNormal())

    def update(self,x):
        return self.readout(self.neuron(self.drive(x)*u.nA))


def trials(cues):
    inputs = u.math.zeros((20,len(cues),1))
    inputs = inputs.at[:5,:,0].set(cues)
    mask = u.math.concatenate((u.math.zeros(15),u.math.ones(5)))
    targets = u.math.where(cues[:,None]<1.,-0.5,0.5)
    return inputs,targets,mask


def masked_loss(outputs,targets,mask):
    per_trial = u.math.mean((outputs-targets)**2,axis=-1)
    return u.math.sum(per_trial*mask[:,None])/(u.math.sum(mask)*outputs.shape[1])


def sequence_loss(model,inputs,targets,mask,checkpointed=False):
    brainstate.nn.reset_all_states(model,batch_size=inputs.shape[1])
    if checkpointed:
        outputs = brainstate.transform.checkpointed_for_loop(model,inputs,base=4)
    else:
        outputs = brainstate.transform.for_loop(model,inputs)
    return masked_loss(outputs,targets,mask)


def train(epochs=120):
    brainstate.environ.set(precision=64,dt=1*u.ms)
    brainstate.random.seed(7)
    inputs,targets,mask = trials(u.math.asarray([0.6,0.8,1.2,1.4]))
    model = DelayedCueSNN()
    brainstate.nn.init_all_states(model,batch_size=4)
    params = model.states(brainstate.ParamState)
    loss = lambda: sequence_loss(model,inputs,targets,mask,checkpointed=True)
    initial = float(loss())
    optimizer = braintools.optim.Adam(lr=0.02)
    optimizer.register_trainable_weights(params)
    gradient = brainstate.transform.grad(loss,params,return_value=True)
    @brainstate.transform.jit
    def step():
        grads,value = gradient()
        optimizer.update(grads)
        return value
    for _ in range(epochs):step()
    final = float(loss())
    held_inputs,held_targets,held_mask = trials(u.math.asarray([0.7,1.3]))
    def evaluate(xs):
        brainstate.nn.reset_all_states(model,batch_size=2)
        outputs = brainstate.transform.for_loop(model,xs)
        return u.math.mean(outputs[-5:],axis=0)
    prediction = evaluate(held_inputs)
    ablated = evaluate(u.math.zeros_like(held_inputs))
    accuracy = u.math.mean((prediction[:,0]>0)==(held_targets[:,0]>0))
    ablated_accuracy = u.math.mean((ablated[:,0]>0)==(held_targets[:,0]>0))
    metrics = {"initial_loss":initial,"final_loss":final,"held_accuracy":float(accuracy),
               "cue_ablated_accuracy":float(ablated_accuracy),"prediction":prediction[:,0].tolist()}
    return model,metrics


if __name__ == "__main__":
    _,metrics = train()
    print(json.dumps(metrics))
