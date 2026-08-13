Implemented the runnable BrainX-native example in [online_working_memory.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.HYywGq/workspace/online_working_memory.py:1).

It uses BrainPy ALIF neurons, BrainEvent CSR recurrent connectivity, BrainTrace `pp_prop` eligibility traces, BrainState `vmap`/`scan`/`for_loop`, and BrainUnit quantities. The default run produced:

```text
initial_accuracy=0.250
final_loss=0.4458
intact_accuracy=1.000
first_cue_ablated_accuracy=0.500
etrace_parameters=4
```

The chance-level ablation confirms the trained network depends on the remembered first cue. Run with:

```bash
python online_working_memory.py
```