Implemented [online_working_memory.py](/private/var/folders/r9/y5dsw3w97zgg3xts8fnl6g2c0000gn/T/brainx-skill-eval.hABfQ8/workspace/online_working_memory.py:1).

It includes unit-aware LIF neurons, BrainEvent CSR recurrence, pp-prop eligibility traces, stateful batch mapping, silent-delay assertions, report-masked online updates, and `for_loop` streaming.

Verified with `python online_working_memory.py`:

```text
Before: loss=0.6781, accuracy=0.500
After 320 updates: loss=0.2611, accuracy=1.000
Silent delay: 40 ms
```

The deterministic training assertions and syntax check pass.