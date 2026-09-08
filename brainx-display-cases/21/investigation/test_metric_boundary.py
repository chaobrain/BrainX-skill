import json
from pathlib import Path

import braintools
import brainunit as u
import numpy as np


if __name__ == '__main__':
    spikes = np.zeros((1000, 2), dtype=bool)
    spikes[::10] = True
    native = braintools.metric.firing_rate(spikes, width=10*u.ms, dt=1*u.ms)
    rates = np.asarray(native)
    count_rate = spikes.sum(0) / 1.
    assert np.allclose(rates[20:-20].mean(), count_rate.mean(), atol=1e-4)
    result = dict(status='pass', native_rate_hz=float(rates[20:-20].mean()),
                  per_cell_count_rate_hz=count_rate.tolist(),
                  window='1 s count; compare time-averaged native output excluding edges',
                  native_return='unitless Hz array, verified against known event count')
    Path('metric-boundary-validation.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result)
