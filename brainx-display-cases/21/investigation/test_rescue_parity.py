"""Verify BX-001's input-API correction changes no scientific result."""

import json
import hashlib
from pathlib import Path

import numpy as np


if __name__ == '__main__':
    result = {}
    with np.load('runs/acute_rescue/raw.npz') as old:
        with np.load('runs/acute_rescue_native/raw.npz') as new:
            for name in old.files:
                assert np.array_equal(old[name], new[name]), name
                result[name] = 'bit-identical'
            t = old['t_ms']
            expected = np.repeat(np.where(t < 100, 0., 120.)[:, None], 3, axis=1)
            expected[t >= 400, 0] = 20.
            assert np.array_equal(new['currents_uA_cm2'], expected)
            result['current_waveform'] = 'exact 32000-by-3 match to declared original protocol'
    old_summary = json.loads(Path('runs/acute_rescue/summary.json').read_text())
    new_summary = json.loads(Path('runs/acute_rescue_native/summary.json').read_text())
    assert old_summary == new_summary
    result['summary'] = 'identical'
    result['status'] = 'pass'
    Path('rescue-parity.json').write_text(json.dumps(result, indent=2)+'\n')
    manifest = json.loads(Path('artifact-manifest.json').read_text())
    for path in sorted(Path('runs/acute_rescue_native').iterdir()):
        manifest.append(dict(path=str(path), bytes=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    Path('artifact-manifest-iteration-2.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps(result, indent=2))
