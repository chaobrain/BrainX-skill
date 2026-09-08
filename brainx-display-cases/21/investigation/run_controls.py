"""Run the frozen follow-up controls serially; never overwrite a result."""

import json
from pathlib import Path

from experiment import run


if __name__ == '__main__':
    for name in ['release_validation', 'block_uncoupled', 'block_residual',
                 'block_dt_half', 'release_rescue']:
        run(json.loads(Path(f'configs/{name}.json').read_text()), Path(f'runs/{name}'))
