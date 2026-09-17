"""Validate the explicitly strong transmission-loss electrical reversal."""

import json
from pathlib import Path

from experiment import run


if __name__ == '__main__':
    for name in ['release_strong_validation', 'release_strong_partial', 'release_strong_rescue']:
        run(json.loads(Path(f'configs/{name}.json').read_text()), Path(f'runs/{name}'))
