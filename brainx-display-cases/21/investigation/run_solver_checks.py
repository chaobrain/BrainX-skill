import json
from pathlib import Path

from experiment import run


if __name__ == '__main__':
    for name in ['cell_exp_euler', 'cell_exp_euler_half']:
        run(json.loads(Path(f'configs/{name}.json').read_text()), Path(f'runs/{name}'))
