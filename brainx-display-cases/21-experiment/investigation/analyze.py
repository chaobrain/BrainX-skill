"""Deterministic seed summaries and an unfitted optical-mixture sensitivity scan."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with path.open() as handle:
        return [{k: (v if k == 'group' else float(v)) for k, v in row.items()}
                for row in csv.DictReader(handle)]


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    assessment, summary_rows, manifest = {}, [], []
    for run in sorted(Path('runs').iterdir()):
        if not (run/'status.json').exists():
            continue
        config = json.loads((run/'config.json').read_text())
        if config['mode'] != 'network':
            continue
        rows = read_rows(run/'metrics.csv')
        sensitivity = []
        for seed in config['seeds']:
            for current in config['currents_uA_cm2']:
                subset = [r for r in rows if r['seed'] == seed and r['current_uA_cm2'] == current]
                baseline = next(r for r in subset if r['fraction'] == 0 and r['group'] == 'all')
                cv0, cs0 = baseline['calcium_voltage_proxy'], baseline['synaptic_inward_uA_cm2']
                for r in subset:
                    r['cv_normalized'] = r['calcium_voltage_proxy']/cv0
                    r['cs_normalized'] = r['synaptic_inward_uA_cm2']/cs0 if cs0 > 0 else float('nan')
                    r['optical_alpha_half'] = .5*(r['cv_normalized']+r['cs_normalized'])
                kd = next(r for r in subset if r['fraction'] == .5 and r['group'] == 'KD')
                wt = next(r for r in subset if r['fraction'] == .5 and r['group'] == 'control')
                full = next(r for r in subset if r['fraction'] == 1 and r['group'] == 'all')
                alphas = np.linspace(0, 1, 1001)
                within = alphas*(kd['cv_normalized']-wt['cv_normalized']) + (1-alphas)*(kd['cs_normalized']-wt['cs_normalized'])
                between = alphas*(full['cv_normalized']-1) + (1-alphas)*(full['cs_normalized']-1)
                valid = alphas[(within > 0) & (between < 0)]
                sensitivity.append(dict(seed=seed, current_uA_cm2=current,
                    alpha_interval_sampled=([float(valid.min()), float(valid.max())] if len(valid) else None),
                    electrical_within_mosaic_hz=kd['rate_hz']-wt['rate_hz'],
                    electrical_full_minus_baseline_hz=full['rate_hz']-baseline['rate_hz'],
                    optical_within_mosaic_at_half=kd['optical_alpha_half']-wt['optical_alpha_half'],
                    optical_full_at_half=full['optical_alpha_half']))
        assessment[run.name] = sensitivity
        for current in config['currents_uA_cm2']:
            for fraction in config['fractions']:
                for group in ['all', 'KD', 'control']:
                    selected = [r for r in rows if r['current_uA_cm2'] == current
                                and r['fraction'] == fraction and r['group'] == group]
                    if not selected:
                        continue
                    row = dict(run=run.name, current_uA_cm2=current, fraction=fraction, group=group,
                               seeds=len(selected))
                    for key in ['rate_hz', 'block', 'late_voltage_mV', 'h_available',
                                'synaptic_inward_uA_cm2', 'cv_normalized', 'cs_normalized',
                                'optical_alpha_half']:
                        values = np.array([r[key] for r in selected])
                        row[key+'_mean'] = float(values.mean())
                        row[key+'_seed_sd'] = float(values.std(ddof=1)) if len(values)>1 else 0.
                    summary_rows.append(row)
    Path('assessment.json').write_text(json.dumps(json_safe(assessment), indent=2, allow_nan=False)+'\n')
    with Path('fraction-summary.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    for path in sorted(Path('runs').rglob('*')):
        if path.is_file():
            manifest.append(dict(path=str(path), bytes=path.stat().st_size,
                                 sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    Path('artifact-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Analyzed:', ', '.join(assessment))


if __name__ == '__main__':
    main()
