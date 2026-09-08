"""Explicitly exploratory parameter search, preserving every sampled result."""

import itertools
import json
from pathlib import Path

import numpy as np
import brainunit as u

from models import condition, simulate


def summarize(result):
    times = np.asarray(result["ts"].to_decimal(u.ms))
    e, i = np.asarray(result["E"]), np.asarray(result["I"])
    return dict(base_mean=e[(times > 600) & (times < 950)].mean(0),
                base_ptp=np.ptp(e[(times > 600) & (times < 950)], axis=0),
                mean=e[times > 2200].mean(0), ptp=np.ptp(e[times > 2200], axis=0),
                i_mean=i[times > 2200].mean(0),
                minimum=np.minimum(e.min(0), i.min(0)))


def main():
    out = Path("runs/exploration")
    out.mkdir(parents=True, exist_ok=False)
    grid = list(itertools.product([12., 14., 16., 18.], [1., 1.5, 2., 2.5],
                                 [1., 2., 3., 4.], [0.25, 0.5, 1.], [0.5, 1., 2.]))
    conditions = []
    for wee, de, di, se, si in grid:
        common = dict(wEE=wee, drive_E=de, drive_I=di)
        for label, eff_e, eff_i in [("baseline", 0, 0), ("syn", se, si),
                                    ("cam", se, 0), ("dlx", 0, si)]:
            conditions.append(condition(**common, label=label, soma_E=eff_e, soma_I=eff_i))
    (out / "conditions.json").write_text(json.dumps(conditions, indent=2))
    result = simulate(conditions)
    metrics = summarize(result)
    np.savez_compressed(out / "raw.npz", ts_ms=result["ts"].to_decimal(u.ms),
                        E=np.asarray(result["E"]), I=np.asarray(result["I"]))
    np.savez_compressed(out / "metrics.npz", **metrics)
    m = {k:v.reshape(-1,4) for k,v in metrics.items()}
    valid = (m["base_ptp"].max(1) < 0.005) & (m["minimum"].min(1) >= -0.0001)
    positive = valid & (m["ptp"][:,1] > 0.12) & (m["ptp"][:,2] < 0.005)
    hits = [{"grid":grid[j], **{k:v[j].tolist() for k,v in m.items()}}
            for j in np.flatnonzero(positive)]
    (out / "candidates.json").write_text(json.dumps(hits, indent=2))
    print(json.dumps({"conditions":len(conditions), "candidate_count":len(hits),
                      "first_candidates":hits[:15]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
