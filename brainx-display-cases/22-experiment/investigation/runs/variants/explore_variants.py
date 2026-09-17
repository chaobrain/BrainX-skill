"""Select illustrative regional and synaptic examples; not data fitting."""

import itertools
import json
from pathlib import Path

import brainunit as u
import numpy as np

from models import condition, simulate
from explore import summarize


def main():
    out = Path("runs/variants")
    out.mkdir(parents=True, exist_ok=False)
    conditions = []
    for wee in [10., 11., 12., 13., 14.]:
        for label, se, si in [("base",0,0),("syn",.25,.5),("cam",.25,0),("dlx",0,1.)]:
            conditions.append(condition(label=f"somatic_{wee}_{label}",wEE=wee,soma_E=se,soma_I=si))
    for wee, le, li in itertools.product([10., 12., 14.], [0., .05, .1, .2], [.2,.3,.4,.5,.6]):
        for label, e, i in [("base",0,0),("syn",le,li),("cam",le,0),("dlx",0,li)]:
            conditions.append(condition(label=f"release_{wee}_{le}_{li}_{label}",wEE=wee,release_E=e,release_I=i))
    (out / "conditions.json").write_text(json.dumps(conditions,indent=2))
    result = simulate(conditions)
    metrics = summarize(result)
    np.savez_compressed(out / "raw.npz",ts_ms=result["ts"].to_decimal(u.ms),
                        E=result["E"],I=result["I"],activation=result["activation"])
    np.savez_compressed(out / "metrics.npz",**metrics)
    records=[dict(label=c["label"],**{k:float(v[j]) for k,v in metrics.items()}) for j,c in enumerate(conditions)]
    (out / "summary.json").write_text(json.dumps(records,indent=2))
    print(json.dumps(records[:20],indent=2),flush=True)
    print("Release candidates:",flush=True)
    print([r for r in records[20:] if r["ptp"]>.15 and r["base_ptp"]<.005 and "syn" in r["label"]],flush=True)


if __name__ == "__main__":
    main()
