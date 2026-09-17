"""Freeze and execute one deterministic BrainMass experiment."""

import argparse
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

import brainmass
import brainstate
import brainunit as u
import jax
import numpy as np

from models import simulate


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metrics(raw, config):
    t, e, i = raw["ts_ms"], raw["E"], raw["I"]
    base = (t >= 600) & (t < 950)
    windows = [(t >= start) & (t < end) for start, end in config["metric_windows_ms"]]
    records = []
    for j, c in enumerate(config["conditions"]):
        row = {"label": c["label"], "family":c.get("family",""),
               "baseline_E":float(e[base,j].mean()), "baseline_I":float(i[base,j].mean()),
               "baseline_ptp":float(np.ptp(e[base,j])),
               "min_activity":float(min(e[:,j].min(),i[:,j].min())),
               "max_activity":float(max(e[:,j].max(),i[:,j].max()))}
        for k, mask in enumerate(windows):
            ev, iv = e[mask,j], i[mask,j]
            h = raw["activation"][mask,j]
            threshold = float((ev.min()+ev.max())/2)
            crossings = np.count_nonzero((ev[1:]>=threshold)&(ev[:-1]<threshold))
            row.update({f"E_mean_{k}":float(ev.mean()), f"I_mean_{k}":float(iv.mean()),
                        f"E_ptp_{k}":float(np.ptp(ev)), f"crossings_{k}":int(crossings),
                        f"I_output_mean_{k}":float(((1-h*c["release_I"])*iv).mean())})
        row["persistent_oscillation"] = bool(all(
            row[f"E_ptp_{k}"]>config["amplitude_threshold"] and row[f"crossings_{k}"]>=3
            for k in range(len(windows))))
        row["valid_baseline"] = bool(row["baseline_ptp"]<.005 and row["min_activity"]>=-1e-6)
        records.append(row)
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",required=True)
    parser.add_argument("--run-id",required=True)
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    out=Path("runs")/args.run_id
    out.mkdir(parents=True,exist_ok=False)
    brainstate.environ.set(precision=64)
    brainstate.environ.set_platform("cpu")
    assert jax.default_backend()=="cpu"
    assert any(m.name=="WilsonCowanStep" for m in brainmass.list_models())
    for src in ["models.py","experiment.py","NeuroSpecification.md",args.config]:
        shutil.copy2(src,out/Path(src).name)
    if Path(args.config).name!="config.json":
        shutil.copy2(args.config,out/"config.json")
    command=" ".join([sys.executable,*sys.argv])
    (out/"command.txt").write_text(command+"\n")
    (out/"code.diff").write_text(subprocess.check_output(["git","diff","--","."],text=True))
    env={"python":sys.executable,"backend":jax.default_backend(),"devices":[str(d) for d in jax.devices()],
         "precision":64,"seed":None,"stochastic":False,
         "versions":{p:importlib.metadata.version(p) for p in ["brainx","brainmass","brainstate","brainunit","braintools","jax","jaxlib","numpy"]},
         "git_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
         "source_hashes":{s:digest(s) for s in ["models.py","experiment.py","NeuroSpecification.md",args.config]}}
    (out/"environment.json").write_text(json.dumps(env,indent=2))
    (out/"RUN_SPEC.md").write_text(
        f"# Run specification\n\nRun: {args.run_id}; level: {config['level']}; new immutable CPU run.\n\n"
        f"Command: `{command}`.\n\nConfiguration SHA256: `{digest(args.config)}`.\n\n"
        "No supplied raw data; deterministic illustrative population dynamics; no seed. "
        "Expected raw.npz, metrics.csv, summary.json, run.log, status.json and exit_code. "
        "Estimated < 2 minutes and < 1 GB RAM. Stop on exception/nonfinite output; no automatic retry.\n")
    log=out/"run.log"
    log.write_text("Starting CPU run with native independent condition axis.\n")
    (out/"status.json").write_text(json.dumps({"status":"running"}))
    start=time.perf_counter()
    try:
        result=simulate(config["conditions"],**config["simulation"])
        raw={k:np.asarray(v) for k,v in result.items() if k!="ts"}
        raw["ts_ms"]=np.asarray(result["ts"].to_decimal(u.ms))
        assert all(np.isfinite(v).all() for v in raw.values())
        assert raw["E"].shape==raw["I"].shape==raw["activation"].shape
        np.savez_compressed(out/"raw.npz",**raw)
        rows=metrics(raw,config)
        (out/"summary.json").write_text(json.dumps(rows,indent=2))
        with (out/"metrics.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        elapsed=time.perf_counter()-start
        with log.open("a") as stream:
            stream.write(f"Done: {len(rows)} conditions; {elapsed:.3f} s including compilation, transfer and serialization.\n")
        (out/"status.json").write_text(json.dumps({"status":"done","seconds":elapsed,"conditions":len(rows)}))
        (out/"exit_code").write_text("0\n")
        print(json.dumps({"run":args.run_id,"seconds":elapsed,"conditions":len(rows),
                          "oscillating":[r["label"] for r in rows if r["persistent_oscillation"]]},indent=2))
    except Exception:
        with log.open("a") as stream: stream.write(traceback.format_exc())
        (out/"status.json").write_text(json.dumps({"status":"failed"}))
        (out/"exit_code").write_text("1\n")
        raise


if __name__=="__main__":
    main()
