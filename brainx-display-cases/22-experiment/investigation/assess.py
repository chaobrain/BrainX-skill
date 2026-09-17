"""Recompute claim evidence and numerical agreement from saved trajectories."""

import json
from pathlib import Path

import numpy as np

from experiment import digest, metrics


def main():
    runs=["central","half_dt","solver","sensitivity"]
    data={}
    for name in runs:
        root=Path("runs")/name
        config=json.loads((root/"config.json").read_text())
        raw=np.load(root/"raw.npz")
        rows=metrics(raw,config)
        assert rows==json.loads((root/"summary.json").read_text())
        assert all(r["valid_baseline"] for r in rows)
        data[name]={r["label"]:r for r in rows}
    central=data["central"]
    checks={}
    for family in ["somatic_circuit","somatic_targeting","release_circuit"]:
        for region,intervention,expected in [("S2","Syn",True),("S2","CaMKII",False),
                                             ("S1","Syn",False),("S1","hDlx",True),
                                             ("S2","hDlx",True),("S2","I_spared",False)]:
            label=f"{family}/{region}/{intervention}"
            checks[label]=central[label]["persistent_oscillation"]==expected
    checks["remote_fails_CaMKII_control"]=central["remote/CaMKII_S2"]["persistent_oscillation"]
    checks["remote_gate_preserved_stable"]=not central["remote/gate_preserved"]["persistent_oscillation"]
    for family in ["equal_circuit","equal_targeting"]:
        checks[f"{family}_removes_regional_difference"]=all(central[f"{family}/{r}/Syn"]["persistent_oscillation"] for r in ["S1","S2"])
    count={}
    for family in sorted({r["family"] for r in data["sensitivity"].values()}):
        ok=(data["sensitivity"][f"{family}/S2/Syn"]["persistent_oscillation"]
            and not data["sensitivity"][f"{family}/S2/CaMKII"]["persistent_oscillation"]
            and not data["sensitivity"][f"{family}/S1/Syn"]["persistent_oscillation"])
        kind=family.split("_grid_")[0]
        count.setdefault(kind,{"matches":0,"total":0,"failed_settings":[]})
        count[kind]["total"]+=1
        count[kind]["matches"]+=int(ok)
        if not ok: count[kind]["failed_settings"].append(family)
    numerical={}
    for other in ["half_dt","solver"]:
        changes=[k for k,r in central.items() if r["persistent_oscillation"]!=data[other][k]["persistent_oscillation"]]
        numerical[other]={"classification_changes":changes,
                          "max_late_E_mean_error":max(abs(r["E_mean_1"]-data[other][k]["E_mean_1"]) for k,r in central.items()),
                          "max_late_E_amplitude_error":max(abs(r["E_ptp_1"]-data[other][k]["E_ptp_1"]) for k,r in central.items())}
        checks[f"{other}_classifications_preserved"]=not changes
    raw=np.load("runs/central/raw.npz")
    config=json.loads(Path("runs/central/config.json").read_text())
    labels=[c["label"] for c in config["conditions"]]
    checks["remote_Syn_CaMKII_exact_same_trace"]=np.array_equal(raw["E"][:,labels.index("remote/Syn_S2")],raw["E"][:,labels.index("remote/CaMKII_S2")])
    checks["E_suppression_reduces_mean_but_induces_oscillation"]=(central["E_only_counterexample/E_suppressed"]["E_mean_1"]<central["E_only_counterexample/baseline"]["E_mean_1"] and central["E_only_counterexample/E_suppressed"]["persistent_oscillation"])
    assert all(checks.values()),checks
    report={"checks":{k:bool(v) for k,v in checks.items()},"sensitivity":count,"numerical":numerical,
            "scope":"Qualitative population-oscillation sufficiency; not measured spikes, electrographic seizure, or convulsion reproduction."}
    Path("assessment.json").write_text(json.dumps(report,indent=2))
    paths=[p for name in runs+["smoke"] for p in (Path("runs")/name).iterdir() if p.is_file()]
    paths += [Path(p) for p in ["models.py","experiment.py","make_configs.py","test_models.py","validation.json","NeuroSpecification.md","assessment.json","assess.py"]]
    Path("artifact-manifest.json").write_text(json.dumps({str(p):digest(p) for p in sorted(paths)},indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
