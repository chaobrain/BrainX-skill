"""Freeze illustrative comparisons and a neighboring-parameter validation grid."""

import itertools
import json
from pathlib import Path

from models import condition


def group(family, s2_wee=14., s1_wee=12., se=.25, si=.6, le=0., li=0., s1_si=None):
    rows=[]
    for region,wee in [("S2",s2_wee),("S1",s1_wee)]:
        inhib=si if region=="S2" or s1_si is None else s1_si
        for label,params in [
            ("baseline",{}),
            ("Syn",dict(soma_E=se,soma_I=inhib,release_E=le,release_I=li)),
            ("CaMKII",dict(soma_E=se,release_E=le)),
            ("hDlx",dict(soma_I=1. if si else 0.,release_I=.5 if li else 0.)),
            ("I_spared",dict(soma_E=se,release_E=le)),
        ]:
            rows.append(condition(label=f"{family}/{region}/{label}",family=family,wEE=wee,**params))
    return rows


def write(name,conditions,**simulation):
    data={"level":"production","conditions":conditions,
          "simulation":dict(dt_ms=.2,duration_ms=6000.,onset_ms=1000.,ramp_ms=200.,sample_every=5,**simulation),
          "metric_windows_ms":[[4000,5000],[5000,6000]],"amplitude_threshold":.1,
          "selection":"Illustrative central settings selected after exploratory runs; neighboring grid frozen before validation.",
          "units":{"ts_ms":"ms","E":"dimensionless population activation","I":"dimensionless population activation"}}
    (Path("configs")/f"{name}.json").write_text(json.dumps(data,indent=2))
    return data


def main():
    Path("configs").mkdir(exist_ok=True)
    central=group("somatic_circuit")
    central+=group("somatic_targeting",s1_wee=14.,s1_si=.2)
    central+=group("release_circuit",se=0.,si=0.,le=.1,li=.5)
    for label,loss in [("baseline",0.),("Syn_S2",.6),("CaMKII_S2",.6),("Syn_S1",.2),("gate_preserved",0.)]:
        central.append(condition(label=f"remote/{label}",family="remote",wEE=14.,gate_loss=loss))
    for label,suppression in [("baseline",0.),("E_suppressed",.5)]:
        central.append(condition(label=f"E_only_counterexample/{label}",family="E_only_counterexample",
                                 wEE=18.,drive_E=2.5,soma_E=suppression))
    central+=group("equal_circuit",s1_wee=14.)
    central+=group("equal_targeting",s1_wee=14.,s1_si=.6)
    config=write("central",central)
    smoke=dict(config,level="smoke",conditions=central[:3])
    (Path("configs")/"smoke.json").write_text(json.dumps(smoke,indent=2))
    half=json.loads(json.dumps(config)); half["simulation"].update(dt_ms=.1,sample_every=10)
    (Path("configs")/"half_dt.json").write_text(json.dumps(half,indent=2))
    other=json.loads(json.dumps(config)); other["simulation"].update(dt_ms=.05,sample_every=20,method="exp_euler")
    (Path("configs")/"solver.json").write_text(json.dumps(other,indent=2))
    rows=[]
    for wee,si in itertools.product([13.5,13.75,14.,14.25,14.5],[.5,.6,.7]):
        rows+=group(f"somatic_grid_{wee}_{si}",s2_wee=wee,si=si)
    for le,li in itertools.product([.05,.1,.15],[.45,.5,.55]):
        rows+=group(f"release_grid_{le}_{li}",se=0.,si=0.,le=le,li=li)
    for s1_si in [.1,.2,.3]:
        rows+=group(f"targeting_grid_{s1_si}",s1_wee=14.,s1_si=s1_si)
    write("sensitivity",rows)
    print(f"Frozen {len(central)} central and {len(rows)} sensitivity conditions.")


if __name__=="__main__":
    main()
