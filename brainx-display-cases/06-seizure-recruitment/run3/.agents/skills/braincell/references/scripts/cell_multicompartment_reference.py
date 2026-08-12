"""
This script is mirrored and reorganized from:
https://brainx.chaobrain.com/braincell/tutorials/cell.html

It is the primary full-script reference for BrainCell multicompartment modeling.
All BrainCell reference scripts live in this directory.
"""

# 1. Imports and CPU/JAX setup.
# Set the platform before importing JAX-backed BrainCell/BrainState packages.
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import braincell
import brainstate
import brainunit as u
import matplotlib.pyplot as plt

from braincell import Cell, Morphology
from braincell import CVPerBranch, MaxCVLen
from braincell.filter import BranchSlice, RootLocation, at
from braincell.mech import CableProperty, Channel, CurrentClamp, StateProbe


# 2. Load morphology from SWC.
# Canonical tutorial path. Adjust this path only if your repository layout differs.
MORPHOLOGY_PATH = "../../data/morphology/example_tree.swc"
morpho = Morphology.from_swc(MORPHOLOGY_PATH)

print("Morphology topology:")
print(morpho.topo())


# 3. Build Cell(morpho) and inspect Cell summary fields.
cell = Cell(morpho)
print("\nDefault Cell summary:")
print(cell)


# 4. Inspect cell.cvs.
# A CV is the discretized compartment interval that BrainCell solves over.
print("\nCV collection:")
print("n_cv:", cell.n_cv)
print("len(cell.cvs):", len(cell.cvs))
print("first three CV intervals:")
for cv in cell.cvs[:3]:
    print(
        f"  CV {cv.id}: branch_id={cv.branch_id}, "
        f"branch_type={cv.branch_type}, prox={cv.prox}, dist={cv.dist}"
    )


# 5. Print important CV attributes.
# CVs carry topology, geometry, cable properties, and mechanism containers.
cv0 = cell.cvs[0]
print("\nFirst CV detail:")
print(cv0)

print("id:", cv0.id)
print("branch_id:", cv0.branch_id)
print("branch_type:", cv0.branch_type)
print("parent_cv:", cv0.parent_cv)
print("children_cv:", cv0.children_cv)

print("length:", cv0.length)
print("area:", cv0.area)
print("r_axial:", cv0.r_axial)

print("cm:", cv0.cm)
print("ra:", cv0.ra)
print("v:", cv0.v)
print("temp:", cv0.temp)

print("density_mech:", cv0.density_mech)
print("point_mech:", cv0.point_mech)


# 6. Compare CV policies.
default_cell = Cell(morpho)
split_cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2))
auto_cell = Cell(morpho, cv_policy=MaxCVLen(max_cv_len=20.0 * u.um))

print("\nCV policy comparison:")
print("default policy")
print(default_cell)
print()
print("CVPerBranch(cv_per_branch=2)")
print(split_cell)
print()
print("MaxCVLen(max_cv_len=20 um)")
print(auto_cell)
print()
print("first four CV intervals under CVPerBranch(cv_per_branch=2):")
for cv in split_cell.cvs[:4]:
    print(f"  CV {cv.id}: branch_id={cv.branch_id}, prox={cv.prox}, dist={cv.dist}")


# 7. Initialize state and inspect the node tree.
# init_state() builds runtime structures such as node_tree from the CV layout.
split_cell.init_state()
node_tree = split_cell.node_tree
print("\nNode tree after init_state():")
print(node_tree)
print("n_nodes:", len(node_tree.nodes))
print("n_edges:", len(node_tree.edges))
print(
    "expected n_nodes = n_cv + n_branches + 1:",
    split_cell.n_cv + len(split_cell.morpho.branches) + 1,
)


# 8. paint(...): cable properties.
# paint(region, CableProperty(...)) assigns passive cable values over CV midpoints.
cable_cell = Cell(morpho, cv_policy=CVPerBranch(cv_per_branch=2))
cable_cell.paint(
    BranchSlice(branch_index=0, prox=0.0, dist=1.0),
    CableProperty(
        resting_potential=-70.0 * u.mV,
        membrane_capacitance=2.0 * (u.uF / u.cm ** 2),
        axial_resistivity=200.0 * (u.ohm * u.cm),
        temperature=u.celsius2kelvin(20.0),
    ),
)

print("\nCable-property paint result:")
print(cable_cell)
print("painted soma CV example:")
print("v:", cable_cell.cvs[0].v)
print("cm:", cable_cell.cvs[0].cm)
print("ra:", cable_cell.cvs[0].ra)
print("temp:", cable_cell.cvs[0].temp)


# 9. paint(...): density mechanisms.
# Channel("IL", ...) is declarative; init_state() lowers it into runtime layouts.
channel_cell = Cell(morpho)
channel_cell.paint(
    BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
    Channel("IL", g_max=4.0 * (u.mS / u.cm ** 2), E=-68.0 * u.mV),
)

channel_cell.init_state()
layout = channel_cell.layouts[0]
point_id = int(layout.point_index[0])

print("\nDensity mechanism layout:")
print("layout.kind:", layout.kind)
print("layout.target:", layout.target)
print("layout.n_active:", layout.n_active)
print("layout.source_cv_ids:", layout.source_cv_ids)
print("layout.point_index:", layout.point_index.tolist())
print("sample g_max at one active point:", channel_cell.get_point_state(point_id)[layout.id]["g_max"])
print("sample E at one active point:", channel_cell.get_point_state(point_id)[layout.id]["E"])


# 10. place(...): point mechanisms.
# place(locset, mechanism) attaches point mechanisms such as clamps and probes.
place_cell = Cell(morpho, cv_policy=CVPerBranch(2))
place_cell.place(
    RootLocation(x=0.5),
    CurrentClamp(delay=1.0 * u.ms, durations=2.0 * u.ms, amplitudes=0.1 * u.nA),
)

place_cell.init_state()
layout = place_cell.layouts[0]

print("\nPoint mechanism layout:")
print(place_cell)
print("layout.kind:", layout.kind)
print("layout.target:", layout.target)
print("layout.n_active:", layout.n_active)
print("layout.point_index:", layout.point_index.tolist())
print("amplitudes:", place_cell.get_state(layout.id, "amplitudes"))
print("durations:", place_cell.get_state(layout.id, "durations"))
print("start:", place_cell.get_state(layout.id, "start"))


# 11. Minimal runnable multicompartment simulation.
# This is the canonical Cell(morpho) simulation shape: choose CV policy and solver,
# paint density mechanisms, place point mechanisms/probes, initialize, reset, run.
sim_cell = Cell(
    morpho,
    cv_policy=CVPerBranch(cv_per_branch=2),
    solver="staggered",
)

sim_cell.paint(
    BranchSlice(branch_index=[0, 1], prox=0.0, dist=1.0),
    Channel("IL", g_max=0.03 * (u.mS / u.cm ** 2), E=-54.387 * u.mV),
    Channel("Na_HH1952", g_max=120.0 * (u.mS / u.cm ** 2)),
    Channel("K_HH1952", g_max=36.0 * (u.mS / u.cm ** 2)),
)

sim_cell.place(
    RootLocation(x=0.5),
    CurrentClamp(delay=20.0 * u.ms, durations=60.0 * u.ms, amplitudes=0.2 * u.nA),
)

sim_cell.place(
    at("soma", 0.5),
    StateProbe(),
)

sim_cell.init_state()
sim_cell.reset_state()

dt = 0.1 * u.ms
duration = 100.0 * u.ms
result = sim_cell.run(dt=dt, duration=duration)

times_ms = u.math.concatenate([result.time])
soma_v_mV = u.math.concatenate([result.traces["soma(0.5)_v"]])

plt.figure(figsize=(6, 4))
plt.plot(times_ms, soma_v_mV, label="soma(0.5)_v")
plt.xlabel("Time (ms)")
plt.ylabel("Voltage (mV)")
plt.legend()
plt.tight_layout()
plt.show()
