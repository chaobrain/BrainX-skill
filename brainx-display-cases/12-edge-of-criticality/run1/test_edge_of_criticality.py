import numpy as np

from edge_of_criticality import contiguous_avalanche_sizes, locate_critical_region


def test_contiguous_avalanche_sizes():
    counts = np.array([0, 2, 1, 0, 0, 4, 3, 2, 0])
    np.testing.assert_array_equal(contiguous_avalanche_sizes(counts), [3, 9])


def test_critical_region_requires_adjacent_points():
    rows = [
        {"coupling": 0.8, "susceptibility": 4.6, "unstable_fraction": 0.0},
        {"coupling": 0.9, "susceptibility": 5.0, "unstable_fraction": 0.0},
        {"coupling": 1.0, "susceptibility": 4.7, "unstable_fraction": 0.05},
        {"coupling": 1.1, "susceptibility": 8.0, "unstable_fraction": 0.25},
    ]
    region, optimum = locate_critical_region(rows)
    assert [row["coupling"] for row in region] == [0.8, 0.9, 1.0]
    assert optimum["coupling"] == 0.9


def test_isolated_optimum_is_not_reported_as_a_region():
    rows = [
        {"coupling": 0.8, "susceptibility": 1.0, "unstable_fraction": 0.0},
        {"coupling": 0.9, "susceptibility": 5.0, "unstable_fraction": 0.0},
        {"coupling": 1.0, "susceptibility": 1.0, "unstable_fraction": 0.0},
    ]
    region, optimum = locate_critical_region(rows)
    assert region == []
    assert optimum["coupling"] == 0.9


def test_zero_variability_is_not_critical():
    rows = [
        {"coupling": 0.8, "susceptibility": 0.0, "unstable_fraction": 0.0},
        {"coupling": 0.9, "susceptibility": 0.0, "unstable_fraction": 0.0},
    ]
    assert locate_critical_region(rows) == ([], None)
