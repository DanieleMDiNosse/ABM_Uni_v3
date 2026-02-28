"""Tests for experiment design generation utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from core.experiment_design import (
    DesignSpec,
    ParameterSpec,
    generate_design_points,
    map_unit_to_point,
    stable_content_hash,
)


def test_stable_content_hash_ignores_mapping_order() -> None:
    h1 = stable_content_hash({"b": 2, "a": 1}, n_hex=16)
    h2 = stable_content_hash({"a": 1, "b": 2}, n_hex=16)
    assert h1 == h2


def test_map_unit_to_point_discrete_edges_are_in_range() -> None:
    space = [
        ParameterSpec(name="k", kind="discrete", values=(0, 10, 20), cast="int"),
    ]
    v0, idx0 = map_unit_to_point(space, [0.0])
    v1, idx1 = map_unit_to_point(space, [0.999999999])

    assert v0["k"] in (0, 10, 20)
    assert v1["k"] in (0, 10, 20)
    assert 0 <= idx0["k"] <= 2
    assert 0 <= idx1["k"] <= 2


def test_lhs_is_deterministic_given_seed() -> None:
    space = [
        ParameterSpec(name="k_sigma", kind="continuous", bounds=(0.0, 2.0), transform="linear"),
        ParameterSpec(name="mint_sigma", kind="continuous", bounds=(1.0, 2.0), transform="linear"),
        ParameterSpec(name="k_discrete", kind="discrete", values=(0, 1, 2), cast="int"),
    ]
    design = DesignSpec(type="lhs", n_points=32, seed=123)
    pts_a, meta_a = generate_design_points(space=space, design=design, seed_base=1)
    pts_b, meta_b = generate_design_points(space=space, design=design, seed_base=999)

    assert meta_a["type"] == "lhs"
    assert meta_b["type"] == "lhs"
    assert len(pts_a) == 32
    assert len(pts_b) == 32
    assert [p.values for p in pts_a] == [p.values for p in pts_b]
    assert [p.indices for p in pts_a] == [p.indices for p in pts_b]


def test_sobol_is_deterministic_given_seed() -> None:
    space = [
        ParameterSpec(name="k_sigma", kind="continuous", bounds=(0.0, 2.0), transform="linear"),
        ParameterSpec(name="mint_mu", kind="continuous", bounds=(-1.0, -0.1), transform="linear"),
    ]
    design = DesignSpec(type="sobol", n_points=64, seed=7)
    pts_a, _ = generate_design_points(space=space, design=design, seed_base=1)
    pts_b, _ = generate_design_points(space=space, design=design, seed_base=1)
    assert [p.values for p in pts_a] == [p.values for p in pts_b]


def test_saltelli_point_count_and_roles() -> None:
    space = [
        ParameterSpec(name="k_sigma", kind="continuous", bounds=(0.0, 2.0), transform="linear"),
        ParameterSpec(name="mint_sigma", kind="continuous", bounds=(1.0, 2.0), transform="linear"),
        ParameterSpec(name="k_discrete", kind="discrete", values=(0, 1, 2), cast="int"),
    ]
    d = len(space)
    n_base = 16
    design = DesignSpec(type="sobol_saltelli", n_base=n_base, seed=11)
    pts, meta = generate_design_points(space=space, design=design, seed_base=1)
    assert meta["type"] == "sobol_saltelli"
    assert len(pts) == n_base * (2 + d)

    roles = [p.meta.get("role") for p in pts]
    assert roles[:n_base] == ["A"] * n_base
    assert roles[n_base : 2 * n_base] == ["B"] * n_base
    assert roles[2 * n_base :] == ["AB"] * (n_base * d)

    i_params = [p.meta.get("i_param") for p in pts[2 * n_base :]]
    # There should be exactly n_base occurrences of each i_param in 0..d-1.
    counts = {i: i_params.count(i) for i in range(d)}
    assert all(counts[i] == n_base for i in range(d))
