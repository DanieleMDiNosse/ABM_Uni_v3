from __future__ import annotations

import re
from pathlib import Path

from core.artifacts import build_run_manifest, make_unique_dir, safe_tag, write_csv_rows


def test_safe_tag_sanitizes_and_is_nonempty():
    tag = safe_tag("fee mode: volatility_dex / v1")
    assert tag
    assert re.fullmatch(r"[A-Za-z0-9._-]+", tag) is not None


def test_make_unique_dir_creates_suffix_on_collision(tmp_path: Path):
    base = tmp_path / "run"
    first = make_unique_dir(base)
    second = make_unique_dir(base)
    assert first.exists() and first.is_dir()
    assert second.exists() and second.is_dir()
    assert first != second
    assert second.name.startswith(first.name)


def test_write_csv_rows_writes_union_of_fields(tmp_path: Path):
    out = tmp_path / "summary.csv"
    write_csv_rows(out, [{"a": 1, "b": 2}, {"b": 3, "c": 4}])
    text = out.read_text(encoding="utf-8")
    header = text.splitlines()[0]
    assert header.split(",") == ["a", "b", "c"]


def test_build_run_manifest_is_jsonable_and_has_git_fields(tmp_path: Path):
    manifest = build_run_manifest(script="demo", run_id="demo_001", config_path=None)
    payload = manifest.to_dict()
    assert payload["script"] == "demo"
    assert payload["run_id"] == "demo_001"
    assert isinstance(payload["created_at_utc"], str) and payload["created_at_utc"]
    # git_commit can be None (export without .git) or a hex hash.
    git_commit = payload.get("git_commit")
    assert git_commit is None or re.fullmatch(r"[0-9a-f]{7,40}", str(git_commit)) is not None

