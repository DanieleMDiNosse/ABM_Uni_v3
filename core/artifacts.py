"""Artifact helpers for reproducible, non-overwriting experiment outputs.

This module centralizes small utilities used by scripts to:
- create unique output folders (avoid silently overwriting results),
- snapshot configs and metadata (reproducibility),
- write simple JSON/CSV artifacts in a consistent way.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Sequence


_SAFE_TAG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def safe_tag(value: str, *, max_len: int = 120) -> str:
    """Convert an arbitrary string into a filesystem-friendly tag.

    Parameters
    ----------
    value
        Input string (e.g., scenario label, CLI flag value).
    max_len
        Maximum length of the returned tag.

    Returns
    -------
    str
        A tag containing only `[A-Za-z0-9._-]` characters, with other runs
        replaced by `-` and leading/trailing separators stripped.

    Notes
    -----
    - This is intentionally conservative: it produces stable, readable folder names
      across platforms and shells.
    - If the sanitized tag is empty, returns `"untagged"`.

    Examples
    --------
    >>> safe_tag("fee mode: volatility_dex / v1")
    'fee-mode-volatility_dex-v1'
    """
    raw = str(value)
    sanitized = _SAFE_TAG_RE.sub("-", raw).strip("-._")
    if not sanitized:
        sanitized = "untagged"
    if max_len <= 0:
        return sanitized
    return sanitized[: int(max_len)]


def make_unique_dir(path: Path) -> Path:
    """Create a unique directory, appending a numeric suffix if needed.

    Parameters
    ----------
    path
        Desired directory path.

    Returns
    -------
    pathlib.Path
        The created directory path. If `path` already existed, a suffix `_<n>`
        is appended until an available path is found.

    Notes
    -----
    - Uses atomic `mkdir(exist_ok=False)` attempts, so it is safe under mild
      concurrent usage (e.g., two invocations starting at the same time).
    - Parent directories are created with `exist_ok=True`.

    Examples
    --------
    >>> from pathlib import Path
    >>> d1 = make_unique_dir(Path("/tmp/example_run"))
    >>> d2 = make_unique_dir(Path("/tmp/example_run"))
    >>> d1 != d2
    True
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = path
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = path.with_name(f"{path.name}_{suffix}")
            suffix += 1


def git_commit_hash(repo_root: Path | None = None) -> str | None:
    """Return the current git commit hash if available.

    Parameters
    ----------
    repo_root
        Repository root to run git in. If None, uses the current working directory.

    Returns
    -------
    str | None
        Commit hash (full) if `git` is available and the directory is a git repo,
        otherwise `None`.

    Notes
    -----
    - This function is best-effort by design: scripts should remain runnable
      even when `git` is not installed or when the repo is exported without `.git/`.

    Examples
    --------
    >>> isinstance(git_commit_hash(), (str, type(None)))
    True
    """
    cwd = str(Path(repo_root) if repo_root is not None else Path.cwd())
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None
    return out or None


def utc_now_iso() -> str:
    """Return a UTC timestamp in ISO-8601 format (seconds precision).

    Parameters
    ----------
    None

    Returns
    -------
    str
        Current UTC time formatted as `YYYY-MM-DDTHH:MM:SSZ`.

    Notes
    -----
    - This is metadata only; experiment determinism must not depend on this value.

    Examples
    --------
    >>> isinstance(utc_now_iso(), str)
    True
    """
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def write_json(path: Path, payload: Mapping[str, Any], *, indent: int = 2) -> None:
    """Write a JSON file with UTF-8 encoding, creating parent directories.

    Parameters
    ----------
    path
        Destination path.
    payload
        JSON-serializable mapping.
    indent
        Indentation level for pretty printing.

    Returns
    -------
    None

    Notes
    -----
    - Uses `sort_keys=True` for stable diffs and reproducibility.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_json(Path("/tmp/example.json"), {"a": 1})
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=int(indent), sort_keys=True) + "\n", encoding="utf-8")


def write_csv_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a CSV file (overwrite) from a sequence of row mappings.

    Parameters
    ----------
    path
        Destination `.csv` path.
    rows
        Sequence of rows (each a mapping). Fieldnames are the union of all keys
        encountered, written in sorted order.

    Returns
    -------
    None

    Notes
    -----
    - Overwrites `path` by design; callers should place outputs inside a unique
      run directory created by `make_unique_dir`.
    - Values are written as-is; callers should pre-cast complex objects.

    Examples
    --------
    >>> from pathlib import Path
    >>> write_csv_rows(Path("/tmp/example.csv"), [{"seed": 1, "metric": 0.1}])
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for row in rows:
        keys.update(str(k) for k in row.keys())
    fieldnames = sorted(keys)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({str(k): row.get(k) for k in fieldnames})


def snapshot_file(src: Path, dst: Path) -> None:
    """Copy a file as a reproducibility snapshot.

    Parameters
    ----------
    src
        Source path to read.
    dst
        Destination path to write.

    Returns
    -------
    None

    Notes
    -----
    - This is a byte-for-byte copy via Python I/O (no metadata).
    - Parent directories for `dst` are created automatically.

    Examples
    --------
    >>> from pathlib import Path
    >>> snapshot_file(Path("pyproject.toml"), Path("/tmp/pyproject_snapshot.toml"))  # doctest: +SKIP
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


@dataclass(frozen=True)
class RunManifest:
    """Standard metadata written for script outputs."""

    script: str
    run_id: str
    created_at_utc: str
    config_path: str | None
    git_commit: str | None
    python: str
    platform: str

    def to_dict(self) -> MutableMapping[str, Any]:
        """Convert the manifest into a JSON-serializable mapping.

        Parameters
        ----------
        None

        Returns
        -------
        MutableMapping[str, Any]
            Plain-Python dictionary suitable for `json.dumps(...)`.

        Notes
        -----
        - This method intentionally returns a new mapping (callers may mutate).

        Examples
        --------
        >>> m = RunManifest(
        ...     script="demo",
        ...     run_id="demo_001",
        ...     created_at_utc="2026-01-01T00:00:00Z",
        ...     config_path=None,
        ...     git_commit=None,
        ...     python="3.11.0",
        ...     platform="linux",
        ... )
        >>> isinstance(m.to_dict(), dict)
        True
        """
        return {
            "script": self.script,
            "run_id": self.run_id,
            "created_at_utc": self.created_at_utc,
            "config_path": self.config_path,
            "git_commit": self.git_commit,
            "python": self.python,
            "platform": self.platform,
        }


def build_run_manifest(*, script: str, run_id: str, config_path: Path | None = None) -> RunManifest:
    """Build a standard run manifest for a script invocation.

    Parameters
    ----------
    script
        Script name (e.g., `"run"`, `"run_multiple"`).
    run_id
        Run identifier used for the output folder name.
    config_path
        Optional scenario YAML path used for the run.

    Returns
    -------
    RunManifest
        A manifest containing timestamp, git hash (if available), and basic
        interpreter/platform info.

    Notes
    -----
    - The returned manifest is intended to be written to `metadata.json` in the
      run folder via `write_json`.
    - Uses `sys.version` and `sys.platform` for lightweight provenance.

    Examples
    --------
    >>> m = build_run_manifest(script="demo", run_id="demo_001")
    >>> "script" in m.to_dict()
    True
    """
    python_str = sys.version.split()[0]
    platform_str = sys.platform
    commit = git_commit_hash()
    return RunManifest(
        script=str(script),
        run_id=str(run_id),
        created_at_utc=utc_now_iso(),
        config_path=None if config_path is None else str(Path(config_path).expanduser().resolve()),
        git_commit=commit,
        python=str(python_str),
        platform=str(platform_str),
    )
