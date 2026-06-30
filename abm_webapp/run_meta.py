"""
Run metadata collection for reproducibility.

Every run persists environment fingerprint data so that any result can be
traced back to the exact code, dependencies, and platform that produced it.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _safe_run(cmd: list[str], *, timeout: float = 5.0) -> Optional[str]:
    """Run a subprocess silently, returning stdout or None on failure."""
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def get_git_commit() -> Optional[str]:
    """Return the short git commit hash, or None if not in a git repo."""
    return _safe_run(["git", "rev-parse", "--short", "HEAD"])


def get_git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    status = _safe_run(["git", "status", "--porcelain"])
    if status is None:
        return False
    return len(status.strip()) > 0


def get_pip_freeze() -> str:
    """Capture ``pip freeze`` output (or empty string on failure)."""
    return _safe_run([sys.executable, "-m", "pip", "freeze", "--local"]) or ""


def get_freeze_hash(freeze_text: str) -> str:
    """SHA-256 of the pip freeze output for quick comparison."""
    return hashlib.sha256(freeze_text.encode("utf-8")).hexdigest()[:16]


def collect_run_meta(
    *,
    app_version: str,
    seed: Optional[int] = None,
    config_yaml: str = "",
    schema_version: int = 2,
) -> Dict[str, Any]:
    """
    Build a metadata dict for a new simulation run.

    Parameters
    ----------
    app_version
        Semantic version of the webapp (from ``abm_webapp.__version__``).
    seed
        Random seed used for the run (if known at collection time).
    config_yaml
        Full validated scenario YAML as a string.
    schema_version
        Current DB schema version.

    Returns
    -------
    dict
        Flat metadata mapping ready for JSON serialization / DB storage.
    """
    pip_freeze = get_pip_freeze()
    git_commit = get_git_commit()
    git_dirty = get_git_dirty()

    return {
        "app_version": str(app_version),
        "git_commit": git_commit or "unknown",
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "platform": platform.platform(),
        "platform_machine": platform.machine(),
        "os_name": os.name,
        "pid": os.getpid(),
        "seed": seed,
        "schema_version": schema_version,
        "pip_freeze": pip_freeze,
        "pip_freeze_hash": get_freeze_hash(pip_freeze),
        "config_yaml": config_yaml,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
