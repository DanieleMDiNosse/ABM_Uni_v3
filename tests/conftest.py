"""Pytest configuration.

This project is intentionally runnable without requiring `pip install -e .`.
To keep tests importable from a clean checkout, ensure the repo root is on
`sys.path` so imports like `import core` work under pytest.
"""

from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Prepend the repository root to `sys.path`.

    Notes
    -----
    - Pytest's import mechanics can vary across versions / import modes.
    - Keeping this in a single place avoids sprinkling `sys.path` hacks across tests.
    """

    repo_root = Path(__file__).resolve().parents[1]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


_ensure_repo_root_on_path()

