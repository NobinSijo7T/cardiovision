"""Helpers for running project scripts from the repository root.

If a script is launched with a system interpreter, prepend the project venv's
`site-packages` directory so direct `python scripts/...py` commands still work
without requiring manual activation.
"""

from __future__ import annotations

import sys
from pathlib import Path


def ensure_project_venv() -> None:
    project_root = Path(__file__).resolve().parents[1]
    venv_site_packages = project_root / ".venv" / "Lib" / "site-packages"

    if not venv_site_packages.exists():
        return

    site_packages_str = str(venv_site_packages)
    if site_packages_str not in sys.path:
        sys.path.insert(0, site_packages_str)