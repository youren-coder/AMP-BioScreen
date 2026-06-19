"""Auto-detect project root and define all data/output paths."""
from pathlib import Path
import os

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Core directories (created automatically on first use)
DATA_DIR     = _PROJECT_ROOT / "data"
DATABASE_DIR = DATA_DIR / "databases"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURE_DIR  = PROCESSED_DIR / "features"
FIGURE_DIR   = _PROJECT_ROOT / "figures"

# External tool reference (not included in repo)
# UniDL4BioPep dataset was used for benchmarking only.
# See: https://github.com/username/UniDL4BioPep
TOOLS_DIR = _PROJECT_ROOT / "tools"

# Ensure directories exist
for d in [DATA_DIR, DATABASE_DIR, PROCESSED_DIR, FEATURE_DIR, FIGURE_DIR, TOOLS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Public alias
PROJECT_ROOT = _PROJECT_ROOT
