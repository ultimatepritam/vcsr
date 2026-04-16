"""
Repo-local runtime environment bootstrap.

This keeps caches, temporary files, and other writable runtime artifacts inside
the repository so Windows hosts do not spill data into the user's default C:
profile directories.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
RUNTIME_ROOT = REPO_ROOT / ".local"

_DIRS = {
    "runtime": RUNTIME_ROOT,
    "cache": RUNTIME_ROOT / "cache",
    "tmp": RUNTIME_ROOT / "tmp",
    "home": RUNTIME_ROOT / "home",
    "userprofile": RUNTIME_ROOT / "userprofile",
    "userprofile_temp": RUNTIME_ROOT / "userprofile" / "AppData" / "Local" / "Temp",
    "appdata": RUNTIME_ROOT / "appdata",
    "localappdata": RUNTIME_ROOT / "localappdata",
    "state": RUNTIME_ROOT / "state",
    "pip_cache": RUNTIME_ROOT / "cache" / "pip",
    "hf_home": RUNTIME_ROOT / "cache" / "huggingface",
    "hf_hub": RUNTIME_ROOT / "cache" / "huggingface" / "hub",
    "hf_datasets": RUNTIME_ROOT / "cache" / "huggingface" / "datasets",
    "transformers": RUNTIME_ROOT / "cache" / "huggingface" / "transformers",
    "torch": RUNTIME_ROOT / "cache" / "torch",
    "wandb": RUNTIME_ROOT / "state" / "wandb",
    "wandb_cache": RUNTIME_ROOT / "cache" / "wandb",
    "matplotlib": RUNTIME_ROOT / "cache" / "matplotlib",
    "jupyter_config": RUNTIME_ROOT / "state" / "jupyter" / "config",
    "jupyter_data": RUNTIME_ROOT / "state" / "jupyter" / "data",
    "jupyter_runtime": RUNTIME_ROOT / "state" / "jupyter" / "runtime",
    "ipython": RUNTIME_ROOT / "state" / "ipython",
    "pycache": RUNTIME_ROOT / "cache" / "pycache",
}


def get_runtime_dir(name: str) -> Path:
    """Return a repo-local runtime directory and create it if needed."""
    path = _DIRS[name]
    path.mkdir(parents=True, exist_ok=True)
    return path


def bootstrap_local_storage() -> Path:
    """
    Configure environment variables so caches and temp files stay under
    E:\\Engineering\\vcsr\\.local by default.
    """
    for path in _DIRS.values():
        path.mkdir(parents=True, exist_ok=True)

    defaults = {
        "TMP": str(_DIRS["userprofile_temp"]),
        "TEMP": str(_DIRS["userprofile_temp"]),
        "TMPDIR": str(_DIRS["userprofile_temp"]),
        "PIP_CACHE_DIR": str(_DIRS["pip_cache"]),
        "XDG_CACHE_HOME": str(_DIRS["cache"]),
        "USERPROFILE": str(_DIRS["userprofile"]),
        "APPDATA": str(_DIRS["appdata"]),
        "LOCALAPPDATA": str(_DIRS["localappdata"]),
        "HF_HOME": str(_DIRS["hf_home"]),
        "HUGGINGFACE_HUB_CACHE": str(_DIRS["hf_hub"]),
        "HF_DATASETS_CACHE": str(_DIRS["hf_datasets"]),
        "TRANSFORMERS_CACHE": str(_DIRS["transformers"]),
        "TORCH_HOME": str(_DIRS["torch"]),
        "WANDB_DIR": str(_DIRS["wandb"]),
        "WANDB_CACHE_DIR": str(_DIRS["wandb_cache"]),
        "MPLCONFIGDIR": str(_DIRS["matplotlib"]),
        "JUPYTER_CONFIG_DIR": str(_DIRS["jupyter_config"]),
        "JUPYTER_DATA_DIR": str(_DIRS["jupyter_data"]),
        "JUPYTER_RUNTIME_DIR": str(_DIRS["jupyter_runtime"]),
        "IPYTHONDIR": str(_DIRS["ipython"]),
        "PYTHONPYCACHEPREFIX": str(_DIRS["pycache"]),
        # Some libraries still consult HOME on Windows even when USERPROFILE exists.
        "HOME": str(_DIRS["home"]),
    }

    for key, value in defaults.items():
        os.environ.setdefault(key, value)

    tempfile.tempdir = os.environ["TMP"]
    return RUNTIME_ROOT


# Apply on import so scripts only need to import this module once.
bootstrap_local_storage()
