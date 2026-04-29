"""Bootstrap repo-local cache/temp settings for script-based Python entry points."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from vcsr_env import bootstrap_local_storage

bootstrap_local_storage()
