"""Top-level compatibility exports for OpenEnv CLI push validation.

This repository uses `app/` as the main runtime package, but some OpenEnv CLI
flows expect classic root-level files (`__init__.py`, `client.py`, `models.py`).
These exports bridge that expectation without changing the runtime architecture.
"""

from .client import HiringEnvClient
from .models import HiringAction, HiringObservation

__all__ = [
    "HiringAction",
    "HiringObservation",
    "HiringEnvClient",
]

