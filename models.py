"""Root-level model compatibility shim for OpenEnv CLI structure checks."""

from app.models import Action as HiringAction
from app.models import Observation as HiringObservation

__all__ = [
    "HiringAction",
    "HiringObservation",
]

