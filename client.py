"""Root-level HTTP client compatibility shim for OpenEnv CLI structure checks.

This lightweight client is intentionally simple and uses the existing FastAPI
HTTP contract exposed by this repository (`/health`, `/tasks`, `/reset`,
`/step`, `/state`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from app.models import ResetResponse, StepResponse


@dataclass
class HiringEnvClient:
    base_url: str
    timeout_seconds: float = 30.0

    def _url(self, path: str) -> str:
        return f"{self.base_url.rstrip('/')}{path}"

    def health(self) -> dict[str, Any]:
        resp = requests.get(self._url("/health"), timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def tasks(self) -> list[dict[str, Any]]:
        resp = requests.get(self._url("/tasks"), timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def reset(self, task_id: str | None = None) -> ResetResponse:
        payload: dict[str, Any] = {}
        if task_id is not None:
            payload["task_id"] = task_id
        resp = requests.post(self._url("/reset"), json=payload, timeout=self.timeout_seconds)
        resp.raise_for_status()
        return ResetResponse.model_validate(resp.json())

    def state(self) -> dict[str, Any]:
        resp = requests.get(self._url("/state"), timeout=self.timeout_seconds)
        resp.raise_for_status()
        return resp.json()

    def step(self, action_type: str, payload: dict[str, Any]) -> StepResponse:
        resp = requests.post(
            self._url("/step"),
            json={"action_type": action_type, "payload": payload},
            timeout=self.timeout_seconds,
        )
        resp.raise_for_status()
        return StepResponse.model_validate(resp.json())

