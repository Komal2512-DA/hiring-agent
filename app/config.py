from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env", override=False)


def _env_bool(name: str, default: str = "0") -> bool:
    value = os.getenv(name, default).strip().lower()
    return value in {"1", "true", "yes", "on"}


def _env_api_key() -> str:
    return os.getenv("API_KEY", "").strip() or os.getenv("HF_TOKEN", "").strip()


class Settings(BaseModel):
    api_base_url: str = Field(default_factory=lambda: os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    model_name: str = Field(default_factory=lambda: os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
    api_key: str = Field(default_factory=_env_api_key)
    hf_token: str = Field(default_factory=lambda: os.getenv("HF_TOKEN", ""))

    openenv_seed: int = Field(default_factory=lambda: int(os.getenv("OPENENV_SEED", "42")))
    app_port: int = Field(default_factory=lambda: int(os.getenv("PORT", "7860")))
    use_llm_justification: bool = Field(default_factory=lambda: _env_bool("USE_LLM_JUSTIFICATION", "0"))
    use_llm_scoring: bool = Field(default_factory=lambda: _env_bool("USE_LLM_SCORING", "0"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
