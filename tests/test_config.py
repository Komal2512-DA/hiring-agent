import os
from pathlib import Path

from dotenv import load_dotenv

from app.config import get_settings


def test_env_defaults_or_loaded_values():
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)

    get_settings.cache_clear()
    settings = get_settings()

    assert settings.api_base_url
    assert settings.model_name

    if os.getenv("API_KEY"):
        assert settings.api_key == os.getenv("API_KEY")
    elif os.getenv("HF_TOKEN"):
        assert settings.api_key == os.getenv("HF_TOKEN")
