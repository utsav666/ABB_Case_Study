# core/env_loader.py
import os
from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False  # simple flag to avoid loading multiple times


def _is_non_empty_env_value(env_name: str) -> bool:
    """Return True when an environment variable exists and has non-whitespace content."""
    env_value = os.getenv(env_name)
    return bool(env_value and env_value.strip())


def load_env() -> None:
    """Load environment variables from the project `.env` file when available."""
    global _env_loaded
    if _env_loaded:
        return

    app_dir = Path(__file__).resolve().parents[1]
    workspace_dir = app_dir.parent

    env_paths = [
        app_dir / "config" / ".env",
        app_dir / ".env",
        workspace_dir / ".env",
    ]

    has_non_empty_openai_key = _is_non_empty_env_value("OPENAI_API_KEY")

    for env_path in env_paths:
        if not env_path.exists():
            continue

        # Keep deployment-provided values unless they are present but empty.
        load_dotenv(dotenv_path=env_path, override=not has_non_empty_openai_key)
        break

    _env_loaded = True


def get_openai_api_key() -> str:
    """Return the configured OpenAI API key from environment or .env."""
    load_env()

    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_key = os.getenv("OPENAI_KEY")
    openai_apikey = os.getenv("OPENAI_APIKEY")
    openai_alias = os.getenv("OPENAI")

    api_key = openai_api_key or openai_key or openai_apikey or openai_alias

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Add it to app/config/.env, app/.env, "
            "workspace .env, or export OPENAI_API_KEY."
        )

    return api_key.strip().strip("'\"")


load_env()
