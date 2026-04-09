# core/env_loader.py
import os
from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False  # simple flag to avoid loading multiple times


def load_env():
    """
    Load environment variables from the central .env file
    no matter which module calls it.
    """
    global _env_loaded
    if _env_loaded:
        return  # prevent redundant loading

    # Find project root (assuming this file lives in project/core/)
    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / "config" / ".env"

    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at: {env_path}")

    # Override empty or stale process-level values with the project .env file.
    load_dotenv(dotenv_path=env_path, override=True)
    _env_loaded = True


def get_openai_model() -> str:
    """Return the configured OpenAI model name, defaulting to gpt-4o."""
    load_env()
    return os.getenv("OPENAI_MODEL", "gpt-4o").strip().strip("'\"")


def get_openai_api_key() -> str:
    """Return the configured OpenAI API key from environment or .env."""
    load_env()

    api_key = (
        os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENAI_KEY")
        or os.getenv("OPENAI_APIKEY")
        or os.getenv("OPENAI")
    )

    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set. Add it to app/config/.env or export OPENAI_API_KEY."
        )

    return api_key.strip().strip("'\"")


load_env()
