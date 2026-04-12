# core/env_loader.py
import os
from pathlib import Path

from dotenv import load_dotenv

_env_loaded = False  # simple flag to avoid loading multiple times


def load_env() -> None:
    """Load environment variables from the project `.env` file when available."""
    global _env_loaded
    if _env_loaded:
        return

    root_dir = Path(__file__).resolve().parents[1]
    env_path = root_dir / "config" / ".env"

    if env_path.exists():
        # Keep deployment-provided environment variables as the source of truth.
        load_dotenv(dotenv_path=env_path, override=False)

    _env_loaded = True


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
