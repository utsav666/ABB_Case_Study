# core/env_loader.py
import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

_env_loaded = False  # simple flag to avoid loading multiple times


def _load_from_aws_secrets() -> None:
    """Pull secrets from AWS Secrets Manager and inject into os.environ.

    Triggered only when the AWS_SECRET_NAME environment variable is set.
    The secret value must be a JSON object whose keys map directly to env
    var names, e.g.:
        {
            "OPENAI_API_KEY": "sk-...",
            "MERCK_API_KEY": "...",
            "OAUTH_CLIENT_SECRET": "...",
            "DB_PASSWORD": "...",
            "DATABASE_URL": "postgresql://..."
        }

    On ECS Fargate: set AWS_SECRET_NAME in the task definition environment.
    Locally:        do NOT set AWS_SECRET_NAME — .env file is used instead.
    """
    secret_name = os.getenv("AWS_SECRET_NAME")
    if not secret_name:
        return

    try:
        import boto3
        from botocore.exceptions import BotoCoreError, ClientError
    except ImportError:
        logger.warning("boto3 not installed — skipping AWS Secrets Manager load.")
        return

    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))

    try:
        client = boto3.client("secretsmanager", region_name=region)
        response = client.get_secret_value(SecretId=secret_name)
        secret_str = response.get("SecretString", "{}")
        secrets: dict = json.loads(secret_str)
        for key, value in secrets.items():
            # Only inject — never overwrite values already explicitly set
            if key not in os.environ:
                os.environ[key] = str(value)
        logger.info("Loaded %d secrets from AWS Secrets Manager (%s).", len(secrets), secret_name)
    except (BotoCoreError, ClientError) as exc:
        logger.error("Failed to load secrets from AWS Secrets Manager: %s", exc)
        raise

_PROVIDER_ALIASES = {
    "openai": "openai",
    "gpt": "openai",
    "azure_openai": "azure_openai",
    "azure": "azure_openai",
    "anthropic": "claude",
    "claude": "claude",
    "claude_merck": "claude_merck",
    "google": "gemini",
    "gemini": "gemini",
}

_DEFAULT_MODEL_BY_PROVIDER = {
    "openai": "gpt-4o",
    "azure_openai": "gpt-4o",
    "claude": "claude-sonnet-4-20250514",
    "claude_merck": "claude-opus-4-6-v1",
    "gemini": "gemini-1.5-pro",
}


def _is_non_empty_env_value(env_name: str) -> bool:
    """Return True when an environment variable exists and has non-whitespace content."""
    env_value = os.getenv(env_name)
    return bool(env_value and env_value.strip())


def load_env() -> None:
    """Load environment variables with a two-tier strategy:

    1. AWS Secrets Manager  — used when AWS_SECRET_NAME is set (ECS Fargate / prod).
    2. .env file            — used when AWS_SECRET_NAME is NOT set (local dev).

    The AWS path is tried first so that secrets injected there always take
    precedence over any leftover values in the process environment.
    """
    global _env_loaded
    if _env_loaded:
        return

    # --- Tier 1: AWS Secrets Manager (prod / ECS Fargate) ---
    _load_from_aws_secrets()

    # --- Tier 2: .env file (local dev) ---
    app_dir = Path(__file__).resolve().parents[1]
    workspace_dir = app_dir.parent

    env_paths = [
        app_dir / "config" / ".env",
        app_dir / ".env",
        workspace_dir / ".env",
    ]

    has_any_non_empty_llm_key = any(
        _is_non_empty_env_value(env_name)
        for env_name in (
            "OPENAI_API_KEY",
            "OPENAI_KEY",
            "OPENAI_APIKEY",
            "OPENAI",
            "AZURE_OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "CLAUDE_API_KEY",
            "MERCK_API_KEY",
            "GOOGLE_API_KEY",
            "GEMINI_API_KEY",
        )
    )

    for env_path in env_paths:
        if not env_path.exists():
            continue

        # Keep deployment-provided values unless they are present but empty.
        load_dotenv(dotenv_path=env_path, override=not has_any_non_empty_llm_key)
        break

    _env_loaded = True


def get_openai_model() -> str:
    """Return the configured OpenAI model name, defaulting to gpt-4o."""
    load_env()
    return os.getenv("OPENAI_MODEL", "gpt-4o").strip().strip("'\"")


def get_llm_provider() -> str:
    """Return normalized LLM provider from environment."""
    load_env()
    raw_provider = os.getenv("LLM_PROVIDER", "openai").strip().strip("'\"").lower()
    provider = _PROVIDER_ALIASES.get(raw_provider, raw_provider)
    if provider not in _DEFAULT_MODEL_BY_PROVIDER:
        supported = ", ".join(sorted(_DEFAULT_MODEL_BY_PROVIDER))
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{raw_provider}'. Supported values: {supported}."
        )
    return provider


def get_llm_model(purpose: str | None = None) -> str:
    """Return model name from environment with provider-aware fallbacks."""
    load_env()
    provider = get_llm_provider()

    env_keys: list[str] = []
    if purpose:
        normalized = purpose.strip().upper().replace("-", "_")
        if normalized:
            env_keys.append(f"{normalized}_MODEL")

    env_keys.append("LLM_MODEL")

    if provider == "openai":
        env_keys.append("OPENAI_MODEL")
    elif provider == "azure_openai":
        env_keys.extend(["AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_MODEL"])
    elif provider == "claude":
        env_keys.extend(["CLAUDE_MODEL", "ANTHROPIC_MODEL"])
    elif provider == "claude_merck":
        env_keys.append("MERCK_MODEL")
    else:
        env_keys.extend(["GEMINI_MODEL", "GOOGLE_MODEL"])

    for key in env_keys:
        value = os.getenv(key)
        if value and value.strip():
            return value.strip().strip("'\"")

    return _DEFAULT_MODEL_BY_PROVIDER[provider]


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


def get_anthropic_api_key() -> str:
    """Return configured Anthropic/Claude API key."""
    load_env()
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    if not api_key:
        raise ValueError(
            "CLAUDE_API_KEY/ANTHROPIC_API_KEY environment variable not set."
        )
    return api_key.strip().strip("'\"")


def get_gemini_api_key() -> str:
    """Return configured Gemini/Google API key."""
    load_env()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY/GOOGLE_API_KEY environment variable not set."
        )
    return api_key.strip().strip("'\"")


def get_azure_openai_api_key() -> str:
    """Return Azure OpenAI API key."""
    load_env()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "AZURE_OPENAI_API_KEY environment variable not set for LLM_PROVIDER=azure_openai."
        )
    return api_key.strip().strip("'\"")


def get_azure_openai_endpoint() -> str:
    """Return Azure OpenAI endpoint."""
    load_env()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise ValueError(
            "AZURE_OPENAI_ENDPOINT environment variable not set for LLM_PROVIDER=azure_openai."
        )
    return endpoint.strip().strip("'\"")


def get_azure_openai_api_version() -> str:
    """Return Azure OpenAI API version with default."""
    load_env()
    return os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview").strip().strip("'\"")


def get_merck_api_key() -> str:
    """Return Merck proxy API key."""
    load_env()
    api_key = os.getenv("MERCK_API_KEY")
    if not api_key:
        raise ValueError(
            "MERCK_API_KEY environment variable not set for LLM_PROVIDER=claude_merck."
        )
    return api_key.strip().strip("'\"")


def get_merck_base_url() -> str:
    """Return Merck proxy base URL."""
    load_env()
    base_url = os.getenv("MERCK_BASE_URL")
    if not base_url:
        raise ValueError(
            "MERCK_BASE_URL environment variable not set for LLM_PROVIDER=claude_merck."
        )
    return base_url.strip().strip("'\"")


load_env()
