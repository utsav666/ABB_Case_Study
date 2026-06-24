"""Factory for provider-agnostic chat LLM clients.

Supports OpenAI, Anthropic (Claude), and Google Gemini based on environment
configuration in app.envsetup.env_loader.
"""

from __future__ import annotations

from typing import Any

from app.envsetup.env_loader import (
    get_anthropic_api_key,
    get_azure_openai_api_key,
    get_azure_openai_api_version,
    get_azure_openai_endpoint,
    get_gemini_api_key,
    get_llm_model,
    get_llm_provider,
    get_merck_api_key,
    get_merck_base_url,
    get_openai_api_key,
)


def create_chat_llm(
    purpose: str,
    temperature: float = 0.0,
    model: str | None = None,
    **kwargs: Any,
):
    """Create a LangChain chat model for the configured provider.

    Args:
        purpose: Logical use-case, used to resolve <PURPOSE>_MODEL overrides.
        temperature: Sampling temperature.
        model: Optional explicit model name override.
        **kwargs: Extra provider-specific kwargs passed to the model constructor.

    Returns:
        A LangChain chat model instance.
    """
    provider = get_llm_provider()
    selected_model = model or get_llm_model(purpose)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as error:  # pragma: no cover - dependency issue
            raise ValueError(
                "langchain-openai is required for LLM_PROVIDER=openai. "
                "Install it via pip install langchain-openai."
            ) from error

        return ChatOpenAI(
            model=selected_model,
            temperature=temperature,
            api_key=get_openai_api_key(),
            **kwargs,
        )

    if provider == "azure_openai":
        try:
            from langchain_openai import AzureChatOpenAI
        except ImportError as error:  # pragma: no cover - dependency issue
            raise ValueError(
                "langchain-openai is required for LLM_PROVIDER=azure_openai. "
                "Install it via pip install langchain-openai."
            ) from error

        return AzureChatOpenAI(
            azure_endpoint=get_azure_openai_endpoint(),
            api_key=get_azure_openai_api_key(),
            api_version=get_azure_openai_api_version(),
            azure_deployment=selected_model,
            temperature=temperature,
            **kwargs,
        )

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:  # pragma: no cover - dependency issue
            raise ValueError(
                "langchain-anthropic is required for LLM_PROVIDER=claude. "
                "Install it via pip install langchain-anthropic."
            ) from error

        return ChatAnthropic(
            model=selected_model,
            temperature=temperature,
            api_key=get_anthropic_api_key(),
            **kwargs,
        )

    if provider == "claude_merck":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as error:  # pragma: no cover - dependency issue
            raise ValueError(
                "langchain-anthropic is required for LLM_PROVIDER=claude_merck. "
                "Install it via pip install langchain-anthropic."
            ) from error

        merck_model = selected_model
        merck_base_url = get_merck_base_url().rstrip("/")
        return ChatAnthropic(
            model=merck_model,
            temperature=temperature,
            api_key="unused",
            base_url=f"{merck_base_url}/anthropic/{merck_model}",
            default_headers={"X-Merck-APIKey": get_merck_api_key()},
            **kwargs,
        )

    if provider == "gemini":
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as error:  # pragma: no cover - dependency issue
            raise ValueError(
                "langchain-google-genai is required for LLM_PROVIDER=gemini/google. "
                "Install it via pip install langchain-google-genai."
            ) from error

        return ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=temperature,
            google_api_key=get_gemini_api_key(),
            **kwargs,
        )

    # Defensive fallback; env loader should have rejected unknown provider first.
    raise ValueError(f"Unsupported LLM provider: {provider}")
