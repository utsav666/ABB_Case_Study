"""Utility for tracking LLM API call costs and usage."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def calculate_openai_cost(
    model: str, prompt_tokens: int, completion_tokens: int
) -> float:
    """Calculate LLM API cost in USD.

    Args:
        model: Model name (e.g., "gpt-4", "claude-opus-4-6")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    # Pricing per 1K tokens (input/output).
    #
    # Order matters: the loop below does a SUBSTRING match, so more specific
    # keys must come first. "claude-opus-4-5" would otherwise be caught by a
    # bare "claude-opus" entry.
    pricing = {
        "gpt-4": {"prompt": 0.03, "completion": 0.06},
        "gpt-4-turbo": {"prompt": 0.01, "completion": 0.03},
        "gpt-4o": {"prompt": 0.005, "completion": 0.015},
        "gpt-3.5-turbo": {"prompt": 0.0015, "completion": 0.002},
        # Claude 4.6+ Opus is $5/$25 per 1M = $0.005/$0.025 per 1K.
        # This was previously $0.015/$0.075 — 3x the real rate — which
        # overstated every logged Opus cost.
        "claude-opus": {"prompt": 0.005, "completion": 0.025},
        "claude-sonnet": {"prompt": 0.003, "completion": 0.015},
        "claude-haiku": {"prompt": 0.001, "completion": 0.005},
    }

    # Find matching pricing
    model_lower = model.lower()
    for key, prices in pricing.items():
        if key in model_lower:
            prompt_cost = (prompt_tokens / 1000) * prices["prompt"]
            completion_cost = (completion_tokens / 1000) * prices["completion"]
            return round(prompt_cost + completion_cost, 6)

    # Default fallback (use gpt-4 pricing if unknown)
    logger.warning("Unknown model '%s', using gpt-4 pricing", model)
    prompt_cost = (prompt_tokens / 1000) * 0.03
    completion_cost = (completion_tokens / 1000) * 0.06
    return round(prompt_cost + completion_cost, 6)


def track_llm_call(
    repo: Any,  # ConversationRepository
    message_id: str,
    thread_id: str,
    llm_purpose: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    provider: str = "openai",
    duration_ms: int | None = None,
    metadata: dict | None = None,
) -> None:
    """Track an LLM API call.

    Args:
        repo: ConversationRepository instance
        message_id: UUID of the conversation message
        thread_id: Conversation thread ID
        llm_purpose: Purpose of the call (e.g., "sql-generator", "table-identifier")
        model: Model name (e.g., "gpt-4")
        prompt_tokens: Number of input tokens
        completion_tokens: Number of output tokens
        total_tokens: Total tokens used
        provider: LLM provider (default: "openai")
        duration_ms: Call duration in milliseconds (optional)
        metadata: Additional metadata (optional)
    """
    try:
        cost_usd = calculate_openai_cost(model, prompt_tokens, completion_tokens)

        repo.add_llm_call(
            message_id=message_id,
            thread_id=thread_id,
            llm_model=model,
            llm_provider=provider,
            llm_purpose=llm_purpose,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            request_metadata=metadata,
        )

        logger.debug(
            "Tracked LLM call: purpose=%s, model=%s, tokens=%d, cost=$%.4f",
            llm_purpose,
            model,
            total_tokens,
            cost_usd,
        )
    except Exception:
        logger.exception("Failed to track LLM call for purpose: %s", llm_purpose)


def extract_openai_usage(response: Any) -> dict[str, int]:
    """Extract token usage from OpenAI response object.

    Args:
        response: OpenAI API response

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
    """
    usage = getattr(response, "usage", None)
    if not usage:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0),
        "completion_tokens": getattr(usage, "completion_tokens", 0),
        "total_tokens": getattr(usage, "total_tokens", 0),
    }


def get_model_from_response(response: Any, default: str = "gpt-4") -> str:
    """Extract model name from OpenAI response.

    Args:
        response: OpenAI API response
        default: Default model if not found

    Returns:
        Model name
    """
    return getattr(response, "model", default)


def extract_langchain_usage(response: Any) -> dict[str, int]:
    """Extract token usage from LangChain AIMessage response.

    Supports both OpenAI (token_usage.prompt_tokens) and Anthropic/Claude
    (usage.input_tokens) metadata structures, plus the LangChain-standard
    usage_metadata attribute.

    Args:
        response: LangChain AIMessage response

    Returns:
        Dict with prompt_tokens, completion_tokens, total_tokens
    """
    # Try LangChain-standard usage_metadata first (works for all providers)
    usage_meta = getattr(response, "usage_metadata", None)
    if usage_meta and isinstance(usage_meta, dict):
        input_tok = usage_meta.get("input_tokens", 0)
        output_tok = usage_meta.get("output_tokens", 0)
        if input_tok or output_tok:
            return {
                "prompt_tokens": input_tok,
                "completion_tokens": output_tok,
                "total_tokens": usage_meta.get("total_tokens", input_tok + output_tok),
            }

    metadata = getattr(response, "response_metadata", {})

    # OpenAI format: response_metadata.token_usage
    usage = metadata.get("token_usage", {})
    if usage:
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0),
        }

    # Anthropic/Claude format: response_metadata.usage
    usage = metadata.get("usage", {})
    if usage:
        return {
            "prompt_tokens": usage.get("input_tokens", 0),
            "completion_tokens": usage.get("output_tokens", 0),
            "total_tokens": usage.get("input_tokens", 0)
            + usage.get("output_tokens", 0),
        }

    return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def get_model_from_langchain_response(response: Any, default: str = "gpt-4") -> str:
    """Extract model name from LangChain AIMessage response.

    Args:
        response: LangChain AIMessage response
        default: Default model if not found

    Returns:
        Model name
    """
    metadata = getattr(response, "response_metadata", {})
    return metadata.get("model_name", default)
