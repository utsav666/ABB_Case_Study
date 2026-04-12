from __future__ import annotations

import logging
from typing import Any

import requests
from fastapi import APIRouter, Request

from app.auth.oauth_service import (
    DEFAULT_OAUTH_CALLBACK_PATH,
    DEFAULT_OAUTH_SESSION_PATH,
    build_oauth_authentication_response,
    build_oauth_login_payload,
    can_exchange_oauth_code,
    exchange_oauth_code_for_token,
    extract_token_field,
    fetch_oauth_user_details,
    get_bearer_token,
    get_oauth_provider_name,
    get_refresh_window_seconds,
    get_seconds_until_expiry,
    read_auth_session_parameters,
    read_oauth_callback_parameters,
    refresh_oauth_token,
)

logger = logging.getLogger(__name__)

auth_router = APIRouter(tags=["auth"])


@auth_router.get("/login")
@auth_router.get("/auth/login")
def get_login_url(request: Request) -> dict[str, Any]:
    """Return an OAuth authorization URL for the frontend login flow."""
    return build_oauth_login_payload(request)


@auth_router.get(DEFAULT_OAUTH_CALLBACK_PATH)
@auth_router.post(DEFAULT_OAUTH_CALLBACK_PATH)
@auth_router.get("/login/callback")
@auth_router.post("/login/callback")
async def get_auth_callback(
    request: Request,
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
    error_description: str | None = None,
) -> dict[str, Any]:
    """Authenticate the OAuth callback and return the frontend payload."""
    callback_values = await read_oauth_callback_parameters(request)
    provider_name = get_oauth_provider_name()
    authorization_code = (code or callback_values.get("code", "")).strip() or None
    callback_state = (state or callback_values.get("state", "")).strip() or None
    callback_error = (error or callback_values.get("error", "")).strip() or None
    callback_error_description = (
        error_description or callback_values.get("error_description", "")
    ).strip() or None

    if callback_error:
        return build_oauth_authentication_response(
            provider_name,
            "OAuth callback received with an error.",
            status="error",
            state=callback_state,
            error=callback_error,
            error_description=(
                callback_error_description or "OAuth login was not completed."
            ),
        )

    if not authorization_code:
        return build_oauth_authentication_response(
            provider_name,
            "OAuth callback endpoint is ready to receive the authorization code.",
            status="pending",
            state=callback_state,
        )

    token_payload: dict[str, Any] | None = None
    user_payload: dict[str, Any] | None = None
    message = "OAuth callback received successfully."

    if can_exchange_oauth_code():
        try:
            token_payload = exchange_oauth_code_for_token(authorization_code, request)
            access_token = str(token_payload.get("access_token", "")).strip()
            user_payload = fetch_oauth_user_details(access_token)
            message = "OAuth callback received and user details were loaded."
        except (ValueError, requests.RequestException) as request_error:
            logger.warning(
                "OAuth backend authentication could not be completed: %s",
                request_error,
            )
            message = (
                "OAuth callback received successfully, but backend authentication "
                "could not be completed."
            )
            return build_oauth_authentication_response(
                provider_name,
                message,
                code=authorization_code,
                authorization_code=authorization_code,
                state=callback_state,
                authenticated=False,
                user_payload=None,
                token_payload=None,
                error="authentication_failed",
                error_description=(
                    "OAuth callback was received, but user details could not be loaded."
                ),
            )
    else:
        message = (
            "OAuth callback received, but backend token exchange is not configured yet."
        )

    return build_oauth_authentication_response(
        provider_name,
        message,
        code=authorization_code,
        authorization_code=authorization_code,
        state=callback_state,
        authenticated=bool(user_payload),
        user_payload=user_payload,
        token_payload=token_payload,
    )


@auth_router.get(DEFAULT_OAUTH_SESSION_PATH)
@auth_router.post(DEFAULT_OAUTH_SESSION_PATH)
async def get_auth_session(request: Request) -> dict[str, Any]:
    """Validate an active token, refresh if near expiry, and return auth payload."""
    provider_name = get_oauth_provider_name()
    refresh_window_seconds = get_refresh_window_seconds()
    session_values = await read_auth_session_parameters(request)

    session_state = str(session_values.get("state", "")).strip() or None
    access_token = get_bearer_token(request)
    if not access_token:
        access_token_candidate = extract_token_field(session_values, "access_token")
        access_token = str(access_token_candidate or "").strip()

    refresh_token_candidate = extract_token_field(session_values, "refresh_token")
    refresh_token = str(refresh_token_candidate or "").strip()

    expires_at = extract_token_field(session_values, "expires_at")
    expires_in = extract_token_field(session_values, "expires_in")

    if not access_token and not refresh_token:
        return build_oauth_authentication_response(
            provider_name,
            "Session token details are missing.",
            status="error",
            state=session_state,
            error="missing_token",
            error_description=(
                "Provide access_token (or Authorization Bearer token). "
                "Optionally include refresh_token for renewal."
            ),
        )

    token_payload: dict[str, Any] = {
        "access_token": access_token or None,
        "refresh_token": refresh_token or None,
        "expires_in": expires_in,
        "expires_at": expires_at,
    }

    should_refresh = False
    if access_token and refresh_token:
        seconds_until_expiry = get_seconds_until_expiry(
            access_token,
            expires_at=expires_at,
            expires_in=expires_in,
        )
        if (
            seconds_until_expiry is not None
            and seconds_until_expiry <= refresh_window_seconds
        ):
            should_refresh = True

    if not access_token and refresh_token:
        should_refresh = True

    if should_refresh:
        try:
            refreshed_payload = refresh_oauth_token(refresh_token, request)
            refreshed_payload.setdefault("refresh_token", refresh_token)
            token_payload.update(refreshed_payload)
            access_token = str(token_payload.get("access_token", "")).strip()
        except (ValueError, requests.RequestException) as refresh_error:
            logger.warning("OAuth token refresh failed: %s", refresh_error)
            return build_oauth_authentication_response(
                provider_name,
                "Token refresh failed.",
                status="error",
                state=session_state,
                authenticated=False,
                token_payload=token_payload,
                error="refresh_failed",
                error_description=(
                    "Unable to refresh the access token. Please authenticate again."
                ),
            )

    if not access_token:
        return build_oauth_authentication_response(
            provider_name,
            "No valid access token is available.",
            status="error",
            state=session_state,
            authenticated=False,
            token_payload=token_payload,
            error="missing_access_token",
            error_description=(
                "A refresh token was present, but the provider did not return an access token."
            ),
        )

    try:
        user_payload = fetch_oauth_user_details(access_token)
    except (ValueError, requests.RequestException) as userinfo_error:
        if refresh_token and not should_refresh:
            try:
                refreshed_payload = refresh_oauth_token(refresh_token, request)
                refreshed_payload.setdefault("refresh_token", refresh_token)
                token_payload.update(refreshed_payload)
                access_token = str(token_payload.get("access_token", "")).strip()
                user_payload = fetch_oauth_user_details(access_token)
            except (ValueError, requests.RequestException) as refresh_error:
                logger.warning(
                    "OAuth user validation and refresh both failed: %s / %s",
                    userinfo_error,
                    refresh_error,
                )
                return build_oauth_authentication_response(
                    provider_name,
                    "Session validation failed.",
                    status="error",
                    state=session_state,
                    authenticated=False,
                    token_payload=token_payload,
                    error="session_invalid",
                    error_description=(
                        "Token validation failed and refresh was unsuccessful. "
                        "Please authenticate again."
                    ),
                )

        else:
            logger.warning("OAuth token validation failed: %s", userinfo_error)
            return build_oauth_authentication_response(
                provider_name,
                "Session validation failed.",
                status="error",
                state=session_state,
                authenticated=False,
                token_payload=token_payload,
                error="session_invalid",
                error_description=(
                    "Token validation failed. Provide a valid token or refresh token."
                ),
            )

    message = "Session token is valid."
    if should_refresh:
        message = "Session token was renewed successfully."

    return build_oauth_authentication_response(
        provider_name,
        message,
        state=session_state,
        authenticated=True,
        user_payload=user_payload,
        token_payload=token_payload,
    )
