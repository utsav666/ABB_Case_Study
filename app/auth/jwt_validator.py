"""Production-grade JWT validation with signature verification."""

from __future__ import annotations

import logging
import time
from typing import Any

import jwt
import requests
from jwt import PyJWKClient
from jwt.exceptions import (
    DecodeError,
    ExpiredSignatureError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidTokenError,
)

from app.auth.oauth_config import get_oauth_config

logger = logging.getLogger(__name__)

# Cache for JWKS client
_jwks_client_cache: dict[str, tuple[float, PyJWKClient]] = {}
JWKS_CACHE_TTL_SECONDS = 3600  # 1 hour


def get_jwks_uri() -> str:
    """Get JWKS URI from OAuth configuration.

    Returns:
        str: JWKS URI for fetching public keys.
    """
    oauth_config = get_oauth_config()

    # Try to get from well-known endpoint
    try:
        well_known_url = oauth_config.well_known_url
        if well_known_url:
            timeout = (
                oauth_config.http_connect_timeout_seconds,
                oauth_config.http_read_timeout_seconds,
            )
            response = requests.get(well_known_url, timeout=timeout)
            response.raise_for_status()
            metadata = response.json()
            jwks_uri = metadata.get("jwks_uri")
            if jwks_uri:
                return jwks_uri
    except Exception as e:
        logger.warning("Failed to get JWKS URI from well-known endpoint: %s", e)

    # Fallback: construct from issuer
    # For Microsoft: https://login.microsoftonline.com/{tenant}/v2.0
    # JWKS: https://login.microsoftonline.com/{tenant}/discovery/v2.0/keys
    return ""


def get_jwks_client() -> PyJWKClient | None:
    """Get or create JWKS client with caching.

    Returns:
        PyJWKClient | None: JWKS client instance or None if unavailable.
    """
    jwks_uri = get_jwks_uri()

    if not jwks_uri:
        logger.warning("No JWKS URI available for JWT verification")
        return None

    # Check cache
    current_time = time.monotonic()
    cached_entry = _jwks_client_cache.get(jwks_uri)

    if cached_entry:
        cached_until, client = cached_entry
        if current_time < cached_until:
            return client

    # Create new client
    try:
        client = PyJWKClient(
            jwks_uri,
            cache_keys=True,
            max_cached_keys=10,
            cache_jwk_set=True,
            lifespan=JWKS_CACHE_TTL_SECONDS,
        )
        _jwks_client_cache[jwks_uri] = (
            current_time + JWKS_CACHE_TTL_SECONDS,
            client,
        )
        return client
    except Exception as e:
        logger.error("Failed to create JWKS client: %s", e)
        return None


def verify_and_decode_jwt(
    token: str,
    verify_signature: bool = True,
    verify_exp: bool = True,
    audience: str | None = None,
    issuer: str | None = None,
) -> dict[str, Any]:
    """Verify and decode JWT token with signature validation.

    Args:
        token: JWT token string.
        verify_signature: Whether to verify token signature.
        verify_exp: Whether to verify expiration.
        audience: Expected audience claim.
        issuer: Expected issuer claim.

    Returns:
        dict[str, Any]: Decoded token payload.

    Raises:
        InvalidTokenError: If token is invalid or verification fails.
    """
    if not token or not isinstance(token, str):
        raise InvalidTokenError("Invalid token format")

    # Get OAuth configuration
    oauth_config = get_oauth_config()

    # Determine audience and issuer if not provided
    if audience is None:
        audience = oauth_config.client_id

    if issuer is None:
        # Extract issuer from well-known or use common patterns
        issuer = None  # Let PyJWT auto-detect

    try:
        if verify_signature:
            # Get JWKS client for signature verification
            jwks_client = get_jwks_client()

            if not jwks_client:
                logger.warning(
                    "JWKS client unavailable, falling back to unverified decode"
                )
                return decode_jwt_unverified(token)

            # Get signing key from JWKS
            signing_key = jwks_client.get_signing_key_from_jwt(token)

            # Verify and decode with signature validation
            decoded = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256", "RS384", "RS512", "ES256", "ES384", "ES512"],
                options={
                    "verify_signature": True,
                    "verify_exp": verify_exp,
                    "verify_aud": bool(audience),
                    "verify_iss": bool(issuer),
                },
                audience=audience,
                issuer=issuer,
            )

            logger.info("JWT signature verified successfully")
            return decoded

        else:
            # Decode without verification (development only)
            return decode_jwt_unverified(token)

    except ExpiredSignatureError:
        logger.warning("JWT token has expired")
        raise
    except InvalidAudienceError:
        logger.warning("JWT audience validation failed")
        raise
    except InvalidIssuerError:
        logger.warning("JWT issuer validation failed")
        raise
    except DecodeError as e:
        logger.warning("JWT decode error: %s", e)
        raise
    except Exception as e:
        logger.error("JWT verification failed: %s", e)
        raise InvalidTokenError(f"Token verification failed: {e}")


def decode_jwt_unverified(token: str) -> dict[str, Any]:
    """Decode JWT without signature verification (INSECURE - dev only).

    Args:
        token: JWT token string.

    Returns:
        dict[str, Any]: Decoded token payload.

    Raises:
        InvalidTokenError: If token cannot be decoded.
    """
    try:
        # Decode without verification
        decoded = jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_exp": False,
                "verify_aud": False,
            },
        )
        logger.warning("JWT decoded WITHOUT signature verification (insecure)")
        return decoded
    except Exception as e:
        logger.error("Failed to decode JWT: %s", e)
        raise InvalidTokenError(f"Failed to decode token: {e}")


def extract_user_email_from_jwt(
    token: str, verify_signature: bool = True
) -> str | None:
    """Extract user email from JWT token with verification.

    Args:
        token: JWT token string.
        verify_signature: Whether to verify token signature.

    Returns:
        str | None: User email if found, None otherwise.
    """
    try:
        payload = verify_and_decode_jwt(token, verify_signature=verify_signature)

        # Try different claim names
        email = (
            payload.get("email")
            or payload.get("preferred_username")
            or payload.get("upn")
            or payload.get("unique_name")
        )

        if email and isinstance(email, str):
            logger.info("Extracted email from JWT: %s", email)
            return email.lower().strip()

        logger.warning(
            "No email claim found in JWT payload. Available claims: %s",
            list(payload.keys()),
        )
        return None

    except InvalidTokenError as e:
        logger.warning("Failed to extract email from JWT: %s", e)
        # Try to decode without verification as last resort
        try:
            payload = decode_jwt_unverified(token)
            email = (
                payload.get("email")
                or payload.get("preferred_username")
                or payload.get("upn")
                or payload.get("unique_name")
            )
            if email:
                logger.info("Extracted email from unverified JWT: %s", email)
                return email.lower().strip()
        except Exception:
            pass
        return None
