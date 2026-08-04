"""Authentication dependencies for FastAPI routes."""

from __future__ import annotations

import logging
import os
from uuid import UUID

from fastapi import Depends, HTTPException, Request, status
from jwt.exceptions import InvalidTokenError

from app.settings_db.repositories import SettingsUserRepository

logger = logging.getLogger(__name__)

AUTH_COOKIE_NAME = "dmia_auth_session"


def should_verify_jwt_signature() -> bool:
    """Check if JWT signature verification should be enabled.

    Returns:
        bool: True if production, False if explicitly disabled for development.
    """
    # Allow disabling signature verification in development
    if os.getenv("JWT_VERIFY_SIGNATURE", "").lower() == "false":
        logger.warning(
            "JWT signature verification is DISABLED - only use in development!"
        )
        return False

    # Always verify in production
    if (
        os.getenv("NODE_ENV") == "production"
        or os.getenv("ENVIRONMENT") == "production"
    ):
        return True

    # Default: verify signatures
    return True


async def get_current_user_id(request: Request) -> UUID | None:
    """Extract current user ID from verified JWT access token.

    This implements production-grade security with:
    - JWT signature verification using JWKS
    - Token expiration validation
    - Audience and issuer claims validation
    - Database lookup/creation to get internal user_id
    - Auto-creates user if they don't exist

    Returns:
        UUID | None: User ID if authenticated and valid, None otherwise.
    """
    # Get access token from cookie
    access_token = request.cookies.get(AUTH_COOKIE_NAME)

    if not access_token:
        logger.debug("No auth token cookie found")
        return None

    try:
        # Extract and verify email from JWT
        verify_signature = should_verify_jwt_signature()

        # First, try to get the full JWT payload for user info
        from app.auth.jwt_validator import decode_jwt_unverified, verify_and_decode_jwt

        try:
            if verify_signature:
                jwt_payload = verify_and_decode_jwt(access_token, verify_signature=True)
            else:
                jwt_payload = decode_jwt_unverified(access_token)
        except Exception as e:
            logger.warning("Failed to decode JWT: %s", e)
            return None

        # Extract email from JWT - try multiple fields
        # Microsoft returns different emails in different fields:
        # - upn: Full corporate email (ajay.raghuram.deekonda@msd.com)
        # - preferred_username: Alias email (deekonaj@merck.com)
        # - email: Can be either

        # Log all JWT claims for debugging
        logger.info("JWT payload claims: %s", list(jwt_payload.keys()))
        logger.info(
            "JWT payload: %s",
            {
                k: v
                for k, v in jwt_payload.items()
                if k not in ["iat", "exp", "nbf", "aud", "iss", "sub", "jti", "nonce"]
            },
        )

        # Get all possible email fields
        upn = jwt_payload.get("upn")
        preferred_username = jwt_payload.get("preferred_username")
        email_claim = jwt_payload.get("email")
        unique_name = jwt_payload.get("unique_name")

        # Prefer UPN (most reliable for corporate accounts)
        primary_email = upn or email_claim or preferred_username or unique_name

        if not primary_email:
            logger.warning(
                "No email found in JWT payload. Available claims: %s",
                list(jwt_payload.keys()),
            )
            return None

        primary_email = primary_email.lower().strip()

        # Collect all email aliases for lookup
        email_aliases = [
            primary_email,
            preferred_username.lower().strip() if preferred_username else None,
            upn.lower().strip() if upn else None,
            email_claim.lower().strip() if email_claim else None,
        ]
        # Remove duplicates and None values
        email_aliases = list(set(filter(None, email_aliases)))

        logger.info(
            "Extracted emails from JWT - primary: %s, aliases: %s",
            primary_email,
            email_aliases,
        )

        # Extract additional user info from JWT
        full_name = jwt_payload.get("name") or jwt_payload.get("given_name")

        # Look up user in database by trying all email aliases
        repository = SettingsUserRepository()
        user = None

        for email_alias in email_aliases:
            user = repository.get_user_by_email(email_alias)
            if user:
                logger.info("Found user by email alias: %s", email_alias)
                break

        if not user:
            logger.info(
                "User not found in database with any alias, auto-creating for email: %s",
                primary_email,
            )
            # Auto-create user from JWT claims using primary email (UPN preferred)
            user = repository.upsert_user(
                email=primary_email,
                full_name=full_name,
                avatar_url=None,
            )
            logger.info(
                "User auto-created with id: %s for email: %s",
                user.get("id"),
                primary_email,
            )

        if not user:
            logger.error("Failed to get or create user for email: %s", primary_email)
            return None

        user_id = user.get("id")
        if user_id:
            logger.info("Using user_id for email %s: %s", primary_email, user_id)
            return UUID(str(user_id))

        return None

    except InvalidTokenError as e:
        logger.warning("Invalid JWT token: %s", e)
        return None
    except Exception as e:
        logger.error("Failed to extract user_id from token: %s", e)
        return None


async def require_current_user_id(
    user_id: UUID | None = Depends(get_current_user_id),
) -> UUID:
    """Require authenticated user ID or raise 401.

    Args:
        user_id: User ID from dependency injection.

    Returns:
        UUID: Authenticated user ID.

    Raises:
        HTTPException: If user is not authenticated or token is invalid.
    """
    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user_id
