"""Tests for JWT decoding and email extraction.

Uses real self-signed tokens rather than mocks: `jwt.encode` is cheap, and a
mocked decoder would only assert that the mock was called, not that the token
handling is right. The network-dependent paths (JWKS fetch) are mocked at
``requests.get``.

Auth is worth testing precisely — an email extracted from the wrong claim
silently attributes a session to the wrong user.
"""

from __future__ import annotations

import datetime

import jwt
import pytest
from jwt.exceptions import InvalidTokenError

from app.auth.jwt_validator import (
    decode_jwt_unverified,
    extract_user_email_from_jwt,
    get_jwks_client,
    get_jwks_uri,
)

SECRET = "test-signing-secret-not-real"


def make_token(claims: dict, secret: str = SECRET) -> str:
    """Sign a token with HS256. Signature content is irrelevant to these paths."""
    return jwt.encode(claims, secret, algorithm="HS256")


class TestDecodeJwtUnverified:
    def test_returns_the_payload(self) -> None:
        token = make_token({"email": "user@example.com", "sub": "abc-123"})

        payload = decode_jwt_unverified(token)

        assert payload == {"email": "user@example.com", "sub": "abc-123"}

    def test_decodes_despite_a_wrong_signature(self) -> None:
        """The point of this function: no signature check.

        Pinned so its insecurity is a documented, deliberate property rather
        than an accident — it must only be reachable on a dev path.
        """
        token = make_token({"email": "a@b.com"}, secret="a-completely-different-key")

        assert decode_jwt_unverified(token)["email"] == "a@b.com"

    def test_decodes_an_expired_token(self) -> None:
        """verify_exp is off, so expiry does not block decoding."""
        past = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(
            days=30
        )
        token = make_token({"email": "a@b.com", "exp": past})

        assert decode_jwt_unverified(token)["email"] == "a@b.com"

    def test_empty_payload_decodes_to_an_empty_dict(self) -> None:
        assert decode_jwt_unverified(make_token({})) == {}

    @pytest.mark.parametrize(
        "bad_token", ["", "not-a-jwt", "a.b", "a.b.c", "....", "eyJhbGciOiJIUzI1NiJ9"]
    )
    def test_malformed_tokens_raise_invalid_token_error(self, bad_token: str) -> None:
        """A specific exception type, so callers can distinguish it from a bug."""
        with pytest.raises(InvalidTokenError):
            decode_jwt_unverified(bad_token)

    def test_error_message_names_the_failure(self) -> None:
        with pytest.raises(InvalidTokenError, match="Failed to decode token"):
            decode_jwt_unverified("garbage")


class TestExtractUserEmailFromJwt:
    def test_prefers_the_email_claim(self) -> None:
        token = make_token(
            {
                "email": "primary@example.com",
                "preferred_username": "secondary@example.com",
            }
        )

        assert (
            extract_user_email_from_jwt(token, verify_signature=False)
            == "primary@example.com"
        )

    def test_falls_back_through_the_claim_chain_in_order(self) -> None:
        """Providers differ: Microsoft sends upn, others preferred_username.

        Each assertion removes the preceding claim, so the fallback ORDER is
        pinned, not merely the set of claims consulted.
        """
        chain = ["email", "preferred_username", "upn", "unique_name"]

        for index, claim in enumerate(chain):
            # Only this claim and later ones are present.
            claims = {name: f"{name}@example.com" for name in chain[index:]}
            token = make_token(claims)

            assert (
                extract_user_email_from_jwt(token, verify_signature=False)
                == f"{claim}@example.com"
            )

    def test_returns_none_when_no_email_claim_is_present(self) -> None:
        """None, not a guess: attributing a session to the wrong user is worse."""
        token = make_token({"sub": "abc-123", "name": "No Email User"})

        assert extract_user_email_from_jwt(token, verify_signature=False) is None

    def test_returns_none_for_a_malformed_token(self) -> None:
        """Swallows the decode error rather than propagating it."""
        assert extract_user_email_from_jwt("garbage", verify_signature=False) is None

    def test_returns_none_for_an_empty_token(self) -> None:
        assert extract_user_email_from_jwt("", verify_signature=False) is None

    def test_empty_claim_value_falls_through_to_the_next(self) -> None:
        """An empty string is not a usable email."""
        token = make_token({"email": "", "preferred_username": "real@example.com"})

        assert (
            extract_user_email_from_jwt(token, verify_signature=False)
            == "real@example.com"
        )


class TestGetJwksUri:
    def test_returns_the_uri_from_the_well_known_document(self, monkeypatch) -> None:
        class Response:
            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"jwks_uri": "https://idp.example.com/keys"}

        monkeypatch.setattr(
            "app.auth.jwt_validator.requests.get", lambda *a, **k: Response()
        )
        monkeypatch.setattr(
            "app.auth.jwt_validator.get_oauth_config",
            lambda: type(
                "Config",
                (),
                {
                    "well_known_url": "https://idp.example.com/.well-known",
                    "http_connect_timeout_seconds": 3,
                    "http_read_timeout_seconds": 5,
                },
            )(),
        )

        assert get_jwks_uri() == "https://idp.example.com/keys"

    def test_returns_empty_string_when_the_fetch_fails(self, monkeypatch) -> None:
        """Degrades rather than raising: the caller treats "" as unavailable."""

        def boom(*_args, **_kwargs):
            raise RuntimeError("network unreachable")

        monkeypatch.setattr("app.auth.jwt_validator.requests.get", boom)
        monkeypatch.setattr(
            "app.auth.jwt_validator.get_oauth_config",
            lambda: type(
                "Config",
                (),
                {
                    "well_known_url": "https://idp.example.com/.well-known",
                    "http_connect_timeout_seconds": 3,
                    "http_read_timeout_seconds": 5,
                },
            )(),
        )

        assert get_jwks_uri() == ""

    def test_returns_empty_string_when_no_well_known_url_is_configured(
        self, monkeypatch
    ) -> None:
        monkeypatch.setattr(
            "app.auth.jwt_validator.get_oauth_config",
            lambda: type("Config", (), {"well_known_url": ""})(),
        )

        assert get_jwks_uri() == ""

    def test_returns_empty_string_when_the_document_omits_jwks_uri(
        self, monkeypatch
    ) -> None:
        class Response:
            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict:
                return {"issuer": "https://idp.example.com"}

        monkeypatch.setattr(
            "app.auth.jwt_validator.requests.get", lambda *a, **k: Response()
        )
        monkeypatch.setattr(
            "app.auth.jwt_validator.get_oauth_config",
            lambda: type(
                "Config",
                (),
                {
                    "well_known_url": "https://idp.example.com/.well-known",
                    "http_connect_timeout_seconds": 3,
                    "http_read_timeout_seconds": 5,
                },
            )(),
        )

        assert get_jwks_uri() == ""


class TestGetJwksClient:
    def test_returns_none_when_no_jwks_uri_is_available(self, monkeypatch) -> None:
        """Without a key source, signature verification cannot be attempted."""
        monkeypatch.setattr("app.auth.jwt_validator.get_jwks_uri", lambda: "")

        assert get_jwks_client() is None
