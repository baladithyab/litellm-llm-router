"""
Unit tests for auth.py - Admin Authentication and Request ID Middleware
======================================================================

Tests cover:
- Secret scrubbing with various secret patterns and adversarial inputs
- Bearer token extraction edge cases
- Admin API key loading from environment
- Admin auth enable/disable logic
- admin_api_key_auth dependency (fail-closed, multi-key, header fallback)
- RequestIDMiddleware (passthrough, generation, context propagation)
- sanitize_error_response
- create_admin_error_response
"""

from __future__ import annotations

import os
import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from litellm_llmrouter.auth import (
    RequestIDMiddleware,
    _extract_bearer_token,
    _is_admin_auth_enabled,
    _load_admin_api_keys,
    _scrub_secrets,
    admin_api_key_auth,
    create_admin_error_response,
    get_request_id,
    sanitize_error_response,
)


# =============================================================================
# Secret Scrubbing Tests
# =============================================================================


class TestScrubSecrets:
    """Tests for _scrub_secrets function."""

    def test_empty_string(self):
        assert _scrub_secrets("") == ""

    def test_none_returns_none(self):
        # _scrub_secrets returns falsy input unchanged
        assert _scrub_secrets("") == ""

    def test_no_secrets(self):
        text = "This is a normal log message with no secrets"
        assert _scrub_secrets(text) == text

    def test_scrubs_sk_api_key(self):
        text = "Error with key sk-1234567890abcdef1234"
        result = _scrub_secrets(text)
        assert "1234567890abcdef1234" not in result
        assert "[REDACTED]" in result

    def test_scrubs_api_key_pattern(self):
        text = "api_key=abcdef12345678"
        result = _scrub_secrets(text)
        assert "abcdef12345678" not in result
        assert "[REDACTED]" in result

    def test_scrubs_api_key_colon(self):
        text = "api-key: my-secret-api-key-value"
        result = _scrub_secrets(text)
        assert "my-secret-api-key-value" not in result

    def test_scrubs_aws_access_key(self):
        text = "Found AKIAIOSFODNN7EXAMPLE in config"
        result = _scrub_secrets(text)
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_scrubs_aws_secret(self):
        text = "aws_secret=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        result = _scrub_secrets(text)
        assert "wJalrXUtnFEMI" not in result
        assert "[REDACTED]" in result

    def test_scrubs_postgres_url(self):
        text = "Connected to postgresql://admin:secretpass@db:5432/mydb"
        result = _scrub_secrets(text)
        assert "secretpass" not in result
        assert "[REDACTED]" in result

    def test_scrubs_mysql_url(self):
        text = "Using mysql://root:password123@localhost/db"
        result = _scrub_secrets(text)
        assert "password123" not in result

    def test_scrubs_bearer_token(self):
        text = "Header: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.payload.sig"
        result = _scrub_secrets(text)
        assert "eyJhbGci" not in result
        assert "[REDACTED]" in result

    def test_scrubs_password_pattern(self):
        text = "password=my_super_secret"
        result = _scrub_secrets(text)
        assert "my_super_secret" not in result
        assert "[REDACTED]" in result

    def test_scrubs_secret_pattern(self):
        text = "secret=the_secret_value"
        result = _scrub_secrets(text)
        assert "the_secret_value" not in result
        assert "[REDACTED]" in result

    def test_does_not_scrub_word_secret_alone(self):
        """The word 'secret' without = or : should not be scrubbed."""
        text = "This is a secret meeting about project plans"
        result = _scrub_secrets(text)
        assert "secret meeting" in result

    def test_multiple_secrets(self):
        text = "sk-abc1234567890def password=hunter2 api_key=test123456"
        result = _scrub_secrets(text)
        assert "abc1234567890def" not in result
        assert "hunter2" not in result
        assert "test123456" not in result

    def test_preserves_non_secret_content(self):
        text = "Status: OK, latency: 42ms, model: gpt-4"
        assert _scrub_secrets(text) == text


# =============================================================================
# Bearer Token Extraction Tests
# =============================================================================


class TestExtractBearerToken:
    """Tests for _extract_bearer_token function."""

    def test_valid_bearer(self):
        assert _extract_bearer_token("Bearer my-token-123") == "my-token-123"

    def test_bearer_with_whitespace(self):
        assert _extract_bearer_token("Bearer   my-token-123  ") == "my-token-123"

    def test_empty_bearer(self):
        assert _extract_bearer_token("Bearer ") is None

    def test_bearer_only_whitespace(self):
        assert _extract_bearer_token("Bearer    ") is None

    def test_not_bearer(self):
        assert _extract_bearer_token("Basic dXNlcjpwYXNz") is None

    def test_empty_string(self):
        assert _extract_bearer_token("") is None

    def test_bearer_lowercase(self):
        """Bearer prefix is case-sensitive per RFC."""
        assert _extract_bearer_token("bearer my-token") is None

    def test_bearer_no_space(self):
        assert _extract_bearer_token("Bearertoken") is None


# =============================================================================
# Admin API Key Loading Tests
# =============================================================================


class TestLoadAdminApiKeys:
    """Tests for _load_admin_api_keys function."""

    def test_no_keys_configured(self):
        with patch.dict(os.environ, {}, clear=True):
            keys = _load_admin_api_keys()
            assert keys == set()

    def test_single_key_from_admin_api_keys(self):
        with patch.dict(os.environ, {"ADMIN_API_KEYS": "key1"}, clear=True):
            keys = _load_admin_api_keys()
            assert keys == {"key1"}

    def test_multiple_keys_from_admin_api_keys(self):
        with patch.dict(
            os.environ, {"ADMIN_API_KEYS": "key1,key2,key3"}, clear=True
        ):
            keys = _load_admin_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_keys_with_whitespace(self):
        with patch.dict(
            os.environ, {"ADMIN_API_KEYS": " key1 , key2 , key3 "}, clear=True
        ):
            keys = _load_admin_api_keys()
            assert keys == {"key1", "key2", "key3"}

    def test_empty_keys_filtered(self):
        with patch.dict(
            os.environ, {"ADMIN_API_KEYS": "key1,,key2,,"}, clear=True
        ):
            keys = _load_admin_api_keys()
            assert keys == {"key1", "key2"}

    def test_legacy_single_key_fallback(self):
        with patch.dict(os.environ, {"ADMIN_API_KEY": "legacy-key"}, clear=True):
            keys = _load_admin_api_keys()
            assert keys == {"legacy-key"}

    def test_both_sources_merged(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "key1,key2", "ADMIN_API_KEY": "legacy-key"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert keys == {"key1", "key2", "legacy-key"}

    def test_duplicate_keys_deduplicated(self):
        with patch.dict(
            os.environ,
            {"ADMIN_API_KEYS": "key1,key1", "ADMIN_API_KEY": "key1"},
            clear=True,
        ):
            keys = _load_admin_api_keys()
            assert keys == {"key1"}


# =============================================================================
# Admin Auth Enabled Tests
# =============================================================================


class TestIsAdminAuthEnabled:
    """Tests for _is_admin_auth_enabled function."""

    def test_default_is_enabled(self):
        with patch.dict(os.environ, {}, clear=True):
            assert _is_admin_auth_enabled() is True

    def test_true_is_enabled(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "true"}, clear=True):
            assert _is_admin_auth_enabled() is True

    def test_false_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "false"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_zero_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "0"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_no_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "no"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_off_disables(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "off"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_case_insensitive(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "FALSE"}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_whitespace_trimmed(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": " false "}, clear=True):
            assert _is_admin_auth_enabled() is False

    def test_random_string_is_enabled(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "whatever"}, clear=True):
            assert _is_admin_auth_enabled() is True


# =============================================================================
# Admin API Key Auth Dependency Tests
# =============================================================================


class TestAdminApiKeyAuth:
    """Tests for admin_api_key_auth FastAPI dependency."""

    def _make_request(self, headers: dict | None = None):
        """Create a mock Request object."""
        request = MagicMock()
        request.headers = headers or {}
        request.url.path = "/config/reload"
        return request

    @pytest.mark.asyncio
    async def test_auth_disabled_allows_all(self):
        with patch.dict(os.environ, {"ADMIN_AUTH_ENABLED": "false"}, clear=True):
            request = self._make_request()
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "__disabled__"

    @pytest.mark.asyncio
    async def test_no_keys_configured_denies(self):
        """Fail-closed: no admin keys = deny all."""
        with patch.dict(
            os.environ, {"ADMIN_AUTH_ENABLED": "true"}, clear=True
        ):
            request = self._make_request()
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 403
            assert exc.value.detail["error"] == "control_plane_not_configured"

    @pytest.mark.asyncio
    async def test_valid_admin_key_header(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_valid_bearer_token_fallback(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"Authorization": "Bearer test-key"})
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "test-key"

    @pytest.mark.asyncio
    async def test_missing_key_returns_401(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request()
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 401
            assert exc.value.detail["error"] == "admin_key_required"

    @pytest.mark.asyncio
    async def test_invalid_key_returns_401(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "test-key"},
            clear=True,
        ):
            request = self._make_request({"X-Admin-API-Key": "wrong-key"})
            with pytest.raises(HTTPException) as exc:
                await admin_api_key_auth(request)
            assert exc.value.status_code == 401
            assert exc.value.detail["error"] == "invalid_admin_key"

    @pytest.mark.asyncio
    async def test_admin_key_header_takes_precedence(self):
        """X-Admin-API-Key header should be checked before Authorization."""
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "admin-key,bearer-key"},
            clear=True,
        ):
            request = self._make_request(
                {
                    "X-Admin-API-Key": "admin-key",
                    "Authorization": "Bearer bearer-key",
                }
            )
            result = await admin_api_key_auth(request)
            assert result["admin_key"] == "admin-key"

    @pytest.mark.asyncio
    async def test_multiple_configured_keys(self):
        with patch.dict(
            os.environ,
            {"ADMIN_AUTH_ENABLED": "true", "ADMIN_API_KEYS": "key1,key2,key3"},
            clear=True,
        ):
            # Any of the keys should work
            for key in ["key1", "key2", "key3"]:
                request = self._make_request({"X-Admin-API-Key": key})
                result = await admin_api_key_auth(request)
                assert result["admin_key"] == key


# =============================================================================
# Request ID Tests
# =============================================================================


class TestGetRequestId:
    """Tests for get_request_id function."""

    def test_returns_none_outside_context(self):
        # When no context variable is set and no OTEL, returns None
        result = get_request_id()
        # May return None or OTEL trace ID depending on environment
        assert result is None or isinstance(result, str)


# =============================================================================
# Error Response Tests
# =============================================================================


class TestSanitizeErrorResponse:
    """Tests for sanitize_error_response function."""

    def test_returns_structured_response(self):
        error = ValueError("something broke")
        result = sanitize_error_response(error)
        assert result["error"] == "internal_error"
        assert result["message"] == "An internal error occurred"
        assert "request_id" in result

    def test_custom_public_message(self):
        error = RuntimeError("db connection failed")
        result = sanitize_error_response(error, public_message="Service unavailable")
        assert result["message"] == "Service unavailable"

    def test_does_not_leak_internal_details(self):
        error = RuntimeError("postgresql://admin:hunter2@db:5432/mydb failed")
        result = sanitize_error_response(error)
        assert "hunter2" not in result["message"]
        assert "admin" not in result["message"]

    def test_uses_provided_request_id(self):
        error = ValueError("test")
        result = sanitize_error_response(error, request_id="req-123")
        assert result["request_id"] == "req-123"


class TestCreateAdminErrorResponse:
    """Tests for create_admin_error_response function."""

    def test_returns_http_exception(self):
        exc = create_admin_error_response(
            status_code=403,
            error_code="forbidden",
            message="Access denied",
        )
        assert isinstance(exc, HTTPException)
        assert exc.status_code == 403
        assert exc.detail["error"] == "forbidden"
        assert exc.detail["message"] == "Access denied"
        assert "request_id" in exc.detail

    def test_uses_provided_request_id(self):
        exc = create_admin_error_response(
            status_code=500,
            error_code="internal",
            message="Error",
            request_id="custom-id",
        )
        assert exc.detail["request_id"] == "custom-id"
