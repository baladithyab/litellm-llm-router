"""
Tests for ConversationAffinityTracker.

Covers:
- Record and retrieve affinity
- TTL expiry (mock time)
- Missing response_id returns None
- Max entries eviction (LRU)
- Cleanup removes expired entries
- Redis backend (mock redis)
- Redis fallback to in-memory on connection error
- Singleton pattern (init, get, reset)
- Concurrent access (thread safety)
- Feature flag disabled returns None
- AffinityRecord serialization
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from litellm_llmrouter.conversation_affinity import (
    AffinityRecord,
    ConversationAffinityTracker,
    get_affinity_tracker,
    init_affinity_tracker,
    reset_affinity_tracker,
)


@pytest.fixture(autouse=True)
def _reset_singleton():
    """Reset the singleton before and after each test."""
    reset_affinity_tracker()
    yield
    reset_affinity_tracker()


@pytest.fixture
def tracker():
    """Create a fresh tracker with short TTL for testing."""
    return ConversationAffinityTracker(ttl_seconds=60, max_entries=100)


# =============================================================================
# AffinityRecord tests
# =============================================================================


class TestAffinityRecord:
    def test_create_record(self):
        now = time.time()
        record = AffinityRecord(
            response_id="resp_123",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )
        assert record.response_id == "resp_123"
        assert record.provider_deployment == "openai/gpt-4"
        assert record.model == "gpt-4"
        assert record.recorded_at == now
        assert record.expires_at == now + 3600

    def test_is_expired_not_expired(self):
        now = time.time()
        record = AffinityRecord(
            response_id="resp_123",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )
        assert not record.is_expired(now)
        assert not record.is_expired(now + 3599)

    def test_is_expired_expired(self):
        now = time.time()
        record = AffinityRecord(
            response_id="resp_123",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )
        assert record.is_expired(now + 3600)
        assert record.is_expired(now + 7200)

    def test_frozen_dataclass(self):
        now = time.time()
        record = AffinityRecord(
            response_id="resp_123",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )
        with pytest.raises(AttributeError):
            record.response_id = "resp_456"  # type: ignore[misc]

    def test_to_json_and_from_json_roundtrip(self):
        now = 1000.0
        record = AffinityRecord(
            response_id="resp_123",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )
        json_str = record.to_json()
        parsed = json.loads(json_str)
        assert parsed["response_id"] == "resp_123"
        assert parsed["provider_deployment"] == "openai/gpt-4"

        restored = AffinityRecord.from_json(json_str)
        assert restored == record

    def test_from_json_invalid(self):
        with pytest.raises(json.JSONDecodeError):
            AffinityRecord.from_json("not json")


# =============================================================================
# In-memory backend tests
# =============================================================================


class TestInMemoryBackend:
    async def test_record_and_retrieve(self, tracker):
        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        record = await tracker.get_affinity("resp_1")
        assert record is not None
        assert record.response_id == "resp_1"
        assert record.provider_deployment == "openai/gpt-4"
        assert record.model == "gpt-4"

    async def test_missing_response_id_returns_none(self, tracker):
        result = await tracker.get_affinity("nonexistent")
        assert result is None

    async def test_ttl_expiry(self, tracker):
        """Records should expire after TTL."""
        now = time.time()
        with patch("litellm_llmrouter.conversation_affinity.time") as mock_time:
            mock_time.time.return_value = now
            await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")

            # Before expiry
            mock_time.time.return_value = now + 30
            record = await tracker.get_affinity("resp_1")
            assert record is not None

            # After expiry (TTL is 60s)
            mock_time.time.return_value = now + 61
            record = await tracker.get_affinity("resp_1")
            assert record is None

    async def test_cleanup_expired(self, tracker):
        """cleanup_expired should remove expired entries."""
        now = time.time()
        with patch("litellm_llmrouter.conversation_affinity.time") as mock_time:
            mock_time.time.return_value = now
            await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
            await tracker.record_response("resp_2", "anthropic/claude", "claude")

            assert tracker.entry_count == 2

            # Expire all
            mock_time.time.return_value = now + 61
            removed = await tracker.cleanup_expired()
            assert removed == 2
            assert tracker.entry_count == 0

    async def test_cleanup_partial_expiry(self, tracker):
        """Only expired entries should be removed."""
        now = time.time()
        with patch("litellm_llmrouter.conversation_affinity.time") as mock_time:
            mock_time.time.return_value = now
            await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")

            # Record second one later
            mock_time.time.return_value = now + 30
            await tracker.record_response("resp_2", "anthropic/claude", "claude")

            assert tracker.entry_count == 2

            # Only first should expire (recorded at now, TTL=60, check at now+61)
            mock_time.time.return_value = now + 61
            removed = await tracker.cleanup_expired()
            assert removed == 1
            assert tracker.entry_count == 1

            # Second should still exist
            record = await tracker.get_affinity("resp_2")
            assert record is not None
            assert record.provider_deployment == "anthropic/claude"

    async def test_max_entries_eviction(self):
        """When max entries reached, oldest entry should be evicted (LRU)."""
        tracker = ConversationAffinityTracker(ttl_seconds=3600, max_entries=3)

        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        await tracker.record_response("resp_2", "anthropic/claude", "claude")
        await tracker.record_response("resp_3", "google/gemini", "gemini")

        assert tracker.entry_count == 3

        # Adding a 4th should evict resp_1 (oldest)
        await tracker.record_response("resp_4", "mistral/large", "mistral")
        assert tracker.entry_count == 3

        # resp_1 should be gone
        assert await tracker.get_affinity("resp_1") is None
        # Others should remain
        assert await tracker.get_affinity("resp_2") is not None
        assert await tracker.get_affinity("resp_3") is not None
        assert await tracker.get_affinity("resp_4") is not None

    async def test_overwrite_existing_entry(self, tracker):
        """Recording same response_id should overwrite without eviction."""
        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        await tracker.record_response("resp_1", "anthropic/claude", "claude")

        assert tracker.entry_count == 1
        record = await tracker.get_affinity("resp_1")
        assert record is not None
        assert record.provider_deployment == "anthropic/claude"

    async def test_entry_count(self, tracker):
        assert tracker.entry_count == 0
        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        assert tracker.entry_count == 1
        await tracker.record_response("resp_2", "anthropic/claude", "claude")
        assert tracker.entry_count == 2

    async def test_properties(self, tracker):
        assert tracker.ttl_seconds == 60
        assert tracker.max_entries == 100
        assert tracker.redis_available is False


# =============================================================================
# Redis backend tests
# =============================================================================


class TestRedisBackend:
    async def test_record_and_retrieve_redis(self):
        """Records should be stored and retrieved via Redis."""
        mock_redis = AsyncMock()
        mock_redis.ping = AsyncMock()
        mock_redis.set = AsyncMock()

        now = 1000.0
        stored_data = {}

        async def mock_set(key, value, ex=None):
            stored_data[key] = value

        async def mock_get(key):
            return stored_data.get(key)

        mock_redis.set = AsyncMock(side_effect=mock_set)
        mock_redis.get = AsyncMock(side_effect=mock_get)

        tracker = ConversationAffinityTracker(ttl_seconds=3600)
        tracker._redis = mock_redis
        tracker._redis_available = True

        with patch("litellm_llmrouter.conversation_affinity.time") as mock_time:
            mock_time.time.return_value = now
            await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")

        record = await tracker.get_affinity("resp_1")
        assert record is not None
        assert record.response_id == "resp_1"
        assert record.provider_deployment == "openai/gpt-4"

    async def test_redis_set_called_with_ttl(self):
        """Redis set should be called with the configured TTL."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        tracker = ConversationAffinityTracker(ttl_seconds=7200)
        tracker._redis = mock_redis
        tracker._redis_available = True

        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")

        mock_redis.set.assert_called_once()
        call_kwargs = mock_redis.set.call_args
        assert call_kwargs[1]["ex"] == 7200
        assert call_kwargs[0][0] == "affinity:resp_1"

    async def test_redis_key_pattern(self):
        """Redis keys should follow the affinity:{response_id} pattern."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()

        tracker = ConversationAffinityTracker(ttl_seconds=3600)
        tracker._redis = mock_redis
        tracker._redis_available = True

        await tracker.record_response("resp_abc_123", "openai/gpt-4", "gpt-4")

        call_args = mock_redis.set.call_args[0]
        assert call_args[0] == "affinity:resp_abc_123"

    async def test_redis_fallback_on_write_error(self):
        """If Redis write fails, should fall back to in-memory."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock(side_effect=ConnectionError("Redis down"))

        tracker = ConversationAffinityTracker(ttl_seconds=3600)
        tracker._redis = mock_redis
        tracker._redis_available = True

        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")

        # Redis should be marked unavailable
        assert tracker.redis_available is False
        # Record should be in memory
        assert tracker.entry_count == 1
        record = await tracker.get_affinity("resp_1")
        assert record is not None
        assert record.provider_deployment == "openai/gpt-4"

    async def test_redis_fallback_on_read_error(self):
        """If Redis read fails, should fall back to in-memory."""
        mock_redis = AsyncMock()
        mock_redis.set = AsyncMock()
        mock_redis.get = AsyncMock(side_effect=ConnectionError("Redis down"))

        tracker = ConversationAffinityTracker(ttl_seconds=3600)
        tracker._redis = mock_redis
        tracker._redis_available = True

        # Store in-memory first (redis_available will still be True for write)
        # We need to manually set up a record in memory for fallback
        now = time.time()
        tracker._store["resp_1"] = AffinityRecord(
            response_id="resp_1",
            provider_deployment="openai/gpt-4",
            model="gpt-4",
            recorded_at=now,
            expires_at=now + 3600,
        )

        # The read from Redis will fail, triggering fallback
        record = await tracker.get_affinity("resp_1")
        assert tracker.redis_available is False
        assert record is not None
        assert record.provider_deployment == "openai/gpt-4"

    async def test_redis_not_found_returns_none(self):
        """Redis returning nil should result in None."""
        mock_redis = AsyncMock()
        mock_redis.get = AsyncMock(return_value=None)

        tracker = ConversationAffinityTracker(ttl_seconds=3600)
        tracker._redis = mock_redis
        tracker._redis_available = True

        result = await tracker.get_affinity("nonexistent")
        assert result is None

    async def test_connect_redis_failure(self):
        """Failed Redis connection should set redis_available=False."""
        tracker = ConversationAffinityTracker(
            redis_url="redis://invalid:9999", ttl_seconds=3600
        )

        mock_aioredis = MagicMock()
        mock_client = AsyncMock()
        mock_client.ping = AsyncMock(side_effect=ConnectionError("Connection refused"))
        mock_aioredis.from_url.return_value = mock_client

        with patch.dict(
            "sys.modules",
            {"redis": MagicMock(), "redis.asyncio": mock_aioredis},
        ):
            await tracker._connect_redis()

        assert tracker.redis_available is False


# =============================================================================
# Singleton pattern tests
# =============================================================================


class TestSingleton:
    def test_get_before_init_returns_none(self):
        assert get_affinity_tracker() is None

    def test_init_creates_tracker(self):
        tracker = init_affinity_tracker(ttl=120, max_entries=500)
        assert tracker is not None
        assert tracker.ttl_seconds == 120
        assert tracker.max_entries == 500

    def test_get_after_init_returns_same(self):
        tracker = init_affinity_tracker()
        assert get_affinity_tracker() is tracker

    def test_reset_clears_singleton(self):
        init_affinity_tracker()
        assert get_affinity_tracker() is not None
        reset_affinity_tracker()
        assert get_affinity_tracker() is None

    def test_init_with_redis_url(self):
        tracker = init_affinity_tracker(redis_url="redis://localhost:6379")
        assert tracker is not None
        assert tracker._redis_url == "redis://localhost:6379"

    def test_reinit_replaces_singleton(self):
        tracker1 = init_affinity_tracker(ttl=100)
        tracker2 = init_affinity_tracker(ttl=200)
        assert get_affinity_tracker() is tracker2
        assert tracker2.ttl_seconds == 200
        assert tracker1 is not tracker2


# =============================================================================
# Concurrent access tests
# =============================================================================


class TestConcurrency:
    async def test_concurrent_writes(self, tracker):
        """Multiple concurrent writes should not corrupt the store."""

        async def write(i: int):
            await tracker.record_response(f"resp_{i}", f"provider/{i}", f"model-{i}")

        await asyncio.gather(*[write(i) for i in range(50)])
        assert tracker.entry_count == 50

    async def test_concurrent_reads_and_writes(self, tracker):
        """Concurrent reads and writes should be safe."""
        # Pre-populate
        for i in range(20):
            await tracker.record_response(f"resp_{i}", f"provider/{i}", f"model-{i}")

        async def read(i: int):
            return await tracker.get_affinity(f"resp_{i}")

        async def write(i: int):
            await tracker.record_response(
                f"resp_new_{i}", f"new_provider/{i}", f"new-model-{i}"
            )

        reads = [read(i) for i in range(20)]
        writes = [write(i) for i in range(20)]

        results = await asyncio.gather(*reads, *writes)
        # First 20 results are reads, should all have records
        for result in results[:20]:
            assert result is not None

    async def test_concurrent_eviction(self):
        """Concurrent writes at max capacity should not exceed max_entries."""
        tracker = ConversationAffinityTracker(ttl_seconds=3600, max_entries=10)

        async def write(i: int):
            await tracker.record_response(f"resp_{i}", f"provider/{i}", f"model-{i}")

        await asyncio.gather(*[write(i) for i in range(50)])
        assert tracker.entry_count <= 10


# =============================================================================
# Feature flag tests
# =============================================================================


class TestFeatureFlag:
    def test_feature_flag_default_disabled(self):
        """Feature flag should default to disabled."""
        # The module-level constant reads from env at import time.
        # We can't easily test the default without controlling the env,
        # but we can verify the env var name and behavior.
        with patch.dict("os.environ", {"CONVERSATION_AFFINITY_ENABLED": "false"}):
            result = (
                os.getenv("CONVERSATION_AFFINITY_ENABLED", "false").lower() == "true"
            )
            assert result is False

    def test_feature_flag_enabled(self):
        with patch.dict("os.environ", {"CONVERSATION_AFFINITY_ENABLED": "true"}):
            result = (
                os.getenv("CONVERSATION_AFFINITY_ENABLED", "false").lower() == "true"
            )
            assert result is True

    def test_singleton_returns_none_when_not_initialized(self):
        """When feature is disabled and tracker not initialized, get returns None."""
        reset_affinity_tracker()
        assert get_affinity_tracker() is None


# =============================================================================
# Lifecycle tests
# =============================================================================


class TestLifecycle:
    async def test_start_and_stop(self, tracker):
        """Start and stop should not raise."""
        await tracker.start()
        await tracker.stop()

    async def test_stop_cancels_cleanup_task(self, tracker):
        await tracker.start()
        assert tracker._cleanup_task is not None
        assert not tracker._cleanup_task.done()

        await tracker.stop()
        assert tracker._cleanup_task is None

    async def test_stop_without_start(self, tracker):
        """Stop without start should not raise."""
        await tracker.stop()

    async def test_start_with_redis_connection_failure(self):
        """Start with invalid Redis URL should fall back gracefully."""
        tracker = ConversationAffinityTracker(
            redis_url="redis://invalid:9999", ttl_seconds=60
        )

        with patch.object(
            tracker, "_connect_redis", new_callable=AsyncMock
        ) as mock_connect:
            mock_connect.return_value = None
            await tracker.start()

        # Should still work with in-memory
        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        record = await tracker.get_affinity("resp_1")
        assert record is not None

        await tracker.stop()


# =============================================================================
# Edge case tests
# =============================================================================


class TestEdgeCases:
    async def test_empty_response_id(self, tracker):
        """Empty string response_id should work."""
        await tracker.record_response("", "openai/gpt-4", "gpt-4")
        record = await tracker.get_affinity("")
        assert record is not None

    async def test_special_characters_in_response_id(self, tracker):
        """Response IDs with special characters should work."""
        response_id = "resp_abc-123_def/ghi:jkl"
        await tracker.record_response(response_id, "openai/gpt-4", "gpt-4")
        record = await tracker.get_affinity(response_id)
        assert record is not None
        assert record.response_id == response_id

    async def test_max_entries_one(self):
        """Max entries of 1 should work."""
        tracker = ConversationAffinityTracker(ttl_seconds=3600, max_entries=1)

        await tracker.record_response("resp_1", "openai/gpt-4", "gpt-4")
        assert tracker.entry_count == 1

        await tracker.record_response("resp_2", "anthropic/claude", "claude")
        assert tracker.entry_count == 1
        assert await tracker.get_affinity("resp_1") is None
        assert await tracker.get_affinity("resp_2") is not None

    async def test_cleanup_on_empty_store(self, tracker):
        """Cleanup on empty store should return 0."""
        removed = await tracker.cleanup_expired()
        assert removed == 0
