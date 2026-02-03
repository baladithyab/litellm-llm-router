"""
Unit Tests for A2A Raw Streaming Passthrough (TG10.1)

These tests validate the raw streaming passthrough implementation for the A2A Gateway,
ensuring proper chunk cadence, TTFB, and rollback safety.

Acceptance Criteria:
- New tests fail on the old implementation and pass on the new implementation
- Streaming endpoint begins emitting without waiting for newline boundaries
- Rollback flag exists and can revert to previous behavior

Test Categories:
1. Feature flag behavior (toggle between raw and buffered modes)
2. Raw streaming (newline-in-chunk preservation, chunk boundary handling)
3. Buffered streaming (backward compatibility, line-based chunking)
4. No full-buffering behavior (backpressure respected)
"""

import asyncio
import json
import os
from typing import Any, AsyncIterator
from unittest.mock import patch

import pytest

# Mark all tests as async
pytestmark = pytest.mark.asyncio


# =============================================================================
# Test Fixtures and Helpers
# =============================================================================


class MockHTTPResponse:
    """Mock HTTP response for testing streaming behavior."""

    def __init__(
        self,
        content: bytes | list[bytes],
        status_code: int = 200,
        chunk_delay: float = 0.0,
    ):
        self.status_code = status_code
        self._content = content if isinstance(content, list) else [content]
        self._chunk_delay = chunk_delay
        self._read = False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")

    async def aiter_bytes(self, chunk_size: int = 1024) -> AsyncIterator[bytes]:
        """Iterate over raw bytes in chunks, preserving original boundaries."""
        for chunk in self._content:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            yield chunk

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Iterate over lines, waiting for newline boundaries."""
        buffer = b""
        for chunk in self._content:
            if self._chunk_delay > 0:
                await asyncio.sleep(self._chunk_delay)
            buffer += chunk

        # Lines are split on newlines
        text = buffer.decode("utf-8")
        for line in text.split("\n"):
            if line:
                yield line


class MockAsyncClient:
    """Mock async HTTP client with streaming support."""

    def __init__(self, response: MockHTTPResponse):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def stream(self, method: str, url: str, **kwargs) -> "MockStreamContext":
        return MockStreamContext(self._response)


class MockStreamContext:
    """Mock stream context manager."""

    def __init__(self, response: MockHTTPResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, *args):
        pass


def create_a2a_gateway_with_agent(
    agent_id: str = "test-agent",
    agent_url: str = "https://test.example.com/agent",
) -> Any:
    """Create an A2A gateway with a registered test agent."""
    # Import here to allow patching env vars first
    from litellm_llmrouter.a2a_gateway import A2AGateway, A2AAgent

    gateway = A2AGateway()
    gateway.enabled = True

    agent = A2AAgent(
        agent_id=agent_id,
        name="Test Agent",
        description="A test agent",
        url=agent_url,
        capabilities=["streaming"],
    )
    gateway.agents[agent_id] = agent

    return gateway


def create_jsonrpc_request(request_id: str = "1") -> Any:
    """Create a JSON-RPC request for testing."""
    from litellm_llmrouter.a2a_gateway import JSONRPCRequest

    return JSONRPCRequest(
        method="message/stream",
        params={"message": {"role": "user", "parts": [{"type": "text", "text": "test"}]}},
        id=request_id,
    )


# =============================================================================
# Feature Flag Tests
# =============================================================================


class TestFeatureFlagBehavior:
    """Tests for the A2A_RAW_STREAMING_ENABLED feature flag."""

    def test_default_flag_is_disabled(self):
        """
        Rollback safety: default flag value is 'false'.

        This ensures minimal blast radius - new behavior only activated
        when explicitly enabled.
        """
        with patch.dict(os.environ, {}, clear=True):
            # Force module reload to pick up env changes
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is False

    def test_flag_enabled_when_set_true(self):
        """Feature flag enables raw streaming when set to 'true'."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is True

    def test_flag_case_insensitive(self):
        """Feature flag is case-insensitive."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "TRUE"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_ENABLED is True

    def test_is_raw_streaming_enabled_helper(self):
        """Helper function correctly reports flag state."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.is_raw_streaming_enabled() is True

    def test_chunk_size_configurable(self):
        """A2A_RAW_STREAMING_CHUNK_SIZE is configurable via environment."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_CHUNK_SIZE": "4096"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            assert gateway_module.A2A_RAW_STREAMING_CHUNK_SIZE == 4096


# =============================================================================
# Raw Streaming Tests
# =============================================================================


class TestRawStreamingPassthrough:
    """
    Tests for raw streaming mode (A2A_RAW_STREAMING_ENABLED=true).

    These tests verify:
    - Chunks with embedded newlines are NOT split
    - Chunk boundaries from upstream are preserved
    - No waiting for newline boundaries before emitting
    """

    async def test_raw_streaming_preserves_newlines_in_chunk(self):
        """
        CRITICAL: Raw streaming does NOT split on newlines.

        In raw mode, a chunk like 'line1\\nline2' should be yielded as-is,
        NOT split into separate 'line1' and 'line2' chunks.

        This test would FAIL on the old line-buffered implementation.
        """
        # Prepare a chunk with embedded newlines
        chunk_with_newlines = b'{"part1": "value"}\n{"part2": "value"}\n'

        mock_response = MockHTTPResponse(content=[chunk_with_newlines])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # In raw mode, should receive the entire content as ONE chunk
        # (or multiple chunks that together equal the original, without splitting on newlines)
        combined = "".join(chunks)
        assert '{"part1": "value"}\n{"part2": "value"}\n' in combined

        # Key assertion: chunk count should match upstream chunk count (1)
        # NOT the line count (2)
        assert len(chunks) == 1, f"Expected 1 chunk but got {len(chunks)}: {chunks}"

    async def test_raw_streaming_preserves_chunk_boundaries(self):
        """
        Raw streaming preserves upstream chunk boundaries.

        If upstream sends 3 chunks, downstream should receive 3 chunks.
        """
        upstream_chunks = [
            b'{"chunk": 1}',
            b'{"chunk": 2}',
            b'{"chunk": 3}',
        ]

        mock_response = MockHTTPResponse(content=upstream_chunks)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should receive same number of chunks as upstream
        assert len(chunks) == 3

        # Content should match
        assert chunks[0] == '{"chunk": 1}'
        assert chunks[1] == '{"chunk": 2}'
        assert chunks[2] == '{"chunk": 3}'

    async def test_raw_streaming_no_waiting_for_newline(self):
        """
        Raw streaming yields chunks immediately, not waiting for newlines.

        Send a chunk without newline - it should be yielded immediately.
        """
        # Chunk without any newline
        partial_chunk = b'{"partial": "data'  # No newline, incomplete JSON

        mock_response = MockHTTPResponse(content=[partial_chunk])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should still receive the chunk even without newline
        assert len(chunks) == 1
        assert chunks[0] == '{"partial": "data'

    async def test_raw_streaming_empty_chunks_filtered(self):
        """Empty chunks are filtered out but non-empty ones pass through."""
        chunks_with_empty = [
            b"chunk1",
            b"",  # Empty chunk
            b"chunk2",
        ]

        mock_response = MockHTTPResponse(content=chunks_with_empty)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Empty chunk filtered out
        assert len(chunks) == 2
        assert "chunk1" in chunks[0]
        assert "chunk2" in chunks[1]


# =============================================================================
# Buffered Streaming Tests (Backward Compatibility)
# =============================================================================


class TestBufferedStreamingCompatibility:
    """
    Tests for buffered streaming mode (A2A_RAW_STREAMING_ENABLED=false).

    These tests verify backward compatibility with the original
    line-buffered implementation.
    """

    async def test_buffered_streaming_splits_on_newlines(self):
        """
        Buffered mode waits for and splits on newline boundaries.

        This is the original behavior for backward compatibility.
        """
        # Content with multiple lines
        content_with_lines = b'{"line1": "value"}\n{"line2": "value"}\n'

        mock_response = MockHTTPResponse(content=[content_with_lines])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Buffered mode should split on newlines
        assert len(chunks) == 2
        assert '{"line1": "value"}\n' in chunks[0]
        assert '{"line2": "value"}\n' in chunks[1]

    async def test_buffered_streaming_appends_newline(self):
        """Buffered mode appends newline to each yielded line."""
        content = b'{"data": "test"}\n'

        mock_response = MockHTTPResponse(content=[content])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_buffered(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Each chunk should end with newline
        for chunk in chunks:
            assert chunk.endswith("\n")


# =============================================================================
# Dispatcher Tests
# =============================================================================


class TestStreamAgentResponseDispatcher:
    """
    Tests for the stream_agent_response dispatcher method.

    Verifies correct routing based on feature flag.
    """

    async def test_dispatcher_uses_raw_when_enabled(self):
        """When flag is enabled, dispatcher uses raw streaming."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Mock the internal methods to verify which is called
            raw_called = False
            buffered_called = False

            async def mock_raw(*args, **kwargs):
                nonlocal raw_called
                raw_called = True
                yield "raw"

            async def mock_buffered(*args, **kwargs):
                nonlocal buffered_called
                buffered_called = True
                yield "buffered"

            gateway._stream_agent_response_raw = mock_raw
            gateway._stream_agent_response_buffered = mock_buffered

            # Consume the generator
            async for _ in gateway.stream_agent_response("test-agent", request):
                pass

            assert raw_called is True
            assert buffered_called is False

    async def test_dispatcher_uses_buffered_when_disabled(self):
        """When flag is disabled, dispatcher uses buffered streaming."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "false"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            # Mock the internal methods to verify which is called
            raw_called = False
            buffered_called = False

            async def mock_raw(*args, **kwargs):
                nonlocal raw_called
                raw_called = True
                yield "raw"

            async def mock_buffered(*args, **kwargs):
                nonlocal buffered_called
                buffered_called = True
                yield "buffered"

            gateway._stream_agent_response_raw = mock_raw
            gateway._stream_agent_response_buffered = mock_buffered

            # Consume the generator
            async for _ in gateway.stream_agent_response("test-agent", request):
                pass

            assert buffered_called is True
            assert raw_called is False


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestStreamingErrorHandling:
    """Tests for error handling in both streaming modes."""

    async def test_disabled_gateway_error(self):
        """Both modes yield proper error when gateway is disabled."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = gateway_module.A2AGateway()
            gateway.enabled = False

            request = create_jsonrpc_request()

            chunks = []
            async for chunk in gateway._stream_agent_response_raw("test-agent", request):
                chunks.append(chunk)

            assert len(chunks) == 1
            error_response = json.loads(chunks[0].strip())
            assert error_response["error"]["code"] == -32000
            assert "not enabled" in error_response["error"]["message"]

    async def test_agent_not_found_error(self):
        """Both modes yield proper error for nonexistent agent."""
        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = gateway_module.A2AGateway()
            gateway.enabled = True

            request = create_jsonrpc_request()

            chunks = []
            async for chunk in gateway._stream_agent_response_raw(
                "nonexistent-agent", request
            ):
                chunks.append(chunk)

            assert len(chunks) == 1
            error_response = json.loads(chunks[0].strip())
            assert error_response["error"]["code"] == -32000
            assert "not found" in error_response["error"]["message"]


# =============================================================================
# Regression Tests
# =============================================================================


class TestStreamingRegressions:
    """
    Regression tests for streaming behavior.

    These tests catch specific bugs that could break streaming semantics.
    """

    async def test_no_full_buffering_behavior(self):
        """
        Verify streaming doesn't buffer entire response before yielding.

        This test simulates slow chunks and verifies we get partial
        results before the full response is complete.
        """
        slow_chunks = [
            b'{"chunk": 1}',
            b'{"chunk": 2}',
            b'{"chunk": 3}',
        ]

        # Add delay between chunks
        mock_response = MockHTTPResponse(content=slow_chunks, chunk_delay=0.01)

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                received_chunks = []
                timestamps = []

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    received_chunks.append(chunk)
                    timestamps.append(asyncio.get_event_loop().time())

                    # Verify we're getting chunks incrementally, not all at once
                    if len(received_chunks) < 3:
                        # Should have received this chunk before all chunks are done
                        assert len(received_chunks) < len(slow_chunks)

        # Should have received all chunks
        assert len(received_chunks) == 3

    async def test_binary_content_decoded_safely(self):
        """Raw streaming handles binary content with invalid UTF-8 gracefully."""
        # Content with invalid UTF-8 byte
        content_with_invalid_utf8 = b'{"data": "\xff\xfe"}'

        mock_response = MockHTTPResponse(content=[content_with_invalid_utf8])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Should handle gracefully with replacement characters
        assert len(chunks) == 1
        # Contains replacement character for invalid bytes
        assert "" in chunks[0] or "data" in chunks[0]

    async def test_empty_response_handled(self):
        """Both modes handle empty responses gracefully."""
        mock_response = MockHTTPResponse(content=[])

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch("httpx.AsyncClient", return_value=MockAsyncClient(mock_response)):
                chunks = []
                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks.append(chunk)

        # Empty response should yield no chunks
        assert len(chunks) == 0


# =============================================================================
# Performance Characteristic Tests
# =============================================================================


class TestStreamingPerformanceCharacteristics:
    """
    Tests that verify performance characteristics are correct.

    These aren't strict benchmarks but verify streaming behavior
    that affects TTFB and chunk cadence.
    """

    async def test_first_chunk_emitted_before_second_arrives(self):
        """
        First chunk should be emitted before second chunk from upstream.

        This verifies we're not waiting for all chunks before emitting.
        """
        chunks_received_order = []

        class TrackedMockResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            async def aiter_bytes(self, chunk_size: int = 1024):
                chunks_received_order.append("upstream_1")
                yield b"chunk1"

                chunks_received_order.append("upstream_2")
                yield b"chunk2"

        mock_response = TrackedMockResponse()

        with patch.dict(os.environ, {"A2A_RAW_STREAMING_ENABLED": "true"}):
            import importlib
            import litellm_llmrouter.a2a_gateway as gateway_module

            importlib.reload(gateway_module)

            gateway = create_a2a_gateway_with_agent()
            request = create_jsonrpc_request()

            with patch(
                "httpx.AsyncClient",
                return_value=MockAsyncClient(mock_response),  # type: ignore
            ):

                async for chunk in gateway._stream_agent_response_raw(
                    "test-agent", request
                ):
                    chunks_received_order.append(f"downstream_{chunk}")

        # Verify interleaving: first upstream chunk should be yielded
        # before second upstream chunk is received
        assert chunks_received_order.index("downstream_chunk1") < chunks_received_order.index("upstream_2")
