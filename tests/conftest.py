"""
Pytest configuration for LiteLLM + LLMRouter tests.

Automatically skips integration tests when the local Docker stack is not running.
"""

import os
import socket

import pytest


def is_port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    """Check if a port is open on the given host."""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except (OSError, socket.error):
        return False


def pytest_collection_modifyitems(config, items):
    """Skip integration tests if the local stack is not running."""
    # Check if we're in CI or if the gateway is not available
    in_ci = os.getenv("CI", "").lower() == "true"
    gateway_port = 4010  # Default gateway port from docker-compose.local-test.yml

    # Try to detect if the gateway is running
    gateway_available = is_port_open("localhost", gateway_port)

    if not gateway_available:
        skip_integration = pytest.mark.skip(
            reason="Integration tests skipped: Local Docker stack not running. "
            "Run 'docker compose -f docker-compose.local-test.yml up -d' first."
        )
        for item in items:
            # Skip tests in the integration folder
            if "integration" in str(item.fspath):
                item.add_marker(skip_integration)

    if in_ci and not gateway_available:
        # In CI, we expect integration tests to be skipped
        print("CI environment detected, skipping integration tests")
