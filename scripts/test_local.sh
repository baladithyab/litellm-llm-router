#!/bin/bash
# Local Test Script for LiteLLM + LLMRouter Gateway
# Tests: Health, A2A Gateway, MCP Gateway, Routing, Hot Reload

set -e

BASE_URL="${BASE_URL:-http://localhost:4000}"
MASTER_KEY="${MASTER_KEY:-sk-test-master-key}"

echo "=========================================="
echo "LiteLLM + LLMRouter Local Test Suite"
echo "=========================================="
echo "Base URL: $BASE_URL"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

pass() { echo -e "${GREEN}✓ PASS${NC}: $1"; }
fail() { echo -e "${RED}✗ FAIL${NC}: $1"; exit 1; }
info() { echo -e "${YELLOW}ℹ INFO${NC}: $1"; }

# Wait for server to be ready
wait_for_server() {
    info "Waiting for server to be ready..."
    for _ in {1..30}; do
        if curl -s "$BASE_URL/health" > /dev/null 2>&1; then
            pass "Server is ready"
            return 0
        fi
        sleep 2
    done
    fail "Server did not become ready in 60 seconds"
}

# Test 1: Health Check
test_health() {
    echo ""
    echo "--- Test 1: Health Check ---"

    response=$(curl -s "$BASE_URL/health")
    if echo "$response" | grep -q "healthy"; then
        pass "Health endpoint returns healthy"
    else
        fail "Health check failed: $response"
    fi

    # Liveliness probe
    response=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/health/liveliness")
    if [ "$response" = "200" ]; then
        pass "Liveliness probe returns 200"
    else
        fail "Liveliness probe failed: $response"
    fi
}

# Test 2: Model List
test_models() {
    echo ""
    echo "--- Test 2: Model List ---"

    response=$(curl -s -H "Authorization: Bearer $MASTER_KEY" "$BASE_URL/v1/models")
    if echo "$response" | grep -q "gpt-4\|claude"; then
        pass "Model list returns configured models"
        info "Models: $(echo "$response" | jq -r '.data[].id' 2>/dev/null | tr '\n' ', ')"
    else
        fail "Model list failed: $response"
    fi
}

# Test 3: A2A Gateway (if enabled)
test_a2a_gateway() {
    echo ""
    echo "--- Test 3: A2A Gateway ---"

    # Register a test agent
    response=$(curl -s -X POST "$BASE_URL/a2a/agents" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "agent_id": "test-agent-1",
            "name": "Test Agent",
            "description": "A test agent for validation",
            "url": "http://localhost:9000/a2a",
            "capabilities": ["chat", "code"]
        }')

    if echo "$response" | grep -qE "registered|success|test-agent"; then
        pass "A2A agent registration works"
    else
        info "A2A response: $response"
    fi

    # List agents
    response=$(curl -s -H "Authorization: Bearer $MASTER_KEY" "$BASE_URL/a2a/agents")
    info "A2A agents list: $response"
}

# Test 4: MCP Gateway (if enabled)
test_mcp_gateway() {
    echo ""
    echo "--- Test 4: MCP Gateway ---"

    # Register a test MCP server
    response=$(curl -s -X POST "$BASE_URL/mcp/servers" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json" \
        -d '{
            "server_id": "test-mcp-1",
            "name": "Test MCP Server",
            "url": "http://localhost:8080/mcp",
            "transport": "streamable_http",
            "tools": ["search", "fetch"]
        }')

    if echo "$response" | grep -qE "registered|success|test-mcp"; then
        pass "MCP server registration works"
    else
        info "MCP response: $response"
    fi

    # List MCP servers
    response=$(curl -s -H "Authorization: Bearer $MASTER_KEY" "$BASE_URL/mcp/servers")
    info "MCP servers list: $response"
}

# Test 5: Config Sync Status
test_config_sync() {
    echo ""
    echo "--- Test 5: Config Sync Status ---"

    response=$(curl -s -H "Authorization: Bearer $MASTER_KEY" "$BASE_URL/config/sync/status")
    if echo "$response" | grep -qE "enabled|sync_interval|running"; then
        pass "Config sync status endpoint works"
        info "Sync status: $response"
    else
        info "Config sync response: $response"
    fi
}

# Test 6: Hot Reload
test_hot_reload() {
    echo ""
    echo "--- Test 6: Hot Reload ---"

    # Trigger config reload
    response=$(curl -s -X POST "$BASE_URL/config/reload" \
        -H "Authorization: Bearer $MASTER_KEY" \
        -H "Content-Type: application/json")

    if echo "$response" | grep -qE "success|triggered|reload"; then
        pass "Config reload endpoint works"
    else
        info "Config reload response: $response"
    fi
}

# Run all tests
main() {
    wait_for_server
    test_health
    test_models
    test_a2a_gateway
    test_mcp_gateway
    test_config_sync
    test_hot_reload

    echo ""
    echo "=========================================="
    echo "All tests completed!"
    echo "=========================================="
}

main "$@"
