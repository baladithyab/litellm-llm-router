#!/bin/bash
# Test MLOps Pipeline: Run requests -> Collect Jaeger traces -> Train model
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Configuration
LITELLM_URL="${LITELLM_URL:-http://localhost:4000}"
JAEGER_URL="${JAEGER_URL:-http://localhost:16686}"
API_KEY="${API_KEY:-sk-dev-key}"
NUM_REQUESTS="${NUM_REQUESTS:-50}"

echo "==========================================="
echo "MLOps Pipeline Test"
echo "==========================================="
echo "LiteLLM URL: $LITELLM_URL"
echo "Jaeger URL: $JAEGER_URL"
echo "Requests: $NUM_REQUESTS"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

pass() { echo -e "${GREEN}✓ PASS:${NC} $1"; }
fail() { echo -e "${RED}✗ FAIL:${NC} $1"; exit 1; }
info() { echo -e "${YELLOW}ℹ INFO:${NC} $1"; }

# Step 1: Check services are running
echo "--- Step 1: Check Services ---"

if curl -s "$LITELLM_URL/health/liveliness" | grep -q "alive"; then
    pass "LiteLLM is running"
else
    fail "LiteLLM is not responding at $LITELLM_URL"
fi

if curl -s "$JAEGER_URL/api/services" | grep -q "data"; then
    pass "Jaeger is running"
else
    fail "Jaeger is not responding at $JAEGER_URL"
fi

# Step 2: Generate sample requests
echo ""
echo "--- Step 2: Generate $NUM_REQUESTS Requests ---"

PROMPTS=(
    "What is machine learning?"
    "Explain quantum computing in simple terms"
    "Write a Python function to sort a list"
    "What is the capital of France?"
    "How does photosynthesis work?"
    "Write a haiku about programming"
    "Explain the theory of relativity"
    "What are the benefits of exercise?"
    "How do neural networks learn?"
    "Describe the water cycle"
)

MODELS=("claude-sonnet" "claude-haiku" "claude-3-sonnet" "claude-3-haiku")

success_count=0
for i in $(seq 1 $NUM_REQUESTS); do
    prompt_idx=$((i % ${#PROMPTS[@]}))
    model_idx=$((i % ${#MODELS[@]}))
    prompt="${PROMPTS[$prompt_idx]}"
    model="${MODELS[$model_idx]}"

    response=$(curl -s -w "\n%{http_code}" "$LITELLM_URL/v1/chat/completions" \
        -H "Authorization: Bearer $API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"$model\", \"messages\": [{\"role\": \"user\", \"content\": \"$prompt\"}], \"max_tokens\": 50}" 2>/dev/null)

    http_code=$(echo "$response" | tail -n1)
    if [ "$http_code" = "200" ]; then
        ((success_count++)) || true
        echo -ne "\r   Progress: $success_count/$NUM_REQUESTS successful"
    fi
done
echo ""
pass "Sent $success_count requests successfully"

# Wait for traces to be indexed
info "Waiting 5s for Jaeger to index traces..."
sleep 5

# Step 3: Extract traces from Jaeger
echo ""
echo "--- Step 3: Extract Traces from Jaeger ---"

TRACES_FILE="/tmp/mlops_traces.jsonl"

python3 "$PROJECT_ROOT/examples/mlops/scripts/extract_jaeger_traces.py" \
    --jaeger-url "$JAEGER_URL" \
    --service-name "litellm-gateway" \
    --hours-back 1 \
    --output "$TRACES_FILE" \
    --limit 1000

if [ -f "$TRACES_FILE" ]; then
    trace_count=$(wc -l < "$TRACES_FILE")
    pass "Extracted $trace_count traces to $TRACES_FILE"
else
    fail "Failed to extract traces"
fi

# Step 4: Show trace sample
echo ""
echo "--- Step 4: Sample Traces ---"
head -3 "$TRACES_FILE" | python3 -m json.tool 2>/dev/null || head -3 "$TRACES_FILE"

# Step 5: Summary
echo ""
echo "==========================================="
echo "MLOps Pipeline Test Complete"
echo "==========================================="
echo "Requests sent: $NUM_REQUESTS"
echo "Successful: $success_count"
echo "Traces extracted: $trace_count"
echo "Traces file: $TRACES_FILE"
echo ""
echo "Next steps:"
echo "  1. Label traces with performance scores"
echo "  2. Generate embeddings"
echo "  3. Train router model with:"
echo "     python examples/mlops/scripts/train_router.py --router-type knn --config <config.yaml>"
