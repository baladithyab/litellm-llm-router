#!/usr/bin/env python3
"""
Extract LLM traces from Jaeger for router training.

Supports both:
1. New versioned telemetry events (routeiq.router_decision.v1) 
2. Legacy span attributes for backward compatibility

Usage:
    python extract_jaeger_traces.py --jaeger-url http://localhost:16686 --output traces.jsonl
"""

import json
import click
import requests
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

# Contract constants (matches telemetry_contracts.py)
CONTRACT_VERSION = "v1"
CONTRACT_NAME = "routeiq.router_decision"
CONTRACT_FULL_NAME = f"{CONTRACT_NAME}.{CONTRACT_VERSION}"
ROUTER_DECISION_EVENT_NAME = CONTRACT_FULL_NAME
ROUTER_DECISION_PAYLOAD_KEY = f"{CONTRACT_NAME}.payload"


def extract_router_decision_from_event(event: dict) -> Optional[Dict[str, Any]]:
    """
    Extract routing decision from a versioned span event.
    
    Args:
        event: Jaeger log/event entry
        
    Returns:
        Parsed RouterDecisionEvent dict if found, None otherwise
    """
    fields = {f.get("key"): f.get("value") for f in event.get("fields", [])}
    
    # Check for versioned payload
    payload = fields.get(ROUTER_DECISION_PAYLOAD_KEY)
    if not payload:
        return None
    
    try:
        if isinstance(payload, str):
            data = json.loads(payload)
        else:
            data = payload
        
        # Validate it's our contract
        if data.get("contract_name") != CONTRACT_FULL_NAME:
            return None
        
        return data
    except (json.JSONDecodeError, TypeError):
        return None


def extract_routing_decisions_from_span(span: dict) -> List[Dict[str, Any]]:
    """
    Extract all routing decision events from a span's logs/events.
    
    Args:
        span: Jaeger span object
        
    Returns:
        List of RouterDecisionEvent dicts found in the span
    """
    decisions = []
    logs = span.get("logs", [])
    
    for log in logs:
        decision = extract_router_decision_from_event(log)
        if decision:
            decisions.append(decision)
    
    return decisions


@click.command()
@click.option("--jaeger-url", default="http://localhost:16686", help="Jaeger API URL")
@click.option(
    "--service-name", default="litellm-gateway", help="Service name in Jaeger"
)
@click.option("--hours-back", default=24, help="Hours of traces to extract")
@click.option("--output", default="traces.jsonl", help="Output JSONL file")
@click.option("--limit", default=10000, help="Max traces to extract")
@click.option(
    "--routing-decisions-output",
    default=None,
    help="Separate output for routing decision events (JSONL)",
)
def extract_traces(
    jaeger_url: str,
    service_name: str,
    hours_back: int,
    output: str,
    limit: int,
    routing_decisions_output: Optional[str],
):
    """Extract LLM traces from Jaeger and save as JSONL."""
    print("🔍 Extracting traces from Jaeger")
    print(f"   URL: {jaeger_url}")
    print(f"   Service: {service_name}")
    print(f"   Hours back: {hours_back}")

    end_time = datetime.now()
    start_time = end_time - timedelta(hours=hours_back)

    # Query Jaeger API for traces
    try:
        response = requests.get(
            f"{jaeger_url}/api/traces",
            params={
                "service": service_name,
                "start": int(start_time.timestamp() * 1_000_000),
                "end": int(end_time.timestamp() * 1_000_000),
                "limit": limit,
            },
            timeout=30,
        )
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"❌ Failed to query Jaeger: {e}")
        return

    data = response.json()
    traces = data.get("data", [])
    print(f"   Found {len(traces)} traces")

    # Extract training-relevant data from spans
    training_data = []
    routing_decisions = []
    
    for trace in traces:
        for span in trace.get("spans", []):
            # Extract legacy training data
            record = extract_span_data(span)
            if record and record.get("query"):
                training_data.append(record)
            
            # Extract versioned routing decisions
            decisions = extract_routing_decisions_from_span(span)
            for decision in decisions:
                # Convert routing decision to training format
                training_record = convert_routing_decision_to_training(decision, span)
                if training_record:
                    routing_decisions.append(training_record)

    print(f"   Extracted {len(training_data)} legacy training records")
    print(f"   Extracted {len(routing_decisions)} routing decision events")

    # Save legacy format as JSONL
    with open(output, "w") as f:
        for record in training_data:
            f.write(json.dumps(record) + "\n")
    print(f"✅ Saved legacy traces to: {output}")

    # Save routing decisions if output path provided
    if routing_decisions_output:
        with open(routing_decisions_output, "w") as f:
            for decision in routing_decisions:
                f.write(json.dumps(decision) + "\n")
        print(f"✅ Saved routing decisions to: {routing_decisions_output}")
    elif routing_decisions:
        # Append routing decisions to main output
        with open(output, "a") as f:
            for decision in routing_decisions:
                f.write(json.dumps(decision) + "\n")
        print(f"   (Appended routing decisions to {output})")


def convert_routing_decision_to_training(
    decision: Dict[str, Any], span: dict
) -> Optional[Dict[str, Any]]:
    """
    Convert a RouterDecisionEvent to training data format.
    
    Args:
        decision: Parsed RouterDecisionEvent dict
        span: Parent Jaeger span
        
    Returns:
        Training-compatible dict, or None if conversion fails
    """
    try:
        # Extract key fields from the decision
        input_data = decision.get("input", {})
        timings = decision.get("timings", {})
        outcome = decision.get("outcome", {})
        
        # Get selected model
        selected_model = decision.get("selected_deployment")
        if not selected_model:
            return None
        
        return {
            # Core training fields
            "query": "",  # Not stored for PII safety - use embedding_id instead
            "query_length": input_data.get("query_length", 0),
            "model_name": selected_model,
            "response_time": timings.get("total_ms", 0) / 1000.0,  # Convert to seconds
            
            # Routing metadata
            "strategy_name": decision.get("strategy_name", ""),
            "strategy_version": decision.get("strategy_version"),
            "outcome_status": outcome.get("status", ""),
            
            # Token usage (if available from downstream)
            "token_num": outcome.get("total_tokens"),
            "input_tokens": outcome.get("input_tokens"),
            "output_tokens": outcome.get("output_tokens"),
            
            # Trace correlation
            "trace_id": decision.get("trace_id") or span.get("traceID", ""),
            "span_id": decision.get("span_id") or span.get("spanID", ""),
            "timestamp": decision.get("timestamp_unix_ms", span.get("startTime", 0)),
            "event_id": decision.get("event_id", ""),
            
            # Contract metadata
            "contract_version": decision.get("contract_version", CONTRACT_VERSION),
            "contract_name": decision.get("contract_name", CONTRACT_FULL_NAME),
            
            # Candidate info
            "candidate_count": len(decision.get("candidate_deployments", [])),
            "candidates": [
                c.get("model_name") 
                for c in decision.get("candidate_deployments", [])
            ],
            
            # Performance labeling (needs post-processing)
            "performance": None,
            "ground_truth": "",
            "task_name": "production",
        }
    except Exception as e:
        print(f"   Warning: Failed to convert routing decision: {e}")
        return None


def extract_span_data(span: dict) -> Optional[dict]:
    """Extract training-relevant data from a Jaeger span (legacy format)."""
    tags = {t["key"]: t["value"] for t in span.get("tags", [])}
    logs = span.get("logs", [])
    duration_us = span.get("duration", 0)

    # Extract prompt from logs if available
    prompt = tags.get("gen_ai.prompt", "")
    if not prompt:
        for log in logs:
            for field in log.get("fields", []):
                if field.get("key") == "gen_ai.prompt":
                    prompt = field.get("value", "")
                    break

    # Extract model name
    model_name = tags.get("gen_ai.response.model", tags.get("gen_ai.request.model", ""))

    if not prompt or not model_name:
        return None

    return {
        "query": prompt,
        "model_name": model_name,
        "response": tags.get("gen_ai.completion", ""),
        "response_time": duration_us / 1_000_000,  # Convert to seconds
        "token_num": int(tags.get("gen_ai.usage.total_tokens", 0)),
        "input_tokens": int(tags.get("gen_ai.usage.prompt_tokens", 0)),
        "output_tokens": int(tags.get("gen_ai.usage.completion_tokens", 0)),
        "task_name": tags.get("litellm.task", "production"),
        "trace_id": span.get("traceID", ""),
        "span_id": span.get("spanID", ""),
        "timestamp": span.get("startTime", 0),
        "performance": None,  # Needs labeling
        "ground_truth": "",  # Needs labeling
    }


if __name__ == "__main__":
    extract_traces()
