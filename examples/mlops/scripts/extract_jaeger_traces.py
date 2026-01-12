#!/usr/bin/env python3
"""
Extract LLM traces from Jaeger for router training.

Usage:
    python extract_jaeger_traces.py --jaeger-url http://localhost:16686 --output traces.jsonl
"""

import json
import click
import requests
from datetime import datetime, timedelta
from typing import Optional


@click.command()
@click.option("--jaeger-url", default="http://localhost:16686", help="Jaeger API URL")
@click.option(
    "--service-name", default="litellm-gateway", help="Service name in Jaeger"
)
@click.option("--hours-back", default=24, help="Hours of traces to extract")
@click.option("--output", default="traces.jsonl", help="Output JSONL file")
@click.option("--limit", default=10000, help="Max traces to extract")
def extract_traces(
    jaeger_url: str, service_name: str, hours_back: int, output: str, limit: int
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
    for trace in traces:
        for span in trace.get("spans", []):
            record = extract_span_data(span)
            if record and record.get("query"):
                training_data.append(record)

    print(f"   Extracted {len(training_data)} training records")

    # Save as JSONL
    with open(output, "w") as f:
        for record in training_data:
            f.write(json.dumps(record) + "\n")

    print(f"✅ Saved to: {output}")


def extract_span_data(span: dict) -> Optional[dict]:
    """Extract training-relevant data from a Jaeger span."""
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
