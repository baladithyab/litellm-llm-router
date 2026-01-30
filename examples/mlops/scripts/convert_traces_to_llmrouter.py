#!/usr/bin/env python3
"""
Convert trace data to LLMRouter format.

Supports both:
1. New routing decision event format (routeiq.router_decision.v1)
2. Legacy trace format for backward compatibility

Converts our trace format to the format expected by LLMRouter:
- routing_data_train.jsonl: query, model_name, performance, embedding_id
- query_embeddings.pt: Pre-computed query embeddings
- llm_data.json: LLM metadata

Usage:
    python convert_traces_to_llmrouter.py --input traces.jsonl --output-dir /tmp/llmrouter_data
"""

import json
import os
import sys
from pathlib import Path
import click
import torch
from sentence_transformers import SentenceTransformer
from typing import Dict, Any, Optional

# Add project root to path for imports
_project_root = Path(__file__).parent.parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import contract constants from source-of-truth module
try:
    from litellm_llmrouter.telemetry_contracts import (
        CONTRACT_VERSION,
        CONTRACT_NAME,
        CONTRACT_FULL_NAME,
    )
except ImportError:
    # Fallback for standalone use: duplicate constants must match source
    CONTRACT_VERSION = "v1"
    CONTRACT_NAME = "routeiq.router_decision"
    CONTRACT_FULL_NAME = f"{CONTRACT_NAME}.{CONTRACT_VERSION}"


def is_routing_decision_record(record: Dict[str, Any]) -> bool:
    """Check if a record is from the new routing decision event format."""
    return record.get("contract_name") == CONTRACT_FULL_NAME


def normalize_trace_record(record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Normalize a trace record to a common format.

    Handles both legacy trace format and new routing decision format.

    Args:
        record: Raw trace record from JSONL

    Returns:
        Normalized record dict, or None if invalid
    """
    if is_routing_decision_record(record):
        # New routing decision format
        return {
            "query": record.get("query", ""),
            "query_length": record.get("query_length", 0),
            "model_name": record.get("model_name", ""),
            "performance": record.get("performance"),
            "response_time": record.get("response_time", 0),
            "token_num": record.get("token_num"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "trace_id": record.get("trace_id", ""),
            "span_id": record.get("span_id", ""),
            "timestamp": record.get("timestamp", 0),
            "strategy_name": record.get("strategy_name", ""),
            "strategy_version": record.get("strategy_version"),
            "candidate_count": record.get("candidate_count", 0),
            "candidates": record.get("candidates", []),
            "outcome_status": record.get("outcome_status", ""),
            "contract_version": record.get("contract_version", CONTRACT_VERSION),
            "is_routing_decision": True,
        }
    else:
        # Legacy trace format
        return {
            "query": record.get("query", ""),
            "query_length": len(record.get("query", "")),
            "model_name": record.get("model_name", ""),
            "performance": record.get("performance"),
            "response_time": record.get("response_time", 0),
            "token_num": record.get("token_num"),
            "input_tokens": record.get("input_tokens"),
            "output_tokens": record.get("output_tokens"),
            "trace_id": record.get("trace_id", ""),
            "span_id": record.get("span_id", ""),
            "timestamp": record.get("timestamp", 0),
            "strategy_name": "",
            "strategy_version": None,
            "candidate_count": 0,
            "candidates": [],
            "outcome_status": "",
            "contract_version": None,
            "is_routing_decision": False,
        }


@click.command()
@click.option("--input", "input_file", required=True, help="Input traces JSONL file")
@click.option("--output-dir", required=True, help="Output directory for LLMRouter data")
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence transformer model for embeddings",
)
@click.option("--test-split", default=0.2, help="Fraction of data for test set")
@click.option(
    "--include-routing-metadata",
    is_flag=True,
    default=False,
    help="Include routing strategy metadata in output",
)
def convert_traces(
    input_file: str,
    output_dir: str,
    embedding_model: str,
    test_split: float,
    include_routing_metadata: bool,
):
    """Convert trace data to LLMRouter format."""
    print(f"📂 Loading traces from: {input_file}")

    # Load and normalize traces
    traces = []
    routing_decision_count = 0
    legacy_count = 0

    with open(input_file, "r") as f:
        for line in f:
            raw_record = json.loads(line)
            normalized = normalize_trace_record(raw_record)
            if normalized:
                traces.append(normalized)
                if normalized.get("is_routing_decision"):
                    routing_decision_count += 1
                else:
                    legacy_count += 1

    print(f"   Loaded {len(traces)} total traces")
    print(f"   - {routing_decision_count} routing decision events (v1)")
    print(f"   - {legacy_count} legacy format traces")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique queries and models
    # For routing decisions without query content, use trace_id as query key
    unique_queries = []
    query_to_index = {}

    for t in traces:
        query_key = (
            t["query"]
            if t["query"]
            else f"routing_decision:{t['trace_id']}:{t['span_id']}"
        )
        if query_key not in query_to_index:
            query_to_index[query_key] = len(unique_queries)
            unique_queries.append(query_key)

    unique_models = list(set(t["model_name"] for t in traces if t["model_name"]))

    print(f"   Unique queries/decisions: {len(unique_queries)}")
    print(f"   Unique models: {len(unique_models)}")

    # Generate embeddings for unique queries
    # Note: For routing decisions without query text, we create placeholder embeddings
    print(f"🔧 Generating embeddings with: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    # Filter queries that have actual text content for embedding
    queries_with_text = [
        q for q in unique_queries if not q.startswith("routing_decision:")
    ]
    placeholder_queries = [
        q for q in unique_queries if q.startswith("routing_decision:")
    ]

    if queries_with_text:
        text_embeddings = model.encode(queries_with_text, convert_to_tensor=True)
        print(f"   Generated embeddings for {len(queries_with_text)} text queries")
    else:
        text_embeddings = None

    # Create placeholder embeddings for routing decisions without query text
    embedding_dim = model.get_sentence_embedding_dimension()
    if placeholder_queries:
        # Use zero vectors as placeholders - these need query text for proper training
        placeholder_embeddings = torch.zeros(len(placeholder_queries), embedding_dim)
        print(
            f"   Created {len(placeholder_queries)} placeholder embeddings (need query text)"
        )
    else:
        placeholder_embeddings = None

    # Combine embeddings in correct order
    all_embeddings = []
    text_idx = 0
    placeholder_idx = 0
    for q in unique_queries:
        if q.startswith("routing_decision:"):
            all_embeddings.append(placeholder_embeddings[placeholder_idx])
            placeholder_idx += 1
        else:
            all_embeddings.append(text_embeddings[text_idx])
            text_idx += 1

    embeddings = torch.stack(all_embeddings)

    # Convert traces to routing data format
    routing_data = []
    for trace in traces:
        query_key = (
            trace["query"]
            if trace["query"]
            else f"routing_decision:{trace['trace_id']}:{trace['span_id']}"
        )

        record = {
            "query": trace["query"],
            "model_name": trace["model_name"],
            "performance": trace.get("performance", 0.5),
            "embedding_id": query_to_index[query_key],
        }

        # Include routing metadata if requested
        if include_routing_metadata:
            record.update(
                {
                    "strategy_name": trace.get("strategy_name", ""),
                    "strategy_version": trace.get("strategy_version"),
                    "outcome_status": trace.get("outcome_status", ""),
                    "candidate_count": trace.get("candidate_count", 0),
                    "response_time_ms": trace.get("response_time", 0) * 1000,
                    "trace_id": trace.get("trace_id", ""),
                    "contract_version": trace.get("contract_version"),
                }
            )

        routing_data.append(record)

    # Split into train/test
    split_idx = int(len(routing_data) * (1 - test_split))
    train_data = routing_data[:split_idx]
    test_data = routing_data[split_idx:]

    # Save routing data
    train_path = os.path.join(output_dir, "routing_train.jsonl")
    with open(train_path, "w") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved training data: {train_path} ({len(train_data)} records)")

    test_path = os.path.join(output_dir, "routing_test.jsonl")
    with open(test_path, "w") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved test data: {test_path} ({len(test_data)} records)")

    # Save embeddings
    emb_path = os.path.join(output_dir, "query_embeddings.pt")
    torch.save(embeddings, emb_path)
    print(f"✅ Saved embeddings: {emb_path} (shape: {embeddings.shape})")

    # Create LLM data
    llm_data = {}
    for model_name in unique_models:
        llm_data[model_name] = {
            "name": model_name,
            "provider": _infer_provider(model_name),
        }

    llm_path = os.path.join(output_dir, "llm_data.json")
    with open(llm_path, "w") as f:
        json.dump(llm_data, f, indent=2)
    print(f"✅ Saved LLM data: {llm_path}")

    # Save conversion metadata
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "total_records": len(traces),
        "routing_decision_records": routing_decision_count,
        "legacy_records": legacy_count,
        "unique_queries": len(unique_queries),
        "queries_with_text": len(queries_with_text),
        "placeholder_queries": len(placeholder_queries),
        "unique_models": len(unique_models),
        "train_records": len(train_data),
        "test_records": len(test_data),
        "embedding_model": embedding_model,
        "embedding_dim": embedding_dim,
    }

    metadata_path = os.path.join(output_dir, "conversion_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved conversion metadata: {metadata_path}")

    print(f"\n🎉 Conversion complete! Data saved to: {output_dir}")

    if placeholder_queries:
        print(
            f"\n⚠️  Warning: {len(placeholder_queries)} routing decision events lack query text."
        )
        print(
            "   These use placeholder embeddings and need query content for effective training."
        )
        print(
            "   Consider enabling prompt logging (with PII precautions) for better embeddings."
        )


def _infer_provider(model_name: str) -> str:
    """Infer provider from model name."""
    model_lower = model_name.lower()

    if "claude" in model_lower or "anthropic" in model_lower:
        return "anthropic"
    elif "gpt" in model_lower or "openai" in model_lower:
        return "openai"
    elif "bedrock" in model_lower:
        return "bedrock"
    elif "gemini" in model_lower or "palm" in model_lower:
        return "google"
    elif "mistral" in model_lower:
        return "mistral"
    elif "llama" in model_lower or "meta" in model_lower:
        return "meta"
    elif "cohere" in model_lower:
        return "cohere"
    else:
        return "unknown"


if __name__ == "__main__":
    convert_traces()
