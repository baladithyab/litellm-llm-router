#!/usr/bin/env python3
"""
Convert trace data to LLMRouter format.

Converts our trace format to the format expected by LLMRouter:
- routing_data_train.jsonl: query, model_name, performance, embedding_id
- query_embeddings.pt: Pre-computed query embeddings
- llm_data.json: LLM metadata

Usage:
    python convert_traces_to_llmrouter.py --input traces.jsonl --output-dir /tmp/llmrouter_data
"""

import json
import os
import click
import torch
from sentence_transformers import SentenceTransformer


@click.command()
@click.option("--input", "input_file", required=True, help="Input traces JSONL file")
@click.option("--output-dir", required=True, help="Output directory for LLMRouter data")
@click.option(
    "--embedding-model",
    default="sentence-transformers/all-MiniLM-L6-v2",
    help="Sentence transformer model for embeddings",
)
@click.option("--test-split", default=0.2, help="Fraction of data for test set")
def convert_traces(
    input_file: str, output_dir: str, embedding_model: str, test_split: float
):
    """Convert trace data to LLMRouter format."""
    print(f"📂 Loading traces from: {input_file}")

    # Load traces
    traces = []
    with open(input_file, "r") as f:
        for line in f:
            traces.append(json.loads(line))

    print(f"   Loaded {len(traces)} traces")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Extract unique queries and models
    unique_queries = list(set(t["query"] for t in traces))
    unique_models = list(set(t["model_name"] for t in traces))

    print(f"   Unique queries: {len(unique_queries)}")
    print(f"   Unique models: {len(unique_models)}")

    # Generate embeddings for unique queries
    print(f"🔧 Generating embeddings with: {embedding_model}")
    model = SentenceTransformer(embedding_model)
    embeddings = model.encode(unique_queries, convert_to_tensor=True)

    # Create query to embedding_id mapping
    query_to_id = {q: i for i, q in enumerate(unique_queries)}

    # Convert traces to routing data format
    routing_data = []
    for trace in traces:
        routing_data.append(
            {
                "query": trace["query"],
                "model_name": trace["model_name"],
                "performance": trace.get("performance", 0.5),
                "embedding_id": query_to_id[trace["query"]],
            }
        )

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
            "provider": "bedrock" if "claude" in model_name.lower() else "unknown",
        }

    llm_path = os.path.join(output_dir, "llm_data.json")
    with open(llm_path, "w") as f:
        json.dump(llm_data, f, indent=2)
    print(f"✅ Saved LLM data: {llm_path}")

    print(f"\n🎉 Conversion complete! Data saved to: {output_dir}")


if __name__ == "__main__":
    convert_traces()
