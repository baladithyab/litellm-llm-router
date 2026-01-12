#!/usr/bin/env python3
"""
Generate synthetic LLM traces for testing the MLOps pipeline.

Usage:
    python generate_synthetic_traces.py --output traces.jsonl --count 100
"""

import json
import random
import click
from datetime import datetime


SAMPLE_PROMPTS = [
    "What is machine learning?",
    "Explain neural networks in simple terms",
    "How do transformers work?",
    "What are the benefits of cloud computing?",
    "Describe the water cycle",
    "What is quantum computing?",
    "Explain photosynthesis",
    "How does GPS work?",
    "What is blockchain technology?",
    "Explain the theory of relativity",
]

MODELS = [
    "claude-sonnet",
    "claude-haiku",
    "claude-3-sonnet",
    "claude-3-haiku",
    "claude-opus",
]


@click.command()
@click.option("--output", default="traces.jsonl", help="Output JSONL file")
@click.option("--count", default=100, help="Number of traces to generate")
def generate_traces(output: str, count: int):
    """Generate synthetic LLM traces for pipeline testing."""
    print(f"🔧 Generating {count} synthetic traces")

    traces = []
    base_time = int(datetime.now().timestamp() * 1_000_000)

    for i in range(count):
        model = random.choice(MODELS)
        prompt = random.choice(SAMPLE_PROMPTS)

        # Simulate realistic latencies (faster for haiku, slower for opus)
        if "haiku" in model:
            response_time = random.uniform(0.3, 1.5)
        elif "opus" in model:
            response_time = random.uniform(2.0, 8.0)
        else:
            response_time = random.uniform(0.8, 3.0)

        # Simulate token counts
        input_tokens = len(prompt.split()) * 2 + random.randint(5, 15)
        output_tokens = random.randint(50, 200)

        # Simulate performance scores based on model characteristics
        if "haiku" in model:
            performance = (
                random.uniform(0.7, 0.9)
                if response_time < 1.0
                else random.uniform(0.5, 0.7)
            )
        elif "opus" in model:
            performance = random.uniform(0.85, 0.98)  # High quality
        else:
            performance = random.uniform(0.75, 0.92)

        trace = {
            "query": prompt,
            "model_name": model,
            "response": f"[Simulated response to: {prompt[:30]}...]",
            "response_time": round(response_time, 3),
            "token_num": input_tokens + output_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "task_name": "production",
            "trace_id": f"trace_{i:06d}",
            "span_id": f"span_{i:06d}",
            "timestamp": base_time - (count - i) * 60_000_000,
            "performance": round(performance, 3),
            "ground_truth": "",
        }
        traces.append(trace)

    # Save as JSONL
    with open(output, "w") as f:
        for trace in traces:
            f.write(json.dumps(trace) + "\n")

    print(f"✅ Generated {len(traces)} traces")
    print(f"📁 Saved to: {output}")

    # Print summary
    model_counts = {}
    for t in traces:
        m = t["model_name"]
        model_counts[m] = model_counts.get(m, 0) + 1

    print("\n📊 Model distribution:")
    for m, c in sorted(model_counts.items()):
        print(f"   {m}: {c}")


if __name__ == "__main__":
    generate_traces()
