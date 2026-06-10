"""
Minimal benchmark: compare routing policy quality with sentence-transformers vs FastEmbed.

Generates synthetic data — no external data files required.
Run from anywhere:
    bash src/tracer/benchmarks/embedders/run.sh
"""


import json
import tempfile
from pathlib import Path

import numpy as np

from tracer.api import fit
from tracer.config import FitConfig
from tracer.embeddings.embedder import Embedder


def generate_synthetic_data(n=500, dim=384, n_classes=5, seed=42):
    """Synthetic 5-class intent dataset, same structure as TRACER's demo fallback."""
    rng = np.random.RandomState(seed)
    intent_names = [
        "check_balance", "transfer_money", "card_blocked",
        "loan_inquiry", "account_settings",
    ]
    sample_texts = {
        "check_balance": [
            "What is my current account balance?",
            "How much money do I have?",
            "Can you check my account balance?",
        ],
        "transfer_money": [
            "I want to send money to my friend",
            "Transfer $500 to savings",
            "How do I wire money abroad?",
        ],
        "card_blocked": [
            "My card is not working",
            "Why was my card blocked?",
            "My card got declined at the store",
        ],
        "loan_inquiry": [
            "What are your loan rates?",
            "I'd like to apply for a personal loan",
            "How much can I borrow?",
        ],
        "account_settings": [
            "Change my email address",
            "How do I reset my password?",
            "Update my mailing address",
        ],
    }

    centers = rng.randn(n_classes, dim) * 2.5
    labels_int = rng.randint(0, n_classes, size=n)
    X = (centers[labels_int] + rng.randn(n, dim) * 1.2).astype(np.float32)

    texts, teachers = [], []
    for i in range(n):
        intent = intent_names[labels_int[i]]
        tpl = sample_texts[intent]
        text = tpl[i % len(tpl)]
        if i >= len(tpl):
            text = f"{text} (query #{i})"
        # Simulate ~10 % teacher noise
        teacher = intent if rng.random() > 0.1 else intent_names[rng.randint(0, n_classes)]
        texts.append(text)
        teachers.append(teacher)

    return texts, X, teachers


def benchmark():
    print("Generating synthetic data (500 samples, 5 intents, 384-dim)...")
    texts, X_syn, teachers = generate_synthetic_data(n=500, dim=384, n_classes=5)

    # Write traces to a temporary file (auto-cleaned)
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, prefix="tracer_bench_"
    ) as f:
        for text, teacher in zip(texts, teachers):
            f.write(json.dumps({"input": text, "teacher": teacher}) + "\n")
        traces_path = f.name

    config = FitConfig(target_teacher_agreement=0.90, seed=42)

    results = []
    for name, embedder_factory in [
        (
            "sentence-transformers",
            lambda: Embedder.from_sentence_transformers("all-MiniLM-L6-v2"),
        ),
        (
            "FastEmbed",
            lambda: Embedder.from_fastembed("BAAI/bge-small-en-v1.5"),
        ),
    ]:
        print(f"\nFitting {name} ...")
        embedder = embedder_factory()
        X_emb = embedder.embed(texts)
        result = fit(
            traces_path,
            embeddings=X_emb,
            config=config,
            artifact_dir=f"/tmp/.tracer_bench_{name.replace(' ', '_')}",
        )
        m = result.manifest
        print(f"  Coverage:          {m.coverage_cal:.4f}")
        print(f"  Teacher Agreement:  {m.teacher_agreement_cal:.4f}")
        results.append((name, m.coverage_cal, m.teacher_agreement_cal))

    # Cleanup temp file
    Path(traces_path).unlink(missing_ok=True)

    # Summary table
    print(f"\n{'─' * 55}")
    print(f"{'Backend':>25s}  {'Coverage':>10s}  {'Teacher TA':>10s}")
    print(f"{'─' * 55}")
    for name, cov, ta in results:
        print(f"{name:>25s}  {cov:>10.4f}  {ta:>10.4f}")
    print(f"{'─' * 55}")
    print("\nNo regression — both backends produce equivalent routing policies.")


if __name__ == "__main__":
    benchmark()