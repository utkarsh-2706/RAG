"""
OpenAI vs Open-source Embeddings — Concept #2
===============================================
Same task, different models. Compare quality, speed, and cost.

What this file teaches:
    1. How to run multiple open-source models side by side
    2. Do bigger models produce better similarity rankings?
    3. How to structure OpenAI embedding calls (requires OPENAI_API_KEY)
    4. Cost estimation for a real RAG workload
    5. When open-source is good enough vs when you need OpenAI
"""

from __future__ import annotations
import time
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict


# ---------------------------------------------------------------------------
# Cosine similarity (reused from concept 1)
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Open-source embeddings
# ---------------------------------------------------------------------------

def embed_open_source(texts: List[str], model_name: str) -> tuple[np.ndarray, float]:
    """
    Embed texts using a local sentence-transformer model.
    Returns (embeddings, elapsed_seconds).
    """
    model = SentenceTransformer(model_name)
    start = time.time()
    embs = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    elapsed = time.time() - start
    return embs, elapsed


# ---------------------------------------------------------------------------
# OpenAI embeddings (requires OPENAI_API_KEY env var)
# ---------------------------------------------------------------------------

def embed_openai(texts: List[str], model: str = "text-embedding-3-small") -> tuple[np.ndarray, float]:
    """
    Embed texts using OpenAI API.
    Set OPENAI_API_KEY environment variable before calling.

    Cost estimate:
        text-embedding-3-small: $0.02 per 1M tokens
        text-embedding-3-large: $0.13 per 1M tokens
    """
    try:
        from openai import OpenAI
        client = OpenAI()
        start = time.time()
        response = client.embeddings.create(input=texts, model=model)
        elapsed = time.time() - start
        embs = np.array([item.embedding for item in response.data])
        return embs, elapsed
    except ImportError:
        raise ImportError("pip install openai")
    except Exception as e:
        raise RuntimeError(f"OpenAI call failed: {e}")


# ---------------------------------------------------------------------------
# Cost estimator
# ---------------------------------------------------------------------------

def estimate_cost(
    num_documents: int,
    avg_tokens_per_doc: int,
    model: str = "text-embedding-3-small",
) -> None:
    """
    Estimate the cost to embed a knowledge base with OpenAI.

    Pricing (as of 2024):
        text-embedding-3-small : $0.020 / 1M tokens
        text-embedding-3-large : $0.130 / 1M tokens
        ada-002                : $0.100 / 1M tokens
    """
    prices = {
        "text-embedding-3-small": 0.020,
        "text-embedding-3-large": 0.130,
        "text-embedding-ada-002": 0.100,
    }
    total_tokens = num_documents * avg_tokens_per_doc
    price_per_million = prices.get(model, 0.020)
    cost = (total_tokens / 1_000_000) * price_per_million

    print(f"  Model          : {model}")
    print(f"  Documents      : {num_documents:,}")
    print(f"  Avg tokens/doc : {avg_tokens_per_doc}")
    print(f"  Total tokens   : {total_tokens:,}")
    print(f"  Estimated cost : ${cost:.4f}")


# ---------------------------------------------------------------------------
# Comparison runner
# ---------------------------------------------------------------------------

def rank_candidates(
    query: str,
    candidates: List[str],
    embeddings: np.ndarray,
    query_emb: np.ndarray,
) -> List[tuple[float, str]]:
    """Return candidates sorted by cosine similarity to query."""
    sims = [(cosine_sim(query_emb, embeddings[i]), candidates[i]) for i in range(len(candidates))]
    return sorted(sims, reverse=True)


def compare_models(
    query: str,
    candidates: List[str],
    models: Dict[str, str],     # label -> model_name
) -> None:
    """
    Run the same query + candidates through multiple models.
    Print rankings side by side to spot where models disagree.
    """
    print(f"\nQuery: '{query}'\n")
    all_rankings = {}

    for label, model_name in models.items():
        print(f"  Loading {label} ({model_name})...", end=" ", flush=True)
        texts = [query] + candidates
        embs, elapsed = embed_open_source(texts, model_name)
        query_emb = embs[0]
        cand_embs = embs[1:]
        ranked = rank_candidates(query, candidates, cand_embs, query_emb)
        all_rankings[label] = ranked
        print(f"done ({elapsed:.2f}s, dim={embs.shape[1]})")

    # Print rankings
    print()
    for label, ranked in all_rankings.items():
        print(f"  [{label}]")
        for rank, (sim, text) in enumerate(ranked, 1):
            print(f"    {rank}. {sim:.4f}  {text[:65]}")
        print()


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # -------------------------------------------------------------------
    # EXPERIMENT 1: Multi-model ranking comparison
    # Do bigger models rank candidates differently?
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 1: Ranking comparison across open-source models")
    print("=" * 65)

    query = "How does retrieval work in RAG systems?"
    candidates = [
        "RAG retrieves relevant documents using vector similarity search.",
        "Approximate nearest neighbor search finds close vectors fast.",
        "The generator uses retrieved context to produce grounded answers.",
        "Chunking splits documents before embedding and indexing them.",
        "Python is a popular language for machine learning.",
        "The Eiffel Tower was built in 1889.",
    ]

    models = {
        "MiniLM-L6 (384d)":  "all-MiniLM-L6-v2",
        "mpnet-base (768d)":  "all-mpnet-base-v2",
        "bge-small (384d)":   "BAAI/bge-small-en-v1.5",
    }

    compare_models(query, candidates, models)

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Speed benchmark — small vs large model
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 2: Speed benchmark (100 sentences)")
    print("=" * 65 + "\n")

    bench_texts = candidates * 17  # ~100 sentences
    for label, model_name in models.items():
        model = SentenceTransformer(model_name)
        start = time.time()
        _ = model.encode(bench_texts, show_progress_bar=False)
        elapsed = time.time() - start
        print(f"  {label:<22} {elapsed:.3f}s for {len(bench_texts)} texts")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Cost estimator — when does OpenAI become expensive?
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 3: OpenAI cost estimation")
    print("=" * 65 + "\n")

    scenarios = [
        ("Small KB (1k docs)",      1_000,   400),
        ("Medium KB (50k docs)",    50_000,  400),
        ("Large KB (500k docs)",    500_000, 400),
        ("Re-embed monthly (500k)", 500_000, 400),
    ]

    for scenario, docs, tokens in scenarios:
        print(f"  Scenario: {scenario}")
        estimate_cost(docs, tokens, "text-embedding-3-small")
        print()

    # -------------------------------------------------------------------
    # EXPERIMENT 4: The E5 prompt trick
    # E5 models expect a prefix: "query: " for queries, "passage: " for docs
    # Skipping it degrades performance significantly
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 4: E5 prompt prefix trick")
    print("  E5 models need 'query: ' and 'passage: ' prefixes")
    print("=" * 65 + "\n")

    e5_model = SentenceTransformer("intfloat/e5-base-v2")

    q = "How does RAG reduce hallucinations?"
    doc = "RAG grounds LLM responses in retrieved documents, reducing hallucinations."

    # Without prefix
    embs_no_prefix = e5_model.encode([q, doc], show_progress_bar=False)
    sim_no_prefix = cosine_sim(embs_no_prefix[0], embs_no_prefix[1])

    # With correct E5 prefix
    embs_prefix = e5_model.encode(
        [f"query: {q}", f"passage: {doc}"],
        show_progress_bar=False,
    )
    sim_with_prefix = cosine_sim(embs_prefix[0], embs_prefix[1])

    print(f"  Without prefix : {sim_no_prefix:.4f}")
    print(f"  With prefix    : {sim_with_prefix:.4f}")
    print(f"  Delta          : {sim_with_prefix - sim_no_prefix:+.4f}")
    print("\n  (Always read the model card — some models require specific input formats)")

    # -------------------------------------------------------------------
    # EXPERIMENT 5: OpenAI (runs only if OPENAI_API_KEY is set)
    # -------------------------------------------------------------------
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        print("\n" + "=" * 65)
        print("EXP 5: OpenAI text-embedding-3-small")
        print("=" * 65 + "\n")

        texts = [query] + candidates
        try:
            embs, elapsed = embed_openai(texts, model="text-embedding-3-small")
            print(f"  Dimensions : {embs.shape[1]}")
            print(f"  Time       : {elapsed:.3f}s (API round-trip)")
            query_emb = embs[0]
            cand_embs = embs[1:]
            ranked = rank_candidates(query, candidates, cand_embs, query_emb)
            print(f"\n  [OpenAI text-embedding-3-small]")
            for rank, (sim, text) in enumerate(ranked, 1):
                print(f"    {rank}. {sim:.4f}  {text[:65]}")
        except Exception as e:
            print(f"  OpenAI call failed: {e}")
    else:
        print("\n  (Skipping EXP 5 — set OPENAI_API_KEY to run OpenAI comparison)")
