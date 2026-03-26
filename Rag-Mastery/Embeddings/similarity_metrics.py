"""
Similarity Metrics — Concept #3
=================================
Cosine similarity, Dot product, L2 distance.
When they agree. When they diverge. Which to use when.

Key insight:
    Normalized vectors → all three metrics give IDENTICAL rankings.
    Unnormalized vectors → dot product is biased toward high-magnitude vectors.

    dot product = cosine × ||a|| × ||b||
    So if ||a|| >> ||b||, dot product inflates a's score regardless of angle.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List


# ---------------------------------------------------------------------------
# The three metrics
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Angle between vectors. Scale-invariant."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def dot_product(a: np.ndarray, b: np.ndarray) -> float:
    """Angle × magnitude. Fast. Equals cosine when vectors are unit-normalized."""
    return float(np.dot(a, b))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance. Smaller = more similar."""
    return float(np.linalg.norm(a - b))


def l2_to_similarity(dist: float) -> float:
    """Convert L2 distance to 0-1 similarity for easier comparison."""
    return 1.0 / (1.0 + dist)


def normalize(v: np.ndarray) -> np.ndarray:
    """L2-normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v


def all_metrics(a: np.ndarray, b: np.ndarray, label: str = "") -> None:
    """Print all three metrics for a pair."""
    cos = cosine_similarity(a, b)
    dot = dot_product(a, b)
    l2  = l2_distance(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    print(f"  {label}")
    print(f"    ||a||={norm_a:.3f}  ||b||={norm_b:.3f}")
    print(f"    cosine={cos:.4f}  dot={dot:.4f}  l2={l2:.4f}")


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    model = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------------------------------------------------
    # EXPERIMENT 1: Normalized vectors — all metrics agree on ranking
    # all-MiniLM-L6-v2 outputs unit-normalized vectors
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 1: Normalized vectors — do all metrics agree?")
    print("=" * 65 + "\n")

    query = "How does RAG reduce hallucinations?"
    candidates = [
        "RAG grounds responses in retrieved documents, reducing hallucinations.",
        "Vector databases store embeddings for similarity search.",
        "The Eiffel Tower is in Paris.",
    ]

    embs = model.encode([query] + candidates, show_progress_bar=False)
    q_emb = embs[0]

    print(f"Query: '{query}'\n")
    for i, cand in enumerate(candidates):
        c_emb = embs[i + 1]
        cos = cosine_similarity(q_emb, c_emb)
        dot = dot_product(q_emb, c_emb)
        l2  = l2_distance(q_emb, c_emb)
        print(f"  Candidate {i+1}: '{cand[:55]}'")
        print(f"    cos={cos:.4f}  dot={dot:.4f}  l2={l2:.4f}")
        # Mathematical relationship check for normalized vectors:
        # l2^2 should equal 2 - 2*cosine
        l2_sq = l2 ** 2
        expected = 2 - 2 * cos
        print(f"    l2^2={l2_sq:.4f}  2-2*cos={expected:.4f}  match={abs(l2_sq-expected)<1e-4}")
        print()

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Unnormalized vectors — dot product diverges
    # Artificially scale one vector to show the magnitude bias
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 2: Unnormalized — dot product is biased by magnitude")
    print("=" * 65 + "\n")

    # Two candidates with same DIRECTION but different MAGNITUDE
    base = model.encode(["RAG retrieves relevant documents."], show_progress_bar=False)[0]
    same_direction_small = base.copy()           # original magnitude
    same_direction_large = base * 5.0            # 5x magnitude, same direction

    # Different direction (unrelated topic)
    different = model.encode(["The weather is sunny today."], show_progress_bar=False)[0]
    different_large = different * 10.0           # 10x magnitude, different direction

    q = model.encode(["How does RAG work?"], show_progress_bar=False)[0]

    print("  Same-direction, small magnitude (original):")
    all_metrics(q, same_direction_small, "  cosine should beat dot for this one")

    print()
    print("  Same-direction, 5x magnitude:")
    all_metrics(q, same_direction_large, "  dot inflated, cosine unchanged")

    print()
    print("  Different-direction, 10x magnitude:")
    all_metrics(q, different_large, "  dot is misleadingly high, cosine still low")

    print()
    print("  Key: dot product ranks '10x unrelated' vs '1x related' differently than cosine")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: L2 vs Cosine — same ranking when normalized?
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 3: L2 vs Cosine ranking on normalized vectors")
    print("=" * 65 + "\n")

    texts = [
        "RAG retrieves relevant documents before generating answers.",
        "Vector databases store high-dimensional embeddings.",
        "Chunking splits documents into smaller pieces.",
        "Python is popular for machine learning.",
        "The sky is blue.",
    ]

    embs = model.encode([query] + texts, show_progress_bar=False)
    q_emb = embs[0]
    t_embs = embs[1:]

    cos_ranked = sorted(
        [(cosine_similarity(q_emb, t_embs[i]), texts[i]) for i in range(len(texts))],
        reverse=True
    )
    l2_ranked = sorted(
        [(l2_distance(q_emb, t_embs[i]), texts[i]) for i in range(len(texts))],
        reverse=False     # lower L2 = more similar
    )

    print(f"  {'Rank':<6} {'Cosine':^30} {'L2 (lower=better)':^30}")
    print(f"  {'-'*66}")
    for i in range(len(texts)):
        cos_text = f"{cos_ranked[i][0]:.4f} {cos_ranked[i][1][:20]}"
        l2_text  = f"{l2_ranked[i][0]:.4f} {l2_ranked[i][1][:20]}"
        match = "SAME" if cos_ranked[i][1] == l2_ranked[i][1] else "DIFF"
        print(f"  {i+1:<6} {cos_text:<30} {l2_text:<30} {match}")

    # -------------------------------------------------------------------
    # EXPERIMENT 4: When to use dot product — speed benchmark
    # Dot product skips 2 norm computations → faster for large-scale retrieval
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 4: Speed — cosine vs dot product at scale")
    print("=" * 65 + "\n")

    import time

    N = 10_000
    dim = 384
    rng = np.random.default_rng(42)

    # Simulate a normalized vector DB (as sentence-transformers produces)
    vecs = rng.standard_normal((N, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs_norm = vecs / norms       # unit-normalized

    q_vec = rng.standard_normal(dim).astype(np.float32)
    q_vec = q_vec / np.linalg.norm(q_vec)

    # Cosine: normalize then dot
    start = time.perf_counter()
    for _ in range(100):
        vec_norms = np.linalg.norm(vecs, axis=1)
        cos_sims = vecs @ q_vec / (vec_norms * np.linalg.norm(q_vec))
    cos_time = (time.perf_counter() - start) / 100

    # Dot product on pre-normalized vectors (what FAISS/vector DBs do)
    start = time.perf_counter()
    for _ in range(100):
        dot_sims = vecs_norm @ q_vec
    dot_time = (time.perf_counter() - start) / 100

    print(f"  10k vectors, dim=384, averaged over 100 runs:")
    print(f"  Cosine (with norm computation) : {cos_time*1000:.3f}ms")
    print(f"  Dot product (pre-normalized)   : {dot_time*1000:.3f}ms")
    print(f"  Speedup: {cos_time/dot_time:.1f}x")
    print()
    print("  Production systems pre-normalize at indexing time.")
    print("  At query time: just dot product — fast and identical to cosine.")

    # -------------------------------------------------------------------
    # EXPERIMENT 5: The magnitude trap — a real-world warning
    # Longer documents have higher-magnitude embeddings in some models
    # Using dot product on un-normalized vectors biases toward verbose docs
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 5: Magnitude trap — long docs score higher with raw dot product")
    print("=" * 65 + "\n")

    short_doc = "RAG reduces hallucinations."
    long_doc = (
        "Retrieval-Augmented Generation, commonly known as RAG, is a powerful "
        "technique that reduces hallucinations in large language models by "
        "retrieving relevant documents from an external knowledge base and using "
        "them as context during generation, thereby grounding the model's "
        "responses in factual information rather than relying solely on "
        "parametric knowledge encoded during pretraining."
    )

    embs = model.encode([query, short_doc, long_doc], show_progress_bar=False)
    q_e, short_e, long_e = embs

    # Raw (model already normalizes, so let's artificially un-normalize for demo)
    short_raw = short_e * np.linalg.norm(short_e) * 2   # shorter text → smaller norm in raw models
    long_raw  = long_e  * np.linalg.norm(long_e)  * 8   # longer text → larger norm

    print("  Simulated unnormalized embeddings (longer doc has larger norm):\n")
    print(f"  Short doc: '{short_doc}'")
    print(f"    cosine={cosine_similarity(q_e, short_raw):.4f}  dot={dot_product(q_e, short_raw):.4f}")
    print(f"\n  Long doc: '{long_doc[:60]}...'")
    print(f"    cosine={cosine_similarity(q_e, long_raw):.4f}  dot={dot_product(q_e, long_raw):.4f}")
    print()
    print("  Cosine gives similar scores (correct — both about RAG).")
    print("  Raw dot product inflates the long doc (magnitude bias).")
    print("  Solution: always normalize before storing in vector DB.")
