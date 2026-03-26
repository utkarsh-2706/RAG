"""
Similarity Threshold Filtering
================================

WHAT THIS SOLVES:
  Top-K always returns K chunks — even for irrelevant queries.
  Threshold filtering adds a minimum score gate: chunks below
  the threshold are rejected entirely, even if they're the "best" available.

CORE INSIGHT:
  Retrieval should be allowed to return ZERO results.
  It's better to say "I don't know" than to hallucinate from garbage context.

PARAMETERS:
  threshold : float in [0, 1] — minimum cosine similarity to accept a chunk
  k         : int             — max chunks to return (upper bound, not guaranteed)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Reuse parsing + embedding utilities from top_k_retrieval
import sys
sys.path.insert(0, str(Path(__file__).parent))
from top_k_retrieval import load_chunks, embed_texts, cosine_similarity


# ---------------------------------------------------------------------------
# CORE FUNCTION: Threshold + Top-K combined
# ---------------------------------------------------------------------------

def threshold_retrieval(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 5,
    threshold: float = 0.5,
    verbose: bool = True,
) -> list[dict]:
    """
    Threshold-filtered Top-K retrieval.

    Steps:
      1. Embed query
      2. Score all chunks via cosine similarity
      3. Filter out chunks below `threshold`
      4. Sort survivors by score descending
      5. Return top min(K, survivors) chunks

    Returns empty list if no chunk clears the threshold.
    """
    # Step 1 — embed & normalize query
    query_vec = embed_texts(model, [query])[0]

    # Step 2 — score all chunks
    scores = cosine_similarity(query_vec, chunk_embeddings)

    # Step 3 — apply threshold gate
    eligible_mask = scores >= threshold
    eligible_indices = np.where(eligible_mask)[0]

    # Step 4 — sort eligible chunks by score
    sorted_eligible = eligible_indices[np.argsort(scores[eligible_indices])[::-1]]

    # Step 5 — take top K
    top_indices = sorted_eligible[:k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        chunk = dict(chunks[idx])
        chunk["score"] = float(scores[idx])
        chunk["rank"] = rank
        results.append(chunk)

    if verbose:
        _print_results(query, results, k, threshold, len(chunks), scores)

    return results


# ---------------------------------------------------------------------------
# SCORE DISTRIBUTION INSPECTOR
# (Critical for calibrating your threshold)
# ---------------------------------------------------------------------------

def inspect_score_distribution(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
):
    """
    Show the FULL score distribution for a query.
    This is how you calibrate a threshold in practice —
    look for the natural gap between relevant and irrelevant scores.
    """
    query_vec = embed_texts(model, [query])[0]
    scores = cosine_similarity(query_vec, chunk_embeddings)
    sorted_idx = np.argsort(scores)[::-1]

    print("\n" + "=" * 65)
    print(f"  SCORE DISTRIBUTION for: \"{query}\"")
    print(f"  {'Rank':<6} {'Score':<8} {'Bar':<32} Topic")
    print("-" * 65)

    for rank, idx in enumerate(sorted_idx, start=1):
        score = scores[idx]
        bar_len = int(score * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        topic = chunks[idx]["topic"]
        # Mark chunks that feel like a natural "gap"
        gap_marker = " ◄ GAP?" if rank > 1 and (scores[sorted_idx[rank-2]] - score) > 0.08 else ""
        print(f"  {rank:<6} {score:<8.4f} {bar} {topic}{gap_marker}")

    print("=" * 65)
    print(f"\n  Min score : {scores.min():.4f}")
    print(f"  Max score : {scores.max():.4f}")
    print(f"  Mean score: {scores.mean():.4f}")
    print(f"  Std dev   : {scores.std():.4f}")
    print()


# ---------------------------------------------------------------------------
# PRETTY PRINTER
# ---------------------------------------------------------------------------

def _print_results(
    query: str,
    results: list[dict],
    k: int,
    threshold: float,
    total: int,
    all_scores: np.ndarray,
):
    rejected = int((all_scores < threshold).sum())
    print("\n" + "=" * 65)
    print(f"  QUERY     : {query}")
    print(f"  K         : {k}  |  Threshold: {threshold}  |  Corpus: {total} chunks")
    print(f"  Rejected  : {rejected} chunks below threshold")
    print(f"  Returned  : {len(results)} chunks")
    print("=" * 65)

    if not results:
        print("\n  ⚠  NO CHUNKS passed the threshold.")
        print("  → System should respond: 'I don't have relevant information.'")
        print("  → This is CORRECT behavior for an irrelevant query.\n")
        return

    for r in results:
        bar = "█" * int(r["score"] * 30)
        print(f"\n  Rank #{r['rank']}  [score: {r['score']:.4f}]  topic: {r['topic']}")
        print(f"  {bar}")
        print(f"  {r['text'][:115]}{'...' if len(r['text']) > 115 else ''}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------------

def run_experiments(chunks, embeddings, model):

    # -----------------------------------------------------------------------
    # EXPERIMENT A: Same relevant query — vary threshold, observe what's cut
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT A — Vary threshold on a RELEVANT query")
    print("  Observe how tighter thresholds reduce returned chunks")
    print("▓" * 65)

    query = "How does retrieval ranking work in RAG?"
    for thresh in [0.2, 0.4, 0.5, 0.6, 0.75]:
        threshold_retrieval(query, chunks, embeddings, model, k=5, threshold=thresh)
        input(f"  ↑ threshold={thresh}. Press ENTER to try next...")

    # -----------------------------------------------------------------------
    # EXPERIMENT B: Irrelevant query — Top-K vs Threshold
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT B — Irrelevant query")
    print("  Top-K returns garbage. Threshold correctly returns nothing.")
    print("▓" * 65)

    irr_query = "Tell me about Napoleon's military campaigns"

    print("\n  --- Top-K (no threshold) ---")
    from top_k_retrieval import top_k_retrieval
    top_k_retrieval(irr_query, chunks, embeddings, model, k=3)

    print("\n  --- Threshold=0.5 ---")
    threshold_retrieval(irr_query, chunks, embeddings, model, k=3, threshold=0.5)
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT C: Score distribution — find the natural gap
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT C — Inspect score distribution")
    print("  This is how you CALIBRATE a threshold in practice.")
    print("  Look for the natural 'gap' in scores — set threshold there.")
    print("▓" * 65)

    inspect_score_distribution(
        "What is the difference between sparse and dense retrieval?",
        chunks, embeddings, model
    )

    print("  Notice: scores form two clusters.")
    print("  Set threshold in the gap between them for optimal filtering.\n")

    # -----------------------------------------------------------------------
    # EXPERIMENT D: Threshold too aggressive — false negatives
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT D — Over-aggressive threshold (false negatives)")
    print("  When threshold is too high, REAL answers get rejected.")
    print("▓" * 65)

    threshold_retrieval(
        "What is K in Top-K retrieval?",
        chunks, embeddings, model,
        k=5, threshold=0.85  # artificially high
    )
    print("  The answer IS in the corpus (chunk 4, 11) but threshold blocked it.")
    print("  This is why threshold must be calibrated, not guessed.\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "sample_data.txt"

    print("\n  Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunks = load_chunks(str(DATA_PATH))
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"  Loaded {len(chunks)} chunks. Embedding shape: {embeddings.shape}\n")

    # Quick demo
    print("  === QUICK DEMO ===")
    threshold_retrieval(
        query="What is cosine similarity in vector search?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=4,
        threshold=0.5,
    )

    # Score distribution for calibration intuition
    inspect_score_distribution(
        "What is cosine similarity in vector search?",
        chunks, embeddings, model
    )

    choice = input("  Run full interactive experiments? (y/n): ").strip().lower()
    if choice == "y":
        run_experiments(chunks, embeddings, model)
