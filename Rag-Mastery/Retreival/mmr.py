"""
MMR — Max Marginal Relevance
=============================

WHAT THIS SOLVES:
  Top-K and Threshold return the K most RELEVANT chunks.
  But if the top chunks are all near-duplicates, context window is wasted.

  MMR selects chunks that are:
    (a) relevant to the query, AND
    (b) different from chunks already selected

  It trades a little relevance for a lot of diversity.

CORE FORMULA:
  MMR_score(c) = λ · sim(c, query) - (1-λ) · max_sim(c, selected)

  λ = 1.0 → pure Top-K (no diversity)
  λ = 0.0 → pure diversity (ignores relevance)
  λ = 0.5 → balanced (common starting point)
  λ = 0.7 → mostly relevant, some diversity (production default)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from top_k_retrieval import load_chunks, embed_texts, cosine_similarity


# ---------------------------------------------------------------------------
# CORE FUNCTION: MMR Selection
# ---------------------------------------------------------------------------

def mmr_retrieval(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 5,
    lambda_val: float = 0.7,
    verbose: bool = True,
) -> list[dict]:
    """
    Max Marginal Relevance retrieval.

    Greedy iterative selection:
      - Each step picks the chunk that maximizes MMR score
      - MMR score balances query-relevance vs similarity to already-selected chunks
      - λ controls the relevance/diversity tradeoff

    Parameters:
        query          : user query string
        chunks         : parsed chunk dicts
        chunk_embeddings: precomputed normalized embeddings (N x D)
        model          : embedding model
        k              : number of chunks to select
        lambda_val     : diversity dial [0.0=diverse, 1.0=relevant only]
        verbose        : print selection process

    Returns:
        List of K selected chunks with score, mmr_score, and rank
    """
    n = len(chunks)
    k = min(k, n)

    # Step 0 — embed + normalize query
    query_vec = embed_texts(model, [query])[0]

    # Relevance scores: sim(chunk_i, query) — fixed, computed once
    relevance_scores = cosine_similarity(query_vec, chunk_embeddings)  # shape: (N,)

    selected_indices = []          # indices of already-selected chunks (in order)
    remaining_indices = list(range(n))  # indices not yet selected

    selection_log = []  # for verbose output

    for step in range(k):
        best_idx = None
        best_mmr = -np.inf

        for idx in remaining_indices:
            relevance = relevance_scores[idx]

            # Redundancy penalty: similarity to the most similar selected chunk
            if len(selected_indices) == 0:
                # First selection: no penalty — pure relevance
                redundancy = 0.0
            else:
                # sim(this_chunk, each_selected_chunk) → take the max
                selected_vecs = chunk_embeddings[selected_indices]   # shape: (M x D)
                sims_to_selected = selected_vecs @ chunk_embeddings[idx]  # dot = cosine (normalized)
                redundancy = float(sims_to_selected.max())

            mmr_score = lambda_val * relevance - (1 - lambda_val) * redundancy

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_idx = idx
                best_relevance = relevance
                best_redundancy = redundancy

        selected_indices.append(best_idx)
        remaining_indices.remove(best_idx)

        selection_log.append({
            "rank": step + 1,
            "idx": best_idx,
            "chunk": chunks[best_idx],
            "relevance": float(best_relevance),
            "redundancy": float(best_redundancy),
            "mmr_score": float(best_mmr),
        })

    results = []
    for entry in selection_log:
        chunk = dict(entry["chunk"])
        chunk["rank"] = entry["rank"]
        chunk["score"] = entry["relevance"]       # raw relevance (for comparison with Top-K)
        chunk["redundancy"] = entry["redundancy"] # how similar to prev selections
        chunk["mmr_score"] = entry["mmr_score"]   # final MMR score used to pick this
        results.append(chunk)

    if verbose:
        _print_results(query, results, k, lambda_val)

    return results


# ---------------------------------------------------------------------------
# SIDE-BY-SIDE COMPARISON: Top-K vs MMR
# ---------------------------------------------------------------------------

def compare_topk_vs_mmr(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 5,
    lambda_val: float = 0.7,
):
    """
    Run Top-K and MMR on the same query and print results side by side.
    This is the clearest way to see MMR's diversity effect.
    """
    from top_k_retrieval import top_k_retrieval

    print("\n" + "═" * 65)
    print(f"  SIDE-BY-SIDE: Top-K vs MMR  (K={k}, λ={lambda_val})")
    print(f"  QUERY: {query}")
    print("═" * 65)

    topk_results = top_k_retrieval(query, chunks, chunk_embeddings, model, k=k, verbose=False)
    mmr_results  = mmr_retrieval(query, chunks, chunk_embeddings, model, k=k, lambda_val=lambda_val, verbose=False)

    topk_ids = [r["id"] for r in topk_results]
    mmr_ids  = [r["id"] for r in mmr_results]

    print(f"\n  {'Rank':<6} {'TOP-K chunk id (topic)':<30} {'MMR chunk id (topic)':<30}")
    print("  " + "-" * 60)
    for i in range(k):
        tk = topk_results[i]
        mm = mmr_results[i]
        same = "  ✓" if tk["id"] == mm["id"] else "  ←different"
        print(f"  {i+1:<6} #{tk['id']:<4} {tk['topic']:<24}  #{mm['id']:<4} {mm['topic']:<24}{same}")

    overlap = set(topk_ids) & set(mmr_ids)
    print(f"\n  Overlap: {len(overlap)}/{k} chunks in common")
    print(f"  MMR introduced {k - len(overlap)} new chunk(s) that Top-K missed")

    new_chunks = [r for r in mmr_results if r["id"] not in topk_ids]
    if new_chunks:
        print("\n  Chunks MMR found that Top-K missed:")
        for c in new_chunks:
            print(f"    → #{c['id']} [{c['topic']}] relevance={c['score']:.4f}  redundancy={c['redundancy']:.4f}")
            print(f"       {c['text'][:100]}...")
    print("═" * 65 + "\n")


# ---------------------------------------------------------------------------
# PRETTY PRINTER
# ---------------------------------------------------------------------------

def _print_results(query: str, results: list[dict], k: int, lambda_val: float):
    print("\n" + "=" * 65)
    print(f"  QUERY  : {query}")
    print(f"  K={k}  λ={lambda_val}  ({'pure relevance=Top-K' if lambda_val==1.0 else 'balanced' if lambda_val==0.5 else 'diversity-biased' if lambda_val < 0.5 else 'mostly relevant'})")
    print("=" * 65)
    for r in results:
        rel_bar  = "█" * int(r["score"] * 20)
        red_bar  = "░" * int(r["redundancy"] * 20)
        print(f"\n  Rank #{r['rank']}  topic: {r['topic']}")
        print(f"  relevance  : {r['score']:.4f}  {rel_bar}")
        print(f"  redundancy : {r['redundancy']:.4f}  {red_bar}")
        print(f"  mmr_score  : {r['mmr_score']:.4f}")
        print(f"  {r['text'][:115]}{'...' if len(r['text']) > 115 else ''}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------------

def run_experiments(chunks, embeddings, model):

    # -----------------------------------------------------------------------
    # EXPERIMENT A: Vary λ — see the relevance/diversity dial in action
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT A — Vary λ on a retrieval query")
    print("  λ=1.0 is pure Top-K. λ=0.0 is pure diversity.")
    print("▓" * 65)

    query = "How does retrieval ranking work in RAG?"
    for lam in [1.0, 0.7, 0.5, 0.2]:
        mmr_retrieval(query, chunks, embeddings, model, k=5, lambda_val=lam)
        input(f"  ↑ λ={lam}. Notice the redundancy scores. Press ENTER for next λ...")

    # -----------------------------------------------------------------------
    # EXPERIMENT B: Side-by-side Top-K vs MMR
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT B — Top-K vs MMR side by side")
    print("  Which chunks does MMR swap out? Are the new ones useful?")
    print("▓" * 65)

    compare_topk_vs_mmr(
        query="How does retrieval ranking work in RAG?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=5,
        lambda_val=0.7,
    )
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT C: Broad query — MMR gives better coverage
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT C — Broad query: MMR provides topic coverage")
    print("  Top-K clusters on one subtopic. MMR spreads across subtopics.")
    print("▓" * 65)

    compare_topk_vs_mmr(
        query="What are the main components of a RAG system?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=6,
        lambda_val=0.6,
    )
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT D: λ=0.0 — pure diversity, relevance ignored
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT D — λ=0.0 (pure diversity)")
    print("  MMR ignores relevance entirely. Demonstrates the failure case.")
    print("▓" * 65)

    mmr_retrieval(
        query="How does retrieval ranking work in RAG?",
        chunks=chunks, chunk_embeddings=embeddings, model=model,
        k=5, lambda_val=0.0
    )
    print("  Notice: completely unrelated chunks may appear.")
    print("  This shows why λ must never be set to 0.0 in production.\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "sample_data.txt"

    print("\n  Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    chunks = load_chunks(str(DATA_PATH))
    texts  = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"  Loaded {len(chunks)} chunks. Embedding shape: {embeddings.shape}\n")

    # Quick demo
    print("  === QUICK DEMO: MMR with λ=0.7, K=5 ===")
    mmr_retrieval(
        query="How does retrieval ranking work in RAG?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=5,
        lambda_val=0.7,
    )

    print("  === SIDE-BY-SIDE: Top-K vs MMR ===")
    compare_topk_vs_mmr(
        query="How does retrieval ranking work in RAG?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=5,
        lambda_val=0.7,
    )

    choice = input("  Run full interactive experiments? (y/n): ").strip().lower()
    if choice == "y":
        run_experiments(chunks, embeddings, model)
