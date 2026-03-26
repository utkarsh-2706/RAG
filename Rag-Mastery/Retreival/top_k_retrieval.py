"""
Top-K Retrieval — Baseline Retrieval for RAG Systems
=====================================================

WHAT THIS FILE DOES:
  Simulates the most fundamental retrieval mechanism in RAG:
  given a query, rank all document chunks by cosine similarity
  and return the top K most similar ones.

CORE INSIGHT:
  Retrieval is NOT just "find matching documents."
  It is a RANKING problem — every chunk gets a score,
  and we return the top K by that score.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. LOAD & PARSE CORPUS
# ---------------------------------------------------------------------------

def load_chunks(filepath: str) -> list[dict]:
    """
    Parse sample_data.txt into a list of chunk dicts.
    Each chunk has: id, topic, text
    """
    text = Path(filepath).read_text(encoding="utf-8")
    raw_chunks = text.strip().split("---")

    chunks = []
    for block in raw_chunks:
        block = block.strip()
        if not block or block.startswith("#"):
            continue

        chunk = {}
        for line in block.splitlines():
            line = line.strip()
            if line.startswith("CHUNK_ID:"):
                chunk["id"] = int(line.split(":")[1].strip())
            elif line.startswith("TOPIC:"):
                chunk["topic"] = line.split(":")[1].strip()
            elif line.startswith("TEXT:"):
                chunk["text"] = line[5:].strip()

        if "id" in chunk and "text" in chunk:
            chunks.append(chunk)

    return chunks


# ---------------------------------------------------------------------------
# 2. EMBED CHUNKS + QUERY
# ---------------------------------------------------------------------------

def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    """Return L2-normalized embeddings so dot product == cosine similarity."""
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    # Normalize → cosine similarity becomes a simple dot product
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


# ---------------------------------------------------------------------------
# 3. COSINE SIMILARITY
# ---------------------------------------------------------------------------

def cosine_similarity(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    """
    Since both vectors are already L2-normalized,
    cosine similarity = dot product.

    Result shape: (num_docs,)  — one score per chunk.
    """
    return doc_vecs @ query_vec  # matrix × vector → score per doc


# ---------------------------------------------------------------------------
# 4. TOP-K RETRIEVAL — THE CORE FUNCTION
# ---------------------------------------------------------------------------

def top_k_retrieval(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    k: int = 3,
    verbose: bool = True,
) -> list[dict]:
    """
    Core Top-K Retrieval:
    1. Embed the query
    2. Compute cosine similarity against all chunk embeddings
    3. Sort by score descending
    4. Return top K chunks with their scores

    Parameters:
        query             : the user's question
        chunks            : list of parsed chunk dicts
        chunk_embeddings  : precomputed, normalized embeddings (shape: N x D)
        model             : embedding model
        k                 : number of chunks to return
        verbose           : print ranked results

    Returns:
        List of top-K chunk dicts, each augmented with 'score' and 'rank'
    """
    # Step 1 — embed the query (normalize it too)
    query_vec = embed_texts(model, [query])[0]

    # Step 2 — score every chunk
    scores = cosine_similarity(query_vec, chunk_embeddings)

    # Step 3 — rank: argsort descending
    ranked_indices = np.argsort(scores)[::-1]

    # Step 4 — take top K
    results = []
    for rank, idx in enumerate(ranked_indices[:k], start=1):
        chunk = dict(chunks[idx])       # copy so we don't mutate original
        chunk["score"] = float(scores[idx])
        chunk["rank"] = rank
        results.append(chunk)

    if verbose:
        print_results(query, results, k, len(chunks))

    return results


# ---------------------------------------------------------------------------
# 5. PRETTY PRINTER
# ---------------------------------------------------------------------------

def print_results(query: str, results: list[dict], k: int, total: int):
    print("\n" + "=" * 65)
    print(f"  QUERY : {query}")
    print(f"  TOP-K : K={k}  |  Corpus size: {total} chunks")
    print("=" * 65)
    for r in results:
        bar = "█" * int(r["score"] * 30)
        print(f"\n  Rank #{r['rank']}  [score: {r['score']:.4f}]  topic: {r['topic']}")
        print(f"  {bar}")
        print(f"  {r['text'][:120]}{'...' if len(r['text']) > 120 else ''}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# 6. EXPERIMENTS — run these to build intuition
# ---------------------------------------------------------------------------

def run_experiments(chunks, embeddings, model):
    """
    Three experiments to build intuition around K and ranking.
    """

    # --- Experiment A: Relevant query, vary K ---
    print("\n" + "▓" * 65)
    print("  EXPERIMENT A — Relevant query, vary K")
    print("  Observe how precision drops as K increases")
    print("▓" * 65)

    query_a = "How does retrieval ranking work in RAG?"

    for k in [1, 3, 5, 8]:
        top_k_retrieval(query_a, chunks, embeddings, model, k=k)
        input(f"  ↑ K={k} shown above. Press ENTER to see K={k+1 if k < 8 else 'next exp'}...")

    # --- Experiment B: Partially relevant query ---
    print("\n" + "▓" * 65)
    print("  EXPERIMENT B — Broad query, observe topic scatter")
    print("  Notice how K=5 starts pulling in tangentially related chunks")
    print("▓" * 65)

    query_b = "What are embeddings and how are they used?"
    top_k_retrieval(query_b, chunks, embeddings, model, k=5)
    input("  Press ENTER to continue to Experiment C...")

    # --- Experiment C: Completely irrelevant query ---
    print("\n" + "▓" * 65)
    print("  EXPERIMENT C — Irrelevant query (French history)")
    print("  TOP-K always returns K results — even if all scores are low!")
    print("  This is a KEY FAILURE of pure Top-K.")
    print("▓" * 65)

    query_c = "Tell me about Napoleon's military campaigns"
    top_k_retrieval(query_c, chunks, embeddings, model, k=3)

    print("\n  NOTICE: Even with an irrelevant query, Top-K returned 3 chunks.")
    print("  The scores are low (~0.2–0.4), but the system still retrieved them.")
    print("  This is why Similarity Threshold Filtering exists — coming next!\n")


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATA_PATH = Path(__file__).parent / "sample_data.txt"

    print("\n  Loading embedding model (sentence-transformers/all-MiniLM-L6-v2)...")
    print("  (First run will download ~90MB model — subsequent runs are instant)\n")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("  Parsing corpus...")
    chunks = load_chunks(str(DATA_PATH))
    print(f"  Loaded {len(chunks)} chunks.\n")

    print("  Embedding all chunks (precompute once, reuse for all queries)...")
    texts = [c["text"] for c in chunks]
    embeddings = embed_texts(model, texts)
    print(f"  Embedding matrix shape: {embeddings.shape}\n")

    # -----------------------------------------------------------------------
    # QUICK DEMO — single query, K=3
    # -----------------------------------------------------------------------
    print("  === QUICK DEMO: Single query, K=3 ===")
    top_k_retrieval(
        query="What is Top-K retrieval and how does it affect precision?",
        chunks=chunks,
        chunk_embeddings=embeddings,
        model=model,
        k=3,
    )

    # -----------------------------------------------------------------------
    # FULL EXPERIMENTS — interactive, step-by-step
    # -----------------------------------------------------------------------
    choice = input("  Run full interactive experiments? (y/n): ").strip().lower()
    if choice == "y":
        run_experiments(chunks, embeddings, model)
