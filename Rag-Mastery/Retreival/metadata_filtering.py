"""
Metadata Filtering
===================

WHAT THIS SOLVES:
  Pure vector search is blind to structure — dates, doc types, languages,
  departments, access levels. Metadata filtering scopes retrieval to a
  specific subset of the corpus BEFORE similarity is computed.

CORE IDEA:
  Metadata filter = hard constraint (binary in/out)
  Similarity score = soft ranking (continuous)

  Pipeline: filter candidates → rank survivors by similarity → return top K

FILTER TYPES COVERED:
  - Exact match      (topic == "retrieval")
  - Multi-value      (topic in ["retrieval", "embeddings"])
  - Numeric range    (year >= 2023)
  - Exclude          (source != "draft")
  - Combined AND/OR  (topic == "retrieval" AND year >= 2023)
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from top_k_retrieval import embed_texts, cosine_similarity


# ---------------------------------------------------------------------------
# ENRICHED CORPUS — chunks with structured metadata
# ---------------------------------------------------------------------------

# We extend the corpus with explicit metadata fields.
# In production these come from document parsing, tagging pipelines, or user-supplied labels.

CHUNKS = [
    {
        "id": 1,  "topic": "vector_databases",
        "source": "rag_guide",   "year": 2024, "difficulty": "beginner",   "language": "en",
        "text": "Vector databases store high-dimensional embeddings and enable fast approximate nearest neighbor search. They are the backbone of modern RAG systems. Popular options include Pinecone, Weaviate, Qdrant, and FAISS.",
    },
    {
        "id": 2,  "topic": "embeddings",
        "source": "ml_handbook", "year": 2023, "difficulty": "beginner",   "language": "en",
        "text": "Embeddings are dense numerical representations of text. They capture semantic meaning, allowing similar concepts to have similar vectors. Models like OpenAI's text-embedding-ada-002 or sentence-transformers generate these embeddings.",
    },
    {
        "id": 3,  "topic": "retrieval",
        "source": "rag_guide",   "year": 2024, "difficulty": "intermediate", "language": "en",
        "text": "Retrieval in RAG systems is fundamentally a ranking problem. Given a query, the system ranks all document chunks by relevance score and returns the top K. The quality of retrieval directly determines the quality of generated answers.",
    },
    {
        "id": 4,  "topic": "retrieval",
        "source": "rag_guide",   "year": 2024, "difficulty": "intermediate", "language": "en",
        "text": "Top-K retrieval selects the K most similar chunks based on cosine similarity between the query embedding and document embeddings. The value of K controls the tradeoff between precision and recall.",
    },
    {
        "id": 5,  "topic": "chunking",
        "source": "ml_handbook", "year": 2022, "difficulty": "beginner",   "language": "en",
        "text": "Chunking is the process of splitting documents into smaller pieces before embedding. Chunk size affects retrieval quality — too small loses context, too large introduces noise.",
    },
    {
        "id": 6,  "topic": "retrieval",
        "source": "rag_guide",   "year": 2024, "difficulty": "advanced",   "language": "en",
        "text": "A common failure mode in retrieval is returning duplicate or near-duplicate chunks. This happens when the same information appears multiple times, causing Top-K to return redundant results.",
    },
    {
        "id": 7,  "topic": "llm_generation",
        "source": "llm_docs",    "year": 2023, "difficulty": "intermediate", "language": "en",
        "text": "Large language models generate answers by conditioning on retrieved context. The quality of generation depends heavily on what was retrieved — irrelevant chunks cause hallucinations.",
    },
    {
        "id": 8,  "topic": "retrieval",
        "source": "rag_guide",   "year": 2024, "difficulty": "beginner",   "language": "en",
        "text": "Cosine similarity measures the angle between two vectors, not their magnitude. A score of 1.0 means identical direction, 0.0 means orthogonal. Most semantic similarities fall between 0.6 and 0.95.",
    },
    {
        "id": 9,  "topic": "hybrid_search",
        "source": "rag_guide",   "year": 2024, "difficulty": "advanced",   "language": "en",
        "text": "Hybrid search combines sparse retrieval (BM25, keyword-based) with dense retrieval (embeddings). BM25 is excellent for exact keyword matches, while dense retrieval handles semantic similarity.",
    },
    {
        "id": 10, "topic": "reranking",
        "source": "ml_handbook", "year": 2023, "difficulty": "advanced",   "language": "en",
        "text": "Reranking is a post-retrieval step where a cross-encoder rescores the top-K retrieved chunks. It improves precision significantly but adds latency.",
    },
    {
        "id": 11, "topic": "retrieval",
        "source": "rag_guide",   "year": 2022, "difficulty": "beginner",   "language": "en",
        "text": "The precision-recall tradeoff: a small K gives high precision but low recall. A large K gives high recall but low precision. Many irrelevant chunks enter the context at large K.",
    },
    {
        "id": 12, "topic": "embeddings",
        "source": "ml_handbook", "year": 2024, "difficulty": "intermediate", "language": "en",
        "text": "Embedding models are trained using contrastive learning — similar sentences are pushed close together in vector space. The quality of the embedding model directly affects retrieval performance.",
    },
    {
        "id": 13, "topic": "vector_databases",
        "source": "infra_docs",  "year": 2023, "difficulty": "intermediate", "language": "en",
        "text": "FAISS is an open-source library for efficient similarity search. It supports exact and approximate nearest neighbor search, suitable from small datasets to billion-scale systems.",
    },
    {
        "id": 14, "topic": "retrieval",
        "source": "rag_guide",   "year": 2024, "difficulty": "advanced",   "language": "en",
        "text": "Metadata filtering allows retrieval to be scoped to specific corpus subsets. Filtering by date, author, category, or language before vector similarity dramatically improves precision in large corpora.",
    },
    {
        "id": 15, "topic": "irrelevant",
        "source": "history_book","year": 2019, "difficulty": "beginner",   "language": "en",
        "text": "The French Revolution began in 1789 and fundamentally transformed European political structures. The storming of the Bastille became a symbol of resistance against tyranny.",
    },
]


# ---------------------------------------------------------------------------
# FILTER ENGINE
# ---------------------------------------------------------------------------

def apply_metadata_filter(chunks: list[dict], filters: dict) -> list[dict]:
    """
    Apply hard metadata filters to produce a candidate pool.

    Supported filter types (mirrors Pinecone/Weaviate filter syntax):
        Exact match :  {"topic": "retrieval"}
        Multi-value :  {"topic": {"$in": ["retrieval", "embeddings"]}}
        Exclude     :  {"source": {"$ne": "history_book"}}
        Range       :  {"year": {"$gte": 2023}}
                       {"year": {"$lte": 2023}}
        Combined    :  multiple keys → implicit AND

    Returns list of chunks that pass ALL filter conditions.
    """
    candidates = []
    for chunk in chunks:
        if _chunk_passes(chunk, filters):
            candidates.append(chunk)
    return candidates


def _chunk_passes(chunk: dict, filters: dict) -> bool:
    """Return True if chunk passes ALL filter conditions."""
    for field, condition in filters.items():
        value = chunk.get(field)

        if isinstance(condition, dict):
            # Operator-based condition
            for op, operand in condition.items():
                if op == "$in"  and value not in operand:       return False
                if op == "$nin" and value in operand:            return False
                if op == "$ne"  and value == operand:            return False
                if op == "$gte" and not (value >= operand):      return False
                if op == "$lte" and not (value <= operand):      return False
                if op == "$gt"  and not (value > operand):       return False
                if op == "$lt"  and not (value < operand):       return False
        else:
            # Exact match
            if value != condition:
                return False

    return True


# ---------------------------------------------------------------------------
# METADATA-FILTERED RETRIEVAL
# ---------------------------------------------------------------------------

def metadata_filtered_retrieval(
    query: str,
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    model: SentenceTransformer,
    filters: dict,
    k: int = 4,
    verbose: bool = True,
) -> list[dict]:
    """
    Two-stage retrieval:
      Stage 1 — Metadata filter  → candidate pool (hard constraint)
      Stage 2 — Top-K by cosine  → ranked results (soft ranking)

    Parameters:
        filters : dict of metadata conditions (see apply_metadata_filter)
    """
    # Stage 1: filter
    candidates = apply_metadata_filter(chunks, filters)

    if verbose:
        print(f"\n  Filter: {filters}")
        print(f"  Candidate pool: {len(candidates)}/{len(chunks)} chunks passed")

    if not candidates:
        print("  ⚠  No chunks passed the metadata filter. Check your filter conditions.")
        return []

    # Get indices of surviving chunks so we can look up their embeddings
    candidate_indices = [i for i, c in enumerate(chunks) if c in candidates]
    candidate_embeddings = chunk_embeddings[candidate_indices]

    # Stage 2: rank candidates by similarity
    query_vec = embed_texts(model, [query])[0]
    scores = cosine_similarity(query_vec, candidate_embeddings)
    ranked = np.argsort(scores)[::-1][:k]

    results = []
    for rank, pos in enumerate(ranked, start=1):
        chunk = dict(candidates[pos])
        chunk["score"] = float(scores[pos])
        chunk["rank"] = rank
        results.append(chunk)

    if verbose:
        _print_results(query, results, filters, k)

    return results


# ---------------------------------------------------------------------------
# PRETTY PRINTER
# ---------------------------------------------------------------------------

def _print_results(query: str, results: list[dict], filters: dict, k: int):
    print("\n" + "=" * 65)
    print(f"  QUERY  : {query}")
    print(f"  FILTER : {filters}")
    print(f"  K      : {k}  →  returned {len(results)} chunks")
    print("=" * 65)
    for r in results:
        bar = "█" * int(r["score"] * 30)
        meta = f"source={r['source']}  year={r['year']}  difficulty={r['difficulty']}"
        print(f"\n  Rank #{r['rank']}  [score: {r['score']:.4f}]  {meta}")
        print(f"  {bar}")
        print(f"  {r['text'][:115]}{'...' if len(r['text']) > 115 else ''}")
    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# EXPERIMENTS
# ---------------------------------------------------------------------------

def run_experiments(chunks, embeddings, model):

    query = "How does retrieval work in RAG?"

    # -----------------------------------------------------------------------
    # EXPERIMENT A: No filter vs topic filter — see precision improvement
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT A — No filter vs topic='retrieval' filter")
    print("  Observe how filtering scopes results to the right domain")
    print("▓" * 65)

    print("\n  [NO FILTER — all 15 chunks eligible]")
    from top_k_retrieval import top_k_retrieval
    top_k_retrieval(query, chunks, embeddings, model, k=4)

    print("\n  [FILTER: topic='retrieval' — 6 chunks eligible]")
    metadata_filtered_retrieval(query, chunks, embeddings, model,
        filters={"topic": "retrieval"}, k=4)
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT B: Year range filter — temporal scoping
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT B — Year range filter (only recent content)")
    print("  Simulates: 'only use docs from 2024 onward'")
    print("▓" * 65)

    metadata_filtered_retrieval(query, chunks, embeddings, model,
        filters={"year": {"$gte": 2024}}, k=4)
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT C: Combined filter (AND logic)
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT C — Combined filter: source + difficulty")
    print("  Simulates: 'only advanced content from rag_guide'")
    print("▓" * 65)

    metadata_filtered_retrieval(query, chunks, embeddings, model,
        filters={"source": "rag_guide", "difficulty": "advanced"}, k=4)
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT D: Multi-value filter ($in operator)
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT D — Multi-value filter ($in)")
    print("  Simulates: 'retrieval OR embeddings topics only'")
    print("▓" * 65)

    metadata_filtered_retrieval(query, chunks, embeddings, model,
        filters={"topic": {"$in": ["retrieval", "embeddings"]}}, k=5)
    input("  Press ENTER to continue...")

    # -----------------------------------------------------------------------
    # EXPERIMENT E: Over-filtering failure case
    # -----------------------------------------------------------------------
    print("\n" + "▓" * 65)
    print("  EXPERIMENT E — Over-filtering (too strict = empty pool)")
    print("  This is the #1 failure mode of metadata filtering")
    print("▓" * 65)

    metadata_filtered_retrieval(query, chunks, embeddings, model,
        filters={"topic": "retrieval", "source": "history_book"}, k=4)

    print("  → Empty pool! The filter was logically contradictory.")
    print("  → In production: always validate filter logic and handle empty pools.\n")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n  Loading model...")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    texts = [c["text"] for c in CHUNKS]
    embeddings = embed_texts(model, texts)
    print(f"  Loaded {len(CHUNKS)} chunks. Embedding shape: {embeddings.shape}\n")

    # Quick demo — source filter
    print("  === QUICK DEMO: Filter to rag_guide source, year >= 2024 ===")
    metadata_filtered_retrieval(
        query="How does retrieval ranking work in RAG?",
        chunks=CHUNKS,
        chunk_embeddings=embeddings,
        model=model,
        filters={"source": "rag_guide", "year": {"$gte": 2024}},
        k=4,
    )

    choice = input("  Run full interactive experiments? (y/n): ").strip().lower()
    if choice == "y":
        run_experiments(CHUNKS, embeddings, model)
