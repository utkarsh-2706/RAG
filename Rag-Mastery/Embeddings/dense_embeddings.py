"""
Dense Embeddings — Concept #1
==============================
Sentence-transformers: converting text into semantic vectors

What this file teaches:
    1. How to generate embeddings with sentence-transformers
    2. What the output actually looks like (shape, dtype, range)
    3. Cosine similarity as a quick proxy for meaning closeness
    4. Failure cases: negation, ambiguity, short vs long text
    5. Dimensionality intuition — does bigger always mean better?

Model used: all-MiniLM-L6-v2
    - 384 dimensions
    - ~80MB
    - Fast, good general-purpose baseline
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Core embedding function
# ---------------------------------------------------------------------------

def get_embeddings(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Convert a list of texts into a matrix of embeddings.

    Args:
        texts : list of strings to embed
        model : loaded SentenceTransformer model

    Returns:
        np.ndarray of shape (len(texts), embedding_dim)
        e.g. (5, 384) for all-MiniLM-L6-v2
    """
    return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.

    Formula:  cos(θ) = (a · b) / (||a|| × ||b||)

    Range: -1 (opposite) to +1 (identical direction)
    In practice for sentence embeddings: 0.0 to 1.0

    Note: measures ANGLE between vectors, not magnitude.
    Two vectors pointing the same direction = similarity 1.0
    regardless of how long they are.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def similarity_matrix(embeddings: np.ndarray) -> np.ndarray:
    """
    Compute all pairwise cosine similarities.
    Returns shape (N, N) — entry [i][j] = similarity between text i and text j.
    """
    # Normalize all rows to unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-9)
    # Matrix multiply = all pairwise dot products of normalized vectors = cosine sims
    return normalized @ normalized.T


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def inspect_embedding(text: str, embedding: np.ndarray) -> None:
    """Show what an embedding looks like — shape, range, statistics."""
    print(f"\nText   : '{text[:60]}'")
    print(f"Shape  : {embedding.shape}")
    print(f"Dtype  : {embedding.dtype}")
    print(f"Min    : {embedding.min():.4f}")
    print(f"Max    : {embedding.max():.4f}")
    print(f"Mean   : {embedding.mean():.4f}")
    print(f"Norm   : {np.linalg.norm(embedding):.4f}")
    print(f"First 8 dims: {embedding[:8].round(4)}")


def compare_pair(text_a: str, text_b: str, model: SentenceTransformer) -> float:
    """Embed two texts and return their cosine similarity."""
    embs = get_embeddings([text_a, text_b], model)
    sim = cosine_similarity(embs[0], embs[1])
    print(f"  A: '{text_a[:70]}'")
    print(f"  B: '{text_b[:70]}'")
    print(f"  Cosine similarity: {sim:.4f}")
    return sim


def run_similarity_grid(pairs: List[Tuple[str, str]], model: SentenceTransformer, label: str) -> None:
    """Run similarity on a list of (text_a, text_b) pairs."""
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}\n")
    sims = []
    for a, b in pairs:
        sim = compare_pair(a, b, model)
        sims.append(sim)
        print()
    return sims


# ---------------------------------------------------------------------------
# Main — five targeted experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    print("Loading model: all-MiniLM-L6-v2  (384 dimensions)")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    # -------------------------------------------------------------------
    # EXPERIMENT 1: What does an embedding actually look like?
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 1: Anatomy of an embedding")
    print("=" * 65)

    sample = "Retrieval-Augmented Generation reduces hallucinations in LLMs."
    emb = get_embeddings([sample], model)[0]
    inspect_embedding(sample, emb)

    # Key observation: norm is NOT 1.0 (not normalized yet)
    # Cosine similarity normalizes internally — but dot product would not

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Semantically similar pairs (different words, same meaning)
    # -------------------------------------------------------------------
    similar_pairs = [
        (
            "The patient suffered a heart attack last Tuesday.",
            "The man experienced a myocardial infarction on Tuesday.",
        ),
        (
            "Machine learning models learn patterns from data automatically.",
            "AI systems improve their performance by training on examples.",
        ),
        (
            "How do I reset my password?",
            "I forgot my login credentials and cannot access my account.",
        ),
    ]
    run_similarity_grid(similar_pairs, model, "EXP 2: Similar pairs (should be HIGH ~0.7-1.0)")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Dissimilar pairs (unrelated topics)
    # -------------------------------------------------------------------
    dissimilar_pairs = [
        (
            "Photosynthesis converts sunlight into glucose in plants.",
            "The Federal Reserve raised interest rates by 25 basis points.",
        ),
        (
            "She adopted a golden retriever puppy from the shelter.",
            "The database query took 45 seconds due to a missing index.",
        ),
    ]
    run_similarity_grid(dissimilar_pairs, model, "EXP 3: Dissimilar pairs (should be LOW ~0.0-0.3)")

    # -------------------------------------------------------------------
    # EXPERIMENT 4: FAILURE CASE — negation trap
    # Embeddings are trained on meaning, not logic.
    # "effective" and "not effective" embed very similarly because
    # they share the same topic words and context.
    # -------------------------------------------------------------------
    negation_pairs = [
        (
            "The drug is effective for treating depression.",
            "The drug is not effective for treating depression.",
        ),
        (
            "The system is secure against SQL injection attacks.",
            "The system is vulnerable to SQL injection attacks.",
        ),
    ]
    print(f"\n{'='*65}")
    print("  EXP 4: FAILURE CASE — Negation trap")
    print("  *** These SHOULD be dissimilar (opposite meaning)")
    print("  *** Watch how high the similarity is anyway")
    print(f"{'='*65}\n")
    for a, b in negation_pairs:
        compare_pair(a, b, model)
        print()

    # -------------------------------------------------------------------
    # EXPERIMENT 5: Short vs long — same meaning, very different length
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  EXP 5: Short vs long (same meaning, different length)")
    print(f"{'='*65}\n")

    short = "RAG reduces hallucinations."
    long = (
        "Retrieval-Augmented Generation reduces hallucinations by grounding "
        "language model responses in retrieved documents from an external "
        "knowledge base, ensuring the model has access to factual context "
        "before generating an answer."
    )
    compare_pair(short, long, model)

    # -------------------------------------------------------------------
    # EXPERIMENT 6: Dimensionality — what does 384 numbers encode?
    # Compare the same concept at different positions in the vector
    # -------------------------------------------------------------------
    print(f"\n{'='*65}")
    print("  EXP 6: Similarity ranking — one query vs multiple candidates")
    print(f"{'='*65}\n")

    query = "How does RAG work?"
    candidates = [
        "RAG retrieves relevant documents before generating answers.",          # very close
        "Retrieval-Augmented Generation combines search with generation.",      # close
        "LLMs can hallucinate facts that are not in their training data.",      # related
        "Vector databases store embeddings for similarity search.",             # somewhat related
        "Python is a popular programming language.",                            # unrelated
        "The Eiffel Tower is located in Paris, France.",                       # completely unrelated
    ]

    embs = get_embeddings([query] + candidates, model)
    query_emb = embs[0]
    cand_embs = embs[1:]

    print(f"Query: '{query}'\n")
    print(f"  {'Sim':>6}  Candidate")
    print(f"  {'-'*60}")
    sims = [(cosine_similarity(query_emb, c), candidates[i]) for i, c in enumerate(cand_embs)]
    for sim, cand in sorted(sims, reverse=True):
        bar = "#" * int(sim * 20)
        print(f"  {sim:.4f}  {bar:<20}  {cand[:55]}")

    print("\n")
    print("Key observations to make:")
    print("  1. Norm of the raw embedding (Exp 1) — is it 1.0?")
    print("  2. Similar pairs score (Exp 2) — what's the threshold for 'similar'?")
    print("  3. Negation pairs score (Exp 4) — why are opposite sentences so similar?")
    print("  4. Short vs long (Exp 5) — does length change the similarity?")
    print("  5. Ranking (Exp 6) — does the order feel semantically correct?")
