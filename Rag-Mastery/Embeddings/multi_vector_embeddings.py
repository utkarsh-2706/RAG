"""
Multi-Vector Embeddings — Concept #5
======================================
ColBERT-style: one vector per token instead of one vector per sentence.

Single-vector problem:
    Mean-pooling loses token-level detail. A 10-token sentence collapses
    into one vector — the individual token meanings are averaged out.

Multi-vector solution:
    Keep ALL token vectors. Score query vs document using MaxSim:
        score = sum over query tokens of max(sim(q_token, d_token) for all d_tokens)

    Each query token finds its best-matching document token.
    No information is destroyed by pooling.

This file implements both approaches from scratch to make the
difference tangible.
"""

from __future__ import annotations
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Cosine similarity helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute cosine similarity between every row in A and every row in B.
    Returns shape (len(A), len(B)).
    Used in MaxSim: rows = query tokens, cols = doc tokens.
    """
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


# ---------------------------------------------------------------------------
# Approach 1: Single-vector (baseline)
# ---------------------------------------------------------------------------

def single_vector_score(
    query: str,
    doc: str,
    model: SentenceTransformer,
) -> float:
    """Standard mean-pooled embedding, cosine similarity."""
    embs = model.encode([query, doc], show_progress_bar=False)
    return cosine_sim(embs[0], embs[1])


# ---------------------------------------------------------------------------
# Approach 2: Multi-vector — token-level embeddings
# ---------------------------------------------------------------------------

def get_token_embeddings(
    text: str,
    tokenizer,
    model,
    max_length: int = 64,
) -> Tuple[np.ndarray, List[str]]:
    """
    Get one embedding vector per token (no pooling).

    Returns:
        token_embeddings : shape (num_tokens, hidden_dim)
        tokens           : list of decoded token strings (for inspection)
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False,
    )
    with torch.no_grad():
        outputs = model(**inputs)

    # outputs.last_hidden_state: shape (1, seq_len, hidden_dim)
    token_embs = outputs.last_hidden_state.squeeze(0).numpy()   # (seq_len, hidden_dim)

    # Decode token ids back to readable strings
    token_ids = inputs["input_ids"].squeeze(0).tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)

    return token_embs, tokens


def maxsim_score(
    query_embs: np.ndarray,
    doc_embs: np.ndarray,
) -> float:
    """
    ColBERT MaxSim scoring.

    For each query token, find the maximum cosine similarity
    with any document token, then sum over all query tokens.

    Shape:
        query_embs : (n_query_tokens, dim)
        doc_embs   : (n_doc_tokens, dim)

    Returns scalar score. Higher = more relevant.
    """
    # sim_matrix[i][j] = cosine sim between query token i and doc token j
    sim_matrix = cosine_matrix(query_embs, doc_embs)   # (n_q, n_d)

    # For each query token, take the best-matching doc token
    max_per_query_token = sim_matrix.max(axis=1)        # (n_q,)

    # Sum across all query tokens
    return float(max_per_query_token.sum())


def multi_vector_score(
    query: str,
    doc: str,
    tokenizer,
    model,
) -> Tuple[float, np.ndarray]:
    """
    Score query vs doc using multi-vector MaxSim.
    Returns (score, sim_matrix) for visualization.
    """
    q_embs, q_tokens = get_token_embeddings(query, tokenizer, model)
    d_embs, d_tokens = get_token_embeddings(doc, tokenizer, model)
    sim_matrix = cosine_matrix(q_embs, d_embs)
    score = float(sim_matrix.max(axis=1).sum())
    return score, sim_matrix, q_tokens, d_tokens


# ---------------------------------------------------------------------------
# Visualization helpers
# ---------------------------------------------------------------------------

def print_maxsim_breakdown(
    q_tokens: List[str],
    d_tokens: List[str],
    sim_matrix: np.ndarray,
) -> None:
    """
    Show which document token each query token matched with.
    This makes MaxSim intuitive.
    """
    print(f"\n  MaxSim breakdown:")
    print(f"  {'Query token':<20} {'Best doc match':<25} {'Score':>8}")
    print(f"  {'-'*55}")
    total = 0.0
    for i, qt in enumerate(q_tokens):
        best_j = int(sim_matrix[i].argmax())
        best_score = float(sim_matrix[i][best_j])
        total += best_score
        dt = d_tokens[best_j] if best_j < len(d_tokens) else "?"
        print(f"  {qt:<20} {dt:<25} {best_score:>8.4f}")
    print(f"  {'':20} {'Total (MaxSim score)':>25} {total:>8.4f}")


def compare_approaches(
    query: str,
    candidates: List[str],
    st_model: SentenceTransformer,
    tokenizer,
    hf_model,
    label: str = "",
) -> None:
    """Run both approaches and compare rankings."""
    print(f"\n  {label}")
    print(f"  Query: '{query}'\n")
    print(f"  {'Single-vec':>12}  {'Multi-vec':>12}  Candidate")
    print(f"  {'-'*70}")

    results = []
    for doc in candidates:
        sv = single_vector_score(query, doc, st_model)
        mv, _, _, _ = multi_vector_score(query, doc, tokenizer, hf_model)
        results.append((sv, mv, doc))

    # Sort by single-vector score
    sv_ranked = sorted(results, key=lambda x: x[0], reverse=True)
    mv_ranked = sorted(results, key=lambda x: x[1], reverse=True)

    for sv, mv, doc in sorted(results, key=lambda x: x[0], reverse=True):
        sv_rank = next(i+1 for i,r in enumerate(sv_ranked) if r[2]==doc)
        mv_rank = next(i+1 for i,r in enumerate(mv_ranked) if r[2]==doc)
        flag = " <-- DISAGREE" if abs(sv_rank - mv_rank) >= 2 else ""
        print(f"  {sv:>12.4f}  {mv:>12.4f}  [{sv_rank}vs{mv_rank}] '{doc[:50]}'{flag}")


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModel

    # Load models
    # Single-vector: sentence-transformers (mean-pooled)
    print("Loading models...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Multi-vector: raw BERT (no pooling)
    # We use the same base model for fair comparison
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_model  = AutoModel.from_pretrained(model_name)
    hf_model.eval()
    print("Models loaded.\n")

    # -------------------------------------------------------------------
    # EXPERIMENT 1: Token-level inspection
    # What does a multi-vector representation look like?
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 1: Token-level embeddings — what does multi-vector look like?")
    print("=" * 65 + "\n")

    text = "RAG retrieves documents using vector search."
    embs, tokens = get_token_embeddings(text, tokenizer, hf_model)
    print(f"  Text: '{text}'")
    print(f"  Single vector shape : (384,)")
    print(f"  Multi vector shape  : {embs.shape}  <- one row per token")
    print(f"\n  Tokens: {tokens}")
    print(f"\n  Each token has its own 384-dim vector - context-aware.")
    print(f"  'RAG' here encodes differently than 'RAG' in a cooking recipe.")

    # -------------------------------------------------------------------
    # EXPERIMENT 2: MaxSim breakdown — see exactly which tokens matched
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 2: MaxSim breakdown — which doc token matches each query token?")
    print("=" * 65)

    query = "vector similarity search"
    doc   = "RAG uses cosine similarity to find the nearest embedding vectors."

    _, sim_matrix, q_tokens, d_tokens = multi_vector_score(query, doc, tokenizer, hf_model)
    print(f"\n  Query: '{query}'")
    print(f"  Doc  : '{doc}'")
    print_maxsim_breakdown(q_tokens, d_tokens, sim_matrix)

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Where single-vector fails, multi-vector recovers
    # Case: query contains a rare/specific token that gets diluted in pooling
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 3: Ranking comparison — where do they disagree?")
    print("=" * 65)

    query = "HNSW graph traversal for nearest neighbor search"
    candidates = [
        "HNSW builds layered proximity graphs for fast ANN retrieval.",
        "Approximate nearest neighbor algorithms use graph structures.",
        "Vector databases index embeddings for similarity queries.",
        "Chunking splits documents before embedding and storage.",
        "The weather forecast shows rain tomorrow.",
    ]
    compare_approaches(query, candidates, st_model, tokenizer, hf_model,
                       label="HNSW technical query")

    # -------------------------------------------------------------------
    # EXPERIMENT 4: Single-token query — where multi-vector shines
    # A short query with one precise technical term
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 4: Precise single-term query")
    print("=" * 65)

    query2 = "MaxSim"
    candidates2 = [
        "MaxSim scores each query token against the best-matching document token.",
        "Cosine similarity measures the angle between two vectors.",
        "ColBERT uses late interaction with per-token embeddings.",
        "Mean pooling compresses all token vectors into one.",
        "Paris is the capital of France.",
    ]
    compare_approaches(query2, candidates2, st_model, tokenizer, hf_model,
                       label="Precise single-term: 'MaxSim'")

    # -------------------------------------------------------------------
    # EXPERIMENT 5: Storage cost — the tradeoff
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 5: Storage cost comparison")
    print("=" * 65 + "\n")

    avg_tokens_per_chunk = 64
    num_chunks = 100_000
    dim = 128   # ColBERT uses 128-dim (compressed from 768)
    bytes_per_float = 4

    single_vec_mb = (num_chunks * 384 * bytes_per_float) / (1024**2)
    multi_vec_mb  = (num_chunks * avg_tokens_per_chunk * dim * bytes_per_float) / (1024**2)

    print(f"  {num_chunks:,} chunks, avg {avg_tokens_per_chunk} tokens, ColBERT dim={dim}\n")
    print(f"  Single-vector  : {single_vec_mb:.1f} MB   (1 vector × 384 dims)")
    print(f"  Multi-vector   : {multi_vec_mb:.1f} MB  ({avg_tokens_per_chunk} vectors × {dim} dims)")
    print(f"  Ratio          : {multi_vec_mb/single_vec_mb:.1f}x more storage\n")
    print("  ColBERT compresses token vectors to 128-dim to control this cost.")
    print("  Still ~21x more storage — the price of token-level precision.")
