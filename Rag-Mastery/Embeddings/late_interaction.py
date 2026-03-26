"""
Late Interaction vs Single Vector — Concept #6
================================================
The architecture triangle: bi-encoder vs cross-encoder vs late interaction.

This is the most interview-critical concept in embeddings.

Three architectures:
    1. Bi-encoder    : encode separately, one dot product. Fast, indexable.
    2. Cross-encoder : encode together, full attention. Slow, highest quality.
    3. Late interaction (ColBERT): encode separately, MaxSim at query time.

Production pattern:
    bi-encoder retrieve (top-k) -> cross-encoder re-rank -> LLM
"""

from __future__ import annotations
import time
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
    return A_norm @ B_norm.T


# ---------------------------------------------------------------------------
# Architecture 1: Bi-encoder (single vector)
# ---------------------------------------------------------------------------

def biencoder_scores(
    query: str,
    docs: List[str],
    model: SentenceTransformer,
) -> List[float]:
    """
    Encode query and all docs independently.
    Score = cosine similarity between single vectors.
    """
    all_texts = [query] + docs
    embs = model.encode(all_texts, show_progress_bar=False)
    q_emb = embs[0]
    return [cosine_sim(q_emb, embs[i+1]) for i in range(len(docs))]


# ---------------------------------------------------------------------------
# Architecture 2: Cross-encoder
# ---------------------------------------------------------------------------

def crossencoder_scores(
    query: str,
    docs: List[str],
    model: CrossEncoder,
) -> List[float]:
    """
    Feed each (query, doc) pair through the model together.
    Full attention between query and document tokens.
    Returns a relevance score — not a cosine similarity, but a raw logit.

    Key difference from bi-encoder:
        - The model SEES the query when encoding the document
        - This allows it to catch subtle relevance signals
        - But it cannot pre-compute doc representations at index time
    """
    pairs = [(query, doc) for doc in docs]
    scores = model.predict(pairs, show_progress_bar=False)
    return scores.tolist()


# ---------------------------------------------------------------------------
# Architecture 3: Late interaction (MaxSim)
# ---------------------------------------------------------------------------

def get_token_embeddings(text: str, tokenizer, model, max_length: int = 64) -> np.ndarray:
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       max_length=max_length, padding=False)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).numpy()


def maxsim_scores(
    query: str,
    docs: List[str],
    tokenizer,
    model,
) -> List[float]:
    """
    Late interaction: per-token embeddings + MaxSim scoring.
    Doc token vectors CAN be pre-computed and indexed (unlike cross-encoder).
    """
    q_embs = get_token_embeddings(query, tokenizer, model)
    scores = []
    for doc in docs:
        d_embs = get_token_embeddings(doc, tokenizer, model)
        sim_matrix = cosine_matrix(q_embs, d_embs)
        scores.append(float(sim_matrix.max(axis=1).sum()))
    return scores


# ---------------------------------------------------------------------------
# Two-stage retrieval pipeline
# ---------------------------------------------------------------------------

def two_stage_retrieve(
    query: str,
    corpus: List[str],
    bi_model: SentenceTransformer,
    cross_model: CrossEncoder,
    retrieve_k: int = 10,
    rerank_k: int = 3,
) -> List[Tuple[float, str]]:
    """
    Stage 1: Bi-encoder retrieves top-k candidates fast.
    Stage 2: Cross-encoder re-ranks those k candidates accurately.

    This is the standard production RAG retrieval pattern.

    Args:
        corpus      : full document set
        retrieve_k  : how many to retrieve in stage 1
        rerank_k    : how many to return after stage 2
    """
    # Stage 1: fast bi-encoder over full corpus
    bi_scores = biencoder_scores(query, corpus, bi_model)
    top_k_idx = sorted(range(len(corpus)), key=lambda i: bi_scores[i], reverse=True)[:retrieve_k]
    candidates = [corpus[i] for i in top_k_idx]

    # Stage 2: accurate cross-encoder over candidates only
    ce_scores = crossencoder_scores(query, candidates, cross_model)
    ranked = sorted(zip(ce_scores, candidates), reverse=True)

    return ranked[:rerank_k]


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Load all three models
    print("Loading models...")
    bi_model    = SentenceTransformer("all-MiniLM-L6-v2")
    cross_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    hf_model  = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    hf_model.eval()
    print("Models loaded.\n")

    # -------------------------------------------------------------------
    # EXPERIMENT 1: All three architectures on the same query + docs
    # -------------------------------------------------------------------
    print("=" * 65)
    print("EXP 1: Architecture comparison on same query")
    print("=" * 65)

    query = "How does RAG reduce hallucinations in language models?"
    docs = [
        "RAG grounds LLM outputs in retrieved documents, reducing hallucinations.",
        "Hallucinations occur when models generate unsupported facts from training.",
        "Vector databases store embeddings for fast similarity retrieval.",
        "Fine-tuning adapts model weights to domain-specific data.",
        "The Eiffel Tower is a landmark in Paris, France.",
    ]

    t0 = time.perf_counter()
    bi_scores = biencoder_scores(query, docs, bi_model)
    bi_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    ce_scores = crossencoder_scores(query, docs, cross_model)
    ce_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    li_scores = maxsim_scores(query, docs, tokenizer, hf_model)
    li_time = time.perf_counter() - t0

    print(f"\n  Query: '{query}'\n")
    print(f"  {'Bi-enc':>8}  {'Cross-enc':>10}  {'Late-int':>10}  Doc")
    print(f"  {'-'*75}")

    # Normalize late-interaction scores for display (different scale)
    li_max = max(li_scores)
    li_norm = [s / li_max for s in li_scores]

    for i, doc in enumerate(docs):
        bi_rank  = sorted(range(len(docs)), key=lambda x: bi_scores[x],  reverse=True).index(i) + 1
        ce_rank  = sorted(range(len(docs)), key=lambda x: ce_scores[x],  reverse=True).index(i) + 1
        li_rank  = sorted(range(len(docs)), key=lambda x: li_scores[x],  reverse=True).index(i) + 1
        disagree = " <-- DISAGREE" if (bi_rank != ce_rank or bi_rank != li_rank) else ""
        print(f"  {bi_scores[i]:>8.4f}  {ce_scores[i]:>10.4f}  {li_norm[i]:>10.4f}  "
              f"[bi={bi_rank} ce={ce_rank} li={li_rank}] '{doc[:40]}'{disagree}")

    print(f"\n  Latency: bi={bi_time*1000:.1f}ms  cross={ce_time*1000:.1f}ms  "
          f"late-int={li_time*1000:.1f}ms")

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Cross-encoder catches subtle relevance bi-encoder misses
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 2: Subtle relevance — where cross-encoder wins")
    print("=" * 65)

    query2 = "What are the limitations of RAG?"
    tricky_docs = [
        "RAG systems struggle with multi-hop reasoning and very long documents.",  # directly answers
        "RAG retrieves documents and uses them as context for generation.",         # about RAG, not limitations
        "LLMs have a context window limit that restricts how much text they use.",  # related limitation
        "Chunking strategy significantly affects retrieval quality in RAG.",        # related but indirect
    ]

    bi2  = biencoder_scores(query2, tricky_docs, bi_model)
    ce2  = crossencoder_scores(query2, tricky_docs, cross_model)

    print(f"\n  Query: '{query2}'\n")
    print(f"  {'Bi-enc rank':>12}  {'Cross-enc rank':>15}  Doc")
    print(f"  {'-'*75}")
    bi_order  = sorted(range(len(tricky_docs)), key=lambda i: bi2[i],  reverse=True)
    ce_order  = sorted(range(len(tricky_docs)), key=lambda i: ce2[i],  reverse=True)
    for i, doc in enumerate(tricky_docs):
        br = bi_order.index(i) + 1
        cr = ce_order.index(i) + 1
        flag = " <-- RERANKED" if abs(br - cr) >= 2 else ""
        print(f"  {br:>12}  {cr:>15}  '{doc[:55]}'{flag}")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Two-stage pipeline end to end
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 3: Two-stage retrieval pipeline")
    print("=" * 65)

    corpus = [
        "RAG reduces hallucinations by grounding responses in retrieved documents.",
        "Hallucinations in LLMs occur when models confabulate unsupported facts.",
        "Multi-hop reasoning requires connecting information across multiple documents.",
        "Vector similarity search finds semantically close chunks efficiently.",
        "RAG limitations include context window constraints and retrieval noise.",
        "Fine-tuning adjusts model weights for domain-specific performance.",
        "Embedding drift causes stale indexes when domain vocabulary shifts.",
        "Cross-encoders re-rank retrieved candidates for higher precision.",
        "HNSW enables approximate nearest neighbor search in vector databases.",
        "The sky is blue and the grass is green.",
    ]

    q3 = "What are the failure modes of RAG systems?"

    print(f"\n  Query: '{q3}'")
    print(f"  Corpus: {len(corpus)} docs\n")

    t0 = time.perf_counter()
    results = two_stage_retrieve(q3, corpus, bi_model, cross_model, retrieve_k=5, rerank_k=3)
    total_time = time.perf_counter() - t0

    print(f"  Stage 1 (bi-encoder): retrieved top-5 from {len(corpus)}")
    print(f"  Stage 2 (cross-encoder): re-ranked to top-3\n")
    for rank, (score, doc) in enumerate(results, 1):
        print(f"  {rank}. score={score:.4f}  '{doc}'")
    print(f"\n  Total latency: {total_time*1000:.1f}ms")

    # -------------------------------------------------------------------
    # EXPERIMENT 4: Latency scaling — why bi-encoder retrieval is necessary
    # -------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("EXP 4: Latency scaling — why cross-encoder alone doesn't scale")
    print("=" * 65 + "\n")

    # Simulate per-query latency at different corpus sizes
    single_ce_latency_ms = ce_time * 1000 / len(docs)   # ms per doc
    single_bi_latency_ms = bi_time * 1000 / len(docs)

    print(f"  Estimated per-doc latency:")
    print(f"    Bi-encoder  : {single_bi_latency_ms:.3f} ms/doc")
    print(f"    Cross-encoder: {single_ce_latency_ms:.3f} ms/doc\n")
    print(f"  {'Corpus size':>15}  {'Bi-enc (ms)':>14}  {'Cross-enc (ms)':>16}")
    print(f"  {'-'*50}")
    for n in [1_000, 10_000, 100_000, 1_000_000]:
        bi_t  = single_bi_latency_ms * n
        ce_t  = single_ce_latency_ms * n
        print(f"  {n:>15,}  {bi_t:>14.1f}  {ce_t:>16.1f}")

    print(f"\n  Two-stage: bi-encoder narrows to 100 docs, cross-encoder scores only those.")
    print(f"  Cross-enc at 100 docs: {single_ce_latency_ms * 100:.1f}ms  (fast enough for production)")
