"""
Dynamic / Query-Aware Chunking
==============================
Technique #6 - The Unit You Index != The Unit You Return

Core insight:
    All previous techniques chunk once at ingestion and return exactly
    what was indexed. This forces a tradeoff: small chunks = precise
    retrieval but poor context; large chunks = rich context but noisy retrieval.

    Dynamic chunking breaks the tradeoff:
        - INDEX small (sharp embedding signal -> precise retrieval)
        - RETURN large (rich context -> better LLM reasoning)

Three strategies implemented:
    1. Small-to-Big  (Parent-Document Retrieval)
       Index small child chunks. On retrieval, return the parent chunk.

    2. Sentence Window Retrieval
       Index individual sentences. On retrieval, return sentence + k neighbors.

    3. Contextual Compression  (simulated - requires LLM in production)
       Retrieve a normal chunk. Then extract only the query-relevant sentences.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import numpy as np


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class IndexChunk:
    """Small chunk stored in the vector DB. Used for embedding + retrieval."""
    chunk_id: str
    text: str
    parent_id: str
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class ParentChunk:
    """Larger chunk returned to the LLM after retrieval."""
    parent_id: str
    text: str
    children: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class RetrievalResult:
    """What gets returned to the LLM - always a large context chunk."""
    query: str
    index_chunk: IndexChunk
    context_chunk: str
    strategy: str
    similarity: float


# ---------------------------------------------------------------------------
# STRATEGY 1: Small-to-Big (Parent-Document Retrieval)
# ---------------------------------------------------------------------------

def build_small_to_big_index(
    text: str,
    parent_size: int = 512,
    child_size: int = 128,
    parent_overlap: int = 0,
    source: str = "unknown",
) -> Tuple[List[ParentChunk], List[IndexChunk]]:
    """
    Build a two-level chunk hierarchy:
        Parent chunks : large, returned to LLM
        Child chunks  : small, embedded and indexed for retrieval

    Args:
        text         : raw document text
        parent_size  : characters per parent chunk
        child_size   : characters per child chunk (must be < parent_size)
        parent_overlap: overlap between parent chunks
        source       : metadata tag

    Returns:
        (parents, children)
    """
    if child_size >= parent_size:
        raise ValueError("child_size must be smaller than parent_size")

    parents: List[ParentChunk] = []
    children: List[IndexChunk] = []

    parent_step = parent_size - parent_overlap
    parent_id = 0
    pos = 0

    while pos < len(text):
        parent_text = text[pos: pos + parent_size].strip()
        if not parent_text:
            break

        p = ParentChunk(
            parent_id=f"p{parent_id}",
            text=parent_text,
            metadata={"source": source},
        )

        # Carve child chunks from this parent
        child_pos = 0
        child_id = 0
        while child_pos < len(parent_text):
            child_text = parent_text[child_pos: child_pos + child_size].strip()
            if child_text:
                c = IndexChunk(
                    chunk_id=f"p{parent_id}_c{child_id}",
                    text=child_text,
                    parent_id=f"p{parent_id}",
                    metadata={"source": source},
                )
                children.append(c)
                p.children.append(c.chunk_id)
                child_id += 1
            child_pos += child_size

        parents.append(p)
        parent_id += 1
        pos += parent_step

    return parents, children


def small_to_big_retrieve(
    query: str,
    parents: List[ParentChunk],
    children: List[IndexChunk],
    model,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    1. Embed all child chunks + query
    2. Find top-k most similar child chunks
    3. Return their parent chunks (larger context) to the LLM
    """
    parent_map: Dict[str, ParentChunk] = {p.parent_id: p for p in parents}

    child_texts = [c.text for c in children]
    child_embeddings = model.encode(child_texts, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    norms = np.linalg.norm(child_embeddings, axis=1, keepdims=True)
    child_norm = child_embeddings / (norms + 1e-9)
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    similarities = child_norm @ q_norm

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    seen_parents = set()

    for idx in top_indices:
        child = children[idx]
        sim = float(similarities[idx])
        parent = parent_map[child.parent_id]

        if parent.parent_id in seen_parents:
            continue
        seen_parents.add(parent.parent_id)

        results.append(RetrievalResult(
            query=query,
            index_chunk=child,
            context_chunk=parent.text,
            strategy="small_to_big",
            similarity=sim,
        ))

    return results


# ---------------------------------------------------------------------------
# STRATEGY 2: Sentence Window Retrieval
# ---------------------------------------------------------------------------

def split_sentences(text: str) -> List[str]:
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


def build_sentence_window_index(
    text: str,
    source: str = "unknown",
) -> Tuple[List[str], List[IndexChunk]]:
    """
    Index each sentence individually.
    The full sentence list is kept for window expansion at retrieval time.

    Returns:
        (sentences, index_chunks)
    """
    sentences = split_sentences(text)
    index_chunks = [
        IndexChunk(
            chunk_id=f"sent_{i}",
            text=sent,
            parent_id=f"sent_{i}",
            metadata={"source": source, "sentence_index": i},
        )
        for i, sent in enumerate(sentences)
    ]
    return sentences, index_chunks


def sentence_window_retrieve(
    query: str,
    sentences: List[str],
    index_chunks: List[IndexChunk],
    model,
    window_size: int = 2,
    top_k: int = 3,
) -> List[RetrievalResult]:
    """
    1. Embed all sentences + query
    2. Find top-k most similar sentences
    3. Return each matched sentence + window_size sentences on each side

    Args:
        window_size : sentences to add BEFORE and AFTER the matched sentence
    """
    sent_texts = [c.text for c in index_chunks]
    sent_embeddings = model.encode(sent_texts, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
    sent_norm = sent_embeddings / (norms + 1e-9)
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    similarities = sent_norm @ q_norm

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        matched = index_chunks[idx]
        sent_idx = matched.metadata["sentence_index"]
        sim = float(similarities[idx])

        # Expand window
        start = max(0, sent_idx - window_size)
        end = min(len(sentences), sent_idx + window_size + 1)
        context = " ".join(sentences[start:end])

        results.append(RetrievalResult(
            query=query,
            index_chunk=matched,
            context_chunk=context,
            strategy=f"sentence_window(k={window_size})",
            similarity=sim,
        ))

    return results


# ---------------------------------------------------------------------------
# STRATEGY 3: Contextual Compression (simulated)
# ---------------------------------------------------------------------------

def contextual_compression_retrieve(
    query: str,
    chunks: List[str],
    model,
    top_k: int = 3,
    top_n_sentences: int = 2,
) -> List[RetrievalResult]:
    """
    1. Retrieve full-size chunks by vector similarity
    2. Compress each: keep only the most query-relevant sentences
       (In production: replace compression with an LLM call)

    Production version:
        compressed = llm(
            f"Extract only sentences relevant to: '{query}'\\n\\n{chunk}"
        )
    """
    chunk_embeddings = model.encode(chunks, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    norms = np.linalg.norm(chunk_embeddings, axis=1, keepdims=True)
    chunk_norm = chunk_embeddings / (norms + 1e-9)
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    similarities = chunk_norm @ q_norm

    top_indices = np.argsort(similarities)[::-1][:top_k]

    results = []
    for idx in top_indices:
        chunk_text = chunks[idx]
        sim = float(similarities[idx])
        compressed = _embedding_compress(chunk_text, query, model, top_n=top_n_sentences)

        ic = IndexChunk(
            chunk_id=f"chunk_{idx}",
            text=chunk_text,
            parent_id=f"chunk_{idx}",
        )
        results.append(RetrievalResult(
            query=query,
            index_chunk=ic,
            context_chunk=compressed,
            strategy="contextual_compression",
            similarity=sim,
        ))

    return results


def _embedding_compress(chunk: str, query: str, model, top_n: int = 2) -> str:
    """Keep only the top_n most query-relevant sentences from a chunk."""
    sentences = split_sentences(chunk)
    if len(sentences) <= top_n:
        return chunk

    sent_embeddings = model.encode(sentences, show_progress_bar=False)
    query_embedding = model.encode([query], show_progress_bar=False)[0]

    norms = np.linalg.norm(sent_embeddings, axis=1, keepdims=True)
    sent_norm = sent_embeddings / (norms + 1e-9)
    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-9)
    sims = sent_norm @ q_norm

    # Keep top_n sentences in their original order (not sorted by similarity)
    top_indices = sorted(np.argsort(sims)[::-1][:top_n].tolist())
    return " ".join(sentences[i] for i in top_indices)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def print_results(results: List[RetrievalResult], label: str = "") -> None:
    print(f"\n{'='*60}")
    if label:
        print(f"Strategy: {label}")
    print(f"{'='*60}\n")
    for i, r in enumerate(results):
        print(f"--- Result {i+1} | sim={r.similarity:.3f} ---")
        print(f"  Query      : {r.query}")
        print(f"  Matched on : [{r.index_chunk.word_count}w] "
              f"{r.index_chunk.text[:80].replace(chr(10),' ')}...")
        print(f"  LLM gets   : [{len(r.context_chunk.split())}w] "
              f"{r.context_chunk[:160].replace(chr(10),' ')}...")
        print()


def size_comparison(
    query: str,
    s2b: List[RetrievalResult],
    sw: List[RetrievalResult],
    cc: List[RetrievalResult],
) -> None:
    """Compare indexed size vs returned context size for each strategy."""
    print(f"\n{'='*60}")
    print("Context size comparison (top result per strategy)")
    print(f"{'='*60}\n")
    print(f"  Query: '{query}'\n")
    print(f"  {'Strategy':<28} {'Indexed':>8} {'Returned':>10} {'Expansion':>10}")
    print(f"  {'-'*58}")

    for name, results in [("Small-to-Big", s2b), ("Sentence Window", sw), ("Contextual Compression", cc)]:
        if results:
            r = results[0]
            indexed = r.index_chunk.word_count
            returned = len(r.context_chunk.split())
            expansion = returned / indexed if indexed else 0
            print(f"  {name:<28} {indexed:>7}w {returned:>9}w {expansion:>9.1f}x")


# ---------------------------------------------------------------------------
# Main - experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    from sentence_transformers import SentenceTransformer

    base_dir = os.path.dirname(os.path.abspath(__file__))
    text = load_text(os.path.join(base_dir, "sample_data.txt"))

    print(f"Document: {len(text)} chars | ~{len(text.split())} words\n")
    print("Loading model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Model loaded.\n")

    QUERY = "How does RAG reduce hallucinations?"

    # -------------------------------------------------------------------
    # EXPERIMENT 1: Small-to-Big
    # -------------------------------------------------------------------
    print(">>> EXP 1: Small-to-Big  (parent=400, child=100)")
    parents, children = build_small_to_big_index(text, parent_size=400, child_size=100)
    print(f"  Built: {len(parents)} parents, {len(children)} children")
    s2b_results = small_to_big_retrieve(QUERY, parents, children, model, top_k=2)
    print_results(s2b_results, "Small-to-Big")

    # -------------------------------------------------------------------
    # EXPERIMENT 2: Sentence Window
    # -------------------------------------------------------------------
    print(">>> EXP 2: Sentence Window  (window_size=2)")
    sentences, sent_index = build_sentence_window_index(text)
    print(f"  Built: {len(sent_index)} indexed sentences")
    sw_results = sentence_window_retrieve(QUERY, sentences, sent_index, model, window_size=2, top_k=2)
    print_results(sw_results, "Sentence Window (k=2)")

    # -------------------------------------------------------------------
    # EXPERIMENT 3: Contextual Compression
    # -------------------------------------------------------------------
    print(">>> EXP 3: Contextual Compression  (keep top 2 sentences)")
    raw_chunks = [p.strip() for p in text.split("\n\n") if p.strip()]
    cc_results = contextual_compression_retrieve(QUERY, raw_chunks, model, top_k=2)
    print_results(cc_results, "Contextual Compression")

    # -------------------------------------------------------------------
    # EXPERIMENT 4: Size comparison across all three strategies
    # -------------------------------------------------------------------
    print(">>> EXP 4: Size comparison")
    size_comparison(QUERY, s2b_results, sw_results, cc_results)

    # -------------------------------------------------------------------
    # EXPERIMENT 5: Window size sweep
    # -------------------------------------------------------------------
    print("\n>>> EXP 5: Sentence window sweep")
    print(f"  {'Window':>8}  {'Words returned to LLM':>24}")
    print(f"  {'-'*34}")
    for w in [0, 1, 2, 3, 5]:
        res = sentence_window_retrieve(QUERY, sentences, sent_index, model, window_size=w, top_k=1)
        if res:
            words = len(res[0].context_chunk.split())
            print(f"  {w:>8}  {words:>24}")

    # -------------------------------------------------------------------
    # EXPERIMENT 6: Different query - context shifts accordingly
    # -------------------------------------------------------------------
    print("\n>>> EXP 6: Query shift - same index, different context returned")
    QUERY2 = "What are vector databases used for?"
    sw2 = sentence_window_retrieve(QUERY2, sentences, sent_index, model, window_size=2, top_k=1)
    print_results(sw2, f"Query: '{QUERY2}'")
