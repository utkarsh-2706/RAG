"""
Semantic Chunking
=================
Technique #4 — Split Where Meaning Changes, Not Where Characters Run Out

Core idea:
    1. Split text into sentences
    2. Embed each sentence using a small local model
    3. Compute cosine similarity between consecutive sentence embeddings
    4. Find "valleys" — points where similarity drops sharply
    5. Split at those valleys → each chunk covers one coherent topic

Two splitting strategies:
    - Threshold  : split where similarity < fixed cutoff (e.g. 0.5)
    - Percentile : split at the bottom N% of similarity scores (adaptive)

Why percentile is usually better:
    - Different documents have different baseline similarity ranges
    - A threshold that works for one doc may over/under-split another
    - Percentile adapts: always splits at the N% lowest-coherence points

Dependencies:
    pip install sentence-transformers numpy
"""

from __future__ import annotations
import re
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    sentence_range: Tuple[int, int]     # (start_sentence_idx, end_sentence_idx)
    avg_similarity: float               # mean similarity within the chunk
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:70].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id}, "
            f"sents={self.sentence_range[0]}-{self.sentence_range[1]}, "
            f"words={self.word_count}, "
            f"avg_sim={self.avg_similarity:.3f}, "
            f"preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Step 1: Sentence splitting
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex.
    Handles common abbreviations imperfectly — use spaCy in production.
    """
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in raw if len(s.strip()) > 10]


# ---------------------------------------------------------------------------
# Step 2: Embedding
# ---------------------------------------------------------------------------

def load_embedding_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Load a local SentenceTransformer model.
    'all-MiniLM-L6-v2' is small (80MB), fast, and good enough for chunking.
    """
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {model_name} ...")
    model = SentenceTransformer(model_name)
    print("Model loaded.\n")
    return model


def embed_sentences(sentences: List[str], model) -> np.ndarray:
    """
    Embed a list of sentences.
    Returns shape: (num_sentences, embedding_dim)
    """
    embeddings = model.encode(sentences, show_progress_bar=False)
    return np.array(embeddings)


# ---------------------------------------------------------------------------
# Step 3: Cosine similarity between consecutive sentences
# ---------------------------------------------------------------------------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def compute_similarity_scores(embeddings: np.ndarray) -> List[float]:
    """
    Compute cosine similarity between each consecutive pair of sentence embeddings.
    Result has length = len(embeddings) - 1.
    Index i = similarity between sentence i and sentence i+1.
    """
    scores = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(embeddings[i], embeddings[i + 1])
        scores.append(sim)
    return scores


# ---------------------------------------------------------------------------
# Step 4: Find split points
# ---------------------------------------------------------------------------

def find_breakpoints_by_threshold(
    similarity_scores: List[float],
    threshold: float = 0.5,
) -> List[int]:
    """
    Split AFTER sentence i if similarity(i, i+1) < threshold.
    Returns list of indices where splits occur (index = sentence BEFORE the split).
    """
    return [i for i, sim in enumerate(similarity_scores) if sim < threshold]


def find_breakpoints_by_percentile(
    similarity_scores: List[float],
    percentile: float = 25.0,
) -> List[int]:
    """
    Split at the bottom `percentile`% of similarity scores.
    More adaptive than threshold — works across different document types.

    E.g., percentile=25 means: split at the 25% lowest-similarity transitions.
    """
    if not similarity_scores:
        return []
    cutoff = float(np.percentile(similarity_scores, percentile))
    return [i for i, sim in enumerate(similarity_scores) if sim <= cutoff]


# ---------------------------------------------------------------------------
# Step 5: Build chunks from breakpoints
# ---------------------------------------------------------------------------

def build_chunks_from_breakpoints(
    sentences: List[str],
    breakpoints: List[int],
    similarity_scores: List[float],
    source: str = "unknown",
    strategy: str = "unknown",
) -> List[Chunk]:
    """
    Group sentences between breakpoints into Chunk objects.
    A breakpoint at index i means: end current chunk after sentence i,
    start new chunk at sentence i+1.
    """
    chunks: List[Chunk] = []
    split_points = sorted(set(breakpoints))

    start = 0
    chunk_id = 0

    for bp in split_points:
        end = bp + 1    # inclusive end
        group = sentences[start:end]
        if group:
            # Mean similarity within this chunk's sentence transitions
            internal_sims = similarity_scores[start:end - 1] if end - start > 1 else [1.0]
            avg_sim = float(np.mean(internal_sims)) if internal_sims else 1.0

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=" ".join(group),
                sentence_range=(start, end - 1),
                avg_similarity=avg_sim,
                metadata={"source": source, "strategy": strategy},
            ))
            chunk_id += 1
        start = end

    # Final chunk (after last breakpoint)
    if start < len(sentences):
        group = sentences[start:]
        internal_sims = similarity_scores[start:] if len(group) > 1 else [1.0]
        avg_sim = float(np.mean(internal_sims)) if internal_sims else 1.0
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=" ".join(group),
            sentence_range=(start, len(sentences) - 1),
            avg_similarity=avg_sim,
            metadata={"source": source, "strategy": strategy},
        ))

    return chunks


# ---------------------------------------------------------------------------
# Public API — two strategies
# ---------------------------------------------------------------------------

def semantic_chunk_threshold(
    text: str,
    model,
    threshold: float = 0.5,
    source: str = "unknown",
) -> List[Chunk]:
    """Semantic chunking using a fixed similarity threshold."""
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences, model)
    scores = compute_similarity_scores(embeddings)
    breakpoints = find_breakpoints_by_threshold(scores, threshold)
    return build_chunks_from_breakpoints(
        sentences, breakpoints, scores, source, strategy=f"threshold={threshold}"
    )


def semantic_chunk_percentile(
    text: str,
    model,
    percentile: float = 25.0,
    source: str = "unknown",
) -> List[Chunk]:
    """Semantic chunking using a percentile cutoff (adaptive)."""
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences, model)
    scores = compute_similarity_scores(embeddings)
    breakpoints = find_breakpoints_by_percentile(scores, percentile)
    return build_chunks_from_breakpoints(
        sentences, breakpoints, scores, source, strategy=f"percentile={percentile}"
    )


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def print_chunks(chunks: List[Chunk], show_full: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")
    for c in chunks:
        print(f"--- Chunk {c.chunk_id} | sents={c.sentence_range} | "
              f"words={c.word_count} | avg_sim={c.avg_similarity:.3f} ---")
        if show_full:
            print(c.text)
        else:
            print(c.text[:180].replace("\n", " ") + "...")
        print()


def print_similarity_scores(sentences: List[str], scores: List[float]) -> None:
    """Show the similarity score between each consecutive sentence pair."""
    print(f"\n{'='*60}")
    print("Similarity scores between consecutive sentences")
    print(f"{'='*60}\n")
    for i, (sim, sent) in enumerate(zip(scores, sentences[:-1])):
        bar = "#" * int(sim * 20)
        marker = " <<< SPLIT" if sim < 0.5 else ""
        print(f"  [{i:02d}->{i+1:02d}] {sim:.3f} |{bar:<20}|{marker}")
        print(f"         '{sent[:60]}...'")
    print()


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "sample_data.txt")
    text = load_text(data_path)

    print(f"Document: {len(text)} chars | ~{len(text.split())} words\n")

    # Load model once — reuse for all experiments
    model = load_embedding_model("all-MiniLM-L6-v2")

    # Pre-compute sentences + scores for inspection
    sentences = split_into_sentences(text)
    embeddings = embed_sentences(sentences, model)
    scores = compute_similarity_scores(embeddings)

    print(f"Total sentences found: {len(sentences)}")

    # -----------------------------------------------------------------------
    # EXPERIMENT 1: Similarity score visualization
    #   See where topic shifts actually occur in the document
    # -----------------------------------------------------------------------
    print("\n>>> EXP 1: Similarity scores between consecutive sentences")
    print_similarity_scores(sentences, scores)

    # -----------------------------------------------------------------------
    # EXPERIMENT 2: Threshold strategy
    # -----------------------------------------------------------------------
    print(">>> EXP 2: Semantic chunking — threshold=0.5")
    chunks_thresh = semantic_chunk_threshold(text, model, threshold=0.5)
    print_chunks(chunks_thresh)

    # -----------------------------------------------------------------------
    # EXPERIMENT 3: Percentile strategy
    # -----------------------------------------------------------------------
    print(">>> EXP 3: Semantic chunking — percentile=25")
    chunks_pct = semantic_chunk_percentile(text, model, percentile=25)
    print_chunks(chunks_pct)

    # -----------------------------------------------------------------------
    # EXPERIMENT 4: Compare chunk counts across strategies
    # -----------------------------------------------------------------------
    print(">>> EXP 4: Sensitivity sweep — percentile vs chunk count")
    print(f"  {'Percentile':>12}  {'Chunks':>8}  {'Avg words/chunk':>16}")
    print(f"  {'-'*40}")
    for pct in [10, 25, 40, 60, 75]:
        c = semantic_chunk_percentile(text, model, percentile=pct)
        avg_words = sum(ch.word_count for ch in c) / len(c)
        print(f"  {pct:>12}  {len(c):>8}  {avg_words:>16.1f}")

    # -----------------------------------------------------------------------
    # EXPERIMENT 5: Show a detected topic-shift split
    #   Find the split with the lowest similarity — the clearest boundary
    # -----------------------------------------------------------------------
    print("\n>>> EXP 5: Clearest topic shift detected")
    min_idx = int(np.argmin(scores))
    print(f"  Lowest similarity: {scores[min_idx]:.3f} between sentences {min_idx} and {min_idx+1}")
    print(f"\n  End of topic A:  '{sentences[min_idx]}'")
    print(f"\n  Start of topic B: '{sentences[min_idx+1]}'")
