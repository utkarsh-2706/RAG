"""
Sliding Window Chunking
=======================
Technique #2 — Overlap as a First-Class Citizen

Core insight: think in terms of WINDOW SIZE + STRIDE, not chunk_size + overlap.
    stride   = how far the window moves each step
    overlap  = window_size - stride   (derived, not set directly)

This reframing forces you to reason about COVERAGE DENSITY:
  - Small stride  → dense sampling → high recall, high redundancy
  - Large stride  → sparse sampling → lower recall, low redundancy
  - stride = window_size → zero overlap (degenerates to fixed chunking)

Two modes implemented:
  1. Character-level  (fast, no NLP dependencies)
  2. Sentence-level   (linguistically aware — better boundaries)
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Shared data structure (same as fixed_chunking but extended)
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: int          # character offset in the original document
    end: int
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
            f"Chunk(id={self.chunk_id}, words={self.word_count}, "
            f"chars={self.char_count}, preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Mode 1: Character-level sliding window
# ---------------------------------------------------------------------------

def sliding_window_char(
    text: str,
    window_size: int = 300,
    stride: int = 150,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Slide a character window across the text.

    Args:
        text        : raw input text
        window_size : number of characters in each window
        stride      : how many characters to advance each step
                      overlap = window_size - stride
        source      : metadata tag

    Returns:
        List of Chunk objects
    """
    if stride <= 0 or stride > window_size:
        raise ValueError("stride must be > 0 and <= window_size.")

    overlap = window_size - stride
    chunks: List[Chunk] = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + window_size
        chunk_text = text[start:end]

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start=start,
            end=min(end, len(text)),
            metadata={
                "source": source,
                "mode": "char",
                "window_size": window_size,
                "stride": stride,
                "overlap_chars": overlap,
            },
        ))

        chunk_id += 1
        start += stride

    return chunks


# ---------------------------------------------------------------------------
# Mode 2: Sentence-level sliding window (better boundaries)
# ---------------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Naive sentence splitter using regex.
    Splits on '.', '!', '?' followed by whitespace or end-of-string.
    Good enough for demonstration; use spaCy/NLTK in production.
    """
    # Split on sentence-ending punctuation followed by space or newline
    raw = re.split(r'(?<=[.!?])\s+', text.strip())
    # Filter out empty strings
    return [s.strip() for s in raw if s.strip()]


def sliding_window_sentences(
    text: str,
    window_sentences: int = 3,
    stride_sentences: int = 1,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Slide a window of N sentences across the text, advancing by stride sentences.

    Args:
        text               : raw input text
        window_sentences   : number of sentences per chunk
        stride_sentences   : how many sentences to advance each step
                             overlap = window_sentences - stride_sentences
        source             : metadata tag

    Returns:
        List of Chunk objects
    """
    if stride_sentences <= 0 or stride_sentences > window_sentences:
        raise ValueError("stride_sentences must be > 0 and <= window_sentences.")

    sentences = split_into_sentences(text)
    overlap_sentences = window_sentences - stride_sentences

    chunks: List[Chunk] = []
    chunk_id = 0
    i = 0

    # Reconstruct character offsets by searching in original text
    # (simple approach: track position via cumulative search)
    sentence_offsets = _compute_sentence_offsets(text, sentences)

    while i < len(sentences):
        window = sentences[i : i + window_sentences]
        if not window:
            break

        chunk_text = " ".join(window)
        start_char = sentence_offsets[i]
        end_idx = min(i + window_sentences - 1, len(sentences) - 1)
        end_char = sentence_offsets[end_idx] + len(sentences[end_idx])

        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=chunk_text,
            start=start_char,
            end=end_char,
            metadata={
                "source": source,
                "mode": "sentence",
                "window_sentences": window_sentences,
                "stride_sentences": stride_sentences,
                "overlap_sentences": overlap_sentences,
                "sentence_indices": list(range(i, i + len(window))),
            },
        ))

        chunk_id += 1
        i += stride_sentences

    return chunks


def _compute_sentence_offsets(text: str, sentences: List[str]) -> List[int]:
    """Find the character offset of each sentence in the original text."""
    offsets = []
    search_from = 0
    for sent in sentences:
        idx = text.find(sent, search_from)
        if idx == -1:
            idx = search_from   # fallback
        offsets.append(idx)
        search_from = idx + len(sent)
    return offsets


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def load_text(filepath: str) -> str:
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def overlap_ratio(window_size: int, stride: int) -> float:
    """What fraction of each window is repeated from the previous one?"""
    return round((window_size - stride) / window_size, 3)


def print_chunks(chunks: List[Chunk], show_full: bool = False) -> None:
    print(f"\n{'='*60}")
    print(f"Total chunks: {len(chunks)}")
    print(f"{'='*60}\n")
    for c in chunks:
        print(f"--- Chunk {c.chunk_id} | words={c.word_count} | chars={c.char_count} ---")
        if show_full:
            print(c.text)
        else:
            print(c.text[:120].replace("\n", " ") + "...")
        print()


def overlap_inspection(chunks: List[Chunk], pair_count: int = 3) -> None:
    """Show the overlapping region between consecutive chunk pairs."""
    print(f"\n{'='*60}")
    print("Overlap Inspection — shared content between adjacent chunks")
    print(f"{'='*60}\n")
    for i in range(min(pair_count, len(chunks) - 1)):
        a, b = chunks[i].text, chunks[i + 1].text
        # Find common suffix/prefix overlap
        overlap_len = 0
        for k in range(1, min(len(a), len(b)) + 1):
            if a[-k:] == b[:k]:
                overlap_len = k
        print(f"Chunk {i} -> Chunk {i+1}:")
        print(f"  End of chunk {i}   : ...{a[-80:].replace(chr(10), ' ')!r}")
        print(f"  Start of chunk {i+1}: {b[:80].replace(chr(10), ' ')!r}...")
        if overlap_len > 0:
            print(f"  Exact overlap    : {overlap_len} chars")
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

    # -----------------------------------------------------------------------
    # EXPERIMENT 1: High overlap (stride = window/2)
    #   50% of each chunk is shared with the next
    # -----------------------------------------------------------------------
    print(">>> EXP 1: window=300, stride=150  (50% overlap)")
    print(f"    Overlap ratio: {overlap_ratio(300, 150)}")
    chunks_50 = sliding_window_char(text, window_size=300, stride=150)
    print_chunks(chunks_50)

    # -----------------------------------------------------------------------
    # EXPERIMENT 2: Low overlap (stride = window * 0.9)
    #   Only 10% overlap — nearly fixed chunking
    # -----------------------------------------------------------------------
    print(">>> EXP 2: window=300, stride=270  (10% overlap)")
    print(f"    Overlap ratio: {overlap_ratio(300, 270)}")
    chunks_10 = sliding_window_char(text, window_size=300, stride=270)
    print_chunks(chunks_10)

    print(f"\nChunk count comparison:")
    print(f"  50% overlap : {len(chunks_50)} chunks")
    print(f"  10% overlap : {len(chunks_10)} chunks")

    # -----------------------------------------------------------------------
    # EXPERIMENT 3: Sentence-level sliding window
    #   window=3 sentences, stride=1 → 2 sentences overlap per chunk pair
    # -----------------------------------------------------------------------
    print("\n>>> EXP 3: Sentence window=3, stride=1  (2-sentence overlap)")
    chunks_sent = sliding_window_sentences(text, window_sentences=3, stride_sentences=1)
    print_chunks(chunks_sent)
    print(f"Total sentence-based chunks: {len(chunks_sent)}")

    # -----------------------------------------------------------------------
    # EXPERIMENT 4: Overlap inspection — see shared content visually
    # -----------------------------------------------------------------------
    print("\n>>> EXP 4: Overlap inspection — char-level, 50% overlap")
    overlap_inspection(chunks_50, pair_count=2)

    # -----------------------------------------------------------------------
    # EXPERIMENT 5: Stride sweep — see how stride affects chunk count
    # -----------------------------------------------------------------------
    print(">>> EXP 5: Stride sweep (window=300, vary stride)")
    print(f"  {'Stride':>8}  {'Overlap%':>10}  {'Chunks':>8}")
    print(f"  {'-'*32}")
    for stride in [30, 60, 150, 210, 270, 300]:
        c = sliding_window_char(text, window_size=300, stride=stride)
        print(f"  {stride:>8}  {overlap_ratio(300, stride)*100:>9.0f}%  {len(c):>8}")
