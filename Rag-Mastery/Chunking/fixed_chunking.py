"""
Fixed Chunking
==============
Technique #1 — The Baseline

Split text into fixed-size character windows with optional overlap.
No awareness of sentence or paragraph boundaries.

Key Parameters:
    chunk_size  : number of characters per chunk
    overlap     : number of characters shared between consecutive chunks
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


# ---------------------------------------------------------------------------
# Data structure for a single chunk
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    start: int          # character offset in original document
    end: int            # character offset in original document
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id}, "
            f"chars={len(self.text)}, "
            f"start={self.start}, "
            f"preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Core chunking function
# ---------------------------------------------------------------------------

def fixed_chunk(
    text: str,
    chunk_size: int = 300,
    overlap: int = 50,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Split text into fixed-size character chunks with optional overlap.

    Args:
        text       : raw input text
        chunk_size : size of each chunk in characters
        overlap    : number of overlapping characters between consecutive chunks
        source     : metadata — name/path of the source document

    Returns:
        List of Chunk objects
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size.")

    chunks: List[Chunk] = []
    step = chunk_size - overlap      # how far we advance the window each time
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size
        chunk_text = text[start:end]

        chunks.append(
            Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                start=start,
                end=min(end, len(text)),
                metadata={"source": source},
            )
        )

        chunk_id += 1
        start += step

    return chunks


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def load_text(filepath: str) -> str:
    """Read a plain-text file and return its contents."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()


def print_chunks(chunks: List[Chunk], show_full_text: bool = False) -> None:
    """Pretty-print chunk list for inspection."""
    print(f"\n{'='*60}")
    print(f"Total chunks produced: {len(chunks)}")
    print(f"{'='*60}\n")

    for chunk in chunks:
        print(f"--- Chunk {chunk.chunk_id} ---")
        print(f"  Characters : {len(chunk.text)}")
        print(f"  Offset     : [{chunk.start} : {chunk.end}]")
        if show_full_text:
            print(f"  Text       :\n{chunk.text}")
        else:
            preview = chunk.text[:120].replace("\n", " ")
            print(f"  Preview    : {preview}...")
        print()


def chunk_stats(chunks: List[Chunk]) -> dict:
    """Return basic statistics about the chunks produced."""
    sizes = [len(c.text) for c in chunks]
    return {
        "total_chunks": len(chunks),
        "min_chars": min(sizes),
        "max_chars": max(sizes),
        "avg_chars": round(sum(sizes) / len(sizes), 1),
    }


# ---------------------------------------------------------------------------
# Main — experiment entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    # Locate sample data relative to this script
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "sample_data.txt")

    text = load_text(data_path)
    print(f"Document length: {len(text)} characters\n")

    # -----------------------------------------------------------------------
    # EXPERIMENT 1: No overlap
    # -----------------------------------------------------------------------
    print(">>> EXPERIMENT 1: chunk_size=300, overlap=0")
    chunks = fixed_chunk(text, chunk_size=300, overlap=0, source="sample_data.txt")
    print_chunks(chunks)
    print("Stats:", chunk_stats(chunks))

    # -----------------------------------------------------------------------
    # EXPERIMENT 2: With overlap
    # -----------------------------------------------------------------------
    print("\n>>> EXPERIMENT 2: chunk_size=300, overlap=50")
    chunks_overlap = fixed_chunk(text, chunk_size=300, overlap=50, source="sample_data.txt")
    print_chunks(chunks_overlap)
    print("Stats:", chunk_stats(chunks_overlap))

    # -----------------------------------------------------------------------
    # EXPERIMENT 3: Show a boundary problem
    #   Look at the last line of chunk N and first line of chunk N+1
    #   to observe mid-sentence cuts.
    # -----------------------------------------------------------------------
    print("\n>>> EXPERIMENT 3: Boundary inspection (no overlap)")
    boundary_chunks = fixed_chunk(text, chunk_size=300, overlap=0)
    for i in range(min(3, len(boundary_chunks) - 1)):
        print(f"\n  --- End of Chunk {i} ---")
        print(f"  ...{boundary_chunks[i].text[-80:]!r}")
        print(f"  --- Start of Chunk {i+1} ---")
        print(f"  {boundary_chunks[i+1].text[:80]!r}...")
