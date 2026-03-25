"""
Recursive Chunking
==================
Technique #3 — Respect Document Structure, Don't Just Count Characters

Core idea:
    Try to split on the "best" (most meaningful) separator first.
    If any piece is still too large, recurse with the next separator.
    Stop when everything fits within chunk_size.

Default separator priority (same as LangChain's RecursiveCharacterTextSplitter):
    ["\n\n", "\n", ". ", " ", ""]
     ^^^^^^   ^^   ^^^   ^   ^
     para   line  sent  word  char (last resort)

This means a chunk will:
  - Prefer to end at a paragraph boundary
  - Fall back to a line boundary
  - Fall back to a sentence boundary
  - Fall back to a word boundary
  - Only break mid-word as a last resort (empty string separator)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
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
# Core recursive splitting logic
# ---------------------------------------------------------------------------

def _split_text(text: str, separator: str) -> List[str]:
    """Split text by separator, keeping non-empty pieces."""
    if separator == "":
        # Character-level fallback: split into individual chars
        return list(text)
    return [s for s in text.split(separator) if s.strip()]


def _merge_splits(
    splits: List[str],
    separator: str,
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Merge small splits back together (with overlap) so we don't produce
    hundreds of tiny fragments when splitting on spaces or sentences.

    This is the 'packing' step: greedily pack splits into chunks up to
    chunk_size, then start a new chunk with `overlap` chars of lookback.
    """
    merged: List[str] = []
    current_pieces: List[str] = []
    current_len = 0
    sep_len = len(separator)

    for piece in splits:
        piece_len = len(piece)
        # +sep_len for the separator we'd add between pieces
        projected = current_len + piece_len + (sep_len if current_pieces else 0)

        if projected > chunk_size and current_pieces:
            # Flush current chunk
            merged.append(separator.join(current_pieces))

            # Apply overlap: drop pieces from the front until we're within overlap budget
            while current_pieces and current_len > overlap:
                current_len -= len(current_pieces[0]) + sep_len
                current_pieces.pop(0)

        current_pieces.append(piece)
        current_len += piece_len + (sep_len if len(current_pieces) > 1 else 0)

    # Flush remainder
    if current_pieces:
        merged.append(separator.join(current_pieces))

    return merged


def _recursive_split(
    text: str,
    separators: List[str],
    chunk_size: int,
    overlap: int,
) -> List[str]:
    """
    Recursively split text using the separator hierarchy.

    For each separator (tried in order):
      1. Split the text.
      2. For pieces still too large, recurse with the remaining separators.
      3. Merge small pieces back up to chunk_size with overlap.
    """
    final_chunks: List[str] = []

    # Pick the first separator that actually exists in the text
    separator = separators[-1]   # fallback: char-level
    remaining_separators = []

    for i, sep in enumerate(separators):
        if sep == "" or sep in text:
            separator = sep
            remaining_separators = separators[i + 1:]
            break

    splits = _split_text(text, separator)

    # Separate: pieces that are small enough vs. pieces that need further splitting
    good_splits: List[str] = []

    for split in splits:
        if len(split) <= chunk_size:
            good_splits.append(split)
        else:
            # This piece is still too large — flush accumulated good splits first
            if good_splits:
                merged = _merge_splits(good_splits, separator, chunk_size, overlap)
                final_chunks.extend(merged)
                good_splits = []

            if not remaining_separators:
                # No more separators — forced hard cut
                final_chunks.append(split[:chunk_size])
            else:
                # Recurse deeper
                sub_chunks = _recursive_split(
                    split, remaining_separators, chunk_size, overlap
                )
                final_chunks.extend(sub_chunks)

    # Merge any remaining good splits
    if good_splits:
        merged = _merge_splits(good_splits, separator, chunk_size, overlap)
        final_chunks.extend(merged)

    return final_chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


def recursive_chunk(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    separators: Optional[List[str]] = None,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Recursively split text respecting natural document boundaries.

    Args:
        text        : raw input text
        chunk_size  : target max characters per chunk
        overlap     : character overlap between consecutive chunks
        separators  : priority-ordered list of split points
                      default: [paragraph, line, sentence, word, char]
        source      : metadata tag

    Returns:
        List of Chunk objects with sequential IDs
    """
    seps = separators if separators is not None else DEFAULT_SEPARATORS
    raw_chunks = _recursive_split(text, seps, chunk_size, overlap)

    return [
        Chunk(
            chunk_id=i,
            text=chunk.strip(),
            metadata={
                "source": source,
                "chunk_size_target": chunk_size,
                "overlap": overlap,
                "separators": seps,
            },
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]


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
        print(f"--- Chunk {c.chunk_id} | words={c.word_count} | chars={c.char_count} ---")
        if show_full:
            print(c.text)
        else:
            print(c.text[:150].replace("\n", " ") + "...")
        print()


def boundary_check(chunks: List[Chunk], count: int = 5) -> None:
    """Show how each chunk ends — does it end cleanly?"""
    print(f"\n{'='*60}")
    print("Boundary Check — how do chunks end?")
    print(f"{'='*60}\n")
    for c in chunks[:count]:
        ending = c.text[-80:].replace("\n", " ")
        starts_clean = c.text[0].isupper() or c.text[0] in ('"', "'", "-")
        ends_clean = c.text.rstrip()[-1] in ".!?:,"
        print(f"  Chunk {c.chunk_id}: ...{ending!r}")
        print(f"    starts_clean={starts_clean}  ends_clean={ends_clean}\n")


def compare_fixed_vs_recursive(text: str, chunk_size: int = 500) -> None:
    """Side-by-side: show how boundary quality differs."""
    from fixed_chunking import fixed_chunk

    fixed = fixed_chunk(text, chunk_size=chunk_size, overlap=0)
    recursive = recursive_chunk(text, chunk_size=chunk_size, overlap=0)

    print(f"\n{'='*60}")
    print(f"Fixed vs Recursive (chunk_size={chunk_size}, overlap=0)")
    print(f"{'='*60}")
    print(f"  Fixed    : {len(fixed)} chunks")
    print(f"  Recursive: {len(recursive)} chunks\n")

    print("-- Fixed Chunk 0 ending --")
    print(repr(fixed[0].text[-100:]))
    print("\n-- Recursive Chunk 0 ending --")
    print(repr(recursive[0].text[-100:]))


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os
    import sys

    base_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, base_dir)          # so compare_fixed_vs_recursive can import
    data_path = os.path.join(base_dir, "sample_data.txt")
    text = load_text(data_path)

    print(f"Document: {len(text)} chars | ~{len(text.split())} words")

    # -----------------------------------------------------------------------
    # EXPERIMENT 1: Default separators, moderate chunk size
    # -----------------------------------------------------------------------
    print("\n>>> EXP 1: chunk_size=500, overlap=50, default separators")
    chunks = recursive_chunk(text, chunk_size=500, overlap=50)
    print_chunks(chunks)

    # -----------------------------------------------------------------------
    # EXPERIMENT 2: Boundary check — do chunks end at sentence/paragraph?
    # -----------------------------------------------------------------------
    print(">>> EXP 2: Boundary quality check")
    boundary_check(chunks, count=6)

    # -----------------------------------------------------------------------
    # EXPERIMENT 3: Fixed vs Recursive — visual comparison
    # -----------------------------------------------------------------------
    print(">>> EXP 3: Fixed vs Recursive boundary comparison")
    compare_fixed_vs_recursive(text, chunk_size=500)

    # -----------------------------------------------------------------------
    # EXPERIMENT 4: Tiny chunk_size — forces deeper recursion
    # -----------------------------------------------------------------------
    print("\n>>> EXP 4: chunk_size=100 — forces sentence/word-level splitting")
    small_chunks = recursive_chunk(text, chunk_size=100, overlap=20)
    print_chunks(small_chunks)
    sizes = [c.char_count for c in small_chunks]
    print(f"Size distribution: min={min(sizes)}, max={max(sizes)}, avg={sum(sizes)/len(sizes):.1f}")

    # -----------------------------------------------------------------------
    # EXPERIMENT 5: Custom separators for code-like content
    #   Useful if your text were Python code or markdown
    # -----------------------------------------------------------------------
    print("\n>>> EXP 5: Custom separators — paragraph only (no sentence split)")
    para_only = recursive_chunk(
        text,
        chunk_size=600,
        overlap=0,
        separators=["\n\n", "\n"],   # stop before sentence-level
    )
    print_chunks(para_only)
    print(f"Paragraph-only chunks: {len(para_only)}")
