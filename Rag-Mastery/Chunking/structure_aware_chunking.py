"""
Structure-Aware Chunking
========================
Technique #5 — Use the Document's Own Skeleton as Chunk Boundaries

Core idea:
    Documents like Markdown and HTML already encode structure through
    headers, sections, and tags. Parse that structure first, then chunk
    along those natural boundaries. Each chunk carries its full heading
    path as metadata — enabling retrieval filtering on top of similarity.

Two parsers implemented:
    1. Markdown parser  — splits on # / ## / ### headings, handles code blocks
    2. HTML parser      — splits on <h1>-<h4> tags using BeautifulSoup

Key concept — heading path:
    A chunk from "## 2. Chunking > ### 2.1 Fixed Chunking" gets metadata:
        { "heading": "2.1 Fixed Chunking",
          "heading_path": "RAG Guide > 2. Chunking > 2.1 Fixed Chunking",
          "level": 3 }

    This lets you do:
        results = db.query(query_vec, filter={"heading_path": {"$contains": "Chunking"}})
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Data structure — richer metadata than previous techniques
# ---------------------------------------------------------------------------

@dataclass
class Chunk:
    chunk_id: int
    text: str
    heading: str                    # immediate section heading
    heading_level: int              # 1=h1, 2=h2, 3=h3, 0=no heading
    heading_path: str               # full breadcrumb: "Doc > Section > Subsection"
    metadata: dict = field(default_factory=dict)

    @property
    def char_count(self) -> int:
        return len(self.text)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    def __repr__(self) -> str:
        preview = self.text[:60].replace("\n", " ")
        return (
            f"Chunk(id={self.chunk_id}, "
            f"h{self.heading_level}='{self.heading}', "
            f"words={self.word_count}, "
            f"preview='{preview}...')"
        )


# ---------------------------------------------------------------------------
# Parser 1: Markdown-aware chunking
# ---------------------------------------------------------------------------

# Regex: matches lines like "# Heading", "## Sub", "### Deep"
HEADING_RE = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)


def _parse_markdown_sections(text: str):
    """
    Walk the markdown text and yield (level, heading, content) tuples.
    Content = everything between this heading and the next same-or-higher heading.
    """
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        # No headings found — treat the whole doc as one section
        yield (0, "Document", text.strip())
        return

    # Content before the first heading
    preamble = text[:matches[0].start()].strip()
    if preamble:
        yield (0, "Preamble", preamble)

    for i, match in enumerate(matches):
        level = len(match.group(1))        # number of # chars
        heading = match.group(2).strip()

        # Content runs from end of this heading line to start of next heading
        content_start = match.end()
        content_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[content_start:content_end].strip()

        yield (level, heading, content)


def _build_heading_path(heading_stack: List[tuple], current_heading: str) -> str:
    """Build a breadcrumb path like: 'Doc > Section > Subsection'."""
    path_parts = [h for _, h in heading_stack] + [current_heading]
    return " > ".join(path_parts)


def markdown_chunk(
    text: str,
    max_chunk_size: Optional[int] = None,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Split Markdown into chunks aligned with heading boundaries.

    Each chunk = one section (heading + its body text).
    If a section exceeds max_chunk_size, it is split further using
    paragraph boundaries (a lightweight recursive fallback).

    Args:
        text           : raw Markdown text
        max_chunk_size : optional character limit per chunk (None = no limit)
        source         : metadata tag

    Returns:
        List of Chunk objects with heading metadata
    """
    sections = list(_parse_markdown_sections(text))
    chunks: List[Chunk] = []
    chunk_id = 0

    # Track heading hierarchy for path building
    heading_stack: List[tuple] = []   # list of (level, heading_text)

    for level, heading, content in sections:
        # Maintain the heading stack (pop levels >= current)
        if level > 0:
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_path = _build_heading_path(heading_stack, heading)
            heading_stack.append((level, heading))
        else:
            heading_path = heading  # Preamble

        # Full chunk text = heading line + body
        full_text = (f"{'#' * level} {heading}\n\n{content}").strip() if level > 0 else content

        # If content fits (or no size limit) → single chunk
        if max_chunk_size is None or len(full_text) <= max_chunk_size:
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=full_text,
                heading=heading,
                heading_level=level,
                heading_path=heading_path,
                metadata={"source": source, "parser": "markdown"},
            ))
            chunk_id += 1
        else:
            # Section too large → split on paragraph boundaries, keep heading in each
            paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
            buffer = f"{'#' * level} {heading}\n\n"
            for para in paragraphs:
                if len(buffer) + len(para) + 2 > max_chunk_size and buffer.strip():
                    chunks.append(Chunk(
                        chunk_id=chunk_id,
                        text=buffer.strip(),
                        heading=heading,
                        heading_level=level,
                        heading_path=heading_path,
                        metadata={"source": source, "parser": "markdown", "split": True},
                    ))
                    chunk_id += 1
                    buffer = f"[continued] {heading}\n\n"
                buffer += para + "\n\n"
            if buffer.strip():
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=buffer.strip(),
                    heading=heading,
                    heading_level=level,
                    heading_path=heading_path,
                    metadata={"source": source, "parser": "markdown", "split": True},
                ))
                chunk_id += 1

    return chunks


# ---------------------------------------------------------------------------
# Parser 2: HTML-aware chunking
# ---------------------------------------------------------------------------

def html_chunk(
    html: str,
    heading_tags: List[str] = None,
    max_chunk_size: Optional[int] = None,
    source: str = "unknown",
) -> List[Chunk]:
    """
    Split HTML into chunks aligned with heading tag boundaries.

    Uses BeautifulSoup to parse the DOM. Each heading tag starts a new chunk
    that contains the heading text plus all sibling content until the next heading.

    Args:
        html          : raw HTML string
        heading_tags  : which tags to split on (default: h1, h2, h3, h4)
        max_chunk_size: optional character limit per chunk
        source        : metadata tag

    Returns:
        List of Chunk objects with heading metadata
    """
    from bs4 import BeautifulSoup, NavigableString, Tag

    if heading_tags is None:
        heading_tags = ["h1", "h2", "h3", "h4"]

    soup = BeautifulSoup(html, "html.parser")
    body = soup.find("body") or soup

    chunks: List[Chunk] = []
    chunk_id = 0
    heading_stack: List[tuple] = []

    # Walk all top-level elements, group by heading
    current_heading = "Preamble"
    current_level = 0
    current_path = "Preamble"
    current_texts: List[str] = []

    def flush_chunk():
        nonlocal chunk_id
        body_text = " ".join(current_texts).strip()
        if not body_text:
            return
        full_text = (
            f"{'#' * current_level} {current_heading}\n\n{body_text}"
            if current_level > 0 else body_text
        )
        chunks.append(Chunk(
            chunk_id=chunk_id,
            text=full_text,
            heading=current_heading,
            heading_level=current_level,
            heading_path=current_path,
            metadata={"source": source, "parser": "html"},
        ))
        chunk_id += 1

    for element in body.children:
        if not isinstance(element, (Tag,)):
            continue

        tag_name = element.name.lower() if element.name else ""

        if tag_name in heading_tags:
            # Flush previous section
            flush_chunk()
            current_texts = []

            # Update heading state
            new_level = int(tag_name[1])
            new_heading = element.get_text(strip=True)

            # Maintain breadcrumb stack
            while heading_stack and heading_stack[-1][0] >= new_level:
                heading_stack.pop()
            current_path = _build_heading_path(heading_stack, new_heading)
            heading_stack.append((new_level, new_heading))

            current_heading = new_heading
            current_level = new_level
        else:
            # Accumulate content under current heading
            text = element.get_text(separator=" ", strip=True)
            if text:
                current_texts.append(text)

    # Flush final section
    flush_chunk()

    return chunks


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
        print(f"--- Chunk {c.chunk_id} | h{c.heading_level} | words={c.word_count} ---")
        print(f"    Heading : {c.heading}")
        print(f"    Path    : {c.heading_path}")
        if show_full:
            print(f"    Text    :\n{c.text}\n")
        else:
            print(f"    Preview : {c.text[:120].replace(chr(10), ' ')}...")
        print()


def show_document_outline(chunks: List[Chunk]) -> None:
    """Print the heading hierarchy as a document outline."""
    print(f"\n{'='*60}")
    print("Document Outline (derived from chunks)")
    print(f"{'='*60}\n")
    for c in chunks:
        indent = "  " * max(0, c.heading_level - 1)
        marker = f"h{c.heading_level}" if c.heading_level > 0 else "  "
        print(f"  {indent}[{marker}] {c.heading}  ({c.word_count} words)")


def simulate_filtered_retrieval(chunks: List[Chunk], section_keyword: str) -> None:
    """
    Show how heading metadata enables filtered retrieval.
    In a real vector DB: db.query(vec, filter={"heading_path": contains(keyword)})
    """
    print(f"\n{'='*60}")
    print(f"Filtered retrieval — heading_path contains: '{section_keyword}'")
    print(f"{'='*60}\n")
    matches = [c for c in chunks if section_keyword.lower() in c.heading_path.lower()]
    if not matches:
        print("  No matches found.")
    for c in matches:
        print(f"  Chunk {c.chunk_id}: [{c.heading_path}]")
        print(f"  Preview: {c.text[:100].replace(chr(10), ' ')}...")
        print()


# ---------------------------------------------------------------------------
# Main — experiments
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import os

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------------------------------------------------
    # EXPERIMENT 1: Markdown chunking — default (no size limit)
    # -----------------------------------------------------------------------
    print(">>> EXP 1: Markdown chunking — one chunk per section")
    md_text = load_text(os.path.join(base_dir, "sample_markdown.md"))
    md_chunks = markdown_chunk(md_text, source="sample_markdown.md")
    print_chunks(md_chunks)

    # -----------------------------------------------------------------------
    # EXPERIMENT 2: Document outline
    #   See the full heading hierarchy recovered from the document
    # -----------------------------------------------------------------------
    print(">>> EXP 2: Document outline extracted from chunks")
    show_document_outline(md_chunks)

    # -----------------------------------------------------------------------
    # EXPERIMENT 3: Filtered retrieval using heading metadata
    #   Simulate what a vector DB filter would do
    # -----------------------------------------------------------------------
    print("\n>>> EXP 3: Filtered retrieval — only 'Chunking' sections")
    simulate_filtered_retrieval(md_chunks, "Chunking")

    print(">>> EXP 3b: Filtered retrieval — only 'Embedding' sections")
    simulate_filtered_retrieval(md_chunks, "Embedding")

    # -----------------------------------------------------------------------
    # EXPERIMENT 4: Markdown with max_chunk_size (oversized sections split)
    # -----------------------------------------------------------------------
    print(">>> EXP 4: Markdown chunking with max_chunk_size=300")
    small_chunks = markdown_chunk(md_text, max_chunk_size=300, source="sample_markdown.md")
    print_chunks(small_chunks)
    print(f"Without size limit: {len(md_chunks)} chunks")
    print(f"With max 300 chars: {len(small_chunks)} chunks")

    # -----------------------------------------------------------------------
    # EXPERIMENT 5: HTML chunking
    # -----------------------------------------------------------------------
    print("\n>>> EXP 5: HTML chunking")
    html_text = load_text(os.path.join(base_dir, "sample_html.html"))
    html_chunks = html_chunk(html_text, source="sample_html.html")
    print_chunks(html_chunks)
    show_document_outline(html_chunks)

    # -----------------------------------------------------------------------
    # EXPERIMENT 6: Filtered retrieval on HTML chunks
    # -----------------------------------------------------------------------
    print(">>> EXP 6: HTML filtered retrieval — 'Search' sections")
    simulate_filtered_retrieval(html_chunks, "Search")
