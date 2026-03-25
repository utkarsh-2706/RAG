# Chunking in RAG

> **Goal:** Understand how to split documents into meaningful pieces for embedding, indexing, and retrieval.

Chunking is the *most underrated* step in a RAG pipeline. Embedding quality, retrieval precision, and answer coherence all depend heavily on how well you chunk. A state-of-the-art LLM with bad chunking will underperform a simple model with great chunking.

---

## 1. Fixed Chunking

### Concept

Split text into fixed-size windows of N characters (or tokens), advancing by a step size each time. The step size is `chunk_size - overlap`.

```
Text (600 chars):
[  chunk 1 (300)  ][  chunk 2 (300)  ]   ← overlap = 0

With overlap = 50:
[  chunk 1 (300)  ]
             [  chunk 2 (300)  ]
                          [  chunk 3 (300)  ]
```

### Key Parameters

| Parameter    | What it controls                                   |
|--------------|----------------------------------------------------|
| `chunk_size` | Max characters per chunk                           |
| `overlap`    | Characters shared between consecutive chunks       |
| `step`       | `chunk_size - overlap` — how far the window moves  |

### How Overlap Helps

Without overlap, a sentence cut at a boundary loses its tail in chunk N and its head appears orphaned in chunk N+1. With overlap, both chunks contain the boundary region — increasing the chance that a query hitting that region finds coherent context.

### Tradeoffs

| Situation                          | Fixed Chunking behavior                                  |
|------------------------------------|----------------------------------------------------------|
| Simple, uniform text               | Works fine — fast, predictable                           |
| Sentences/paragraphs crossing boundary | **Breaks mid-sentence** — loses coherence            |
| Very small chunk_size              | Too many fragments, high retrieval noise                 |
| Very large chunk_size              | Chunks too broad, low retrieval precision                |
| Code or structured documents       | **Terrible** — ignores structure entirely                |

### When To Use

- Baseline / benchmarking (always start here)
- Quick prototyping when document structure doesn't matter
- Preprocessed text with already-uniform structure

### When It Fails

- Long, complex paragraphs (NLP documents, legal text)
- Any document where structure (headings, code blocks) carries meaning
- When retrieval precision matters more than speed

### Interview Insight

> "Fixed chunking is O(n) and requires no NLP models, making it the fastest approach. Its failure mode is boundary fragmentation — a query about a concept that spans a chunk boundary may retrieve neither chunk as a top result, since neither contains the full concept. Overlap mitigates this at the cost of redundancy (more chunks stored, more tokens processed at retrieval time)."

### Think About This

1. If you increase `chunk_size` from 300 to 1000 — what gets better and what gets worse during retrieval?
2. Would fixed chunking work well for Python source code? Why or why not?
3. Why does overlap increase chunk count? Can you derive the formula for total chunks given text length, chunk size, and overlap?

---

## 2. Sliding Window Chunking

### Concept

Think in terms of **window + stride**, not chunk_size + overlap.

```
Window W=5, Stride S=2:

Text:  [A B C D E F G H I J]

Win 1:  A B C D E
Win 2:      C D E F G         (moved 2 steps)
Win 3:          E F G H I
Win 4:              G H I J

Overlap = W - S = 3 units per pair
```

The stride is the *primary lever*. Overlap is derived from it.

### Key Parameters

| Parameter         | What it controls                            |
|-------------------|---------------------------------------------|
| `window_size`     | Characters (or sentences) per chunk         |
| `stride`          | How far the window advances each step        |
| `overlap`         | `window_size - stride` (derived)            |
| `overlap_ratio`   | `(window_size - stride) / window_size`      |

### The Core Insight: Stride Sweep

| Stride | Overlap % | Chunk Count | Use Case                        |
|--------|-----------|-------------|---------------------------------|
| 30     | 90%       | 138         | Very dense — research retrieval |
| 150    | 50%       | 28          | Balanced — most RAG systems     |
| 270    | 10%       | 16          | Near fixed chunking             |
| 300    | 0%        | 14          | Identical to fixed chunking     |

### Two Modes

- **Character-level**: Fast, no NLP dependencies. Still breaks mid-sentence.
- **Sentence-level**: Better linguistic boundaries. Chunks vary in character count.

### Tradeoffs

| Dimension       | High Overlap                          | Low Overlap                   |
|-----------------|---------------------------------------|-------------------------------|
| Recall          | Higher (boundary concepts recovered)  | Lower                         |
| Redundancy      | High (near-duplicate chunks stored)   | Low                           |
| Storage cost    | High                                  | Low                           |
| Context quality | Risk of returning similar chunks      | Cleaner distinct chunks       |

### When To Use

- When you need dense coverage of a document (legal review, QA over long docs)
- Sentence-level mode when linguistic coherence matters more than speed
- As a baseline with tunable overlap before moving to semantic chunking

### Interview Insight

> "Sliding window reframes fixed chunking by making overlap a design decision, not a patch.
> The stride controls coverage density. The failure mode is redundancy: with high overlap,
> top-k retrieval may return 3 chunks that are 80% identical — wasting your context window.
> A deduplication pass (e.g. MMR — Maximal Marginal Relevance) is often applied post-retrieval
> to solve this."

### Think About This

1. You retrieve top-3 chunks and all 3 have 80% overlap with each other. What is the real cost? (Hint: think context window tokens and the LLM's ability to reason over distinct information.)
2. Sentence-level mode produces variable-size chunks. Does this cause problems when embedding? Why or why not?
3. For a 500-page PDF legal contract, would you prefer high or low stride? What are you optimizing for in each case?

---

## 3. Recursive Chunking

### Concept

Try to split on the most meaningful boundary first. If a piece is still too large, recurse with the next separator. Stop when everything fits.

```
Separator priority: ["\n\n",  "\n",  ". ",  " ",  ""]
                      para    line   sent   word  char (last resort)

800-char paragraph -> split on "\n\n" -> still 800 chars
                   -> recurse with "\n" -> still too big
                   -> recurse with ". " -> sentences fit -> done
```

Key guarantee: **a chunk will never be broken mid-sentence unless it has no other choice.**

### How It Works

```python
separators = ["\n\n", "\n", ". ", " ", ""]

def recursive_split(text, separators, chunk_size):
    sep = first separator that exists in text
    pieces = text.split(sep)
    for piece in pieces:
        if len(piece) <= chunk_size:
            keep it
        else:
            recurse(piece, remaining_separators, chunk_size)
    merge small pieces back up to chunk_size with overlap
```

The `_merge_splits` step is critical — without it, splitting on `. ` would produce one sentence per chunk, all too small.

### Key Parameters

| Parameter     | What it controls                                     |
|---------------|------------------------------------------------------|
| `chunk_size`  | Max characters per chunk                             |
| `overlap`     | Char overlap between merged chunks                   |
| `separators`  | Priority-ordered split points (fully customizable)   |

### Boundary Quality (from Exp 2)

Every chunk in the default run ended with a period — `ends_clean=True` for all 10 chunks. Compare to Fixed chunking where chunk 0 ended mid-word inside "technologies".

### Fixed vs Recursive (Exp 3)

```
-- Fixed Chunk 0 ending --
'...the ability to automatically learn and '    ← mid-sentence!

-- Recursive Chunk 0 ending --
'...from healthcare to finance, manufacturing to education.'  ← clean!
```

### Tradeoffs

| Situation                        | Recursive behavior                                    |
|----------------------------------|-------------------------------------------------------|
| Well-structured paragraphs       | Excellent — one topic per chunk naturally             |
| Uniform text, no clear structure | Falls back to sentence/word splitting (still ok)      |
| chunk_size too small             | Forces word-level splits, many tiny fragments (Exp 4) |
| Custom doc types (code, HTML)    | Swap separator list — fully flexible                  |
| Speed                            | Slightly slower than fixed (recursive calls + merge)  |

### When To Use

- Default choice for most plain-text RAG pipelines
- Any document with paragraph or sentence structure
- When you want LangChain-compatible behavior (`RecursiveCharacterTextSplitter`)
- Before trying semantic chunking — this is the ceiling for rule-based approaches

### Custom Separator Lists (Exp 5)

```python
# For Markdown
separators = ["\n## ", "\n### ", "\n\n", "\n", ". ", " "]

# For Python code
separators = ["\nclass ", "\ndef ", "\n\n", "\n", " "]

# For HTML (after stripping tags)
separators = ["</p>", "</div>", "\n\n", "\n", ". "]
```

### Interview Insight

> "Recursive chunking is a greedy algorithm that walks a separator hierarchy.
> Its key advantage over fixed chunking is boundary coherence — chunks tend to
> end at natural linguistic breaks. The tradeoff is that chunk sizes become
> variable, which is fine for embedding but matters if downstream systems expect
> fixed-size inputs. The separator list is the main customization lever —
> changing it transforms recursive chunking into a structure-aware splitter."

### Think About This

1. In Exp 4, `chunk_size=100` produced 57 chunks with some as short as 19 chars. What's the retrieval problem with 19-char chunks? How would you solve it?
2. The separator list `["\n\n", "\n", ". ", " ", ""]` is designed for English prose. How would you modify it for Python source code? What new separators would you add?
3. Recursive chunking still doesn't understand *meaning* — it just finds the nearest clean boundary. What kind of query would still fail even with perfect boundary splitting?

---

## 4. Semantic Chunking

### Concept

Embed every sentence. Compute cosine similarity between consecutive sentence embeddings. Split where similarity drops sharply — that's where the topic changed.

```
Sentences:   [S1,  S2,  S3,  S4,  S5,  S6]
Similarity:      0.91  0.88  0.21  0.87  0.90
                              ^^^^
                         topic shift → split here

Result: [S1, S2, S3] | [S4, S5, S6]
```

### How It Works (Pipeline)

```
1. Split text into sentences
2. Embed each sentence  →  vectors [E1, E2, ..., En]
3. Compute sim(Ei, Ei+1) for all consecutive pairs
4. Find breakpoints using threshold OR percentile strategy
5. Group sentences between breakpoints into chunks
```

### Two Splitting Strategies

| Strategy    | How it splits                              | Best when                              |
|-------------|-------------------------------------------|----------------------------------------|
| Threshold   | Split where sim < fixed value (e.g. 0.5)  | You know your domain's similarity range|
| Percentile  | Split at bottom N% of similarity scores   | Unknown or varied document types       |

**Percentile is almost always better in practice** — it adapts to each document.

### Sensitivity Sweep (from Exp 4)

```
Percentile    Chunks   Avg words/chunk
        10         5             114.0   ← very coarse
        25         9              63.3   ← balanced
        40        14              40.7
        60        20              28.5
        75        25              22.8   ← very fine-grained
```

### Clearest Topic Shift Detected (Exp 5)

```
sim = 0.108  (lowest in document)

End of topic A:   "Popular vector databases include Pinecone, Weaviate, Chroma, and FAISS."
Start of topic B: "Chunking is the process of splitting large documents..."
```
The model correctly identified the boundary between "vector databases" and "chunking" as the strongest topic shift — with no rules, just geometry.

### Threshold = 0.5 Problem (Exp 2)

With threshold=0.5, 24 chunks were produced — many single-sentence fragments (`avg_sim=1.000` means only 1 sentence). This is **over-splitting**: the threshold was too aggressive. Single-sentence chunks carry almost no context for the LLM.

### Tradeoffs

| Dimension           | Semantic Chunking                                        |
|---------------------|----------------------------------------------------------|
| Boundary quality    | Best of all methods — semantically motivated             |
| Speed               | Slowest — requires embedding every sentence              |
| Cost                | API embedding calls cost money at scale                  |
| Determinism         | Changes if embedding model changes                       |
| Tiny fragments      | Risk with low percentile or high threshold               |
| Long coherent paras | May keep them whole even if large (no size enforcement)  |

### Hybrid Approach (Production Pattern)

```python
# Semantic chunking + max size guard
chunks = semantic_chunk_percentile(text, model, percentile=25)
# Then split any oversized chunks with recursive chunking
final = []
for c in chunks:
    if c.char_count > MAX_CHARS:
        final.extend(recursive_chunk(c.text, chunk_size=MAX_CHARS))
    else:
        final.append(c)
```

### When To Use

- High-quality RAG over dense, multi-topic documents (research papers, reports)
- When retrieval precision matters more than ingestion speed
- As the gold standard to compare other methods against

### Interview Insight

> "Semantic chunking treats the document as a sequence of meaning vectors, not characters.
> It finds natural topic transitions using embedding similarity — the same geometry that
> powers retrieval. The key failure mode is over-splitting: setting the percentile too high
> creates single-sentence chunks that lack enough context for an LLM to reason with.
> In production, semantic chunking is paired with a max-size guard using recursive chunking
> as a fallback. The embedding model choice matters: a domain-specific model will detect
> topic shifts more accurately than a general-purpose one."

### Think About This

1. In Exp 2, threshold=0.5 produced many 1-sentence chunks. Why does `avg_sim=1.000` mean a chunk has only one sentence? How would you fix the over-splitting problem?
2. Exp 5 found the clearest split between "vector databases" and "chunking" with sim=0.108. Why would these two topics have lower similarity than, say, "deep learning" and "NLP"?
3. Semantic chunking uses the same embedding model for both chunking and retrieval. What happens if you chunk with model A but retrieve with model B? Is this a problem?

---

## 5. Structure-Aware Chunking

### Concept

Documents like Markdown and HTML already encode structure through headers. Parse that skeleton first, then chunk along those boundaries. Each chunk carries its full heading **path** as metadata — enabling retrieval filtering on top of vector similarity.

```
## 2. Chunking Strategies        → Chunk { heading: "2. Chunking Strategies",
### 2.1 Fixed Chunking                     heading_path: "RAG Guide > 2. Chunking > 2.1 Fixed" }
### 2.2 Semantic Chunking
```

### The Key Idea: Heading Path

Every chunk knows *where it lives* in the document:

```
"RAG Systems: A Complete Guide > 3. Embedding Models > 3.1 Choosing an Embedding Model"
```

This enables **hybrid retrieval**:
```python
# Vector similarity + structural filter combined
results = db.query(
    query_vector=embed("how to choose an embedding model"),
    filter={"heading_path": {"$contains": "Embedding"}}
)
```
Without structure-aware chunking, you can only do the first half.

### Document Outline (Exp 2)

The chunker automatically reconstructed the full document hierarchy:
```
[h1] RAG Systems: A Complete Guide  (29 words)
  [h2] 1. What is RAG?  (62 words)
    [h3] 1.1 Key Components  (40 words)
    [h3] 1.2 When to Use RAG  (43 words)
  [h2] 2. Chunking Strategies  (33 words)
    [h3] 2.1 Fixed Chunking  (56 words)
    ...
```
No NLP model needed — pure structure parsing.

### Filtered Retrieval (Exp 3)

```
Query: heading_path contains "Chunking"
  → Chunk 4: [> 2. Chunking Strategies]
  → Chunk 5: [> 2. Chunking Strategies > 2.1 Fixed Chunking]
  → Chunk 6: [> 2. Chunking Strategies > 2.2 Semantic Chunking]
```
3 exact-match chunks, zero false positives — from a pure metadata filter before any vector search.

### Two Parsers

| Parser     | Input      | Split signal          | Dependency     |
|------------|------------|-----------------------|----------------|
| Markdown   | `.md` text | `#`, `##`, `###` regex| None           |
| HTML       | `.html`    | `<h1>`–`<h4>` tags    | BeautifulSoup  |

### max_chunk_size Guard (Exp 4)

When a section exceeds `max_chunk_size`, it splits on paragraph boundaries and tags continued chunks with `[continued] heading_name`. This keeps heading metadata intact across splits.

```
Without limit: 16 chunks
With 300 chars: 20 chunks  (4 oversized sections split into 2 each)
```

### Tradeoffs

| Situation                        | Structure-Aware behavior                              |
|----------------------------------|-------------------------------------------------------|
| Well-structured Markdown/HTML    | Excellent — best chunk-to-topic alignment possible    |
| Plain text with no headers       | Fails completely — falls back to one big chunk        |
| Inconsistent heading structure   | Heading path becomes unreliable                       |
| PDF documents                    | Requires PDF parser (pdfminer, PyMuPDF) first         |
| Code files                       | Use `class`/`def` as separators instead of `#`        |

### Production Pattern: Structure + Semantic Hybrid

```python
# Phase 1: Structure-aware split (coarse, with metadata)
sections = markdown_chunk(doc)

# Phase 2: Semantic split within each section (fine, coherent)
final_chunks = []
for section in sections:
    if section.char_count > MAX_SIZE:
        sub = semantic_chunk_percentile(section.text, model, percentile=25)
        for s in sub:
            s.metadata["heading_path"] = section.heading_path  # preserve metadata
        final_chunks.extend(sub)
    else:
        final_chunks.append(section)
```

### When To Use

- Any document with consistent heading structure (docs, reports, wikis, manuals)
- When you need retrieval filtering by section, chapter, or topic
- As the outer layer of a hybrid chunking pipeline

### Interview Insight

> "Structure-aware chunking leverages the document author's own intent — headings are human-annotated
> topic boundaries. The unique advantage over semantic chunking is metadata richness: heading paths
> enable pre-filtering before vector search, dramatically reducing the search space and false positives.
> The failure mode is brittle coupling to formatting — a document with no headers, inconsistent
> heading levels, or generated from a bad PDF extraction will produce garbage chunks."

### Think About This

1. A user queries: *"What are the retrieval metrics in RAG?"* — With heading path metadata, how would you combine vector search AND structural filtering to get a better answer than vector search alone?
2. You receive a PDF document. Structure-aware chunking won't work directly. What preprocessing pipeline would you build to make it work?
3. In Exp 4, the `[continued]` tag was added to split chunks. What problem does this create at retrieval time, and how would you fix it?

---

## 6. Dynamic / Query-Aware Chunking

### Concept

Every previous technique makes a single fixed bet at ingestion time. Dynamic chunking separates the retrieval unit from the generation unit:

```
Traditional:  INDEX [chunk] --> RETRIEVE [chunk] --> LLM gets [chunk]
                                                      (same size always)

Dynamic:      INDEX [small] --> RETRIEVE [small] --> EXPAND --> LLM gets [large]
                                  (precise)                      (rich context)
```

**The unit you index for retrieval != the unit you return to the LLM.**

### Three Strategies

#### Strategy 1: Small-to-Big (Parent-Document Retrieval)

Build a two-level hierarchy at ingestion. Index small children. Return large parents.

```
Parent (400 chars) = "RAG enhances LLMs by grounding responses..."
  Child 1 (100 chars) = "RAG enhances LLMs..."         <- indexed, gets embedding
  Child 2 (100 chars) = "Instead of relying solely..." <- indexed, gets embedding
  Child 3 (100 chars) = "This reduces hallucinations..." <- indexed, gets embedding

Query hits Child 3 -> return full Parent to LLM (4x expansion)
```

#### Strategy 2: Sentence Window Retrieval

Index every sentence. On retrieval, expand to k sentences before + after.

```
Matched sentence: "This approach reduces hallucinations..."  [11 words]
Window (k=2):     [sentence-2] [sentence-1] [MATCH] [sentence+1] [sentence+2]
Returned to LLM:  105 words  (9.5x expansion)
```

The window adapts to the match location — every query gets a different context window.

#### Strategy 3: Contextual Compression

Retrieve a full chunk, then compress it down to only the query-relevant sentences.

```
Chunk retrieved: 59 words (full paragraph about RAG)
After compression: 40 words (only the 2 sentences most relevant to the query)
Expansion: 0.7x  <- smaller than indexed! Opposite direction.
```

In production, compression is done by an LLM call:
```python
compressed = llm(f"Extract only sentences relevant to: '{query}'\n\n{chunk}")
```

### Size Comparison (Exp 4)

```
Query: "How does RAG reduce hallucinations?"

Strategy                  Indexed   Returned   Expansion
Small-to-Big                  13w       52w       4.0x
Sentence Window               11w      105w       9.5x
Contextual Compression        59w       40w       0.7x
```

Three different philosophies: expand 4x, expand 9.5x, or compress to 0.7x — all from the same document, same query.

### Window Size Sweep (Exp 5)

```
Window    Words returned
     0              11    <- just the matched sentence
     1              56
     2             105    <- sweet spot for most RAG systems
     3             124
     5             192    <- risk of context dilution
```

### Exp 6: Same Index, Query-Dependent Context

When the query changes from "hallucinations" to "vector databases", the sentence window retrieval automatically surfaces different sentences and expands a different context window — **zero re-indexing needed**.

### Tradeoffs

| Strategy               | Retrieval precision | Context richness | Cost                      |
|------------------------|---------------------|------------------|---------------------------|
| Small-to-Big           | High (small index)  | High (parent)    | 2x storage (parent+child) |
| Sentence Window        | Very high (sentence)| Tunable via k    | 1x storage, expand at query|
| Contextual Compression | Normal              | Targeted         | Extra LLM call per result |

### When To Use

| Need                                          | Strategy                  |
|-----------------------------------------------|---------------------------|
| Best retrieval precision + full context       | Small-to-Big              |
| Exact sentence match + surrounding context    | Sentence Window           |
| Reduce context window usage, focused answers  | Contextual Compression    |
| All of the above in production                | Combine all three         |

### Interview Insight

> "Dynamic chunking decouples the indexing granularity from the generation context size.
> This resolves the fundamental RAG tradeoff: small chunks give sharp embeddings for
> retrieval, large chunks give coherent context for generation. Small-to-Big and Sentence
> Window are ingestion-time decisions (no extra cost at query time). Contextual Compression
> is a query-time decision (extra LLM call but most targeted context). In production,
> these strategies are combined: structure-aware chunking gives sections, semantic chunking
> splits those sections, small-to-big indexes the sentences, and compression trims the
> returned context per query."

### Think About This

1. In Exp 4, Sentence Window returned 9.5x more words than it indexed. At what point does the expansion *hurt* the LLM? What is the risk when `window_size=5`?
2. Small-to-Big requires 2x storage (both parents and children). When is this storage cost worth paying vs when would you use Sentence Window instead?
3. Contextual Compression gives 0.7x (less than indexed). Could compression ever *remove* the answer from the context? How would you detect and fix this failure?

---

## Tradeoffs Summary

### Chunk Size

| Chunk size | Context quality  | Retrieval precision | Storage cost |
|------------|------------------|---------------------|--------------|
| Large      | High             | Low (too broad)     | Low          |
| Small      | Low (fragmented) | High (too narrow)   | High         |
| Optimal    | Balanced         | Balanced            | Medium       |

> Rule of thumb: chunk size should match the *granularity of the facts* you expect queries to ask about.

### Strategy Selection Guide

| Document type                  | Best strategy                              |
|--------------------------------|--------------------------------------------|
| Plain text, no structure       | Recursive -> Semantic                      |
| Markdown / HTML with headers   | Structure-Aware + Recursive fallback       |
| Dense research papers          | Semantic + Small-to-Big                    |
| Customer support / FAQ docs    | Structure-Aware + Sentence Window          |
| Code files                     | Recursive with code-specific separators    |
| Need highest retrieval quality | Semantic -> Small-to-Big -> Compression    |
| Fast prototype / baseline      | Fixed -> Recursive                         |

### Technique Progression

```
Fixed Chunking
    |-- adds boundary awareness --> Sliding Window
    |-- adds structural respect --> Recursive Chunking
    |-- adds semantic awareness --> Semantic Chunking
    |-- adds document structure --> Structure-Aware Chunking
    |-- adds query-time expand  --> Dynamic Chunking
                                    (combines all the above)
```

Each technique fixes the failure mode of the previous one.
In production, you combine multiple techniques rather than pick just one.

---

## Observations (fill this after experiments)

```
Experiment 1 — Fixed, no overlap:


Experiment 2 — Fixed, with overlap=50:


What I noticed about boundaries:

```
