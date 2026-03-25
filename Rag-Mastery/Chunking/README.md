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

*Coming next...*

---

## 6. Dynamic / Query-Aware Chunking

*Coming next...*

---

## Tradeoffs Summary

| Chunk size | Context quality | Retrieval precision | Storage cost |
|------------|----------------|---------------------|--------------|
| Large      | High            | Low (too broad)     | Low          |
| Small      | Low (fragmented)| High (too narrow)   | High         |
| Optimal    | Balanced        | Balanced            | Medium       |

> Rule of thumb: chunk size should match the *granularity of the facts* you expect queries to ask about.

---

## Observations (fill this after experiments)

```
Experiment 1 — Fixed, no overlap:


Experiment 2 — Fixed, with overlap=50:


What I noticed about boundaries:

```
