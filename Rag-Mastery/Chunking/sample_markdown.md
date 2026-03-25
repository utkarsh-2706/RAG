# RAG Systems: A Complete Guide

Retrieval-Augmented Generation (RAG) is a framework that combines information retrieval with language generation. This guide covers everything from basic concepts to production deployment.

## 1. What is RAG?

RAG enhances large language models by grounding their responses in retrieved documents. Instead of relying solely on parametric knowledge baked into model weights during training, RAG systems dynamically fetch relevant information at inference time.

The core motivation is simple: LLMs hallucinate because they must answer from memory. RAG gives them a reference library to consult before answering.

### 1.1 Key Components

Every RAG system has three core components: a document store, a retriever, and a generator. The document store holds the knowledge base. The retriever finds relevant pieces. The generator uses those pieces to produce an answer.

### 1.2 When to Use RAG

Use RAG when your application requires up-to-date information, domain-specific knowledge not in the base model, or verifiable citations. It is particularly valuable in enterprise settings where proprietary data must be searchable but cannot be used for fine-tuning.

## 2. Chunking Strategies

Chunking is the process of splitting documents into pieces small enough to embed and retrieve efficiently. Poor chunking is the single most common cause of RAG failures in production.

### 2.1 Fixed Chunking

Fixed chunking splits text every N characters regardless of content. It is fast and deterministic but ignores sentence and paragraph boundaries. Best used as a baseline only.

```python
def fixed_chunk(text, chunk_size=500, overlap=50):
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(text), step):
        chunks.append(text[start:start + chunk_size])
    return chunks
```

### 2.2 Semantic Chunking

Semantic chunking embeds sentences and splits where cosine similarity between adjacent sentences drops below a threshold. It produces topically coherent chunks at the cost of embedding computation.

The key parameter is the similarity threshold or percentile cutoff. A 25th percentile split means: split at the 25% lowest-similarity transitions in the document.

## 3. Embedding Models

Embedding models convert text into dense vectors. The choice of model directly impacts retrieval quality. General-purpose models like all-MiniLM-L6-v2 work well for most use cases. Domain-specific models outperform general ones in specialized fields like medicine or law.

### 3.1 Choosing an Embedding Model

Consider three factors: dimensionality (higher is not always better), context window (max tokens the model can encode), and domain alignment (was the model trained on similar text?). For most RAG systems, a 384- or 768-dimensional model with a 512-token context window is sufficient.

### 3.2 Embedding Costs

OpenAI's text-embedding-3-small costs $0.02 per million tokens. For a 1000-document knowledge base averaging 5000 tokens each, that is $0.10 per full re-embedding. Local models like all-MiniLM-L6-v2 are free but require GPU for high throughput.

## 4. Vector Databases

Vector databases store embeddings and support approximate nearest neighbor (ANN) search. They are the retrieval engine of a RAG system.

### 4.1 Comparison

FAISS is a library, not a server — best for offline or single-process use. Chroma is lightweight and easy to set up locally. Pinecone is fully managed and scales to billions of vectors. Weaviate supports hybrid search (vector + keyword) natively.

### 4.2 Indexing Strategies

Most vector databases use HNSW (Hierarchical Navigable Small World) graphs for ANN search. HNSW trades index build time for fast query time. For datasets under 100k vectors, flat (exact) search is often fast enough and avoids approximation errors.

## 5. Evaluation

RAG evaluation is harder than standard NLP evaluation because it has two stages: retrieval and generation. A failure can occur in either stage.

### 5.1 Retrieval Metrics

Measure whether the right chunks were retrieved. Key metrics: Hit Rate (was the answer chunk in top-k?), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain).

### 5.2 Generation Metrics

Measure whether the answer was correct and grounded. RAGAS provides three metrics: faithfulness (answer supported by context?), answer relevance (answer addresses the question?), context precision (was retrieved context useful?).
