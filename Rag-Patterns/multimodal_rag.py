# multimodal_rag.py
# Multimodal RAG — Cross-modal retrieval over text + image embeddings
# Core idea: unified embedding space enables text-to-image and image-to-text search

import numpy as np
from typing import List, Union
from dataclasses import dataclass
from enum import Enum


# ─────────────────────────────────────────────
# 1. MODALITY TYPES
# ─────────────────────────────────────────────

class Modality(Enum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"


# ─────────────────────────────────────────────
# 2. MULTIMODAL DOCUMENT STORE
# In production: images stored as embeddings + thumbnail URLs
# ─────────────────────────────────────────────

@dataclass
class MultimodalDocument:
    id: str
    modality: Modality
    content: str        # Text content OR image caption/description
    metadata: dict      # e.g., {"source": "product_catalog", "url": "..."}


DOCUMENT_STORE: List[MultimodalDocument] = [
    # Text documents
    MultimodalDocument("t1", Modality.TEXT,
        "RAG systems combine retrieval with generation to reduce hallucination.",
        {"source": "research_paper"}),
    MultimodalDocument("t2", Modality.TEXT,
        "CLIP aligns text and image embeddings in a shared vector space.",
        {"source": "research_paper"}),
    MultimodalDocument("t3", Modality.TEXT,
        "Vector databases support approximate nearest neighbor search at scale.",
        {"source": "documentation"}),

    # Image documents (represented by their descriptions/captions)
    MultimodalDocument("i1", Modality.IMAGE,
        "A diagram showing the RAG pipeline: query → retriever → LLM → answer.",
        {"source": "architecture_diagram", "url": "rag_pipeline.png"}),
    MultimodalDocument("i2", Modality.IMAGE,
        "A scatter plot showing text and image embeddings clustered in CLIP space.",
        {"source": "research_figure", "url": "clip_embedding_space.png"}),
    MultimodalDocument("i3", Modality.IMAGE,
        "A product photo: red running shoes with white sole, size indicator visible.",
        {"source": "product_catalog", "url": "product_001.jpg"}),

    # Table documents
    MultimodalDocument("tb1", Modality.TABLE,
        "Table: RAG Pattern | Retrieval Type | Key Innovation\n"
        "Simple RAG | Dense | Basic grounding\nHyDE | Hypothesis | Query transformation",
        {"source": "comparison_table"}),
]


# ─────────────────────────────────────────────
# 3. MULTIMODAL ENCODER (CLIP-style simulation)
# In production: use openai/clip-vit-large-patch14 or sentence-transformers CLIP
# ─────────────────────────────────────────────

class CLIPEncoder:
    """
    Simulates CLIP-style cross-modal encoding.

    Real usage:
    ─────────────────────────────────────────────
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # Encode text
    inputs = processor(text=["a dog"], return_tensors="pt", padding=True)
    text_features = model.get_text_features(**inputs)

    # Encode image
    inputs = processor(images=image, return_tensors="pt")
    image_features = model.get_image_features(**inputs)

    # Both live in same 512-dim space — cross-modal search works directly
    ─────────────────────────────────────────────
    """
    def __init__(self, dim: int = 64):
        self.dim = dim

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into shared embedding space."""
        np.random.seed(abs(hash(text)) % (2**32))
        return np.random.randn(self.dim).astype(np.float32)

    def encode_image_from_description(self, description: str) -> np.ndarray:
        """
        Simulate image encoding via caption.
        In production: encode actual image pixels through CLIP vision encoder.
        The key CLIP property: encode_text("a dog") ≈ encode_image(dog_photo)
        """
        # Simulate alignment: image embeddings are "near" their text descriptions
        # We add small noise to text embedding to simulate vision encoder output
        text_vec = self.encode_text(description)
        noise = np.random.randn(self.dim).astype(np.float32) * 0.1
        return text_vec + noise  # Close but not identical to text encoding

    def encode(self, content: str, modality: Modality) -> np.ndarray:
        if modality == Modality.IMAGE:
            return self.encode_image_from_description(content)
        else:  # TEXT and TABLE both use text encoder
            return self.encode_text(content)


# ─────────────────────────────────────────────
# 4. MULTIMODAL INDEX
# ─────────────────────────────────────────────

class MultimodalIndex:
    def __init__(self, encoder: CLIPEncoder):
        self.encoder = encoder
        self.documents: List[MultimodalDocument] = []
        self.embeddings: List[np.ndarray] = []

    def add_documents(self, docs: List[MultimodalDocument]):
        print(f"Indexing {len(docs)} multimodal documents...")
        for doc in docs:
            embedding = self.encoder.encode(doc.content, doc.modality)
            self.documents.append(doc)
            self.embeddings.append(embedding)
            print(f"  [{doc.modality.value.upper():6s}] {doc.id}: {doc.content[:50]}...")
        print(f"Index built: {len(self.documents)} documents\n")

    def search(self, query_vec: np.ndarray, top_k: int = 3,
               filter_modality: Modality = None) -> List[tuple]:
        scores = []
        for i, emb in enumerate(self.embeddings):
            doc = self.documents[i]
            # Apply modality filter if specified
            if filter_modality and doc.modality != filter_modality:
                continue
            sim = float(np.dot(query_vec, emb) /
                       (np.linalg.norm(query_vec) * np.linalg.norm(emb) + 1e-9))
            scores.append((sim, doc))
        return sorted(scores, reverse=True)[:top_k]


# ─────────────────────────────────────────────
# 5. CONTEXT ASSEMBLER
# Converts multimodal results into LLM-readable format
# ─────────────────────────────────────────────

def assemble_context(results: List[tuple]) -> str:
    """
    Convert retrieved multimodal docs into a unified context string.

    In production for real images:
    - Images → base64 encoded, passed to vision LLM (GPT-4V, Claude 3)
    - Tables → converted to markdown
    - Audio → transcribed to text first
    """
    context_parts = []
    for score, doc in results:
        if doc.modality == Modality.TEXT:
            context_parts.append(f"[TEXT | {score:.3f}]\n{doc.content}")
        elif doc.modality == Modality.IMAGE:
            # In production: include actual image or base64
            context_parts.append(
                f"[IMAGE | {score:.3f}] (URL: {doc.metadata.get('url', 'N/A')})\n"
                f"Visual content: {doc.content}"
            )
        elif doc.modality == Modality.TABLE:
            context_parts.append(f"[TABLE | {score:.3f}]\n{doc.content}")
    return "\n\n".join(context_parts)


# ─────────────────────────────────────────────
# 6. MULTIMODAL LLM CALL (simulated)
# In production: GPT-4V, Claude 3, LLaVA
# ─────────────────────────────────────────────

def call_multimodal_llm(query: str, context: str) -> str:
    """
    In production with vision LLM:
    ─────────────────────────────────────────────
    response = openai_client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": f"Context:\n{text_context}\n\nQuestion: {query}"},
                {"type": "image_url", "image_url": {"url": image_url}}
                # Add more images as needed
            ]
        }]
    )
    ─────────────────────────────────────────────
    """
    print("\n[Multimodal LLM]")
    print("─" * 60)
    print(f"Query  : {query}")
    print(f"Context:\n{context}")
    print("─" * 60)
    return "[Simulated multimodal answer — text + visual context used]"


# ─────────────────────────────────────────────
# 7. MULTIMODAL RAG PIPELINE
# ─────────────────────────────────────────────

class MultimodalRAG:
    def __init__(self, documents: List[MultimodalDocument], top_k: int = 3):
        self.encoder = CLIPEncoder(dim=64)
        self.index = MultimodalIndex(self.encoder)
        self.index.add_documents(documents)
        self.top_k = top_k

    def query(self, user_query: str, query_modality: Modality = Modality.TEXT,
              filter_modality: Modality = None) -> str:
        print(f"\n{'='*60}")
        print(f"QUERY [{query_modality.value.upper()}]: {user_query}")
        if filter_modality:
            print(f"Filter: return only {filter_modality.value.upper()} results")

        # Step 1: Encode query (text or image)
        query_vec = self.encoder.encode(user_query, query_modality)

        # Step 2: Cross-modal search
        results = self.index.search(query_vec, self.top_k, filter_modality)

        print(f"\nTop-{self.top_k} Retrieved (cross-modal):")
        for score, doc in results:
            print(f"  [{doc.modality.value.upper():6s} | {score:.3f}] {doc.content[:60]}...")

        # Step 3: Assemble context
        context = assemble_context(results)

        # Step 4: Generate
        answer = call_multimodal_llm(user_query, context)
        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 8. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = MultimodalRAG(DOCUMENT_STORE, top_k=3)

    print("\n--- Test 1: Text query → retrieves text + images ---")
    rag.query("How does RAG work and show me diagrams?")

    print("\n\n--- Test 2: Text query filtered to images only ---")
    rag.query("Show me visual representations of embeddings", filter_modality=Modality.IMAGE)

    print("\n\n--- Test 3: Image description query → cross-modal retrieval ---")
    rag.query(
        "product photo: athletic footwear red and white",
        query_modality=Modality.IMAGE
    )