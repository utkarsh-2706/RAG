# graph_rag.py
# Graph RAG — Knowledge graph construction + traversal + LLM generation
# Core idea: entities as nodes, relationships as edges, traversal for context

from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np


# ─────────────────────────────────────────────
# 1. GRAPH DATA STRUCTURES
# ─────────────────────────────────────────────

@dataclass
class Entity:
    id: str
    name: str
    entity_type: str    # PERSON, COMPANY, PRODUCT, CONCEPT, EVENT
    description: str = ""

@dataclass
class Relationship:
    source_id: str
    target_id: str
    relation_type: str  # CEO_OF, MADE_BY, COMPETITOR_OF, PART_OF, RELATED_TO
    properties: dict = field(default_factory=dict)


class KnowledgeGraph:
    def __init__(self):
        self.entities: Dict[str, Entity] = {}           # id → Entity
        self.adjacency: Dict[str, List[Relationship]] = defaultdict(list)
        self.reverse_adjacency: Dict[str, List[Relationship]] = defaultdict(list)

    def add_entity(self, entity: Entity):
        self.entities[entity.id] = entity

    def add_relationship(self, rel: Relationship):
        self.adjacency[rel.source_id].append(rel)
        self.reverse_adjacency[rel.target_id].append(rel)

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def get_neighbors(self, entity_id: str, relation_filter: str = None) -> List[Tuple[str, Relationship]]:
        """Returns (neighbor_id, relationship) pairs for outgoing edges."""
        rels = self.adjacency.get(entity_id, [])
        if relation_filter:
            rels = [r for r in rels if r.relation_type == relation_filter]
        return [(r.target_id, r) for r in rels]

    def get_all_entities(self) -> List[Entity]:
        return list(self.entities.values())


# ─────────────────────────────────────────────
# 2. KNOWLEDGE GRAPH CONSTRUCTION
# (Simulates LLM-based entity + relation extraction)
# In production: use LLM prompts to extract from raw text
# ─────────────────────────────────────────────

def build_knowledge_graph() -> KnowledgeGraph:
    """
    In production, this would run over documents:
    ─────────────────────────────────────────────────────────
    for chunk in document_chunks:
        prompt = f\"\"\"Extract entities and relationships from:
        {chunk}

        Return JSON:
        {{
          "entities": [{{"id": ..., "name": ..., "type": ..., "description": ...}}],
          "relationships": [{{"source": ..., "target": ..., "relation": ...}}]
        }}\"\"\"
        result = call_llm(prompt)
        # Add to graph, resolve duplicate entities
    ─────────────────────────────────────────────────────────
    """
    kg = KnowledgeGraph()

    # ── Entities ──
    entities = [
        Entity("e1", "RAG", "CONCEPT", "Retrieval-Augmented Generation — grounds LLMs in retrieved docs"),
        Entity("e2", "LLM", "CONCEPT", "Large Language Model — transformer-based text generator"),
        Entity("e3", "Vector Database", "CONCEPT", "Stores dense embeddings for ANN search"),
        Entity("e4", "FAISS", "PRODUCT", "Meta's library for fast similarity search"),
        Entity("e5", "Pinecone", "PRODUCT", "Managed vector database service"),
        Entity("e6", "OpenAI", "COMPANY", "AI research company, creator of GPT series"),
        Entity("e7", "GPT-4", "PRODUCT", "OpenAI's flagship large language model"),
        Entity("e8", "Self-RAG", "CONCEPT", "RAG variant with LLM self-evaluation via reflection tokens"),
        Entity("e9", "CRAG", "CONCEPT", "Corrective RAG — validates retrieved docs before generation"),
        Entity("e10", "Embedding Model", "CONCEPT", "Converts text to dense vector representations"),
        Entity("e11", "Anthropic", "COMPANY", "AI safety company, creator of Claude"),
        Entity("e12", "Claude", "PRODUCT", "Anthropic's LLM assistant"),
        Entity("e13", "Hallucination", "CONCEPT", "LLM generating factually incorrect information"),
        Entity("e14", "HyDE", "CONCEPT", "Hypothetical Document Embeddings — embed generated hypothesis for retrieval"),
    ]
    for e in entities:
        kg.add_entity(e)

    # ── Relationships ──
    relationships = [
        Relationship("e1", "e2", "USES"),                    # RAG → USES → LLM
        Relationship("e1", "e3", "USES"),                    # RAG → USES → Vector Database
        Relationship("e1", "e13", "REDUCES"),                # RAG → REDUCES → Hallucination
        Relationship("e2", "e13", "CAUSES"),                 # LLM → CAUSES → Hallucination
        Relationship("e3", "e4", "IMPLEMENTED_BY"),          # Vector DB → IMPLEMENTED_BY → FAISS
        Relationship("e3", "e5", "IMPLEMENTED_BY"),          # Vector DB → IMPLEMENTED_BY → Pinecone
        Relationship("e6", "e7", "CREATED"),                 # OpenAI → CREATED → GPT-4
        Relationship("e7", "e2", "IS_A"),                    # GPT-4 → IS_A → LLM
        Relationship("e8", "e1", "EXTENDS"),                 # Self-RAG → EXTENDS → RAG
        Relationship("e9", "e1", "EXTENDS"),                 # CRAG → EXTENDS → RAG
        Relationship("e14", "e1", "EXTENDS"),                # HyDE → EXTENDS → RAG
        Relationship("e10", "e3", "FEEDS_INTO"),             # Embedding Model → FEEDS_INTO → Vector DB
        Relationship("e11", "e12", "CREATED"),               # Anthropic → CREATED → Claude
        Relationship("e12", "e2", "IS_A"),                   # Claude → IS_A → LLM
        Relationship("e8", "e9", "RELATED_TO"),              # Self-RAG ↔ CRAG
        Relationship("e6", "e11", "COMPETITOR_OF"),          # OpenAI ↔ Anthropic
    ]
    for r in relationships:
        kg.add_relationship(r)

    return kg


# ─────────────────────────────────────────────
# 3. VECTOR INDEX OVER ENTITY NAMES
# Used to find entry point into graph from user query
# ─────────────────────────────────────────────

def simulate_embedding(text: str, dim: int = 64) -> np.ndarray:
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.randn(dim).astype(np.float32)

def cosine_similarity(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def find_entry_entities(query: str, kg: KnowledgeGraph, top_k: int = 2) -> List[Entity]:
    """Find the most relevant entities as graph entry points."""
    q_vec = simulate_embedding(query)
    scored = []
    for entity in kg.get_all_entities():
        # Embed entity name + description
        entity_text = f"{entity.name} {entity.description}"
        e_vec = simulate_embedding(entity_text)
        score = cosine_similarity(q_vec, e_vec)
        scored.append((score, entity))
    scored.sort(reverse=True)
    return [e for _, e in scored[:top_k]]


# ─────────────────────────────────────────────
# 4. GRAPH TRAVERSAL (BFS)
# ─────────────────────────────────────────────

def traverse_graph(kg: KnowledgeGraph, start_entities: List[Entity],
                   max_depth: int = 2, max_nodes: int = 10) -> List[str]:
    """
    BFS traversal from entry nodes.
    Returns serialized graph paths as context strings.

    In production: use Cypher queries (Neo4j) for precise control:
    MATCH path = (start)-[*1..2]-(related)
    WHERE start.name IN ['RAG', 'Self-RAG']
    RETURN path LIMIT 20
    """
    visited: Set[str] = set()
    context_lines: List[str] = []
    queue = deque([(entity.id, 0) for entity in start_entities])

    # Add starting entities
    for entity in start_entities:
        visited.add(entity.id)
        context_lines.append(
            f"[ENTITY] {entity.name} ({entity.entity_type}): {entity.description}"
        )

    while queue and len(visited) < max_nodes:
        entity_id, depth = queue.popleft()

        if depth >= max_depth:
            continue

        # Traverse outgoing edges
        for neighbor_id, rel in kg.get_neighbors(entity_id):
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                neighbor = kg.get_entity(neighbor_id)
                source = kg.get_entity(entity_id)

                if neighbor and source:
                    # Serialize relationship as natural language
                    context_lines.append(
                        f"[RELATION] {source.name} ──[{rel.relation_type}]──► {neighbor.name}"
                        + (f": {neighbor.description}" if neighbor.description else "")
                    )
                    queue.append((neighbor_id, depth + 1))

    return context_lines


# ─────────────────────────────────────────────
# 5. CONTEXT SERIALIZER + LLM
# ─────────────────────────────────────────────

def build_graph_prompt(query: str, graph_context: List[str]) -> str:
    context = "\n".join(graph_context)
    return f"""You are answering based on a knowledge graph.
The context below contains entities and their relationships.

=== Knowledge Graph Context ===
{context}

=== Question ===
{query}

Answer using the relationships and entities above:"""

def call_llm(prompt: str) -> str:
    print("\n[LLM — Graph Context]")
    print("─" * 60)
    print(prompt)
    print("─" * 60)
    return "[Simulated answer grounded in knowledge graph traversal]"


# ─────────────────────────────────────────────
# 6. GRAPH RAG PIPELINE
# ─────────────────────────────────────────────

class GraphRAG:
    def __init__(self, max_depth: int = 2, max_nodes: int = 10):
        print("Building knowledge graph...")
        self.kg = build_knowledge_graph()
        self.max_depth = max_depth
        self.max_nodes = max_nodes
        print(f"Graph built: {len(self.kg.entities)} entities\n")

    def query(self, user_query: str) -> str:
        print(f"\n{'='*60}")
        print(f"USER QUERY: {user_query}")

        # Step 1: Find entry entities via vector search
        print("\n[Step 1] Finding graph entry points via vector search...")
        entry_entities = find_entry_entities(user_query, self.kg, top_k=2)
        for e in entry_entities:
            print(f"  Entry: {e.name} ({e.entity_type})")

        # Step 2: BFS graph traversal
        print(f"\n[Step 2] Traversing graph (depth={self.max_depth})...")
        graph_context = traverse_graph(
            self.kg, entry_entities, self.max_depth, self.max_nodes
        )
        print(f"Collected {len(graph_context)} context items:")
        for line in graph_context:
            print(f"  {line}")

        # Step 3: Build prompt + generate
        print("\n[Step 3] Generating answer from graph context...")
        prompt = build_graph_prompt(user_query, graph_context)
        answer = call_llm(prompt)

        print(f"\nFINAL ANSWER: {answer}")
        return answer


# ─────────────────────────────────────────────
# 7. DEMO
# ─────────────────────────────────────────────

if __name__ == "__main__":
    rag = GraphRAG(max_depth=2, max_nodes=12)

    print("\n--- Test 1: Relationship query ---")
    rag.query("How is Self-RAG related to hallucination reduction?")

    print("\n\n--- Test 2: Company + product graph traversal ---")
    rag.query("What products has OpenAI created and who are their competitors?")

    print("\n\n--- Test 3: Multi-hop concept traversal ---")
    rag.query("How does RAG connect to vector databases and what implements them?")
