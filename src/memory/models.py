from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class FactCategory(str, Enum):
    """Classification categories for user memory facts."""
    PREFERENCE = "preference"      # E.g. "prefers dark mode", "likes concise code"
    PROFILE = "profile"            # E.g. "lives in Seattle", "name is Alex"
    SKILL = "skill"                # E.g. "experienced with Python & Qdrant"
    PROJECT = "project"            # E.g. "building a memory engine for AI agents"
    CONSTRAINT = "constraint"      # E.g. "cannot use Docker in production"
    OTHER = "other"


class MemoryScope(str, Enum):
    """Hierarchical scope tiers for memory isolation and lifecycle management."""
    USER = "user"            # Persistent personal facts across all sessions
    SESSION = "session"      # Ephemeral active conversation / task working memory
    WORKSPACE = "workspace"  # Shared team / repository / organizational rules


class EntityTriple(BaseModel):
    """Knowledge graph entity-relation triple (Subject -> Relation -> Object)."""
    subject: str = Field(description="The source entity (e.g., 'User', 'FastAPI', 'Qdrant').")
    relation: str = Field(description="The directed predicate relation (e.g., 'lives_in', 'uses', 'prefers').")
    object: str = Field(description="The target entity or value (e.g., 'Providence', 'Python', 'Dark Mode').")


class Fact(BaseModel):
    """An atomic, durable fact extracted from conversation."""
    text: str = Field(description="The self-contained, standalone fact statement.")
    category: FactCategory = Field(
        default=FactCategory.OTHER,
        description="Category classification for the fact.",
    )
    scope: MemoryScope = Field(
        default=MemoryScope.USER,
        description="Scope tier for the memory (user, session, workspace).",
    )
    triples: List[EntityTriple] = Field(
        default_factory=list,
        description="Extracted graph entity triples for GraphRAG topological retrieval.",
    )


class FactExtractionResponse(BaseModel):
    """Structured response container for extracted facts."""
    facts: List[Fact] = Field(
        default_factory=list,
        description="List of distinct, atomic facts extracted from the input text.",
    )


class MemoryOperationType(str, Enum):
    """Supported atomic operations during memory reconciliation."""
    ADD = "ADD"        # Insert a brand new fact not previously known
    UPDATE = "UPDATE"  # Overwrite an existing fact that changed/evolved
    DELETE = "DELETE"  # Invalidate or remove an existing fact that is no longer true
    NOOP = "NOOP"      # Fact is identical to existing record; do nothing


class MemoryOperation(BaseModel):
    """A single reconciliation action resolved by the LLM."""
    operation: MemoryOperationType = Field(description="Action to take (ADD, UPDATE, DELETE, NOOP).")
    fact: str = Field(description="The fact text to add or updated content.")
    category: str = Field(default="other", description="Category classification (profile, preference, skill, project, constraint, other).")
    scope: MemoryScope = Field(default=MemoryScope.USER, description="Scope tier for the memory.")
    triples: List[EntityTriple] = Field(default_factory=list, description="Knowledge graph triples.")
    target_memory_id: Optional[str] = Field(
        default=None,
        description="The ID of the existing memory to UPDATE or DELETE (null for ADD/NOOP).",
    )
    reason: str = Field(description="Brief justification for why this operation was chosen.")


class ReconciliationResponse(BaseModel):
    """Structured response container for memory reconciliation decisions."""
    operations: List[MemoryOperation] = Field(
        default_factory=list,
        description="List of memory operations to execute against the vector store.",
    )


class MemoryRecord(BaseModel):
    """Representation of a stored memory point in Qdrant."""
    id: str
    user_id: str
    fact: str
    category: str = "other"
    scope: str = "user"
    session_id: Optional[str] = None
    workspace_id: Optional[str] = None
    triples: List[EntityTriple] = Field(default_factory=list)
    created_at: str
    updated_at: Optional[str] = None
    last_accessed_at: Optional[str] = None
    access_count: int = 0
    score: Optional[float] = None              # Raw vector cosine similarity
    recency_score: Optional[float] = None      # Temporal decay score R(t)
    freshness_label: Optional[str] = None      # Human readable freshness (e.g. "🔥 Fresh (Today)")
    composite_score: Optional[float] = None    # Blended ranking score


class AddMemoryResponse(BaseModel):
    """API response when adding/processing new conversation turns."""
    user_id: str
    operations_performed: List[MemoryOperation]
    memories_affected: int


class SearchMemoryResponse(BaseModel):
    """API response when querying memories for a user."""
    query: str
    user_id: str
    results: List[MemoryRecord]
