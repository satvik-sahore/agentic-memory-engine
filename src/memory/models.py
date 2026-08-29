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


class Fact(BaseModel):
    """An atomic, durable fact extracted from conversation."""
    text: str = Field(description="The self-contained, standalone fact statement.")
    category: FactCategory = Field(
        default=FactCategory.OTHER,
        description="Category classification for the fact.",
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
    created_at: str
    updated_at: Optional[str] = None
    score: Optional[float] = None


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
