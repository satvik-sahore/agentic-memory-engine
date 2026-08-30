import json
import logging
from typing import List
from src.llm.client import llm_client
from src.memory.models import (
    Fact,
    MemoryRecord,
    MemoryOperation,
    MemoryOperationType,
    ReconciliationResponse,
)

logger = logging.getLogger(__name__)

RECONCILIATION_SYSTEM_PROMPT = """You are a Memory Reconciliation Engine for an AI agent.
Your job is to compare newly extracted user facts against existing memories retrieved from the user's vector memory store, and output atomic memory operations: ADD, UPDATE, DELETE, or NOOP.

Rules:
1. 'ADD': Choose this if the candidate fact is completely new information that is not captured in any existing memory.
2. 'UPDATE': Choose this if the candidate fact updates, modifies, or clarifies an existing memory. You MUST provide the exact 'target_memory_id' of the existing memory to update.
3. 'DELETE': Choose this if the candidate fact indicates that an existing memory is no longer true, obsolete, or contradicted. You MUST provide the exact 'target_memory_id' to delete.
4. 'NOOP': Choose this if the existing memory already captures this exact fact with equivalent meaning. Do NOT create duplicate memories.

Be precise. Never hallucinate memory IDs. Only reference target_memory_id values from the provided Existing Memories list.
"""


class MemoryReconciler:
    """Reconciles new candidate facts against existing memories to determine state updates."""

    def __init__(self):
        self.llm = llm_client

    def reconcile(
        self,
        new_facts: List[Fact],
        existing_memories: List[MemoryRecord],
    ) -> List[MemoryOperation]:
        """
        Compares new facts against existing vector store memories and decides mutations.
        """
        if not new_facts:
            return []

        # If there are no existing memories, all new facts are ADD operations
        if not existing_memories:
            return [
                MemoryOperation(
                    operation=MemoryOperationType.ADD,
                    fact=fact.text,
                    category=fact.category.value if hasattr(fact.category, "value") else str(fact.category),
                    scope=fact.scope,
                    triples=fact.triples,
                    target_memory_id=None,
                    reason="Initial memory record insertion (no existing memories found).",
                )
                for fact in new_facts
            ]

        # Format prompt with existing memories and new candidate facts
        existing_list_str = "\n".join(
            [f"- [ID: {m.id}] ({m.category}): {m.fact}" for m in existing_memories]
        )
        new_facts_str = "\n".join(
            [f"- ({f.category.value}): {f.text}" for f in new_facts]
        )

        prompt = f"""Compare the following new facts against the existing memories and determine the required operations (ADD, UPDATE, DELETE, NOOP):

=== EXISTING MEMORIES ===
{existing_list_str}

=== NEW CANDIDATE FACTS ===
{new_facts_str}
"""

        result = self.llm.generate_structured(
            prompt=prompt,
            response_model=ReconciliationResponse,
            system_instruction=RECONCILIATION_SYSTEM_PROMPT,
        )
        logger.info(f"Reconciliation decided {len(result.operations)} operations.")
        return result.operations


memory_reconciler = MemoryReconciler()
