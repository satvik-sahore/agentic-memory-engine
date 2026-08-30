import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional, Union, Dict, Any

from qdrant_client.http import models as rest_models

from src.config import settings
from src.db.qdrant import qdrant_manager
from src.llm.client import llm_client
from src.memory.extractor import fact_extractor
from src.memory.reconciler import memory_reconciler
from src.memory.decay import calculate_temporal_decay, compute_composite_score
from src.memory.models import (
    MemoryRecord,
    MemoryOperationType,
    MemoryOperation,
    AddMemoryResponse,
    SearchMemoryResponse,
)

logger = logging.getLogger(__name__)


class MemoryService:
    """Orchestrates extraction, reconciliation, and vector database synchronization."""

    def __init__(self):
        self.db = qdrant_manager
        self.llm = llm_client
        self.extractor = fact_extractor
        self.reconciler = memory_reconciler
        self.collection_name = settings.qdrant_collection_name
        # Ensure collection is ready
        self.db.ensure_collection()

    def process_conversation(
        self,
        user_id: str,
        conversation: Union[str, List[Dict[str, Any]]],
    ) -> AddMemoryResponse:
        """
        Executes the two-phase pipeline:
        1. Extract candidate facts from the conversation.
        2. Query existing memories for the user.
        3. Reconcile differences (ADD / UPDATE / DELETE / NOOP).
        4. Mutate Qdrant vector store.
        """
        # Step 1: Extract atomic facts
        extracted_facts = self.extractor.extract_facts(conversation)
        if not extracted_facts:
            return AddMemoryResponse(
                user_id=user_id,
                operations_performed=[],
                memories_affected=0,
            )

        # Step 2: Retrieve existing memories for this user
        existing_memories = self.get_all_memories(user_id=user_id)

        # Step 3: Reconcile changes
        operations = self.reconciler.reconcile(
            new_facts=extracted_facts,
            existing_memories=existing_memories,
        )

        # Step 4: Apply operations in Qdrant
        affected_count = self._apply_operations(user_id=user_id, operations=operations)

        return AddMemoryResponse(
            user_id=user_id,
            operations_performed=operations,
            memories_affected=affected_count,
        )

    def _apply_operations(self, user_id: str, operations: List[MemoryOperation]) -> int:
        """Executes the reconciliation operations against Qdrant."""
        affected = 0
        now_str = datetime.now(timezone.utc).isoformat()

        for op in operations:
            if op.operation == MemoryOperationType.ADD:
                point_id = str(uuid.uuid4())
                vector = self.llm.embed_text(op.fact)
                payload = {
                    "user_id": user_id,
                    "fact": op.fact,
                    "created_at": now_str,
                    "updated_at": now_str,
                    "last_accessed_at": now_str,
                    "access_count": 0,
                }
                self.db.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        rest_models.PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )
                affected += 1
                logger.info(f"Inserted new memory [{point_id}]: {op.fact}")

            elif op.operation == MemoryOperationType.UPDATE and op.target_memory_id:
                vector = self.llm.embed_text(op.fact)
                payload = {
                    "user_id": user_id,
                    "fact": op.fact,
                    "updated_at": now_str,
                    "last_accessed_at": now_str,
                    "access_count": 0,
                }
                # Qdrant upsert with existing ID replaces the point
                self.db.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        rest_models.PointStruct(
                            id=op.target_memory_id,
                            vector=vector,
                            payload=payload,
                        )
                    ],
                )
                affected += 1
                logger.info(f"Updated memory [{op.target_memory_id}]: {op.fact}")

            elif op.operation == MemoryOperationType.DELETE and op.target_memory_id:
                self.db.client.delete(
                    collection_name=self.collection_name,
                    points_selector=rest_models.PointIdsList(points=[op.target_memory_id]),
                )
                affected += 1
                logger.info(f"Deleted memory [{op.target_memory_id}]")

            elif op.operation == MemoryOperationType.NOOP:
                logger.info(f"NOOP for fact: {op.fact}")

        return affected

    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
        score_threshold: Optional[float] = None,
    ) -> List[MemoryRecord]:
        """
        Performs semantic vector search blended with Ebbinghaus temporal decay recency weighting.
        Also performs spaced repetition reinforcement on retrieved memories.
        """
        threshold = score_threshold if score_threshold is not None else settings.similarity_threshold
        query_vector = self.llm.embed_text(query)

        # Retrieve a slightly wider pool of candidates to re-rank with temporal decay
        fetch_limit = max(limit * 2, 10)
        results = self.db.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="user_id",
                        match=rest_models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=fetch_limit,
            score_threshold=threshold,
        )

        records = []
        now_str = datetime.now(timezone.utc).isoformat()

        for point in results.points:
            payload = point.payload or {}
            created_at = payload.get("created_at", "")
            last_accessed_at = payload.get("last_accessed_at")
            access_count = payload.get("access_count", 0)

            # Compute temporal decay and composite score
            recency_score, freshness_label = calculate_temporal_decay(
                created_at_iso=created_at,
                last_accessed_at_iso=last_accessed_at,
                half_life_days=settings.decay_half_life_days,
            )

            if settings.enable_temporal_decay and point.score is not None:
                composite = compute_composite_score(
                    vector_similarity=point.score,
                    recency_score=recency_score,
                    recency_weight=settings.recency_weight,
                )
            else:
                composite = point.score

            records.append(
                MemoryRecord(
                    id=str(point.id),
                    user_id=payload.get("user_id", user_id),
                    fact=payload.get("fact", ""),
                    category=payload.get("category", "other"),
                    created_at=created_at,
                    updated_at=payload.get("updated_at"),
                    last_accessed_at=last_accessed_at,
                    access_count=access_count,
                    score=point.score,
                    recency_score=recency_score,
                    freshness_label=freshness_label,
                    composite_score=composite,
                )
            )

        # Re-rank candidates by composite score descending
        records.sort(key=lambda r: (r.composite_score if r.composite_score is not None else 0.0), reverse=True)
        top_records = records[:limit]

        # Spaced Reinforcement: Touch top retrieved memories to refresh retention
        for r in top_records:
            try:
                self.db.client.set_payload(
                    collection_name=self.collection_name,
                    payload={
                        "last_accessed_at": now_str,
                        "access_count": r.access_count + 1,
                    },
                    points=[r.id],
                )
            except Exception:
                pass

        return top_records

    def get_all_memories(self, user_id: str, limit: int = 100) -> List[MemoryRecord]:
        """Retrieves all active memory records with freshness metadata for a given user."""
        points, _ = self.db.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=rest_models.Filter(
                must=[
                    rest_models.FieldCondition(
                        key="user_id",
                        match=rest_models.MatchValue(value=user_id),
                    )
                ]
            ),
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )

        records = []
        for point in points:
            payload = point.payload or {}
            created_at = payload.get("created_at", "")
            last_accessed_at = payload.get("last_accessed_at")
            access_count = payload.get("access_count", 0)

            recency_score, freshness_label = calculate_temporal_decay(
                created_at_iso=created_at,
                last_accessed_at_iso=last_accessed_at,
                half_life_days=settings.decay_half_life_days,
            )

            records.append(
                MemoryRecord(
                    id=str(point.id),
                    user_id=payload.get("user_id", user_id),
                    fact=payload.get("fact", ""),
                    category=payload.get("category", "other"),
                    created_at=created_at,
                    updated_at=payload.get("updated_at"),
                    last_accessed_at=last_accessed_at,
                    access_count=access_count,
                    recency_score=recency_score,
                    freshness_label=freshness_label,
                )
            )
        return records

    def delete_memory(self, memory_id: str) -> bool:
        """Deletes a single memory record by its ID."""
        try:
            self.db.client.delete(
                collection_name=self.collection_name,
                points_selector=rest_models.PointIdsList(points=[memory_id]),
            )
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory point {memory_id}: {e}")
            return False

    def clear_user_memories(self, user_id: str) -> int:
        """Clears all memories belonging to a specific user."""
        existing = self.get_all_memories(user_id)
        if not existing:
            return 0
        point_ids = [m.id for m in existing]
        self.db.client.delete(
            collection_name=self.collection_name,
            points_selector=rest_models.PointIdsList(points=point_ids),
        )
        return len(point_ids)


memory_service = MemoryService()
