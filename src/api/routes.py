import logging
from typing import List, Optional, Union, Dict, Any
from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from src.config import settings
from src.db.qdrant import qdrant_manager
from src.memory.service import memory_service
from src.memory.models import (
    AddMemoryResponse,
    SearchMemoryResponse,
    MemoryRecord,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/memories", tags=["Memories"])


class ProcessConversationRequest(BaseModel):
    """Payload to extract facts and reconcile memories from a conversation."""
    user_id: str = Field(..., description="Unique identifier for the user / agent owner.")
    conversation: Union[str, List[Dict[str, Any]]] = Field(
        ...,
        description="Either raw transcript string or list of turn dicts [{'role': 'user', 'content': '...'}]",
    )


class DeleteMemoryResponse(BaseModel):
    status: str
    memory_id: str


class ClearUserMemoriesResponse(BaseModel):
    status: str
    user_id: str
    deleted_count: int


@router.post(
    "/process",
    response_model=AddMemoryResponse,
    status_code=status.HTTP_200_OK,
    summary="Process Conversation & Update Memory State",
)
def process_conversation(request: ProcessConversationRequest):
    """
    Ingests conversation turns, extracts atomic facts, queries existing user memories,
    reconciles conflicts (ADD / UPDATE / DELETE / NOOP), and synchronizes Qdrant.
    """
    try:
        response = memory_service.process_conversation(
            user_id=request.user_id,
            conversation=request.conversation,
        )
        return response
    except Exception as e:
        logger.error(f"Failed to process conversation for user {request.user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing memory pipeline: {str(e)}",
        )


@router.get(
    "/search",
    response_model=SearchMemoryResponse,
    summary="Semantic Vector Memory Search",
)
def search_memories(
    user_id: str = Query(..., description="User ID to search memories for"),
    query: str = Query(..., description="Natural language search query"),
    limit: int = Query(5, ge=1, le=50, description="Max memories to return"),
    score_threshold: Optional[float] = Query(
        None, ge=0.0, le=1.0, description="Similarity threshold cutoff"
    ),
):
    """
    Performs cosine similarity search over a user's memories using vector embeddings.
    """
    try:
        results = memory_service.search_memories(
            user_id=user_id,
            query=query,
            limit=limit,
            score_threshold=score_threshold,
        )
        return SearchMemoryResponse(
            query=query,
            user_id=user_id,
            results=results,
        )
    except Exception as e:
        logger.error(f"Error searching memories for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Memory search failed: {str(e)}",
        )


@router.get(
    "/user/{user_id}",
    response_model=List[MemoryRecord],
    summary="Get All Active Memories For User",
)
def get_user_memories(
    user_id: str,
    limit: int = Query(100, ge=1, le=500, description="Max memories to return"),
):
    """
    Retrieves all active memory records currently persisted for the user.
    """
    try:
        return memory_service.get_all_memories(user_id=user_id, limit=limit)
    except Exception as e:
        logger.error(f"Error fetching memories for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch memories: {str(e)}",
        )


@router.delete(
    "/{memory_id}",
    response_model=DeleteMemoryResponse,
    summary="Delete a Memory Record",
)
def delete_memory(memory_id: str):
    """
    Manually deletes a single memory record by its ID.
    """
    success = memory_service.delete_memory(memory_id=memory_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not delete memory {memory_id}",
        )
    return DeleteMemoryResponse(status="deleted", memory_id=memory_id)


@router.delete(
    "/user/{user_id}",
    response_model=ClearUserMemoriesResponse,
    summary="Clear All Memories For User",
)
def clear_user_memories(user_id: str):
    """
    Deletes all memories belonging to a specific user.
    """
    count = memory_service.clear_user_memories(user_id=user_id)
    return ClearUserMemoriesResponse(
        status="cleared",
        user_id=user_id,
        deleted_count=count,
    )
