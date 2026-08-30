import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, status, Query

from src.memory.service import memory_service
from src.memory.graph import build_user_knowledge_graph

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/graph", tags=["Knowledge Graph"])


@router.get(
    "/{user_id}",
    summary="Get User Entity-Relation Knowledge Graph (GraphRAG)",
)
def get_user_graph(
    user_id: str,
    scope: Optional[str] = Query(None, description="Optional scope filter (user, session, workspace)"),
) -> Dict[str, Any]:
    """
    Returns an interactive node-edge topological representation of the user's memories
    for graph visualization and multi-hop reasoning.
    """
    try:
        memories = memory_service.get_all_memories(user_id=user_id, scope=scope)
        graph_data = build_user_knowledge_graph(memories=memories, user_id=user_id)
        return graph_data
    except Exception as e:
        logger.error(f"Error generating knowledge graph for {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate knowledge graph: {str(e)}",
        )
