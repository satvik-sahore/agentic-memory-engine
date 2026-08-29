import json
import logging
from typing import Optional
from mcp.server.mcpserver import MCPServer

from src.memory.service import memory_service
from src.config import settings

logger = logging.getLogger("mcp_memory_server")

# Initialize MCP 2.x Server
mcp = MCPServer("Agent-LongTerm-Memory")


@mcp.tool()
def remember_conversation(user_id: str, conversation_text: str) -> str:
    """
    Extracts durable facts from conversation text and reconciles them into long-term memory (ADD, UPDATE, DELETE, NOOP).
    
    Args:
        user_id: The unique identifier for the user.
        conversation_text: The conversation transcript or user statement to remember.
    """
    try:
        response = memory_service.process_conversation(
            user_id=user_id,
            conversation=conversation_text,
        )
        operations = [
            f"[{op.operation.value}] {op.fact} ({op.reason})"
            for op in response.operations_performed
        ]
        return json.dumps(
            {
                "status": "success",
                "user_id": user_id,
                "memories_affected": response.memories_affected,
                "operations": operations,
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in remember_conversation tool: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def recall_memories(user_id: str, query: str, limit: int = 5) -> str:
    """
    Retrieves semantically relevant memories from vector storage given a natural language query.
    
    Args:
        user_id: The unique identifier for the user.
        query: The topic, question, or context to retrieve facts for.
        limit: Maximum number of memory records to return (default: 5).
    """
    try:
        results = memory_service.search_memories(
            user_id=user_id,
            query=query,
            limit=limit,
        )
        memories = [
            {
                "id": r.id,
                "fact": r.fact,
                "category": r.category,
                "similarity_score": round(r.score, 4) if r.score else None,
            }
            for r in results
        ]
        return json.dumps(
            {
                "status": "success",
                "query": query,
                "user_id": user_id,
                "count": len(memories),
                "memories": memories,
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in recall_memories tool: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def list_user_memories(user_id: str, limit: int = 50) -> str:
    """
    Lists all active stored memories for a user.
    
    Args:
        user_id: The unique identifier for the user.
        limit: Maximum number of memories to return (default: 50).
    """
    try:
        results = memory_service.get_all_memories(user_id=user_id, limit=limit)
        memories = [
            {
                "id": r.id,
                "fact": r.fact,
                "category": r.category,
                "created_at": r.created_at,
            }
            for r in results
        ]
        return json.dumps(
            {
                "status": "success",
                "user_id": user_id,
                "total_memories": len(memories),
                "memories": memories,
            },
            indent=2,
        )
    except Exception as e:
        logger.error(f"Error in list_user_memories tool: {e}")
        return json.dumps({"status": "error", "message": str(e)})


@mcp.tool()
def forget_memory(memory_id: str) -> str:
    """
    Deletes a specific memory record by its unique ID.
    
    Args:
        memory_id: The UUID of the memory record to remove.
    """
    try:
        success = memory_service.delete_memory(memory_id=memory_id)
        if success:
            return json.dumps({"status": "success", "message": f"Memory {memory_id} deleted."})
        return json.dumps({"status": "error", "message": f"Memory {memory_id} not found."})
    except Exception as e:
        logger.error(f"Error in forget_memory tool: {e}")
        return json.dumps({"status": "error", "message": str(e)})


def main():
    """Runs the MCP server over standard I/O (stdio)."""
    mcp.run()


if __name__ == "__main__":
    main()
