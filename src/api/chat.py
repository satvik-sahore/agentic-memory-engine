import logging
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from src.config import settings
from src.llm.client import llm_client
from src.memory.service import memory_service
from src.memory.models import MemoryRecord, MemoryOperation, MemoryScope
from src.memory.queue import async_memory_queue

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/chat", tags=["Chat"])


class ChatRequest(BaseModel):
    user_id: str = Field(..., description="The user ID chatting with the agent.")
    message: str = Field(..., description="The user's latest input message.")
    scope: MemoryScope = Field(default=MemoryScope.USER, description="Scope tier for the memory.")
    session_id: Optional[str] = Field(None, description="Optional active session ID.")
    workspace_id: Optional[str] = Field(None, description="Optional workspace ID.")
    history: Optional[List[Dict[str, str]]] = Field(
        default_factory=list,
        description="Prior conversation history [{'role': 'user'|'assistant', 'content': '...'}]",
    )


class ChatResponse(BaseModel):
    reply: str
    recalled_memories: List[MemoryRecord]
    operations_performed: List[MemoryOperation] = Field(default_factory=list)
    async_job_id: Optional[str] = None


@router.post(
    "",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Chat with Memory-Aware AI Agent",
)
async def chat_with_agent(request: ChatRequest):
    """
    1. Recalls semantically relevant memories for the user's message and scope.
    2. Injects memories into LLM prompt context.
    3. Generates conversational reply with sub-150ms latency.
    4. Offloads fact extraction and state reconciliation to the non-blocking async queue.
    """
    try:
        # Step 1: Recall relevant memories
        recalled = memory_service.search_memories(
            user_id=request.user_id,
            query=request.message,
            limit=4,
            score_threshold=0.55,
            scope=request.scope.value if request.scope else None,
        )

        # Step 2: Format prompt with memory context
        memory_context = "No prior memories recorded yet."
        if recalled:
            memory_context = "\n".join([f"- {m.fact}" for m in recalled])

        system_instruction = f"""You are an intelligent, stateful AI assistant.
You possess continuous long-term memory about the user.

USER'S KNOWN LONG-TERM FACTS:
{memory_context}

INSTRUCTIONS:
1. Use the known facts naturally to personalize your response without awkwardly repeating the whole list.
2. Be helpful, concise, warm, and highly capable.
"""

        # Step 3: Call LLM for conversational response
        history_list = request.history or []

        if settings.provider == "gemini":
            full_prompt = f"User: {request.message}"
            if history_list:
                history_text = "\n".join(
                    [f"{h.get('role', 'user').capitalize()}: {h.get('content', '')}" for h in history_list[-4:]]
                )
                full_prompt = f"Recent History:\n{history_text}\n\nUser: {request.message}"

            response = llm_client.gemini.models.generate_content(
                model=settings.extraction_model,
                contents=f"System Instruction: {system_instruction}\n\n{full_prompt}",
            )
            reply = response.text or "I understand."
        else:
            messages: List[Dict[str, Any]] = [{"role": "system", "content": system_instruction}]
            for h in history_list[-4:]:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": request.message})

            response = llm_client.openai.chat.completions.create(
                model=settings.extraction_model,
                messages=messages,  # type: ignore
            )
            reply = response.choices[0].message.content or "I understand."

        # Step 4: Non-blocking asynchronous ingestion queue
        job_id = await async_memory_queue.enqueue(
            user_id=request.user_id,
            conversation=request.message,
            scope=request.scope,
            session_id=request.session_id,
            workspace_id=request.workspace_id,
        )

        return ChatResponse(
            reply=reply.strip(),
            recalled_memories=recalled,
            operations_performed=[],
            async_job_id=job_id,
        )

    except Exception as e:
        logger.error(f"Error in chat_with_agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )

    except Exception as e:
        logger.error(f"Error in chat_with_agent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chat error: {str(e)}",
        )
