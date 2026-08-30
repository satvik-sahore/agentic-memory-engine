"""Tests for Knowledge Graph, Async Ingestion Queue, and Multi-Tier Scopes."""

import pytest
import uuid
import asyncio
from fastapi.testclient import TestClient
from src.main import app
from src.memory.service import memory_service
from src.memory.models import MemoryScope
from src.memory.queue import async_memory_queue

client = TestClient(app)


@pytest.fixture
def advanced_test_user():
    user_id = f"adv_user_{uuid.uuid4().hex[:8]}"
    yield user_id
    memory_service.clear_user_memories(user_id)


def test_multi_tier_scope_isolation(advanced_test_user):
    """Verify that memories with different scopes (user vs session vs workspace) are isolated."""
    # 1. Add permanent user memory
    memory_service.process_conversation(
        user_id=advanced_test_user,
        conversation="I live in Boston and my favorite language is Python.",
        scope=MemoryScope.USER,
    )

    # 2. Add ephemeral session memory
    memory_service.process_conversation(
        user_id=advanced_test_user,
        conversation="In this active session, I am debugging an issue with port 8080.",
        scope=MemoryScope.SESSION,
        session_id="session_123",
    )

    # 3. Retrieve only USER scope memories
    user_memories = memory_service.get_all_memories(user_id=advanced_test_user, scope="user")
    assert len(user_memories) >= 1
    assert any("boston" in m.fact.lower() or "python" in m.fact.lower() for m in user_memories)
    assert not any("port 8080" in m.fact.lower() for m in user_memories)

    # 4. Retrieve only SESSION scope memories
    session_memories = memory_service.get_all_memories(user_id=advanced_test_user, scope="session")
    assert len(session_memories) >= 1
    assert any("port 8080" in m.fact.lower() for m in session_memories)


def test_knowledge_graph_endpoint(advanced_test_user):
    """Verify that the knowledge graph endpoint returns topological nodes and edges."""
    memory_service.process_conversation(
        user_id=advanced_test_user,
        conversation="My name is Satvik and I work as a software engineer at Owting.",
        scope=MemoryScope.USER,
    )

    res = client.get(f"/v1/graph/{advanced_test_user}")
    assert res.status_code == 200
    graph_data = res.json()

    assert "nodes" in graph_data
    assert "edges" in graph_data
    assert len(graph_data["nodes"]) >= 2
    assert len(graph_data["edges"]) >= 1


@pytest.mark.asyncio
async def test_async_ingestion_queue(advanced_test_user):
    """Verify that the async memory queue enqueues and completes ingestion jobs in the background."""
    async_memory_queue.start()

    job_id = await async_memory_queue.enqueue(
        user_id=advanced_test_user,
        conversation="I am testing the non-blocking async queue background worker.",
        scope=MemoryScope.USER,
    )

    assert job_id is not None
    assert len(job_id) > 10

    # Wait briefly for worker to complete
    await asyncio.sleep(2.5)

    memories = memory_service.get_all_memories(user_id=advanced_test_user)
    assert len(memories) >= 1
