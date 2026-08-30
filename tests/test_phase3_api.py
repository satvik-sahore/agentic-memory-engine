"""Phase 3 Tests: FastAPI REST Endpoints."""

import pytest
import uuid
from fastapi.testclient import TestClient
from src.main import app
from src.config import settings
from src.memory.service import memory_service


client = TestClient(app)


@pytest.fixture
def test_user():
    user_id = f"api_test_user_{uuid.uuid4().hex[:8]}"
    yield user_id
    memory_service.clear_user_memories(user_id)


def test_healthz_and_root():
    """Verify system health check and root dashboard endpoints."""
    # Test Root Dashboard
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Agentic Memory Engine" in res_root.text

    # Test Healthz
    res_health = client.get("/healthz")
    assert res_health.status_code == 200
    data_health = res_health.json()
    assert data_health["status"] in ["healthy", "ok"]
    assert data_health["qdrant_connected"] is True
    assert data_health["collection"] == settings.qdrant_collection_name


def test_api_memory_lifecycle(test_user):
    """
    Test REST API lifecycle:
    1. POST /v1/memories/process
    2. GET /v1/memories/search
    3. GET /v1/memories/user/{user_id}
    4. DELETE /v1/memories/{memory_id}
    5. DELETE /v1/memories/user/{user_id}
    """
    # 1. Process conversation
    payload = {
        "user_id": test_user,
        "conversation": "I am working on building a scalable vector search engine using FastAPI and Qdrant in Python.",
    }
    res_proc = client.post("/v1/memories/process", json=payload)
    assert res_proc.status_code == 200
    proc_data = res_proc.json()
    assert proc_data["user_id"] == test_user
    assert proc_data["memories_affected"] >= 1

    # 2. Get user memories
    res_list = client.get(f"/v1/memories/user/{test_user}")
    assert res_list.status_code == 200
    memories = res_list.json()
    assert len(memories) >= 1
    target_mem_id = memories[0]["id"]

    # 3. Search memories
    res_search = client.get(f"/v1/memories/search?user_id={test_user}&query=What technologies does the user use?")
    assert res_search.status_code == 200
    search_data = res_search.json()
    assert len(search_data["results"]) >= 1

    # 4. Delete specific memory
    res_del = client.delete(f"/v1/memories/{target_mem_id}")
    assert res_del.status_code == 200
    assert res_del.json()["status"] == "deleted"

    # 5. Clear all user memories
    res_clear = client.delete(f"/v1/memories/user/{test_user}")
    assert res_clear.status_code == 200
    assert res_clear.json()["status"] == "cleared"
