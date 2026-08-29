"""Tests for Memory-Aware Chat Endpoint and Visual Dashboard."""

import pytest
import uuid
from fastapi.testclient import TestClient
from src.main import app
from src.memory.service import memory_service

client = TestClient(app)


@pytest.fixture
def chat_test_user():
    user_id = f"chat_user_{uuid.uuid4().hex[:8]}"
    yield user_id
    memory_service.clear_user_memories(user_id)


def test_dashboard_static_routes():
    """Verify that dashboard and root serve the interactive frontend UI."""
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "Agentic Memory Engine" in res_root.text

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "Agent Chat Playground" in res_dash.text


def test_chat_with_memory_recall(chat_test_user):
    """
    Verify chat workflow:
    1. Teach agent a fact.
    2. Chat with agent to verify that the fact is recalled and returned in the response.
    """
    # Step 1: Tell the agent a personal detail
    payload_1 = {
        "user_id": chat_test_user,
        "message": "Hello! I am a Cloud Architect based in Seattle, and I love hiking Mount Rainier.",
    }
    res_1 = client.post("/v1/chat", json=payload_1)
    assert res_1.status_code == 200
    data_1 = res_1.json()
    assert "reply" in data_1
    assert data_1["reply"] is not None

    # Step 2: Ask the agent a follow-up question
    payload_2 = {
        "user_id": chat_test_user,
        "message": "What outdoor activities do I enjoy?",
    }
    res_2 = client.post("/v1/chat", json=payload_2)
    assert res_2.status_code == 200
    data_2 = res_2.json()
    assert "reply" in data_2
    # Verify that the hiking memory was recalled
    assert len(data_2["recalled_memories"]) >= 1
    assert any("hiking" in m["fact"].lower() or "rainier" in m["fact"].lower() for m in data_2["recalled_memories"])
