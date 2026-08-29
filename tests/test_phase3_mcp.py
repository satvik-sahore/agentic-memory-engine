"""Phase 3 Tests: Model Context Protocol (MCP) Tools."""

import pytest
import json
import uuid
from src.mcp_server.server import (
    remember_conversation,
    recall_memories,
    list_user_memories,
    forget_memory,
)
from src.memory.service import memory_service


@pytest.fixture
def mcp_test_user():
    user_id = f"mcp_test_user_{uuid.uuid4().hex[:8]}"
    yield user_id
    memory_service.clear_user_memories(user_id)


def test_mcp_remember_and_recall_tools(mcp_test_user):
    """Verify that remember_conversation and recall_memories MCP tools work properly."""
    # 1. Test remember tool
    conv_text = "I am a DevOps engineer specializing in Kubernetes and Terraform on AWS."
    raw_res = remember_conversation(user_id=mcp_test_user, conversation_text=conv_text)
    data = json.loads(raw_res)
    assert data["status"] == "success"
    assert data["memories_affected"] >= 1

    # 2. Test recall tool
    raw_search = recall_memories(user_id=mcp_test_user, query="What cloud infrastructure does the user know?")
    search_data = json.loads(raw_search)
    assert search_data["status"] == "success"
    assert search_data["count"] >= 1
    assert any("kubernetes" in m["fact"].lower() or "terraform" in m["fact"].lower() or "aws" in m["fact"].lower() for m in search_data["memories"])

    # 3. Test list memories tool
    raw_list = list_user_memories(user_id=mcp_test_user)
    list_data = json.loads(raw_list)
    assert list_data["status"] == "success"
    assert list_data["total_memories"] >= 1
    mem_id = list_data["memories"][0]["id"]

    # 4. Test forget tool
    raw_forget = forget_memory(memory_id=mem_id)
    forget_data = json.loads(raw_forget)
    assert forget_data["status"] == "success"
