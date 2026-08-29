"""Phase 2 Tests: Fact Extraction, Reconciliation Logic, and Memory Service Lifecycle."""

import pytest
import uuid
from src.memory.extractor import fact_extractor
from src.memory.reconciler import memory_reconciler
from src.memory.service import memory_service
from src.memory.models import Fact, MemoryRecord, MemoryOperationType


@pytest.fixture(autouse=True)
def clean_test_user_memories():
    """Ensure test users are cleaned before and after tests."""
    test_user_id = f"test_user_{uuid.uuid4().hex[:8]}"
    yield test_user_id
    memory_service.clear_user_memories(test_user_id)


def test_fact_extraction_from_conversation():
    """Verify that atomic facts are extracted and conversational noise is filtered out."""
    conversation = [
        {"role": "user", "content": "Hi there! Good morning."},
        {"role": "assistant", "content": "Good morning! How can I help you today?"},
        {"role": "user", "content": "I am a Senior Backend Engineer based in San Francisco, and I love building distributed systems with Python and Rust."},
        {"role": "assistant", "content": "That sounds great!"},
        {"role": "user", "content": "Thanks! Can you help me write a quick script?"},
    ]

    facts = fact_extractor.extract_facts(conversation)
    assert len(facts) >= 1

    extracted_texts = " ".join([f.text.lower() for f in facts])
    assert "san francisco" in extracted_texts or "engineer" in extracted_texts or "python" in extracted_texts


def test_fact_extraction_ignores_pure_noise():
    """Verify that pure greetings/pleasantries produce no durable facts."""
    conversation = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi! How can I help you?"},
        {"role": "user", "content": "Just saying thanks!"},
    ]
    facts = fact_extractor.extract_facts(conversation)
    assert len(facts) == 0


def test_reconciliation_initial_add():
    """Verify that new facts with empty existing memories resolve to ADD."""
    facts = [
        Fact(text="User lives in Seattle"),
        Fact(text="User prefers PostgreSQL over MongoDB"),
    ]
    operations = memory_reconciler.reconcile(new_facts=facts, existing_memories=[])
    assert len(operations) == 2
    assert all(op.operation == MemoryOperationType.ADD for op in operations)


def test_reconciliation_update_fact():
    """Verify that updating a user's location resolves to UPDATE targeting the existing ID."""
    existing = [
        MemoryRecord(
            id="mem-loc-123",
            user_id="user_test",
            fact="User lives in New York City",
            category="profile",
            created_at="2026-01-01T00:00:00Z",
        )
    ]
    new_facts = [
        Fact(text="User moved and now lives in Tokyo")
    ]

    operations = memory_reconciler.reconcile(new_facts=new_facts, existing_memories=existing)
    assert len(operations) == 1
    assert operations[0].operation == MemoryOperationType.UPDATE
    assert operations[0].target_memory_id == "mem-loc-123"


def test_end_to_end_memory_lifecycle(clean_test_user_memories):
    """
    Test complete lifecycle:
    1. Initial conversation -> ADD facts.
    2. Search facts -> verify retrieval.
    3. Update conversation -> UPDATE existing fact without duplication.
    """
    user_id = clean_test_user_memories

    # Step 1: Initial conversation
    conv_1 = [
        {"role": "user", "content": "I am currently living in Austin, Texas and working on an AI agent project."}
    ]
    res_1 = memory_service.process_conversation(user_id=user_id, conversation=conv_1)
    assert res_1.memories_affected >= 1

    # Verify stored memories
    memories_step1 = memory_service.get_all_memories(user_id=user_id)
    assert len(memories_step1) >= 1
    austin_mem = next((m for m in memories_step1 if "austin" in m.fact.lower()), None)
    assert austin_mem is not None

    # Step 2: Semantic search
    search_results = memory_service.search_memories(user_id=user_id, query="Where does the user live?")
    assert len(search_results) >= 1
    assert any("austin" in r.fact.lower() for r in search_results)

    # Brief buffer for API rate limit budgeting
    import time
    time.sleep(3)

    # Step 3: Update conversation (User moved)
    conv_2 = [
        {"role": "user", "content": "Update: I just moved from Austin to London last week!"}
    ]
    res_2 = memory_service.process_conversation(user_id=user_id, conversation=conv_2)
    assert res_2.memories_affected >= 1

    # Verify that the location was updated rather than duplicated
    memories_step2 = memory_service.get_all_memories(user_id=user_id)
    london_mem = next((m for m in memories_step2 if "london" in m.fact.lower()), None)
    assert london_mem is not None

    # Check that old Austin fact was updated / removed
    austin_mem_after = next((m for m in memories_step2 if "austin" in m.fact.lower() and "moved" not in m.fact.lower()), None)
    # The old standalone Austin location fact should no longer be active as current location
    assert london_mem.id == austin_mem.id or austin_mem_after is None
