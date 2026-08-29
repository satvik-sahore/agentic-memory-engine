#!/usr/bin/env python3
"""
Interactive / Visual Demo of the Two-Phase Memory Pipeline:
1. Simulates initial conversation (extracts facts and adds them to memory).
2. Performs semantic similarity retrieval on user facts.
3. Simulates a follow-up conversation that updates existing facts.
4. Demonstrates that old facts are updated/reconciled in place rather than duplicated.
"""

import sys
import json
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory.service import memory_service


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title} ")
    print("=" * 70)


def print_section(title: str):
    print(f"\n👉 {title}")
    print("-" * 50)


def main():
    test_user = "demo_developer_42"
    memory_service.clear_user_memories(test_user)

    print_banner("🧠 Self-Learning AI Agent: Two-Phase Memory Pipeline Demo")
    print(f"Active User ID: {test_user}")

    # =========================================================================
    # Step 1: Initial Conversation
    # =========================================================================
    print_section("Step 1: Ingesting Initial Conversation")
    conversation_1 = [
        {"role": "user", "content": "Hey! My name is Alex. I'm a Senior AI Engineer currently living in San Francisco."},
        {"role": "assistant", "content": "Nice to meet you, Alex! What are you working on?"},
        {"role": "user", "content": "I primarily build backends with Python and FastAPI, and I prefer PostgreSQL over NoSQL."},
        {"role": "assistant", "content": "Got it! Let me know if you need any database schemas designed."},
    ]
    print("Transcript Ingested:")
    for turn in conversation_1:
        print(f"  [{turn['role']}]: {turn['content']}")

    print("\nProcessing Extraction & Reconciliation Pipeline...")
    res_1 = memory_service.process_conversation(user_id=test_user, conversation=conversation_1)

    print(f"\nOperations Resolved ({len(res_1.operations_performed)}):")
    for op in res_1.operations_performed:
        print(f"  • [{op.operation.value}] {op.fact} (Reason: {op.reason})")

    # Display Current State
    current_memories = memory_service.get_all_memories(test_user)
    print(f"\nCurrent Memory Bank ({len(current_memories)} records):")
    for m in current_memories:
        print(f"  📌 [ID: {m.id[:8]}...] {m.fact}")

    # =========================================================================
    # Step 2: Semantic Memory Retrieval
    # =========================================================================
    print_section("Step 2: Semantic Memory Search")
    queries = [
        "What programming languages does the user use?",
        "Where is the user located?",
        "What database does the user prefer?",
    ]

    for q in queries:
        print(f"\n🔍 Query: '{q}'")
        results = memory_service.search_memories(user_id=test_user, query=q, limit=2)
        if results:
            for r in results:
                print(f"   Score: {r.score:.4f} | Fact: {r.fact}")
        else:
            print("   No matching memories found.")

    # =========================================================================
    # Step 3: Follow-Up Conversation with Fact Conflicts & Updates
    # =========================================================================
    print_section("Step 3: Follow-up Conversation (State Evolution & Conflict Resolution)")
    conversation_2 = [
        {"role": "user", "content": "Big update: I just moved from San Francisco to Tokyo! Also, I recently switched my primary language from Python to Rust."},
        {"role": "assistant", "content": "Congratulations on the move to Tokyo and the switch to Rust!"}
    ]
    print("New Transcript Ingested:")
    for turn in conversation_2:
        print(f"  [{turn['role']}]: {turn['content']}")

    print("\nRunning Dynamic Reconciliation...")
    res_2 = memory_service.process_conversation(user_id=test_user, conversation=conversation_2)

    print(f"\nReconciliation Decisions ({len(res_2.operations_performed)}):")
    for op in res_2.operations_performed:
        target = f" -> Target: {op.target_memory_id[:8]}..." if op.target_memory_id else ""
        print(f"  ⚡ [{op.operation.value}{target}] {op.fact} (Reason: {op.reason})")

    # Display Updated State
    updated_memories = memory_service.get_all_memories(test_user)
    print(f"\nUpdated Memory Bank ({len(updated_memories)} records):")
    for m in updated_memories:
        print(f"  📌 [ID: {m.id[:8]}...] {m.fact}")

    # Clean up
    memory_service.clear_user_memories(test_user)
    memory_service.db.close()
    print_banner("✅ Demo Complete! Memory engine successfully updated state without duplication.")


if __name__ == "__main__":
    main()
