#!/usr/bin/env python3
"""
Verification Script for Phase 1:
1. Connect to Qdrant.
2. Ensure the collection exists.
3. Insert a test vector with sample memory payload.
4. Perform a similarity search on the test vector.
5. Clean up the test point.
"""

import sys
import uuid
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qdrant_client.http import models as rest_models
from src.config import settings
from src.db.qdrant import qdrant_manager


def main():
    print("=" * 60)
    print("Self-Learning AI Agent: Phase 1 Setup Verification")
    print("=" * 60)
    print(f"Target Qdrant Endpoint : http://{settings.qdrant_host}:{settings.qdrant_port}")
    print(f"Collection Name        : {settings.qdrant_collection_name}")
    print(f"Vector Dimension       : {settings.embedding_dimension}")
    print("-" * 60)

    # 1. Health check
    print("[1/5] Checking Qdrant connection...")
    if not qdrant_manager.is_healthy():
        print("❌ Error: Unable to connect to Qdrant instance.")
        print("   Make sure Qdrant is running. Try: docker compose up -d")
        sys.exit(1)
    print("✅ Qdrant connection successful!")

    # 2. Ensure collection
    print(f"[2/5] Initializing collection '{settings.qdrant_collection_name}'...")
    created = qdrant_manager.ensure_collection()
    if created:
        print(f"✅ Created new collection '{settings.qdrant_collection_name}'.")
    else:
        print(f"✅ Collection '{settings.qdrant_collection_name}' is ready.")

    # 3. Insert test vector
    test_id = str(uuid.uuid4())
    test_vector = [0.01] * settings.embedding_dimension
    test_payload = {
        "user_id": "test_user_001",
        "fact": "User is a backend engineer who loves Python and Qdrant.",
        "category": "preferences",
        "created_at": "2026-08-29T18:00:00Z",
        "is_test": True,
    }

    print(f"[3/5] Upserting test memory record (ID: {test_id})...")
    qdrant_manager.client.upsert(
        collection_name=settings.qdrant_collection_name,
        points=[
            rest_models.PointStruct(
                id=test_id,
                vector=test_vector,
                payload=test_payload,
            )
        ],
    )
    print("✅ Test record inserted successfully.")

    # 4. Query vector
    print("[4/5] Querying vector similarity with payload filter (user_id='test_user_001')...")
    results = qdrant_manager.client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=test_vector,
        query_filter=rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="user_id",
                    match=rest_models.MatchValue(value="test_user_001"),
                )
            ]
        ),
        limit=1,
    )

    if results.points and results.points[0].id == test_id:
        matched_point = results.points[0]
        print(f"✅ Search verified! Found point with score: {matched_point.score:.4f}")
        print(f"   Payload: {matched_point.payload}")
    else:
        print("❌ Error: Failed to retrieve the inserted test point.")
        sys.exit(1)

    # 5. Clean up
    print("[5/5] Cleaning up test point...")
    qdrant_manager.client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=rest_models.PointIdsList(points=[test_id]),
    )
    print("✅ Test point deleted successfully.")

    # Clean teardown
    qdrant_manager.close()

    print("-" * 60)
    print("🎉 Phase 1 Verification Passed! Environment and Qdrant are fully ready.")
    print("=" * 60)


if __name__ == "__main__":
    main()
