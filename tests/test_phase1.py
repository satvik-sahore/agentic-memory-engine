"""Phase 1 Verification Tests: Environment, Qdrant DB, and Model Connectivity."""

import pytest
import uuid
from qdrant_client.http import models as rest_models
from src.config import settings
from src.db.qdrant import qdrant_manager


def test_settings_loaded():
    """Verify that configuration loads properly from environment/.env."""
    assert settings.active_api_key is not None, "API key should be set in .env"
    assert settings.embedding_dimension == 3072
    assert settings.qdrant_collection_name == "agent_memories"


def test_qdrant_health_and_collection():
    """Verify that Qdrant is reachable and the collection is created."""
    assert qdrant_manager.is_healthy() is True
    # Ensure collection
    qdrant_manager.ensure_collection()
    col_info = qdrant_manager.client.get_collection(settings.qdrant_collection_name)
    assert col_info.config.params.vectors.size == settings.embedding_dimension


def test_qdrant_vector_crud_lifecycle():
    """Verify inserting, querying, and deleting a point in Qdrant."""
    test_id = str(uuid.uuid4())
    test_vector = [0.05] * settings.embedding_dimension
    test_payload = {
        "user_id": "test_user_phase1",
        "fact": "User is validating Phase 1 setup.",
        "category": "testing",
    }

    # 1. Upsert
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

    # 2. Query
    results = qdrant_manager.client.query_points(
        collection_name=settings.qdrant_collection_name,
        query=test_vector,
        query_filter=rest_models.Filter(
            must=[
                rest_models.FieldCondition(
                    key="user_id",
                    match=rest_models.MatchValue(value="test_user_phase1"),
                )
            ]
        ),
        limit=1,
    )
    assert len(results.points) == 1
    assert results.points[0].id == test_id
    assert results.points[0].payload["fact"] == "User is validating Phase 1 setup."

    # 3. Delete
    qdrant_manager.client.delete(
        collection_name=settings.qdrant_collection_name,
        points_selector=rest_models.PointIdsList(points=[test_id]),
    )


def test_gemini_api_connectivity():
    """Verify that Gemini model generates content and embeddings."""
    from google import genai

    client = genai.Client(api_key=settings.active_api_key)

    # Test text generation
    response = client.models.generate_content(
        model=settings.extraction_model,
        contents="Say 'Phase 1 verified' in 3 words.",
    )
    assert response.text is not None and len(response.text) > 0

    # Test embedding
    emb_res = client.models.embed_content(
        model=settings.embedding_model,
        contents="Self-learning AI agent memory",
    )
    assert len(emb_res.embeddings[0].values) == settings.embedding_dimension
