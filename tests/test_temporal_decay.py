"""Unit tests for Temporal Decay and Ebbinghaus Recency Weighting."""

import pytest
from datetime import datetime, timezone, timedelta
from src.memory.decay import calculate_temporal_decay, compute_composite_score


def test_decay_score_zero_days():
    """Verify that a brand new memory has 100% freshness score (1.0)."""
    now = datetime.now(timezone.utc)
    score, label = calculate_temporal_decay(
        created_at_iso=now.isoformat(),
        half_life_days=30.0,
        reference_time=now,
    )
    assert score == 1.0
    assert "Fresh (Today)" in label


def test_decay_score_half_life():
    """Verify that after exactly 1 half-life (30 days), score drops to ~0.50."""
    now = datetime.now(timezone.utc)
    past_30d = now - timedelta(days=30)
    score, label = calculate_temporal_decay(
        created_at_iso=past_30d.isoformat(),
        half_life_days=30.0,
        reference_time=now,
    )
    assert 0.49 <= score <= 0.51


def test_decay_score_two_half_lives():
    """Verify that after 2 half-lives (60 days), score drops to ~0.25."""
    now = datetime.now(timezone.utc)
    past_60d = now - timedelta(days=60)
    score, label = calculate_temporal_decay(
        created_at_iso=past_60d.isoformat(),
        half_life_days=30.0,
        reference_time=now,
    )
    assert 0.24 <= score <= 0.26


def test_composite_score_blending():
    """Verify composite formula: 0.8 * similarity + 0.2 * recency."""
    similarity = 0.80
    recency = 1.0  # Fresh today
    comp_fresh = compute_composite_score(vector_similarity=similarity, recency_score=recency, recency_weight=0.20)
    assert comp_fresh == round(0.80 * 0.80 + 0.20 * 1.0, 4)  # 0.64 + 0.20 = 0.84

    recency_old = 0.25  # 2 months old
    comp_old = compute_composite_score(vector_similarity=similarity, recency_score=recency_old, recency_weight=0.20)
    assert comp_old == round(0.80 * 0.80 + 0.20 * 0.25, 4)  # 0.64 + 0.05 = 0.69

    # Fresh memory outranks old memory with identical vector similarity
    assert comp_fresh > comp_old
