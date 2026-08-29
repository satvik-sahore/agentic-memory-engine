import math
from datetime import datetime, timezone
from typing import Optional, Tuple


def calculate_temporal_decay(
    created_at_iso: Optional[str] = None,
    last_accessed_at_iso: Optional[str] = None,
    half_life_days: float = 30.0,
    reference_time: Optional[datetime] = None,
) -> Tuple[float, str]:
    """
    Calculates Ebbinghaus temporal retention score R(t) in range (0.0, 1.0].
    
    Formula:
        R(t) = e^(-lambda * delta_t_days)
        lambda = ln(2) / half_life_days
    
    Returns:
        (decay_score: float, human_label: str)
    """
    now = reference_time or datetime.now(timezone.utc)
    
    # Pick the most recent touch point (access time or creation time)
    timestamp_str = last_accessed_at_iso or created_at_iso
    if not timestamp_str:
        return 1.0, "🔥 Fresh (New)"

    try:
        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
        delta_seconds = max((now - dt).total_seconds(), 0.0)
        delta_days = delta_seconds / 86400.0
    except Exception:
        return 1.0, "🔥 Fresh"

    # Decay rate lambda
    decay_rate = math.log(2) / max(half_life_days, 1.0)
    decay_score = math.exp(-decay_rate * delta_days)
    clamped_score = max(min(decay_score, 1.0), 0.05)

    # Human-readable freshness label
    if delta_days < 1.0:
        label = "🔥 Fresh (Today)"
    elif delta_days < 7.0:
        label = f"⚡ {int(delta_days)}d ago"
    elif delta_days < 30.0:
        label = f"📅 {int(delta_days // 7)}w ago"
    else:
        label = f"⏳ {int(delta_days // 30)}mo ago"

    return round(clamped_score, 4), label


def compute_composite_score(
    vector_similarity: float,
    recency_score: float,
    recency_weight: float = 0.20,
) -> float:
    """
    Blends vector cosine similarity and temporal recency into a unified composite ranking score.
    
    Formula:
        Composite = (1 - w) * VectorSimilarity + w * RecencyScore
    """
    w = max(min(recency_weight, 1.0), 0.0)
    composite = (1.0 - w) * vector_similarity + w * recency_score
    return round(composite, 4)
