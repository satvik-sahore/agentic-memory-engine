import re
import logging
from typing import List, Dict, Any, Set, Tuple
from src.memory.models import MemoryRecord, EntityTriple

logger = logging.getLogger(__name__)


def build_user_knowledge_graph(memories: List[MemoryRecord], user_id: str = "User") -> Dict[str, Any]:
    """
    Constructs a topological entity-relation graph (GraphRAG) from stored user memories.
    
    Returns:
        {
            "nodes": [{"id": str, "label": str, "type": str, "size": int, "color": str}],
            "edges": [{"source": str, "target": str, "label": str, "id": str}]
        }
    """
    nodes_map: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []
    seen_edge_keys: Set[str] = set()

    # Root Node for User
    user_node_id = user_id.capitalize()
    nodes_map[user_node_id] = {
        "id": user_node_id,
        "label": user_node_id,
        "type": "user",
        "size": 22,
        "color": "#6366f1",
    }

    # Color palette for entity types
    category_colors = {
        "profile": "#3b82f6",     # Blue
        "preference": "#a855f7",  # Purple
        "skill": "#10b981",       # Emerald Green
        "project": "#f59e0b",     # Amber
        "constraint": "#f43f5e",  # Rose
        "other": "#94a3b8",       # Slate
    }

    for mem in memories:
        cat = (mem.category or "other").lower()
        color = category_colors.get(cat, "#94a3b8")

        # 1. Process explicit Entity Triples if available
        if mem.triples:
            for triple in mem.triples:
                sub = triple.subject.strip()
                rel = triple.relation.strip()
                obj = triple.object.strip()

                if not sub or not obj:
                    continue

                if sub not in nodes_map:
                    nodes_map[sub] = {"id": sub, "label": sub, "type": "entity", "size": 14, "color": "#6366f1" if sub.lower() == "user" else color}
                if obj not in nodes_map:
                    nodes_map[obj] = {"id": obj, "label": obj, "type": cat, "size": 14, "color": color}

                edge_key = f"{sub}->{rel}->{obj}"
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    edges.append({
                        "id": f"e_{len(edges)}",
                        "source": sub,
                        "target": obj,
                        "label": rel,
                    })

        # 2. Fallback heuristic extraction from declarative fact text
        else:
            fact_text = mem.fact
            # Derive target entity phrase from fact
            target_entity, relation = _extract_entity_and_relation(fact_text, cat)
            if target_entity:
                if target_entity not in nodes_map:
                    nodes_map[target_entity] = {
                        "id": target_entity,
                        "label": target_entity,
                        "type": cat,
                        "size": 14,
                        "color": color,
                    }

                edge_key = f"{user_node_id}->{relation}->{target_entity}"
                if edge_key not in seen_edge_keys:
                    seen_edge_keys.add(edge_key)
                    edges.append({
                        "id": f"e_{len(edges)}",
                        "source": user_node_id,
                        "target": target_entity,
                        "label": relation,
                    })

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges,
        "total_nodes": len(nodes_map),
        "total_edges": len(edges),
    }


def _extract_entity_and_relation(fact: str, category: str) -> Tuple[str, str]:
    """Helper heuristic to extract entity and relation if explicit triples are absent."""
    fact_clean = fact.strip().rstrip(".")
    
    # Common relation patterns
    patterns = [
        (r"^User lives in (.*)$", "lives_in"),
        (r"^User's name is (.*)$", "named"),
        (r"^User works as an? (.*)$", "works_as"),
        (r"^User prefers (.*)$", "prefers"),
        (r"^User is learning about (.*)$", "learning"),
        (r"^User is (\d+ years old)$", "age"),
        (r"^User uses (.*)$", "uses"),
        (r"^User builds (.*)$", "builds"),
        (r"^User knows (.*)$", "skilled_in"),
    ]

    for pat, rel in patterns:
        match = re.match(pat, fact_clean, re.IGNORECASE)
        if match:
            return match.group(1).strip(), rel

    # Default fallback
    words = fact_clean.replace("User ", "").replace("User's ", "")
    return words[:35], category
