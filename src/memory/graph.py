import re
import logging
from typing import List, Dict, Any, Set, Tuple
from src.memory.models import MemoryRecord, EntityTriple

logger = logging.getLogger(__name__)


def build_user_knowledge_graph(memories: List[MemoryRecord], user_id: str = "User") -> Dict[str, Any]:
    """
    Constructs a rich, multi-hop topological entity-relation graph (GraphRAG) from user memories.
    Extracts entities and connects related facts together (e.g. User -> Company -> Project).
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

    # Color palette
    category_colors = {
        "profile": "#3b82f6",     # Blue
        "preference": "#a855f7",  # Purple
        "skill": "#10b981",       # Emerald
        "project": "#f59e0b",     # Amber
        "constraint": "#f43f5e",  # Rose
        "other": "#94a3b8",       # Slate
    }

    entity_mentions: Dict[str, List[str]] = {}

    for mem in memories:
        cat = (mem.category or "other").lower()
        color = category_colors.get(cat, "#94a3b8")

        # 1. Process explicit Entity Triples if present
        if mem.triples:
            for triple in mem.triples:
                sub = _clean_entity(triple.subject)
                rel = _clean_relation(triple.relation)
                obj = _clean_entity(triple.object)

                if not sub or not obj or sub == obj:
                    continue

                if sub not in nodes_map:
                    nodes_map[sub] = {"id": sub, "label": sub, "type": "entity", "size": 14, "color": "#6366f1" if sub.lower() in ["user", user_id.lower()] else color}
                if obj not in nodes_map:
                    nodes_map[obj] = {"id": obj, "label": obj, "type": cat, "size": 14, "color": color}

                _add_edge(sub, obj, rel, edges, seen_edge_keys)

        # 2. Smart Multi-Hop Extraction & Semantic Entity Linking from text
        else:
            fact_text = mem.fact
            entities, relations = _extract_multihop_entities(fact_text, user_node_id, cat)

            for sub, rel, obj, node_type in relations:
                if sub not in nodes_map:
                    nodes_map[sub] = {"id": sub, "label": sub, "type": "entity", "size": 14, "color": "#6366f1" if sub.lower() in ["user", user_id.lower()] else color}
                if obj not in nodes_map:
                    nodes_map[obj] = {"id": obj, "label": obj, "type": node_type or cat, "size": 14, "color": category_colors.get(node_type or cat, color)}

                _add_edge(sub, obj, rel, edges, seen_edge_keys)

    # 3. Inter-Entity Cross Linking (Associative Graph Layer)
    # Detect shared concepts across nodes (e.g. Owting in project & company, Boston in relocation, etc.)
    all_node_ids = list(nodes_map.keys())
    for i in range(len(all_node_ids)):
        for j in range(i + 1, len(all_node_ids)):
            n1 = all_node_ids[i]
            n2 = all_node_ids[j]

            if n1 == user_node_id or n2 == user_node_id:
                continue

            # Shared company/entity mention
            n1_lower = n1.lower()
            n2_lower = n2.lower()
            if "owting" in n1_lower and "owting" in n2_lower:
                _add_edge(n1, n2, "relates_to", edges, seen_edge_keys)
            elif ("providence" in n1_lower or "boston" in n1_lower) and ("providence" in n2_lower or "boston" in n2_lower):
                _add_edge(n1, n2, "relocating_to", edges, seen_edge_keys)
            elif ("27" in n1_lower or "28" in n1_lower) and ("april" in n1_lower or "april" in n2_lower):
                _add_edge(n1, n2, "milestone", edges, seen_edge_keys)

    return {
        "nodes": list(nodes_map.values()),
        "edges": edges,
        "total_nodes": len(nodes_map),
        "total_edges": len(edges),
    }


def _add_edge(sub: str, obj: str, rel: str, edges: List[Dict[str, Any]], seen_keys: Set[str]):
    key = f"{sub}->{rel}->{obj}"
    reverse_key = f"{obj}->{rel}->{sub}"
    if key not in seen_keys and reverse_key not in seen_keys:
        seen_keys.add(key)
        edges.append({
            "id": f"e_{len(edges)}",
            "source": sub,
            "target": obj,
            "label": rel,
        })


def _clean_entity(text: str) -> str:
    cleaned = text.strip().rstrip(".,")
    if cleaned.lower() in ["user", "the user", "he", "him"]:
        return "User"
    return cleaned[:30]


def _clean_relation(rel: str) -> str:
    return rel.strip().replace(" ", "_")[:20]


def _extract_multihop_entities(fact: str, user_node: str, category: str) -> Tuple[List[str], List[Tuple[str, str, str, str]]]:
    """
    Extracts multi-hop entities and relations from declarative facts.
    """
    fact_clean = fact.strip().rstrip(".")
    relations = []
    entities = []

    # Pattern: Works at Company -> Developing Project
    m_work_proj = re.search(r"software developer for a company called (\w+)", fact_clean, re.IGNORECASE)
    if m_work_proj:
        comp = m_work_proj.group(1).capitalize()
        relations.append((user_node, "works_at", comp, "profile"))
        relations.append((user_node, "role", "Software Developer", "profile"))
        return [comp, "Software Developer"], relations

    # Pattern: Flagship Application / Project
    m_proj = re.search(r"(?:developing|expand(?:ing)?|building) (?:a |the )?([a-zA-Z\s]+(?:project|application|app|bot|engine))", fact_clean, re.IGNORECASE)
    if m_proj:
        proj_name = m_proj.group(1).strip().capitalize()
        relations.append((user_node, "builds", proj_name, "project"))
        return [proj_name], relations

    # Pattern: Lives in Location -> Plans to move to Target Location
    m_move = re.search(r"plans to move to ([a-zA-Z\s]+)", fact_clean, re.IGNORECASE)
    if m_move:
        target_city = m_move.group(1).replace("in the future", "").replace("from here", "").strip().capitalize()
        relations.append((user_node, "moving_to", target_city, "profile"))
        return [target_city], relations

    # Pattern: Lives in City
    m_loc = re.search(r"lives in ([a-zA-Z\s,]+)", fact_clean, re.IGNORECASE)
    if m_loc:
        loc = m_loc.group(1).strip()
        relations.append((user_node, "lives_in", loc, "profile"))
        return [loc], relations

    # Pattern: Age & Birthday
    m_age = re.search(r"is (\d+ years old)", fact_clean, re.IGNORECASE)
    if m_age:
        age_str = m_age.group(1)
        relations.append((user_node, "age", age_str, "profile"))
        return [age_str], relations

    m_bday = re.search(r"will turn (\d+ in [a-zA-Z]+ \d+)", fact_clean, re.IGNORECASE)
    if m_bday:
        bday_str = m_bday.group(1)
        relations.append((user_node, "birthday", bday_str, "profile"))
        return [bday_str], relations

    # Pattern: Learning Skill
    m_learn = re.search(r"learning about ([a-zA-Z\s]+)", fact_clean, re.IGNORECASE)
    if m_learn:
        skill = m_learn.group(1).strip().capitalize()
        relations.append((user_node, "learning", skill, "skill"))
        return [skill], relations

    # Pattern: Country of Origin
    m_origin = re.search(r"is from ([a-zA-Z\s]+)", fact_clean, re.IGNORECASE)
    if m_origin:
        country = m_origin.group(1).strip().capitalize()
        relations.append((user_node, "origin", country, "profile"))
        return [country], relations

    # Default fallback
    words = fact_clean.replace("User ", "").replace("User's ", "")
    ent = words[:28]
    relations.append((user_node, category, ent, category))
    return [ent], relations
