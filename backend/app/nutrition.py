from __future__ import annotations

import json
import re
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .prompting import build_prompt
from .rag import combine_retrieval_results, rag_store
from .schemas import PromptSections


APP_NAME = "Nutrition Healthcare Assistant Backend"
APP_DOMAIN = "nutrition_healthcare"
DEFAULT_SESSION_ID = "demo-nutrition-session"


NUTRITION_SYSTEM_PROMPT = """You are a virtual nutrition assistant focused on safe, practical, evidence-aligned, and personalized nutrition guidance. Your role is to help users make better food, hydration, and lifestyle choices based on their goals, preferences, dietary patterns, allergies, intolerances, health-related context, and prior conversation history.

Personalize your guidance using available chat history, derived memory, health context, and retrieved knowledge when present. Recommendations should be realistic, sustainable, and easy to follow. Adapt suggestions to food preferences, cultural eating patterns, budget, schedule, and nutrition goals.

You may help with balanced meals, protein, fiber, hydration, portion guidance, meal timing, snack ideas, substitutions, grocery guidance, simple habit formation, and nutrition-aware adjustments for user-reported conditions, allergies, and dietary restrictions.

You must remain supportive, clear, and practical; prioritize safety and evidence-aligned nutrition guidance; account for user-reported allergies, intolerances, diseases, and special conditions when relevant; and clearly distinguish nutrition guidance from medical diagnosis or treatment.

You must not diagnose disease, prescribe medicines, replace professional medical advice, recommend unsafe restrictions, starvation-style diets, detoxes, or extreme rapid-weight-loss plans, or ignore stated allergies, intolerances, or medical cautions.

If the user mentions severe symptoms, urgent issues, disordered eating risk, or a medical condition needing clinical supervision, advise consultation with a qualified healthcare professional.

Response style: warm and encouraging, concise but useful, structured and readable, personalized when context exists, and focused on sustainable progress over perfection."""


TOOL_CATALOG = [
    {
        "name": "food_nutrition_lookup",
        "active": True,
        "description": "Returns approximate macro and micronutrient facts for common foods.",
        "input_schema": {"food": "string", "serving_size": "string"},
        "output_schema": {"calories": "number", "protein_g": "number", "fiber_g": "number", "notes": "string"},
    },
    {
        "name": "hydration_calculator",
        "active": True,
        "description": "Estimates a simple daily hydration target from body weight and activity.",
        "input_schema": {"weight_kg": "number", "activity_level": "low|moderate|high"},
        "output_schema": {"water_liters_per_day": "number", "adjustments": "array"},
    },
    {
        "name": "calorie_estimator",
        "active": True,
        "description": "Produces a rough calorie range for maintenance or a modest goal.",
        "input_schema": {"age": "number", "sex": "string", "activity_level": "string", "goal": "string"},
        "output_schema": {"estimated_range_kcal": "string", "safety_note": "string"},
    },
    {
        "name": "protein_target_estimator",
        "active": True,
        "description": "Suggests an evidence-aligned daily protein range for general wellness goals.",
        "input_schema": {"weight_kg": "number", "goal": "string", "medical_context": "string"},
        "output_schema": {"protein_range_g_per_day": "string", "distribution_tip": "string"},
    },
    {
        "name": "grocery_recommendation",
        "active": True,
        "description": "Suggests simple grocery staples based on preferences and restrictions.",
        "input_schema": {"dietary_pattern": "string", "allergies": "array", "budget": "string"},
        "output_schema": {"items": "array", "substitutions": "array"},
    },
    {
        "name": "meal_plan_helper",
        "active": True,
        "description": "Creates a simple meal or snack structure from goals and available foods.",
        "input_schema": {"goal": "string", "foods_available": "array", "meal_timing": "string"},
        "output_schema": {"meal_ideas": "array", "prep_notes": "array"},
    },
    {
        "name": "allergen_checker",
        "active": True,
        "description": "Flags obvious conflicts between foods and user-reported allergies or intolerances.",
        "input_schema": {"foods": "array", "allergies": "array", "intolerances": "array"},
        "output_schema": {"warnings": "array", "safer_swaps": "array"},
    },
]


SEEDED_CHAT_HISTORY = [
    {
        "id": "seed-user-1",
        "role": "user",
        "content": "I am trying to eat healthier for steady energy. I prefer vegetarian Indian meals, avoid eggs, and I am allergic to peanuts.",
        "timestamp": "2026-04-21T09:00:00+00:00",
        "session_id": DEFAULT_SESSION_ID,
        "source": "seed",
        "is_seed": True,
    },
    {
        "id": "seed-assistant-1",
        "role": "assistant",
        "content": "Great. I will keep your vegetarian preference, egg avoidance, and peanut allergy in mind while suggesting balanced meals.",
        "timestamp": "2026-04-21T09:00:20+00:00",
        "session_id": DEFAULT_SESSION_ID,
        "source": "seed",
        "is_seed": True,
    },
    {
        "id": "seed-user-2",
        "role": "user",
        "content": "I also have low vitamin D history and occasional bloating after very spicy dinners.",
        "timestamp": "2026-04-21T09:01:00+00:00",
        "session_id": DEFAULT_SESSION_ID,
        "source": "seed",
        "is_seed": True,
    },
    {
        "id": "seed-assistant-2",
        "role": "assistant",
        "content": "Noted. I can suggest gentler dinner options, fiber pacing, and vitamin-D-aware food ideas while reminding you to follow clinical guidance for deficiencies.",
        "timestamp": "2026-04-21T09:01:25+00:00",
        "session_id": DEFAULT_SESSION_ID,
        "source": "seed",
        "is_seed": True,
    },
]


FALLBACK_MEMORY = {
    "nutrition_goals": ["steady energy", "balanced meals"],
    "dietary_preferences": ["vegetarian"],
    "cuisine_preferences": ["Indian"],
    "disliked_foods": [],
    "allergies": ["peanuts"],
    "intolerances": [],
    "diseases_history": [],
    "specific_conditions": [],
    "deficiency_history": ["vitamin D"],
    "digestive_issues": ["occasional bloating after very spicy dinners"],
    "pregnancy_or_postpartum_flags": [],
    "food_restrictions": ["avoids eggs"],
    "meal_timing_habits": [],
    "hydration_habits": [],
    "activity_level": "not provided",
    "supplement_or_medication_mentions": [],
    "safety_flags": ["peanut allergy"],
    "personalization_notes": [
        "Use vegetarian Indian-friendly examples.",
        "Avoid peanuts and eggs unless the user updates this preference.",
    ],
}


FIELD_PATTERNS = {
    "allergies": [
        r"allergic to ([a-zA-Z0-9, /-]+)",
        r"allergy to ([a-zA-Z0-9, /-]+)",
        r"([a-zA-Z0-9 -]+) allergy",
    ],
    "intolerances": [
        r"intolerant to ([a-zA-Z0-9, /-]+)",
        r"([a-zA-Z0-9 -]+) intolerance",
        r"can't tolerate ([a-zA-Z0-9, /-]+)",
    ],
    "diseases_history": [
        r"i have ([a-zA-Z0-9, /-]*(?:diabetes|hypertension|cholesterol|pcos|thyroid|celiac|kidney disease|heart disease)[a-zA-Z0-9, /-]*)",
        r"history of ([a-zA-Z0-9, /-]+)",
        r"diagnosed with ([a-zA-Z0-9, /-]+)",
    ],
    "specific_conditions": [
        r"\b(pcos|thyroid|hypothyroid|hyperthyroid|celiac|ibs|gerd|acid reflux|fatty liver|kidney disease|heart disease)\b",
    ],
    "deficiency_history": [
        r"low ([a-zA-Z0-9, /-]*(?:vitamin d|b12|iron|ferritin|calcium)[a-zA-Z0-9, /-]*)",
        r"deficient in ([a-zA-Z0-9, /-]+)",
        r"([a-zA-Z0-9 -]+) deficiency",
    ],
    "digestive_issues": [
        r"\b(bloating|constipation|diarrhea|acid reflux|heartburn|gas|indigestion)\b",
    ],
    "pregnancy_or_postpartum_flags": [
        r"\b(pregnant|pregnancy|postpartum|breastfeeding|lactating)\b",
    ],
    "food_restrictions": [
        r"\b(vegan|vegetarian|gluten-free|dairy-free|no eggs|avoid eggs|avoids eggs|no meat|halal|kosher)\b",
        r"avoid ([a-zA-Z0-9, /-]+)",
    ],
    "supplement_or_medication_mentions": [
        r"taking ([a-zA-Z0-9, /-]*(?:supplement|tablet|medicine|medication|metformin|insulin|thyroxine|vitamin|iron|calcium)[a-zA-Z0-9, /-]*)",
        r"\b(metformin|insulin|thyroxine|levothyroxine|vitamin d supplement|b12 supplement|iron supplement)\b",
    ],
}


NEGATION_PATTERNS = {
    "allergies": [
        r"(?:not|no longer|never) allergic to ([a-zA-Z0-9, /-]+)",
        r"remove ([a-zA-Z0-9, /-]+) allergy",
    ],
    "intolerances": [
        r"(?:not|no longer|never) intolerant to ([a-zA-Z0-9, /-]+)",
        r"can tolerate ([a-zA-Z0-9, /-]+)",
    ],
    "diseases_history": [
        r"(?:do not|don't|no longer) have ([a-zA-Z0-9, /-]+)",
        r"not diagnosed with ([a-zA-Z0-9, /-]+)",
    ],
    "specific_conditions": [
        r"(?:do not|don't|no longer) have ([a-zA-Z0-9, /-]+)",
    ],
    "food_restrictions": [
        r"(?:do not|don't|no longer) avoid ([a-zA-Z0-9, /-]+)",
        r"i eat ([a-zA-Z0-9, /-]+) now",
    ],
}


sessions: dict[str, dict[str, Any]] = {}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_envelope(success: bool, message: str, data: Any, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "success": success,
        "message": message,
        "data": data,
        "metadata": {
            "timestamp": now_iso(),
            **(metadata or {}),
        },
    }


def get_session(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    if session_id not in sessions:
        sessions[session_id] = {
            "session_id": session_id,
            "created_at": now_iso(),
            "updated_at": now_iso(),
            "chat_history": deepcopy(SEEDED_CHAT_HISTORY),
            "latest_user_prompt": _prompt_from_message(SEEDED_CHAT_HISTORY[-2]),
            "latest_retrieval": empty_retrieval(),
            "latest_generation": None,
            "latest_assembled_prompt": "",
            "memory": {},
        }
        refresh_memory(session_id)
    return sessions[session_id]


def _prompt_from_message(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "raw_text": message["content"],
        "timestamp": message["timestamp"],
        "session_id": message["session_id"],
        "source": message.get("source", "chat"),
    }


def add_message(role: str, content: str, source: str = "chat", timestamp: str | None = None, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    session = get_session(session_id)
    message = {
        "id": str(uuid.uuid4()),
        "role": role,
        "content": content.strip(),
        "timestamp": timestamp or now_iso(),
        "session_id": session_id,
        "source": source,
        "is_seed": False,
    }
    session["chat_history"].append(message)
    session["updated_at"] = now_iso()
    if role == "user":
        session["latest_user_prompt"] = _prompt_from_message(message)
        refresh_memory(session_id)
    return message


def refresh_memory(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    session = get_session(session_id) if session_id in sessions else sessions.setdefault(session_id, {})
    history = session.get("chat_history", deepcopy(SEEDED_CHAT_HISTORY))
    derived = derive_state_and_memory(history, session_id)
    session["memory"] = derived
    return derived


def derive_state_and_memory(history: list[dict[str, Any]], session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    memory = deepcopy(FALLBACK_MEMORY)
    evidence: list[dict[str, Any]] = []
    runtime_meaningful = False

    for message in history:
        if message.get("role") != "user":
            continue
        text = message.get("content", "")
        if not text:
            continue
        changes = extract_health_context(text)
        if changes["added"] or changes["removed"] or _has_personalization_signal(text):
            if not message.get("is_seed"):
                runtime_meaningful = True
            evidence.append(
                {
                    "message_id": message.get("id"),
                    "timestamp": message.get("timestamp"),
                    "source": message.get("source", "chat"),
                    "is_seed": bool(message.get("is_seed")),
                    "text": text,
                    "derived": changes,
                }
            )
        for field, values in changes["removed"].items():
            for value in values:
                memory[field] = _remove_matching(memory.get(field, []), value)
        for field, values in changes["added"].items():
            for value in values:
                _append_unique(memory.setdefault(field, []), value)

        lower = text.lower()
        if "weight loss" in lower or "lose weight" in lower:
            _append_unique(memory["nutrition_goals"], "weight management")
        if "muscle" in lower or "protein" in lower:
            _append_unique(memory["nutrition_goals"], "protein support")
        if "hydration" in lower or "water" in lower:
            _append_unique(memory["hydration_habits"], text)
        if any(word in lower for word in ["walk", "gym", "workout", "exercise", "sedentary"]):
            memory["activity_level"] = text
        if "budget" in lower:
            _append_unique(memory["personalization_notes"], "Consider budget-friendly options.")

    safety_flags = []
    if memory.get("allergies"):
        safety_flags.append("Respect reported allergies: " + ", ".join(memory["allergies"]))
    if memory.get("diseases_history") or memory.get("specific_conditions"):
        safety_flags.append("Nutrition guidance should stay within non-diagnostic support for reported health conditions.")
    if memory.get("pregnancy_or_postpartum_flags"):
        safety_flags.append("Pregnancy/postpartum context may require clinician-guided nutrition decisions.")
    memory["safety_flags"] = _dedupe(memory.get("safety_flags", []) + safety_flags)

    return {
        "session_id": session_id,
        "session_metadata": {
            "chat_message_count": len(history),
            "user_message_count": len([item for item in history if item.get("role") == "user"]),
            "last_updated": now_iso(),
        },
        **memory,
        "extraction_source_evidence": evidence,
        "fallback_flags": {
            "used_seeded_dummy_values": not runtime_meaningful,
            "reason": "No meaningful runtime user health context yet." if not runtime_meaningful else "Runtime chat history is contributing to memory.",
        },
        "timestamps": {
            "derived_at": now_iso(),
            "latest_message_at": history[-1].get("timestamp") if history else None,
        },
    }


def extract_health_context(text: str) -> dict[str, dict[str, list[str]]]:
    lower = text.lower()
    added: dict[str, list[str]] = {field: [] for field in FIELD_PATTERNS}
    removed: dict[str, list[str]] = {field: [] for field in FIELD_PATTERNS}

    for field, patterns in NEGATION_PATTERNS.items():
        for pattern in patterns:
            for match in re.findall(pattern, lower):
                for value in split_values(match):
                    _append_unique(removed.setdefault(field, []), value)

    for field, patterns in FIELD_PATTERNS.items():
        for pattern in patterns:
            for match in re.findall(pattern, lower):
                for value in split_values(match):
                    if not any(_matches(value, removed_value) for removed_value in removed.get(field, [])):
                        _append_unique(added.setdefault(field, []), value)

    return {
        "added": {key: value for key, value in added.items() if value},
        "removed": {key: value for key, value in removed.items() if value},
    }


def split_values(value: str) -> list[str]:
    cleaned = re.sub(r"\b(and|or|but|now|anymore|please|for nutrition|for diet)\b", ",", value.lower())
    pieces = re.split(r"[,/]+", cleaned)
    return [piece.strip(" .:-") for piece in pieces if 2 <= len(piece.strip(" .:-")) <= 80]


def _append_unique(values: list[str], value: str) -> None:
    normalized = value.strip().lower()
    if normalized and not any(_matches(item, normalized) for item in values):
        values.append(normalized)


def _dedupe(values: list[str]) -> list[str]:
    deduped: list[str] = []
    for value in values:
        _append_unique(deduped, value)
    return deduped


def _remove_matching(values: list[str], value: str) -> list[str]:
    return [item for item in values if not _matches(item, value)]


def _matches(left: str, right: str) -> bool:
    left_norm = left.strip().lower()
    right_norm = right.strip().lower()
    return left_norm == right_norm or left_norm in right_norm or right_norm in left_norm


def _has_personalization_signal(text: str) -> bool:
    lower = text.lower()
    return any(
        keyword in lower
        for keyword in [
            "prefer",
            "avoid",
            "goal",
            "allergic",
            "intolerant",
            "diagnosed",
            "history",
            "pregnant",
            "postpartum",
            "bloating",
            "constipation",
            "vegetarian",
            "vegan",
            "supplement",
        ]
    )


def get_memory(session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    session = get_session(session_id)
    return refresh_memory(session_id) if not session.get("memory") else session["memory"]


def empty_retrieval(query: str = "") -> dict[str, Any]:
    return {
        "query": query,
        "results": [],
        "combined_text": "",
        "message": "No uploaded documents are available or no relevant retrieved knowledge was found.",
        "retrieved_at": now_iso(),
    }


def retrieve_knowledge(query: str, top_k: int = 5, session_id: str = DEFAULT_SESSION_ID) -> dict[str, Any]:
    session = get_session(session_id)
    results = rag_store.retrieve(query, top_k)
    retrieval = {
        "query": query,
        "results": [result.model_dump() for result in results],
        "combined_text": combine_retrieval_results(results) if results else "",
        "message": f"Retrieved {len(results)} relevant chunk(s)." if results else "No relevant retrieved knowledge was found for this query.",
        "retrieved_at": now_iso(),
    }
    session["latest_retrieval"] = retrieval
    return retrieval


def render_chat_history(history: list[dict[str, Any]], limit: int | None = None) -> str:
    items = history[-limit:] if limit else history
    return "\n".join(f"{item['role'].title()} [{item['timestamp']}]: {item['content']}" for item in items)


def render_state_memory(memory: dict[str, Any]) -> str:
    prompt_memory = {key: value for key, value in memory.items() if key not in {"extraction_source_evidence"}}
    return json.dumps(prompt_memory, indent=2, ensure_ascii=True)


def render_tools() -> str:
    return json.dumps(TOOL_CATALOG, indent=2, ensure_ascii=True)


def get_prompt_sections(session_id: str = DEFAULT_SESSION_ID) -> dict[str, str]:
    session = get_session(session_id)
    latest_prompt = session.get("latest_user_prompt") or {}
    memory = get_memory(session_id)
    retrieval = session.get("latest_retrieval") or empty_retrieval()
    return {
        "system_instructions": NUTRITION_SYSTEM_PROMPT,
        "user_input": latest_prompt.get("raw_text", ""),
        "conversation_history": render_chat_history(session["chat_history"], limit=12),
        "retrieved_knowledge": retrieval.get("combined_text", ""),
        "tool_definitions": render_tools(),
        "state_and_memory": render_state_memory(memory),
    }


def assemble_latest_prompt(session_id: str = DEFAULT_SESSION_ID, include_retrieved_knowledge: bool = True) -> dict[str, Any]:
    session = get_session(session_id)
    sections = get_prompt_sections(session_id)
    prompt, included_sections = build_prompt(
        PromptSections(**sections),
        include_retrieved_knowledge=include_retrieved_knowledge,
    )
    session["latest_assembled_prompt"] = prompt
    return {
        "prompt": prompt,
        "included_sections": included_sections,
        "sections": sections,
    }


def generate_local_response(user_text: str, memory: dict[str, Any], retrieval: dict[str, Any]) -> str:
    restrictions = []
    for field in ["allergies", "intolerances", "food_restrictions", "diseases_history", "specific_conditions"]:
        values = memory.get(field) or []
        if values:
            restrictions.append(f"{field.replace('_', ' ')}: {', '.join(values)}")

    context_line = " I will keep in mind " + "; ".join(restrictions[:4]) + "." if restrictions else ""
    knowledge_line = " I also found relevant uploaded knowledge to consider." if retrieval.get("combined_text") else ""
    return (
        f"Here is a practical nutrition-focused response to your question: {user_text.strip()}\n\n"
        f"- Start with a balanced plate: protein, high-fiber carbohydrate, vegetables or fruit, and a small amount of healthy fat.\n"
        f"- Keep the plan realistic for your routine, preferences, and budget.{context_line}{knowledge_line}\n"
        f"- Because nutrition can interact with medical conditions, allergies, pregnancy/postpartum needs, or medicines, use this as general guidance and check with a qualified clinician for individualized medical care."
    )
