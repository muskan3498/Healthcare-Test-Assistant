from .schemas import PromptSections


SECTION_LABELS = {
    "system_instructions": "System Instructions",
    "user_input": "Current User Input",
    "conversation_history": "Conversation History",
    "retrieved_knowledge": "Retrieved Knowledge",
    "tool_definitions": "Tool Definitions",
    "state_and_memory": "State & Memory",
}


def build_prompt(sections: PromptSections, include_retrieved_knowledge: bool = True) -> tuple[str, list[str]]:
    ordered_sections = [
        ("system_instructions", sections.system_instructions),
        ("user_input", sections.user_input),
        ("conversation_history", sections.conversation_history),
        ("retrieved_knowledge", sections.retrieved_knowledge if include_retrieved_knowledge else ""),
        ("tool_definitions", sections.tool_definitions),
        ("state_and_memory", sections.state_and_memory),
    ]

    rendered_sections: list[str] = []
    included_sections: list[str] = []

    for key, value in ordered_sections:
        normalized = value.strip()
        if not normalized:
            continue
        included_sections.append(key)
        rendered_sections.append(f"{SECTION_LABELS[key]}\n{'=' * len(SECTION_LABELS[key])}\n{normalized}")

    if not rendered_sections:
        rendered_sections.append(
            "User Input\n==========\nNo prompt content was provided. Ask the user to supply at least one non-empty section."
        )
        included_sections.append("user_input")

    return "\n\n".join(rendered_sections), included_sections
