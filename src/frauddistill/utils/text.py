from __future__ import annotations


def normalize_context(context: str | None) -> str:
    return context.strip() if context and context.strip() else "N/A"


def build_detector_input(
    user_query: str,
    target_model_answer: str,
    context: str | None = None,
    mode: str = "q_y_c",
) -> str:
    parts = {
        "q_only": f"[USER QUESTION]\n{user_query}",
        "y_only": f"[MODEL ANSWER]\n{target_model_answer}",
        "q_y": f"[USER QUESTION]\n{user_query}\n\n[MODEL ANSWER]\n{target_model_answer}",
        "q_y_c": (
            f"[USER QUESTION]\n{user_query}\n\n"
            f"[MODEL ANSWER]\n{target_model_answer}\n\n"
            f"[CONTEXT]\n{normalize_context(context)}"
        ),
    }
    if mode not in parts:
        raise ValueError(f"unknown input mode: {mode}")
    return parts[mode]
