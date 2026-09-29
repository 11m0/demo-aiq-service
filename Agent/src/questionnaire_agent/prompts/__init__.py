"""
Объединяет версионированные промпты агента.
"""

from questionnaire_agent.prompts.next_question import (
    NEXT_QUESTION_SCHEMA_NAME,
    NextQuestionChoice,
    build_next_question_prompt,
    parse_next_question_choice,
)
from questionnaire_agent.prompts.questionnaire_turn import (
    AGENT_DECISION_SCHEMA_NAME,
    QUESTIONNAIRE_TURN_PROMPT_VERSION,
    QUESTIONNAIRE_TURN_SYSTEM_PROMPT,
    build_questionnaire_turn_prompt,
)

__all__ = [
    "AGENT_DECISION_SCHEMA_NAME",
    "QUESTIONNAIRE_TURN_PROMPT_VERSION",
    "QUESTIONNAIRE_TURN_SYSTEM_PROMPT",
    "build_questionnaire_turn_prompt",
    "NEXT_QUESTION_SCHEMA_NAME",
    "NextQuestionChoice",
    "build_next_question_prompt",
    "parse_next_question_choice",
]
