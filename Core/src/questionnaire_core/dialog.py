"""
Формирует каноническую реплику следующего шага без участия языковой модели.
"""

import re
import unicodedata
from collections import Counter

from questionnaire_contracts import NextActionType, Question, QuestionnaireDefinition

COMPLETION_MESSAGE = "Спасибо! Анкета заполнена."
COMPLETION_WITH_SKIPS_MESSAGE = "Спасибо! Анкета завершена, некоторые вопросы пропущены."
FALLBACK_MESSAGE = "Продолжим анкету."
_QUESTION_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)


def repeats_official_question(transition_text: str, question_text: str) -> bool:
    """
    Находит в подводке повтор официального вопроса без опоры на его пунктуацию.
    """
    transition_words = _dialog_words(transition_text)
    question_words = _dialog_words(question_text)
    if not transition_words or not question_words:
        return False

    shared_run_length = min(4, len(question_words))
    if shared_run_length < 4 and len(" ".join(question_words)) < 20:
        return False
    question_runs = {
        question_words[index : index + shared_run_length]
        for index in range(len(question_words) - shared_run_length + 1)
    }
    if any(
        transition_words[index : index + shared_run_length] in question_runs
        for index in range(len(transition_words) - shared_run_length + 1)
    ):
        return True

    if len(question_words) < 5:
        return False
    shared_words = sum((Counter(transition_words) & Counter(question_words)).values())
    return shared_words >= max(4, (4 * len(question_words) + 4) // 5)


def _dialog_words(value: str) -> tuple[str, ...]:
    """
    Убирает различия регистра и пунктуации при сравнении двух реплик.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return tuple(_QUESTION_WORD_PATTERN.findall(normalized))


def partial_multi_choice_prompt(
    question: Question,
    selected_option_codes: tuple[str, ...],
) -> str:
    """
    Кратко спрашивает об остальных любимых занятиях после раннего выбора.
    """
    selected, _remaining = _partial_choice_labels(question, selected_option_codes)
    return (
        f"Я отметил твои любимые занятия: {selected}. "
        "Есть ещё что-то, чем тебе особенно нравится заниматься, или это всё?"
    )


def partial_multi_choice_followup(
    question: Question,
    selected_option_codes: tuple[str, ...],
    attempt: int,
) -> str:
    """
    Уточняет оставшиеся варианты без повторения предыдущей формулировки.
    """
    selected, remaining = _partial_choice_labels(question, selected_option_codes)
    if not remaining:
        return (
            f"Скажи, пожалуйста, я всё правильно записал: {selected}?"
            if attempt == 1
            else f"Нужно ли ещё что-нибудь добавить к твоим любимым занятиям: {selected}?"
        )
    if attempt == 1:
        return f"Спасибо! А ещё что-нибудь из этого тебе нравится: {remaining}, или это всё?"
    return (
        "Хочу ничего не забыть: тебе нравится ещё что-нибудь из этого — "
        f"{remaining}, или больше ничего?"
    )


def _partial_choice_labels(
    question: Question,
    selected_option_codes: tuple[str, ...],
) -> tuple[str, str]:
    """
    Разделяет проверенные варианты на отмеченные и ещё не названные.
    """
    selected_codes = set(selected_option_codes)
    selected = ", ".join(
        option.text for option in question.options if option.option_code in selected_codes
    )
    remaining = ", ".join(
        option.text for option in question.options if option.option_code not in selected_codes
    )
    return selected, remaining


def canonical_turn_reply(
    questionnaire: QuestionnaireDefinition,
    action: NextActionType,
    question_code: str | None,
    *,
    clarification_text: str | None = None,
    pending_clarification: bool = False,
    corrected_answer: bool = False,
    recorded_other_answer: bool = False,
) -> str:
    """
    Возвращает официальный вопрос, уточнение или сообщение о завершении.
    """
    if action in {
        NextActionType.ASK_QUESTION,
        NextActionType.REASK_QUESTION,
        NextActionType.CLARIFY,
    }:
        question = next(
            (item for item in questionnaire.questions if item.question_code == question_code),
            None,
        )
        if question is None:
            return FALLBACK_MESSAGE
        if action == NextActionType.REASK_QUESTION:
            trusted_question = (
                question.followup_clarification_text
                if pending_clarification and question.followup_clarification_text is not None
                else question.clarification_text or question.question_text
            )
            if corrected_answer:
                return f"Спасибо за уточнение. {trusted_question}"
            if recorded_other_answer:
                return f"Понял тебя. {trusted_question}"
            return f"Давай попробуем ещё раз: {trusted_question}"
        if action == NextActionType.CLARIFY:
            return clarification_text or question.clarification_text or question.question_text
        return question.question_text
    if action == NextActionType.COMPLETE_WITH_SKIPS:
        return COMPLETION_WITH_SKIPS_MESSAGE
    return COMPLETION_MESSAGE
