"""
Хранит серверные правила начала и продолжения диалога анкеты.
"""

from enum import StrEnum

from questionnaire_contracts import QuestionnaireDefinition


class DialogMode(StrEnum):
    """
    Перечисляет поддерживаемые способы начать содержательную часть анкеты.
    """

    UNSELECTED = "UNSELECTED"
    GUIDED = "GUIDED"
    FREE_STORY = "FREE_STORY"


OPENING_MESSAGE = (
    "Привет! Как тебе удобнее начать: сначала немного рассказать о себе "
    "или отвечать на мои вопросы по одному?"
)
FREE_STORY_CHOICE_MESSAGE = "Сначала расскажу о себе"
GUIDED_CHOICE_MESSAGE = "Задавай вопросы"
FREE_STORY_INVITATION = (
    "Хорошо! Расскажи, что тебе нравится, чем ты любишь заниматься и что "
    "у тебя хорошо получается. Можешь рассказать сразу о нескольких вещах "
    "— если понадобится, я потом кое-что уточню."
)


def dialog_mode_user_message(dialog_mode: DialogMode) -> str:
    """
    Возвращает каноническую пользовательскую подпись выбранного сценария.
    """
    if dialog_mode == DialogMode.FREE_STORY:
        return FREE_STORY_CHOICE_MESSAGE
    if dialog_mode == DialogMode.GUIDED:
        return GUIDED_CHOICE_MESSAGE
    raise ValueError("Нельзя выбрать незаданный сценарий диалога")


def dialog_mode_assistant_message(
    dialog_mode: DialogMode,
    questionnaire: QuestionnaireDefinition,
) -> str:
    """
    Возвращает приглашение к рассказу либо первый официальный вопрос.
    """
    if dialog_mode == DialogMode.FREE_STORY:
        return FREE_STORY_INVITATION
    if dialog_mode == DialogMode.GUIDED:
        return f"Хорошо, давай по одному. {questionnaire.questions[0].question_text}"
    raise ValueError("Нельзя начать незаданный сценарий диалога")
