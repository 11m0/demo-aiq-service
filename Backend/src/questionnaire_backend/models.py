"""
Задаёт компактный публичный контракт между браузером и backend.
"""

from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints
from pydantic.alias_generators import to_camel

from questionnaire_backend.dialog import DialogMode
from questionnaire_contracts import (
    AnswerDefaultReason,
    AnswerOrigin,
    DialogRole,
    QuestionnaireStatus,
    QuestionStatus,
    RespondentMode,
)

MessageId = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=128),
]
MessageContent = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=4000),
]


class PublicModel(BaseModel):
    """
    Настраивает camelCase и строгую проверку публичных JSON-моделей.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class CreateSessionRequest(PublicModel):
    """
    Передаёт минимальный контекст респондента для новой сессии.
    """

    age: int = Field(ge=6, le=9)
    respondent_mode: Literal[RespondentMode.CHILD]


class SubmitMessageRequest(PublicModel):
    """
    Передаёт идемпотентную пользовательскую команду к известной версии сессии.
    """

    message_id: MessageId
    content: MessageContent
    state_version: int = Field(ge=0)


class SelectDialogModeRequest(PublicModel):
    """
    Передаёт идемпотентный выбор способа начать анкетирование.
    """

    message_id: MessageId
    dialog_mode: Literal[DialogMode.GUIDED, DialogMode.FREE_STORY]
    state_version: int = Field(ge=0)


class PublicDialogMessage(PublicModel):
    """
    Представляет одну безопасную реплику в браузерной истории.
    """

    message_id: str
    role: DialogRole
    content: str
    created_at: datetime


class PublicQuestionnaireOption(PublicModel):
    """
    Представляет допустимый вариант ответа с понятной пользователю подписью.
    """

    option_code: str
    text: str
    numeric_value: float | None = None


class PublicQuestionnaireAnswer(PublicModel):
    """
    Представляет сохранённый ответ без внутренних идентификаторов источников.
    """

    option_codes: tuple[str, ...]
    evidence: str | None = None
    origin: AnswerOrigin = AnswerOrigin.USER_EVIDENCE
    default_reason: AnswerDefaultReason | None = None


class PublicQuestion(PublicModel):
    """
    Представляет состояние одного вопроса без служебных доказательств.
    """

    question_code: str
    question_text: str
    status: QuestionStatus
    options: tuple[PublicQuestionnaireOption, ...]
    answer: PublicQuestionnaireAnswer | None
    skip_reason: str | None = None


class PublicQuestionnaireIdentity(PublicModel):
    """
    Идентифицирует опубликованную анкету и её неизменяемую версию.
    """

    questionnaire_code: str
    questionnaire_version: str


class PublicSessionSnapshot(PublicModel):
    """
    Возвращает полный браузерный снимок одной сессии.
    """

    session_id: str
    state_version: int
    status: QuestionnaireStatus
    completed: bool
    dialog_mode: DialogMode
    active_question_code: str | None
    companion_question_code: str | None = Field(
        default=None,
        exclude_if=lambda value: value is None,
    )
    questionnaire: PublicQuestionnaireIdentity
    questions: tuple[PublicQuestion, ...]
    dialog_history: tuple[PublicDialogMessage, ...]
