"""
Проверяет и ограничивает формулировки контекстных уточнений агента.
"""

import re
import unicodedata
from collections.abc import Iterable

from questionnaire_contracts import DialogRole, Question, TurnRequest
from questionnaire_contracts.contract_types import (
    has_invalid_dialog_joiner_context,
    is_forbidden_dialog_text_character,
)

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_DURATION_RANGE_PATTERN = re.compile(r"(?<!\d)\d+\s*[-–—]\s*\d+(?!\d)")
_OPTION_CODE_PATTERN = re.compile(r"(?<!\w)o\d+(?!\w)", flags=re.IGNORECASE)
_NO_FAVOURITE_ACTIVITY_OPTION_PATTERN = re.compile(
    r"или (?:"
    r"ничего (?:из (?:этого|перечисленного|этих вариантов)|не подходит)"
    r"|ни один (?:вариант|из (?:этих )?вариантов) не подходит"
    r")"
)
_PHONE_WORD_PATTERN = re.compile(r"\bтелефон\w*\b", flags=re.IGNORECASE)
_SAFE_DEVICE_PHONE_PATTERN = re.compile(
    r"\b(?:на|в)\s+телефон\w*\b",
    flags=re.IGNORECASE,
)
_URL_PATTERN = re.compile(r"(?:https?://|www\.)\S+", flags=re.IGNORECASE)
_EMAIL_PATTERN = re.compile(
    r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
    flags=re.IGNORECASE,
)
_PHONE_NUMBER_PATTERN = re.compile(r"(?<!\w)\+?\d[\d\s().-]{6,}\d(?!\w)")
_GEOLOCATION_WORD_PATTERN = re.compile(
    r"\b(?:геолокац\w*|местополож\w*|координат\w*|локац\w*)\b",
    flags=re.IGNORECASE,
)
_PERSONAL_DETAILS_REQUEST_PATTERN = re.compile(
    r"\b(?:назови|сообщи|напиши|пришли|отправь)\s+(?:своих?\s+)?родител\w*\b",
    flags=re.IGNORECASE,
)
_FORBIDDEN_MARKERS = ("http://", "https://", "www.", "```", "<", ">")
_FRIENDLY_CLARIFICATION_SYMBOLS = frozenset({"🙂", "😊"})
_MAX_SAFE_CONTEXT_TERMS = 12
_MIN_CONTEXT_TERM_LENGTH = 4
_MAX_CONTEXT_TERM_LENGTH = 32
_FUNCTION_WORDS = frozenset(
    {
        "а",
        "без",
        "бы",
        "в",
        "вам",
        "вас",
        "ведь",
        "во",
        "вот",
        "все",
        "вы",
        "для",
        "до",
        "его",
        "ее",
        "если",
        "еще",
        "же",
        "за",
        "и",
        "из",
        "или",
        "им",
        "как",
        "когда",
        "который",
        "ли",
        "либо",
        "мне",
        "мы",
        "на",
        "наверно",
        "наверное",
        "не",
        "нет",
        "но",
        "ну",
        "о",
        "об",
        "он",
        "она",
        "они",
        "от",
        "по",
        "при",
        "про",
        "пока",
        "пожалуй",
        "похоже",
        "с",
        "со",
        "сомневаюсь",
        "сомневаешься",
        "так",
        "такой",
        "там",
        "тебе",
        "тебя",
        "то",
        "тогда",
        "тот",
        "ты",
        "у",
        "чем",
        "что",
        "чтобы",
        "это",
        "уверен",
        "уверена",
        "возможно",
        "кажется",
        "хотя",
        "я",
    }
)
_SAFE_FORMULATION_WORDS = frozenset(
    {
        "ближе",
        "больше",
        "вариант",
        "выбери",
        "говорить",
        "другой",
        "именно",
        "меньше",
        "назови",
        "например",
        "несколько",
        "несмотря",
        "обычно",
        "один",
        "ответ",
        "ответь",
        "пожалуйста",
        "подходит",
        "примерно",
        "пропустить",
        "скажи",
        "сколько",
        "совсем",
        "спокойно",
        "скорее",
        "смотреть",
        "уточни",
        "чаще",
    }
)
_FORBIDDEN_OUTPUT_WORDS = frozenset(
    {
        "адрес",
        "ask",
        "backend",
        "email",
        "evidence",
        "ignore",
        "llm",
        "location",
        "логин",
        "имя",
        "инструкция",
        "зовут",
        "карта",
        "местоположение",
        "номер",
        "option",
        "password",
        "пароль",
        "паспорт",
        "пинкод",
        "покажи",
        "правило",
        "пришли",
        "prompt",
        "секрет",
        "ссылка",
        "сохрани",
        "токен",
        "фамилия",
    }
)
_CONTEXT_HARD_DENY_PREFIXES = (
    "адрес",
    "address",
    "ask",
    "assistant",
    "answer",
    "backend",
    "choose",
    "clarify",
    "developer",
    "email",
    "evidence",
    "ignore",
    "игнорир",
    "инструкц",
    "instruction",
    "json",
    "координат",
    "геолокац",
    "локац",
    "location",
    "llm",
    "логин",
    "login",
    "местополож",
    "модел",
    "model",
    "name",
    "назов",
    "напиш",
    "номер",
    "option",
    "отправ",
    "ответ",
    "отчеств",
    "password",
    "парол",
    "паспорт",
    "phone",
    "пинкод",
    "покаж",
    "почт",
    "правил",
    "пришл",
    "prompt",
    "reveal",
    "раскро",
    "реквизит",
    "секрет",
    "secret",
    "select",
    "send",
    "show",
    "скажи",
    "спрос",
    "систем",
    "system",
    "снилс",
    "сохран",
    "телефон",
    "tell",
    "токен",
    "token",
    "установ",
    "уточн",
    "выбер",
    "фамили",
)
_CONTEXT_HARD_DENY_WORDS = frozenset(
    {
        "выполни",
        "забудь",
        "запиши",
        "зовут",
        "имя",
        "код",
        "поставь",
        "считай",
        "укажи",
        "верни",
    }
)
_CONTEXT_EXCLUDED_PREFIXES = (
    "быва",
    "всег",
    "видел",
    "вижу",
    "долго",
    "замеч",
    "игра",
    "иног",
    "каки",
    "крас",
    "любл",
    "люби",
    "могу",
    "нрав",
    "обыч",
    "петь",
    "получ",
    "рассм",
    "рисо",
    "рису",
    "смотр",
    "смот",
    "хочу",
    "част",
    "чита",
)
_CURATED_CONTEXT_TERM_FORMS = {
    "манга": ("манга", "манги", "манге", "мангу", "мангой", "мангою"),
    "комикс": (
        "комикс",
        "комикса",
        "комиксу",
        "комиксом",
        "комиксе",
        "комиксы",
        "комиксов",
        "комиксам",
        "комиксами",
        "комиксах",
    ),
    "аниме": ("аниме",),
    "фотография": (
        "фотография",
        "фотографии",
        "фотографию",
        "фотографией",
        "фотографиею",
        "фотографий",
        "фотографиям",
        "фотографиями",
        "фотографиях",
    ),
    "картина": (
        "картина",
        "картины",
        "картине",
        "картину",
        "картиной",
        "картиною",
        "картин",
        "картинам",
        "картинами",
        "картинах",
    ),
    "иллюстрация": (
        "иллюстрация",
        "иллюстрации",
        "иллюстрацию",
        "иллюстрацией",
        "иллюстрациею",
        "иллюстраций",
        "иллюстрациям",
        "иллюстрациями",
        "иллюстрациях",
    ),
    "рисунок": (
        "рисунок",
        "рисунка",
        "рисунку",
        "рисунком",
        "рисунке",
        "рисунки",
        "рисунков",
        "рисункам",
        "рисунками",
        "рисунках",
    ),
}
_CURATED_CONTEXT_FORM_TO_TERM = {
    form: term
    for term, forms in _CURATED_CONTEXT_TERM_FORMS.items()
    for form in forms
}


def extract_safe_context_terms(texts: Iterable[str]) -> tuple[str, ...]:
    """
    Извлекает ограниченный набор безопасных предметных слов из контекста.
    """
    terms: list[str] = []
    for text in texts:
        sanitized = _sanitize_context_text(text)
        raw_tokens = tuple(_WORD_PATTERN.findall(sanitized))
        normalized_tokens = tuple(_normalize_text(token) for token in raw_tokens)
        if any(_is_hard_denied_context_token(token) for token in normalized_tokens):
            continue
        for position, (raw_token, token) in enumerate(
            zip(raw_tokens, normalized_tokens, strict=True)
        ):
            if not _is_safe_context_term_candidate(
                raw_token,
                token,
                allow_initial_titlecase=position == 0,
            ):
                continue
            term = _CURATED_CONTEXT_FORM_TO_TERM.get(token, token)
            if term in terms:
                continue
            terms.append(term)
            if len(terms) >= _MAX_SAFE_CONTEXT_TERMS:
                return tuple(terms)
    return tuple(terms)


def build_safe_context_term_forms(
    texts: Iterable[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """
    Связывает найденный термин с точной или доверенной группой словоформ.
    """
    return tuple(
        (
            term,
            _CURATED_CONTEXT_TERM_FORMS.get(term, (term,)),
        )
        for term in extract_safe_context_terms(texts)
    )


def build_option_clarification_text(question: Question) -> str:
    """
    Формирует доверенное уточнение с допустимыми вариантами ответа.
    """
    option_texts = tuple(option.text for option in question.options)
    joined_options = (
        option_texts[0]
        if len(option_texts) == 1
        else f"{', '.join(option_texts[:-1])} или {option_texts[-1]}"
    )
    selection_kind = (
        "один или несколько вариантов"
        if question.question_type.value == "MULTI_CHOICE"
        else "один вариант"
    )
    detailed = f"Выбери {selection_kind}: {joined_options}; либо «пропустить»?"
    if len(detailed) <= 240:
        return detailed
    return (
        f"Выбери {selection_kind}, который подходит больше всего, "
        "либо «пропустить»?"
    )


def relevant_clarification_context_texts(
    request: TurnRequest,
    question_code: str,
) -> tuple[str, ...]:
    """
    Собирает пользовательские реплики текущей цепочки уточнения вопроса.
    """
    pending = request.state.pending_clarification
    context_message_ids = (
        pending.context_message_ids
        if pending is not None and pending.question_code == question_code
        else ()
    )
    messages_by_id = {
        message.message_id: message
        for message in request.dialog_history
        if message.role == DialogRole.USER
    }
    previous_texts = tuple(
        messages_by_id[message_id].content
        for message_id in context_message_ids
        if message_id in messages_by_id
    )
    return (*previous_texts, request.current_message.content)


def _sanitize_context_text(value: str) -> str:
    """
    Удаляет ссылки, адреса почты и похожие на телефон числовые значения.
    """
    sanitized = unicodedata.normalize("NFKC", value)
    for pattern in (_URL_PATTERN, _EMAIL_PATTERN, _PHONE_NUMBER_PATTERN):
        sanitized = pattern.sub(" ", sanitized)
    return sanitized


def _is_hard_denied_context_token(token: str) -> bool:
    """
    Распознаёт команды, служебные слова и маркеры чувствительных данных.
    """
    return token in _CONTEXT_HARD_DENY_WORDS or token.startswith(
        _CONTEXT_HARD_DENY_PREFIXES
    )


def _is_safe_context_term_candidate(
    raw_token: str,
    token: str,
    *,
    allow_initial_titlecase: bool,
) -> bool:
    """
    Оставляет предметные слова одного алфавита с обычным регистром.
    """
    if not _MIN_CONTEXT_TERM_LENGTH <= len(token) <= _MAX_CONTEXT_TERM_LENGTH:
        return False
    ordinary_initial_capital = (
        allow_initial_titlecase
        and raw_token[:1].isupper()
        and raw_token[1:] == raw_token[1:].lower()
    )
    if raw_token != raw_token.lower() and not ordinary_initial_capital:
        return False
    if not token.isalpha():
        return False
    if not _is_cyrillic_word(token):
        return False
    if token in _FUNCTION_WORDS or token in _SAFE_FORMULATION_WORDS:
        return False
    if _is_hard_denied_context_token(token):
        return False
    return not token.startswith(_CONTEXT_EXCLUDED_PREFIXES)


def _is_cyrillic_word(value: str) -> bool:
    """
    Проверяет, что слово состоит только из кириллических букв.
    """
    return bool(value) and all(
        "CYRILLIC" in unicodedata.name(character, "") for character in value
    )


def _contains_unsafe_phone_word(value: str) -> bool:
    """
    Отклоняет каждое упоминание телефона вне контекста устройства.
    """
    without_device_mentions = _SAFE_DEVICE_PHONE_PATTERN.sub(" ", value)
    return _PHONE_WORD_PATTERN.search(without_device_mentions) is not None


def _has_safe_favourite_activity_no_option_followup(
    question: Question,
    text: str,
) -> bool:
    """
    Разрешает E1 короткий второй вопрос об отсутствии подходящего варианта.
    """
    if (
        question.question_code != "E1"
        or question.answer_meaning != "FAVOURITE_ACTIVITIES"
        or question.question_type.value != "MULTI_CHOICE"
    ):
        return False
    first_mark_position = next(
        (position for position, character in enumerate(text) if _is_question_mark(character)),
        None,
    )
    if first_mark_position is None or not text[:first_mark_position].strip():
        return False
    followup = text[first_mark_position + 1 :].strip()
    if not followup or not _is_question_mark(followup[-1]):
        return False
    return _NO_FAVOURITE_ACTIVITY_OPTION_PATTERN.fullmatch(
        _normalize_text(followup[:-1])
    ) is not None


def is_safe_contextual_clarification(
    question: Question,
    text: str,
    *,
    previous_assistant_text: str | None,
    context_texts: Iterable[str] = (),
) -> bool:
    """
    Проверяет форму и явную безопасность живого уточнения.

    Тему определяет модель; код проверяет допустимость questionCode отдельно и
    не отклоняет перефразировки из-за отсутствия слов анкеты или точных словоформ.
    context_texts сохранён для совместимости вызовов, но не ограничивает лексику.
    """
    if text != text.strip() or not 1 <= len(text) <= 240:
        return False
    if any(is_forbidden_dialog_text_character(character) for character in text):
        return False
    if has_invalid_dialog_joiner_context(text):
        return False
    symbols = tuple(
        character
        for character in text
        if unicodedata.category(character).startswith("S")
    )
    if len(symbols) > 1 or any(
        character not in _FRIENDLY_CLARIFICATION_SYMBOLS for character in symbols
    ):
        return False
    lowered_text = text.casefold()
    if any(marker in lowered_text for marker in _FORBIDDEN_MARKERS):
        return False
    if (
        _EMAIL_PATTERN.search(text)
        or _PHONE_NUMBER_PATTERN.search(text)
        or _GEOLOCATION_WORD_PATTERN.search(text)
        or _PERSONAL_DETAILS_REQUEST_PATTERN.search(text)
    ):
        return False
    if _OPTION_CODE_PATTERN.search(lowered_text):
        return False
    if _contains_unsafe_phone_word(lowered_text):
        return False
    question_marks = tuple(character for character in text if _is_question_mark(character))
    question_ending = text.rstrip(" 🙂😊")
    if (
        not (
            len(question_marks) == 1
            or (
                len(question_marks) == 2
                and _has_safe_favourite_activity_no_option_followup(
                    question,
                    question_ending,
                )
            )
        )
        or not question_ending
        or not _is_question_mark(question_ending[-1])
    ):
        return False
    if previous_assistant_text is not None:
        if _normalize_text(previous_assistant_text) == _normalize_text(text):
            return False
        if _has_same_word_multiset(previous_assistant_text, text):
            return False

    normalized_text = _normalize_text(text)
    if (
        question.answer_meaning == "FOCUS_DURATION"
        and _lists_complete_duration_scale(question, normalized_text)
    ):
        return False
    return not _contains_forbidden_output_word(text, question)


def _lists_complete_duration_scale(question: Question, text: str) -> bool:
    """
    Отклоняет чтение всей числовой шкалы вместо живого уточнения T3.
    """
    option_ranges = {
        tuple(_WORD_PATTERN.findall(match.group()))
        for option in question.options
        for match in _DURATION_RANGE_PATTERN.finditer(option.text)
    }
    proposed_ranges = {
        tuple(_WORD_PATTERN.findall(match.group()))
        for match in _DURATION_RANGE_PATTERN.finditer(text)
    }
    return len(option_ranges) >= 3 and len(option_ranges & proposed_ranges) >= 3


def normalize_dialog_text(value: str) -> str:
    """
    Нормализует отображаемый текст для точного сравнения реплик.
    """
    return _normalize_text(value)


def _word_stems(value: str) -> frozenset[str]:
    """
    Приводит русские словоформы к консервативным пятибуквенным основам.
    """
    stems: set[str] = set()
    for token in _word_tokens(value):
        if token in _FUNCTION_WORDS:
            continue
        stems.add(_word_stem(token))
    return frozenset(stems)


def _word_tokens(value: str) -> tuple[str, ...]:
    """
    Возвращает нормализованные буквенно-цифровые слова строки.
    """
    return tuple(_WORD_PATTERN.findall(_normalize_text(value)))


def _has_same_word_multiset(first: str, second: str) -> bool:
    """
    Находит повтор реплики, в котором лишь переставлены те же слова.
    """
    first_words = sorted(_word_tokens(first))
    return bool(first_words) and first_words == sorted(_word_tokens(second))


def _word_stem(token: str) -> str:
    """
    Возвращает короткую основу одного уже нормализованного слова.
    """
    return token[:5] if len(token) >= 6 else token


def _contains_forbidden_output_word(text: str, question: Question) -> bool:
    """
    Отличает запрещённое слово от общей короткой основы легального варианта.
    """
    forbidden_by_stem: dict[str, list[str]] = {}
    for word in _FORBIDDEN_OUTPUT_WORDS:
        forbidden_by_stem.setdefault(_word_stem(word), []).append(word)
    option_stems = _word_stems(" ".join(option.text for option in question.options))

    for token in _word_tokens(text):
        stem = _word_stem(token)
        forbidden_matches = forbidden_by_stem.get(stem, ())
        if not forbidden_matches:
            continue
        if stem not in option_stems or any(
            token.startswith(word[: max(5, len(word) - 3)])
            for word in forbidden_matches
        ):
            return True
    return False


def _normalize_text(value: str) -> str:
    """
    Нормализует Unicode, регистр, букву ё и последовательности пробелов.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(normalized.split())


def _is_question_mark(character: str) -> bool:
    """
    Распознаёт вопросительные знаки разных письменностей и интерробанги.
    """
    return "QUESTION" in unicodedata.name(character, "")
