"""
Проверяет безопасные границы пользовательского основания ответа.
"""

import re
import unicodedata

_WORD_PATTERN = re.compile(r"[^\W_]+", flags=re.UNICODE)
_CLAUSE_SEPARATOR_PATTERN = re.compile(r"[,.!?;:]+")
_OPTION_CODE_PATTERN = re.compile(r"(?<!\w)o\d+(?!\w)", flags=re.IGNORECASE)
_OVERRIDE_COMMAND_PATTERN = re.compile(
    r"\b(?:disregard|forget|ignore|забуд(?:ь|ьте)|игнорир\w*)\b",
)
_ROLE_PREFIX_PATTERN = re.compile(
    r"(?:^|\s)(?:assistant|developer|system|user)\s*:",
)
_PROMPT_EXTRACTION_PATTERN = re.compile(
    (
        r"\b(?:покаж\w*|раскро\w*|повтор\w*|выда\w*)\b"
        r".{0,80}\b(?:prompt|промпт\w*|системн\w*\s+инструкц\w*)\b"
    ),
)
_DECISION_COMMANDS = frozenset(
    {
        "верни",
        "выбери",
        "выполни",
        "запиши",
        "поставь",
        "сохрани",
        "считай",
        "укажи",
        "установи",
    }
)
_INTERNAL_TARGETS = frozenset(
    {
        "answer",
        "backend",
        "балл",
        "json",
        "option",
        "prompt",
        "вариант",
        "инструкция",
        "код",
        "максимальный",
        "максимум",
        "модель",
        "ответ",
        "оценка",
        "правило",
        "система",
    }
)
_SKIP_PATTERNS = (
    re.compile(r"\b(?:перейдем|переходим|давай)\b.*\bдальше\b"),
    re.compile(r"\bследующ\w*\b.*\bвопрос\w*\b"),
    re.compile(r"\bне\s+(?:хочу|буду)\s+отвеч\w*\b"),
    re.compile(r"\bотказываюсь\s+отвеч\w*\b"),
)
_NEGATED_SKIP_PATTERNS = (
    re.compile(r"\bне\s+(?:надо\s+|стоит\s+)?пропус\w*\b"),
    re.compile(r"\bне\s+хочу\s+пропус\w*\b"),
)
_EXPLICIT_SKIP_WORDS = frozenset(
    {"пропуск", "пропускай", "пропускаем", "пропусти", "пропустим", "пропустить"}
)
_SKIP_POLITENESS_CLAUSES = frozenset({"пожалуйста"})


def is_safe_semantic_answer_evidence(evidence: str) -> bool:
    """
    Проверяет только содержательность и безопасность основания SAVE_ANSWER.

    Соответствие естественной реплики варианту ответа определяет LLM. Core не
    сравнивает слова пользователя с текстом вопроса и не переоценивает optionCode.
    Дословную принадлежность основания исходным сообщениям отдельно проверяет
    валидатор ответа хода.
    """
    normalized = _normalize_text(evidence)
    words = frozenset(_WORD_PATTERN.findall(normalized))
    return (
        bool(words)
        and not _contains_unsafe_answer_instruction(normalized, words)
        and not _contains_only_explicit_skip_clauses(evidence)
    )


def is_explicit_skip_evidence(evidence: str) -> bool:
    """
    Распознаёт только явный отказ отвечать или просьбу перейти дальше.
    """
    normalized = _normalize_text(evidence)
    words = tuple(_WORD_PATTERN.findall(normalized))
    if any(pattern.search(normalized) for pattern in _NEGATED_SKIP_PATTERNS):
        return False
    if any(word in _EXPLICIT_SKIP_WORDS for word in words):
        return True
    if words in {
        ("дальше",),
        ("следующий",),
        ("следующий", "вопрос"),
        ("следующий", "вопрос", "пожалуйста"),
    }:
        return True
    return any(pattern.search(normalized) for pattern in _SKIP_PATTERNS)


def _contains_only_explicit_skip_clauses(evidence: str) -> bool:
    """
    Отличает чистый отказ от содержательного ответа с просьбой идти дальше.
    """
    clauses = tuple(
        clause.strip()
        for clause in _CLAUSE_SEPARATOR_PATTERN.split(evidence)
        if clause.strip()
    )
    return bool(clauses) and all(
        is_explicit_skip_evidence(clause)
        or _normalize_text(clause) in _SKIP_POLITENESS_CLAUSES
        for clause in clauses
    )


def _contains_unsafe_answer_instruction(
    normalized: str,
    words: frozenset[str],
) -> bool:
    """
    Находит внутренний код или команду, замаскированную под ответ анкеты.
    """
    if _OPTION_CODE_PATTERN.search(normalized):
        return True
    if (
        _OVERRIDE_COMMAND_PATTERN.search(normalized)
        or _ROLE_PREFIX_PATTERN.search(normalized)
        or _PROMPT_EXTRACTION_PATTERN.search(normalized)
    ):
        return True
    return bool(
        words & _DECISION_COMMANDS
        and (words & _INTERNAL_TARGETS or any(word.isdigit() for word in words))
    )


def _normalize_text(value: str) -> str:
    """
    Нормализует Unicode, регистр, букву ё и пробелы перед сравнением.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold().replace("ё", "е")
    return " ".join(normalized.split())
