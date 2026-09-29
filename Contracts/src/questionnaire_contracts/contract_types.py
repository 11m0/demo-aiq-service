"""
Задаёт общие типы и правила предметных моделей внешнего контракта.
"""

import unicodedata
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints
from pydantic.alias_generators import to_camel

NonEmptyString = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
_KNOWN_VISUALLY_EMPTY_CHARACTERS = frozenset({"\u115f", "\u1160", "\u2800", "\u3164", "\uffa0"})
_FORBIDDEN_DIALOG_TEXT_CATEGORIES = frozenset({"Cc", "Cn", "Co", "Cs", "Zl", "Zp"})
_ALLOWED_DIALOG_FORMAT_CHARACTERS = frozenset({"\u200c", "\u200d"})
_FORBIDDEN_DIALOG_INVISIBLE_CHARACTERS = frozenset({"\u034f"})


def is_known_visually_empty_character(character: str) -> bool:
    """
    Распознаёт символы с буквенной или символьной категорией, но пустым глифом.
    """
    return character in _KNOWN_VISUALLY_EMPTY_CHARACTERS


def is_visibly_meaningful_character(character: str) -> bool:
    """
    Определяет символ, который сам даёт пользователю видимое содержимое.
    """
    return unicodedata.category(character)[0] in {
        "L",
        "N",
        "P",
        "S",
    } and not is_known_visually_empty_character(character)


def is_semantic_text_character(character: str) -> bool:
    """
    Определяет букву или цифру, несущую самостоятельное текстовое содержание.
    """
    return unicodedata.category(character)[0] in {
        "L",
        "N",
    } and not is_known_visually_empty_character(character)


def is_forbidden_dialog_text_character(character: str) -> bool:
    """
    Определяет небезопасный символ в отображаемом тексте диалога.
    """
    category = unicodedata.category(character)
    return (
        category in _FORBIDDEN_DIALOG_TEXT_CATEGORIES
        or (category == "Cf" and character not in _ALLOWED_DIALOG_FORMAT_CHARACTERS)
        or character in _FORBIDDEN_DIALOG_INVISIBLE_CHARACTERS
        or is_known_visually_empty_character(character)
    )


def has_invalid_dialog_joiner_context(value: str) -> bool:
    """
    Проверяет наличие ZWJ или ZWNJ без соседних букв, знаков или символов.
    """
    allowed_neighbor_categories = {"L", "M", "S"}
    last_index = len(value) - 1
    for index, character in enumerate(value):
        if character not in _ALLOWED_DIALOG_FORMAT_CHARACTERS:
            continue
        if index == 0 or index == last_index:
            return True
        previous_category = unicodedata.category(value[index - 1])[0]
        next_category = unicodedata.category(value[index + 1])[0]
        if (
            previous_category not in allowed_neighbor_categories
            or next_category not in allowed_neighbor_categories
        ):
            return True
    return False


def _preserve_non_empty_string(value: str) -> str:
    """
    Проверяет видимую строку, не изменяя исходные пробелы и переводы строк.
    """
    if not any(is_visibly_meaningful_character(character) for character in value):
        raise ValueError("строка должна содержать хотя бы один видимый символ")
    return value


PreservedNonEmptyString = Annotated[
    str,
    StringConstraints(min_length=1),
    AfterValidator(_preserve_non_empty_string),
]


def _validate_canonical_prompt_text(value: str) -> str:
    """
    Проверяет безопасный однострочный текст вопроса без его изменения.
    """
    if any(is_forbidden_dialog_text_character(character) for character in value):
        raise ValueError("канонический текст содержит запрещённый Unicode-символ")
    if value != value.strip():
        raise ValueError("канонический текст не должен содержать пробелы по краям")
    if has_invalid_dialog_joiner_context(value):
        raise ValueError("канонический текст содержит ZWJ или ZWNJ без контекста")
    if not any(is_visibly_meaningful_character(character) for character in value):
        raise ValueError("канонический текст должен содержать видимый символ")
    if not any(is_semantic_text_character(character) for character in value):
        raise ValueError("канонический текст должен содержать букву или цифру")
    return value


CanonicalPromptText = Annotated[
    str,
    StringConstraints(min_length=1, max_length=4000),
    AfterValidator(_validate_canonical_prompt_text),
]


class ContractModel(BaseModel):
    """
    Задаёт единые правила чтения и сериализации внешнего JSON.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )
