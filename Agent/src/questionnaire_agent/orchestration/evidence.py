"""
Строит ограниченный набор дословных оснований из текущей реплики.
"""
import re
from dataclasses import dataclass

_SEPARATOR_PATTERN = re.compile(
    r"[,;.!?]+(?:\s+|$)?|\r?\n+|\s+(?:и|а|но)\s+",
    flags=re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
_MAX_SPAN_PARTS = 3
_MAX_CANDIDATES = 64
_MAX_EXCERPT_CHARACTERS = 240
_MAX_FRAGMENT_CHARACTERS = 6000
FULL_MESSAGE_EVIDENCE_ID = "FULL_MESSAGE"


@dataclass(frozen=True, slots=True)
class EvidenceCandidate:
    """
    Связывает короткий стабильный идентификатор с точной цитатой.
    """
    candidate_id: str
    text: str
    part_count: int


def build_evidence_candidates(content: str) -> tuple[EvidenceCandidate, ...]:
    """
    Возвращает ограниченный набор коротких точных фрагментов реплики.
    """
    full_content = content.strip()
    part_spans = _split_part_spans(content)
    ranked_candidates: list[tuple[int, int, int, str]] = []

    for width in range(1, _MAX_SPAN_PARTS + 1):
        for start_index in range(0, len(part_spans) - width + 1):
            start = part_spans[start_index][0]
            end = part_spans[start_index + width - 1][1]
            candidate = content[start:end].strip()
            if (
                candidate == full_content
                or len(candidate) > _MAX_EXCERPT_CHARACTERS
                or not _WORD_PATTERN.search(candidate)
            ):
                continue
            ranked_candidates.append((len(candidate), start, width, candidate))

    selected: list[tuple[str, int]] = []
    selected_values: set[str] = set()
    used_characters = 0
    for _, _, width, candidate in sorted(ranked_candidates):
        if candidate in selected_values:
            continue
        if len(selected) >= _MAX_CANDIDATES:
            break
        if used_characters + len(candidate) > _MAX_FRAGMENT_CHARACTERS:
            continue
        selected.append((candidate, width))
        selected_values.add(candidate)
        used_characters += len(candidate)

    return tuple(
        EvidenceCandidate(
            candidate_id=f"E{index}",
            text=candidate,
            part_count=part_count,
        )
        for index, (candidate, part_count) in enumerate(selected)
    )


def select_verified_evidence(
    content: str,
    candidate_id: str,
    *,
    required_concepts: tuple[str, ...] = (),
) -> str:
    """
    Проверяет идентификатор и покрытие обязательных понятий в цитате.
    """
    candidates = build_evidence_candidates(content)
    concept_candidates = _candidates_covering_concepts(
        candidates,
        required_concepts,
    )
    if concept_candidates:
        return min(
            concept_candidates,
            key=lambda candidate: (len(candidate.text), candidate.candidate_id),
        ).text
    if required_concepts:
        return content
    if candidate_id == FULL_MESSAGE_EVIDENCE_ID:
        return content
    candidates_by_id = {
        candidate.candidate_id: candidate
        for candidate in candidates
    }
    selected = candidates_by_id.get(candidate_id)
    if selected is None:
        return content
    return selected.text


def _candidates_covering_concepts(
    candidates: tuple[EvidenceCandidate, ...],
    required_concepts: tuple[str, ...],
) -> tuple[EvidenceCandidate, ...]:
    """
    Находит фрагменты, в которых представлены все выбранные понятия.
    """
    concept_token_groups = tuple(
        _normalized_token_stems(concept)
        for concept in required_concepts
        if concept.strip()
    )
    if not concept_token_groups or any(not group for group in concept_token_groups):
        return ()

    matching: list[EvidenceCandidate] = []
    for candidate in candidates:
        candidate_tokens = _normalized_token_stems(candidate.text)
        if all(candidate_tokens & group for group in concept_token_groups):
            matching.append(candidate)
    return tuple(matching)


def _normalized_token_stems(value: str) -> frozenset[str]:
    """
    Получает грубые регистронезависимые основы слов для поиска опечаток и форм.
    """
    return frozenset(
        token[:4] if len(token) >= 5 else token
        for token in (item.casefold() for item in _WORD_PATTERN.findall(value))
        if len(token) >= 3
    )


def _split_part_spans(content: str) -> tuple[tuple[int, int], ...]:
    """
    Находит границы смысловых частей без изменения исходных символов.
    """
    spans: list[tuple[int, int]] = []
    cursor = 0
    for separator in _SEPARATOR_PATTERN.finditer(content):
        _append_trimmed_span(content, cursor, separator.start(), spans)
        cursor = separator.end()
    _append_trimmed_span(content, cursor, len(content), spans)
    return tuple(spans)


def _append_trimmed_span(
    content: str,
    start: int,
    end: int,
    spans: list[tuple[int, int]],
) -> None:
    """
    Добавляет непустой диапазон после удаления только краевых пробелов.
    """
    while start < end and content[start].isspace():
        start += 1
    while end > start and content[end - 1].isspace():
        end -= 1
    if start < end:
        spans.append((start, end))
