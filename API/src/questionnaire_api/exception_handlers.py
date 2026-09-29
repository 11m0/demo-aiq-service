"""
Преобразует ошибки HTTP-валидации без отражения пользовательских данных.
"""

from typing import Any

from fastapi import Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from questionnaire_api.http_errors import (
    APIErrorDetail,
    APIErrorResponse,
    RequestValidationDetail,
    RequestValidationErrorResponse,
    RequestValidationIssue,
)
from questionnaire_contracts import TurnRequest

_MAX_VALIDATION_ISSUES = 20
_MAX_LOCATION_SEGMENTS = 16
_SAFE_LOCATION_ORIGINS = frozenset({"body", "cookie", "header", "path", "query"})
_REDACTED_LOCATION_SEGMENT = "unknown"


async def handle_unexpected_exception(
    request: Request,
    error: Exception,
) -> JSONResponse:
    """
    Возвращает безопасный некешируемый 500 без внутренних деталей исключения.
    """
    del request, error
    response = APIErrorResponse(
        detail=APIErrorDetail(
            code="INTERNAL_SERVER_ERROR",
            message="Внутренняя ошибка сервиса.",
        )
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=response.model_dump(mode="json", by_alias=True),
        headers={"Cache-Control": "no-store"},
    )


def _collect_schema_property_names(value: object) -> frozenset[str]:
    """
    Собирает только доверенные имена полей из JSON Schema входного контракта.
    """
    collected: set[str] = set()
    if isinstance(value, dict):
        properties = value.get("properties")
        if isinstance(properties, dict):
            collected.update(key for key in properties if isinstance(key, str))
        for nested_value in value.values():
            collected.update(_collect_schema_property_names(nested_value))
    elif isinstance(value, list):
        for nested_value in value:
            collected.update(_collect_schema_property_names(nested_value))
    return frozenset(collected)


_SAFE_LOCATION_SEGMENTS = _SAFE_LOCATION_ORIGINS | _collect_schema_property_names(
    TurnRequest.model_json_schema(by_alias=True)
)


async def handle_request_validation_error(
    request: Request,
    error: RequestValidationError,
) -> JSONResponse:
    """
    Возвращает только расположение и тип ошибок без входных значений и контекста.
    """
    del request
    issues = tuple(
        _build_validation_issue(raw_error) for raw_error in error.errors()[:_MAX_VALIDATION_ISSUES]
    )
    response = RequestValidationErrorResponse(
        detail=RequestValidationDetail(
            code="REQUEST_VALIDATION_FAILED",
            message="HTTP-запрос не соответствует контракту.",
            issues=issues,
        )
    )
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
        content=response.model_dump(mode="json", by_alias=True),
    )


def _build_validation_issue(raw_error: dict[str, Any]) -> RequestValidationIssue:
    """
    Очищает одну ошибку Pydantic до безопасных структурных полей.
    """
    raw_location_value = raw_error.get("loc", ())
    raw_location = raw_location_value if isinstance(raw_location_value, list | tuple) else ()
    location = tuple(
        _sanitize_location_segment(item)
        for item in raw_location[:_MAX_LOCATION_SEGMENTS]
        if isinstance(item, str | int) and not isinstance(item, bool)
    )
    raw_type = raw_error.get("type")
    error_type = (
        raw_type
        if isinstance(raw_type, str)
        and 1 <= len(raw_type) <= 100
        and raw_type.isascii()
        and all(
            character.islower() or character.isdigit() or character in "_."
            for character in raw_type
        )
        else "validation_error"
    )
    return RequestValidationIssue(location=location, error_type=error_type)


def _sanitize_location_segment(item: str | int) -> str | int:
    """
    Сохраняет индекс или известное имя поля и скрывает пользовательский ключ.
    """
    if isinstance(item, int):
        return item
    if item in _SAFE_LOCATION_SEGMENTS:
        return item
    return _REDACTED_LOCATION_SEGMENT
