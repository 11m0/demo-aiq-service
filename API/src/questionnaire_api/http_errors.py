"""
Описывает безопасные HTTP-ошибки вне предметного контракта обработки хода.
"""

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class HTTPErrorModel(BaseModel):
    """
    Задаёт строгую сериализацию моделей HTTP-ошибок в camelCase.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        populate_by_name=True,
        extra="forbid",
        frozen=True,
    )


class APIErrorDetail(HTTPErrorModel):
    """
    Содержит стабильный код и безопасное описание HTTP-ошибки.
    """

    code: str
    message: str


class APIErrorResponse(HTTPErrorModel):
    """
    Оборачивает ошибку в стандартное поле FastAPI detail.
    """

    detail: APIErrorDetail


class RequestValidationIssue(HTTPErrorModel):
    """
    Указывает расположение и тип ошибки без исходного значения пользователя.
    """

    location: tuple[str | int, ...] = Field(max_length=16)
    error_type: str = Field(
        min_length=1,
        max_length=100,
        pattern=r"^[a-z0-9_.]+$",
    )


class RequestValidationDetail(APIErrorDetail):
    """
    Добавляет ограниченный список безопасных нарушений входного контракта.
    """

    issues: tuple[RequestValidationIssue, ...] = Field(max_length=20)


class RequestValidationErrorResponse(HTTPErrorModel):
    """
    Возвращает очищенное описание ошибки проверки HTTP-запроса.
    """

    detail: RequestValidationDetail
