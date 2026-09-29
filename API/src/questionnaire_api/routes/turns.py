"""
Публикует stateless-обработку одного хода анкеты.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, status

from questionnaire_agent.orchestration import QuestionnaireTurnController
from questionnaire_api.authentication import require_service_authentication
from questionnaire_api.dependencies import get_runtime_turn_controller
from questionnaire_api.http_errors import APIErrorResponse, RequestValidationErrorResponse
from questionnaire_contracts import TurnRequest, TurnResponse

router = APIRouter(
    prefix="/v1/turns",
    tags=["turns"],
    dependencies=[Depends(require_service_authentication)],
)


@router.post(
    "/process",
    response_model=TurnResponse,
    response_model_by_alias=True,
    status_code=status.HTTP_200_OK,
    responses={
        status.HTTP_401_UNAUTHORIZED: {
            "model": APIErrorResponse,
            "description": "Bearer-токен отсутствует или не прошёл проверку.",
        },
        status.HTTP_400_BAD_REQUEST: {
            "model": APIErrorResponse,
            "description": "Служебные заголовки HTTP-запроса заданы некорректно.",
        },
        status.HTTP_413_CONTENT_TOO_LARGE: {
            "model": APIErrorResponse,
            "description": "Сырое тело HTTP-запроса превышает настроенный предел.",
        },
        status.HTTP_422_UNPROCESSABLE_CONTENT: {
            "model": RequestValidationErrorResponse,
            "description": "HTTP-запрос не соответствует входному контракту.",
        },
        status.HTTP_503_SERVICE_UNAVAILABLE: {
            "model": APIErrorResponse,
            "description": "Контроллер обработки ходов временно недоступен.",
        },
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "model": APIErrorResponse,
            "description": "Непредвиденная внутренняя ошибка сервиса.",
        },
    },
)
async def process_turn(
    turn_request: TurnRequest,
    controller: Annotated[
        QuestionnaireTurnController,
        Depends(get_runtime_turn_controller),
    ],
) -> TurnResponse:
    """
    Возвращает проверенный патч или контролируемую предметную ошибку.
    """
    return await controller.process_turn(turn_request)
