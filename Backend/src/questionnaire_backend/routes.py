"""
Публикует браузерные маршруты управления сессией анкетирования.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, Request, status
from fastapi.responses import Response

from questionnaire_backend.models import (
    CreateSessionRequest,
    PublicSessionSnapshot,
    SelectDialogModeRequest,
    SubmitMessageRequest,
)
from questionnaire_backend.service import QuestionnaireService

router = APIRouter(prefix="/api", tags=["questionnaire"])


def get_questionnaire_service(request: Request) -> QuestionnaireService:
    """
    Возвращает сервис, созданный внутри lifespan текущего приложения.
    """
    service = getattr(request.app.state, "questionnaire_service", None)
    if not isinstance(service, QuestionnaireService):
        raise RuntimeError("Backend-сервис ещё не готов")
    return service


@router.post(
    "/sessions",
    response_model=PublicSessionSnapshot,
    response_model_by_alias=True,
    status_code=status.HTTP_201_CREATED,
)
async def create_session(
    payload: CreateSessionRequest,
    service: Annotated[QuestionnaireService, Depends(get_questionnaire_service)],
) -> PublicSessionSnapshot:
    """
    Создаёт новую серверную сессию и предлагает выбрать сценарий.
    """
    return await service.create_session(payload)


@router.get(
    "/sessions/{session_id}",
    response_model=PublicSessionSnapshot,
    response_model_by_alias=True,
)
async def get_session(
    session_id: str,
    service: Annotated[QuestionnaireService, Depends(get_questionnaire_service)],
) -> PublicSessionSnapshot:
    """
    Возвращает актуальный снимок известной активной сессии.
    """
    return await service.get_session(session_id)


@router.post(
    "/sessions/{session_id}/dialog-mode",
    response_model=PublicSessionSnapshot,
    response_model_by_alias=True,
)
async def select_dialog_mode(
    session_id: str,
    payload: SelectDialogModeRequest,
    service: Annotated[QuestionnaireService, Depends(get_questionnaire_service)],
) -> PublicSessionSnapshot:
    """
    Фиксирует выбранный сценарий и возвращает его первую реплику.
    """
    return await service.select_dialog_mode(session_id, payload)


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_session(
    session_id: str,
    service: Annotated[QuestionnaireService, Depends(get_questionnaire_service)],
) -> Response:
    """
    Завершает браузерную сессию и сразу освобождает место в хранилище.
    """
    await service.delete_session(session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.post(
    "/sessions/{session_id}/messages",
    response_model=PublicSessionSnapshot,
    response_model_by_alias=True,
)
async def submit_message(
    session_id: str,
    payload: SubmitMessageRequest,
    service: Annotated[QuestionnaireService, Depends(get_questionnaire_service)],
) -> PublicSessionSnapshot:
    """
    Обрабатывает одну идемпотентную пользовательскую реплику.
    """
    return await service.submit_message(session_id, payload)
