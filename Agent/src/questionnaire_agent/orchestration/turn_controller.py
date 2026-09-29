"""
Оркестрирует один stateless-ход от запроса до проверенного ответа.
"""

import json
import logging
import re
import time
import unicodedata
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pydantic import ValidationError

from questionnaire_agent.llm import (
    LLMClient,
    LLMClientError,
    LLMCompletion,
    LLMCompletionRequest,
    LLMMessage,
    LLMMessageRole,
)
from questionnaire_agent.orchestration.agent_decision import (
    AgentClarification,
    AgentConfirmAnswerChange,
    AgentDecision,
    AgentExplanation,
    AgentIndirectSignal,
    AgentInterpretAnswerChange,
    AgentOptionSelectionInterpretation,
    AgentQuestionDraft,
    AgentSaveAnswerChange,
    AgentScaleInterpretation,
    AgentSkipQuestionChange,
    AgentStateChange,
)
from questionnaire_agent.orchestration.choice_verifier import (
    ChoiceVerificationOutcome,
    build_additive_choice_verification_prompt,
    build_choice_reconciliation_prompt,
    build_choice_verification_prompt,
    build_partial_choice_clarification_prompt,
    build_partial_choice_question_prompt,
    build_partial_choice_verification_prompt,
    parse_choice_reconciliation,
    parse_choice_verification,
    parse_partial_choice_clarification,
    parse_partial_choice_question,
)
from questionnaire_agent.orchestration.errors import TurnProcessingErrorCode
from questionnaire_agent.orchestration.evidence import (
    FULL_MESSAGE_EVIDENCE_ID,
    select_verified_evidence,
)
from questionnaire_agent.orchestration.handcraft_verifier import (
    build_handcraft_verification_prompt,
    parse_handcraft_verification,
)
from questionnaire_agent.orchestration.limits import (
    TurnProcessingLimits,
    measure_llm_output_bytes,
    measure_llm_request_bytes,
)
from questionnaire_agent.orchestration.semantic_mapping import (
    SemanticMappingError,
    SemanticMappingErrorCode,
    map_semantic_answer,
)
from questionnaire_contracts import (
    AnswerCorrectionMode,
    AnswerDefaultReason,
    ApplyDefaultAnswerChange,
    ConfirmAnswerChange,
    DialogMessage,
    DialogRole,
    NextAction,
    NextActionType,
    PolicySkipReason,
    Question,
    QuestionStatus,
    QuestionType,
    RequestClarificationChange,
    ReviseMultiChoiceChange,
    SaveAnswerChange,
    SkipQuestionChange,
    TurnAnswerScope,
    TurnFailureResponse,
    TurnRequest,
    TurnSuccessResponse,
)
from questionnaire_core.answer_evidence import is_explicit_skip_evidence
from questionnaire_core.clarification import (
    build_option_clarification_text,
    is_safe_contextual_clarification,
    normalize_dialog_text,
    relevant_clarification_context_texts,
)
from questionnaire_core.dialog import (
    canonical_turn_reply,
    partial_multi_choice_followup,
    partial_multi_choice_prompt,
    repeats_official_question,
)
from questionnaire_core.validation import (
    InvalidTurnResponseError,
    TurnResponseViolationCode,
    validate_turn_response,
)

Clock = Callable[[], datetime]

_LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class _TurnTiming:
    """
    Собирает обезличенные показатели одного хода для итогового журнала.
    """

    started_ns: int
    llm_calls: int = 0
    llm_elapsed_ns: int = 0
    usage_calls: int = 0
    input_tokens_known: int = 0
    output_tokens_known: int = 0
    draft_proposed: bool = False
    draft_accepted: bool = False


_ACTIVE_TURN_TIMING: ContextVar[_TurnTiming | None] = ContextVar(
    "questionnaire_active_turn_timing", default=None
)

_INVALID_AGENT_OUTPUT_MESSAGE = "Языковая модель вернула ответ в недопустимом формате."
_REJECTED_AGENT_OUTPUT_MESSAGE = "Предложение языковой модели не прошло проверку правил анкеты."
_CONTEXT_TOO_LARGE_MESSAGE = "Контекст хода превышает допустимый размер."
_OUTPUT_TOO_LARGE_MESSAGE = "Ответ языковой модели превышает допустимый размер."
_MAX_LLM_ATTEMPTS = 2
_MAX_CLARIFICATION_ATTEMPTS = 2
_UNKNOWN_QUESTION_FALLBACK = "Уточни, пожалуйста, свой ответ?"
_ASK_REPLY_BRIDGE_FALLBACK = "Мне интересно узнать ещё кое-что."
_MECHANICAL_TRANSITION_PREFIXES = frozenset(
    {"понял", "поняла", "ясно", "хорошо", "принято"}
)
_PERFORMANCE_POSITIVE_EVIDENCE_CONCEPT = (
    "люблю любит нравится нравятся хочу хочет желаю желание обожаю"
)
_PERFORMANCE_INHIBITION_EVIDENCE_CONCEPT = (
    "стесняюсь стесняется стыдно боюсь боязнь страшно страх нервничаю"
)
_MAX_SHORT_PERSISTENCE_EVIDENCE_CHARACTERS = 240
_MAX_SHORT_QUALIFIED_EVIDENCE_CHARACTERS = 180
_SAFE_VALIDATION_PATH_PARTS = frozenset(
    {
        "changes",
        "clarification",
        "transitionText",
        "explanation",
        "preferredNextQuestionCode",
        "operation",
        "questionCode",
        "interpretation",
        "kind",
        "directLevel",
        "indirectSignals",
        "side",
        "strength",
        "optionRefs",
        "optionCodes",
        "evidenceCandidateId",
        "replaceExisting",
        "reason",
        "clarificationText",
        "explanationText",
    }
)
_QUALIFYING_CONNECTOR = re.compile(
    r"\b(?:но|зато|хотя|однако|только|при\s+этом|несмотря\s+на)\b",
    flags=re.IGNORECASE,
)


class _DuplicateJsonKeyError(ValueError):
    """
    Сообщает о неоднозначном объекте с повторяющимся JSON-ключом.
    """


class _DuplicateQuestionTargetError(ValueError):
    """
    Сообщает о нескольких изменениях одного разрешённого вопроса.
    """

    def __init__(self, question_codes: tuple[str, ...]) -> None:
        """
        Сохраняет только доверенные коды вопросов без текста ответа модели.
        """
        self.question_codes = question_codes
        super().__init__("решение содержит повторные изменения одного вопроса")


class _InvalidChangesByQuestionError(ValueError):
    """
    Сообщает о несоответствии ключа и изменения в объекте ответов.
    """


class _InvalidDecisionShapeError(ValueError):
    """
    Сообщает о нарушении формы или разрешённой области отдельных коррекций.
    """


def _utc_now() -> datetime:
    """
    Возвращает текущее время UTC для служебного поля ответа.
    """
    return datetime.now(timezone.utc)


class QuestionnaireTurnController:
    """
    Преобразует один ответ LLM в проверенный внешний контракт хода.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        *,
        reply_writer_client: LLMClient | None = None,
        clock: Clock | None = None,
        limits: TurnProcessingLimits | None = None,
    ) -> None:
        """
        Сохраняет клиент и внедряемые часы без владения их жизненным циклом.
        """
        self._llm_client = llm_client
        self._reply_writer_client = reply_writer_client
        self._clock = clock or _utc_now
        self._limits = limits or TurnProcessingLimits()

    async def process_turn(
        self,
        request: TurnRequest,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Измеряет полный ход контроллера, включая ранние выходы и ошибки.
        """
        timing = _TurnTiming(started_ns=time.perf_counter_ns())
        token = _ACTIVE_TURN_TIMING.set(timing)
        outcome = "exception"
        error_code: str | None = None
        try:
            response = await self._process_turn(request)
            if isinstance(response, TurnFailureResponse):
                outcome = "failure"
                error_code = response.error.code
            else:
                outcome = "success"
            return response
        except BaseException as error:
            error_code = type(error).__name__
            raise
        finally:
            _ACTIVE_TURN_TIMING.reset(token)
            total_ns = time.perf_counter_ns() - timing.started_ns
            _LOGGER.info(
                "event=turn_total correlation_id=%s outcome=%s error_code=%s "
                "duration_ms=%.1f llm_duration_ms=%.1f other_duration_ms=%.1f "
                "llm_calls=%s usage_calls=%s input_tokens_known=%s "
                "output_tokens_known=%s draft_proposed=%s draft_accepted=%s",
                request.correlation_id,
                outcome,
                error_code,
                total_ns / 1_000_000,
                timing.llm_elapsed_ns / 1_000_000,
                max(0, total_ns - timing.llm_elapsed_ns) / 1_000_000,
                timing.llm_calls,
                timing.usage_calls,
                timing.input_tokens_known,
                timing.output_tokens_known,
                timing.draft_proposed,
                timing.draft_accepted,
            )

    async def _complete_llm(
        self,
        request: TurnRequest,
        prompt: LLMCompletionRequest,
        *,
        client: LLMClient,
        stage: str,
        reason: str,
    ) -> LLMCompletion:
        """
        Измеряет один вызов модели без записи текста ребёнка и сырого ответа.
        """
        timing = _ACTIVE_TURN_TIMING.get()
        call_index = 0
        if timing is not None:
            timing.llm_calls += 1
            call_index = timing.llm_calls
        started_ns = time.perf_counter_ns()
        completion: LLMCompletion | None = None
        outcome = "success"
        error_code: str | None = None
        try:
            completion = await client.complete(prompt)
            return completion
        except BaseException as error:
            outcome = "error"
            error_code = (
                error.code.value if isinstance(error, LLMClientError) else type(error).__name__
            )
            raise
        finally:
            elapsed_ns = time.perf_counter_ns() - started_ns
            usage = None if completion is None else completion.usage
            if timing is not None:
                timing.llm_elapsed_ns += elapsed_ns
                if usage is not None:
                    timing.usage_calls += 1
                    timing.input_tokens_known += usage.input_tokens or 0
                    timing.output_tokens_known += usage.output_tokens or 0
            _LOGGER.info(
                "event=llm_call correlation_id=%s call=%s stage=%s reason=%s "
                "outcome=%s error_code=%s duration_ms=%.1f "
                "input_tokens=%s output_tokens=%s cached_input_tokens=%s",
                request.correlation_id,
                call_index,
                stage,
                reason,
                outcome,
                error_code,
                elapsed_ns / 1_000_000,
                None if usage is None else usage.input_tokens,
                None if usage is None else usage.output_tokens,
                None if usage is None else usage.cached_input_tokens,
            )

    async def _process_turn(
        self,
        request: TurnRequest,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Выполняет основной и не более одного исправляющего вызова модели.
        """
        from questionnaire_agent.prompts.questionnaire_turn import (
            build_questionnaire_turn_prompt,
        )

        partial_question = _active_partial_multi_choice_question(request)
        if partial_question is not None:
            return await self._resolve_partial_multi_choice(request, partial_question)

        grounded_decision = _build_grounded_explicit_skip(request)
        if grounded_decision is not None:
            processed_at = self._read_processed_at()
            grounded_response = self._build_success_response(
                request,
                grounded_decision,
                processed_at,
            )
            try:
                validate_turn_response(request, grounded_response)
            except InvalidTurnResponseError as error:
                _log_rejected_decision(request, error, repaired=False)
            else:
                return await self._finalize_assistant_reply(request, grounded_response)

        prompt = build_questionnaire_turn_prompt(
            request,
            separate_reply=self._reply_writer_client is not None,
        )
        request_size = measure_llm_request_bytes(prompt)
        if request_size > self._limits.max_llm_request_bytes:
            return self._build_failure_response(
                request,
                processed_at=self._read_processed_at(),
                code=TurnProcessingErrorCode.CONTEXT_TOO_LARGE.value,
                message=_CONTEXT_TOO_LARGE_MESSAGE,
                retryable=False,
            )

        active_prompt = prompt
        prior_correction_codes, favourite_addition_code = _authorized_correction_targets(
            request
        )
        processed_at: datetime | None = None
        repair_reason = "none"
        for attempt_index in range(_MAX_LLM_ATTEMPTS):
            repaired = attempt_index > 0
            try:
                completion = await self._complete_llm(
                    request,
                    active_prompt,
                    client=self._llm_client,
                    stage="decision",
                    reason=repair_reason if repaired else "initial",
                )
            except LLMClientError as error:
                return self._build_failure_response(
                    request,
                    processed_at=processed_at or self._read_processed_at(),
                    code=error.code.value,
                    message=str(error),
                    retryable=error.retryable,
                )

            try:
                output_size = measure_llm_output_bytes(completion.content)
            except UnicodeEncodeError:
                return self._recover_technical_output_or_fail(
                    request,
                    processed_at or self._read_processed_at(),
                    code=TurnProcessingErrorCode.INVALID_AGENT_OUTPUT,
                    message=_INVALID_AGENT_OUTPUT_MESSAGE,
                )
            if output_size > self._limits.max_llm_output_bytes:
                return self._recover_technical_output_or_fail(
                    request,
                    processed_at or self._read_processed_at(),
                    code=TurnProcessingErrorCode.OUTPUT_TOO_LARGE,
                    message=_OUTPUT_TOO_LARGE_MESSAGE,
                )

            try:
                decision = _parse_agent_decision(
                    completion.content,
                    allowed_question_codes=frozenset(
                        question.question_code for question in request.questionnaire.questions
                    ),
                    authorized_question_codes=frozenset(
                        prompt.response_schema.schema["properties"]["changes"]["properties"]
                    ),
                    authorized_prior_correction_codes=prior_correction_codes,
                    authorized_favourite_addition_code=favourite_addition_code,
                )
            except (
                json.JSONDecodeError,
                ValidationError,
                _DuplicateJsonKeyError,
                _DuplicateQuestionTargetError,
                _InvalidChangesByQuestionError,
                _InvalidDecisionShapeError,
            ) as error:
                failure_code = _decision_format_failure_code(error)
                _log_invalid_decision_format(
                    request,
                    failure_code,
                    repaired=repaired,
                    error=error,
                )
                if not repaired:
                    repair_prompt = _build_repair_prompt(
                        prompt,
                        error=None,
                        format_failure_code=failure_code,
                        format_error=error,
                    )
                    if (
                        measure_llm_request_bytes(repair_prompt)
                        <= self._limits.max_llm_request_bytes
                    ):
                        active_prompt = repair_prompt
                        repair_reason = failure_code
                        continue
                return self._recover_technical_output_or_fail(
                    request,
                    processed_at or self._read_processed_at(),
                    code=TurnProcessingErrorCode.INVALID_AGENT_OUTPUT,
                    message=_INVALID_AGENT_OUTPUT_MESSAGE,
                )

            if self._reply_writer_client is not None and decision.transition_text is not None:
                decision = decision.model_copy(update={"transition_text": None})

            has_prior_answer_correction = any(
                isinstance(change, (AgentInterpretAnswerChange, AgentSaveAnswerChange))
                and change.replace_existing
                and change.question_code in request.replaceable_question_codes
                for change in decision.changes
            )
            decision = await self._verify_handcraft_engagement(request, decision)
            decision = await self._upgrade_saved_handcraft_engagement(request, decision)
            if len(_nonadditive_replacements(request, decision)) > 1 and not repaired:
                _LOGGER.warning(
                    "LLM предложила несколько замен прежних ответов: "
                    "correlation_id=%s; запрошено исправление",
                    request.correlation_id,
                )
                repair_prompt = _build_repair_prompt(
                    prompt,
                    error=None,
                    format_failure_code="MULTIPLE_REPLACEMENTS",
                )
                if (
                    measure_llm_request_bytes(repair_prompt)
                    <= self._limits.max_llm_request_bytes
                ):
                    active_prompt = repair_prompt
                    repair_reason = "MULTIPLE_REPLACEMENTS"
                    continue
            decision = _discard_conflicting_replacements(request, decision)
            has_prior_answer_correction = has_prior_answer_correction and any(
                isinstance(change, (AgentInterpretAnswerChange, AgentSaveAnswerChange))
                and change.replace_existing
                and change.question_code in request.replaceable_question_codes
                for change in decision.changes
            )

            if processed_at is None:
                processed_at = self._read_processed_at()
            try:
                response = self._build_success_response(request, decision, processed_at)
            except (ValidationError, SemanticMappingError) as error:
                failure_code = _decision_format_failure_code(error)
                _log_invalid_decision_format(
                    request,
                    failure_code,
                    repaired=repaired,
                    error=error,
                )
                if not repaired:
                    repair_prompt = _build_repair_prompt(
                        prompt,
                        error=None,
                        format_failure_code=failure_code,
                    )
                    if (
                        measure_llm_request_bytes(repair_prompt)
                        <= self._limits.max_llm_request_bytes
                    ):
                        active_prompt = repair_prompt
                        repair_reason = failure_code
                        continue
                return self._recover_technical_output_or_fail(
                    request,
                    processed_at,
                    code=TurnProcessingErrorCode.INVALID_AGENT_OUTPUT,
                    message=_INVALID_AGENT_OUTPUT_MESSAGE,
                )
            if has_prior_answer_correction:
                decision, response = await self._reconcile_saved_multi_choice(
                    request,
                    decision,
                    response,
                    processed_at,
                )
            if _proposed_additive_multi_choice_question(request, response) is not None:
                verified_addition = await self._verify_additive_multi_choice(
                    request,
                    decision,
                    response,
                    processed_at,
                )
                if verified_addition is None:
                    return self._recover_technical_output_or_fail(
                        request,
                        processed_at,
                        code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                        message=_REJECTED_AGENT_OUTPUT_MESSAGE,
                    )
                decision, response = verified_addition
            if _pending_multi_choice_question(request, response) is not None:
                verified_response = await self._verify_pending_multi_choice(
                    request,
                    decision,
                    response,
                    processed_at,
                )
                return await self._finalize_assistant_reply(request, verified_response)
            try:
                validate_turn_response(request, response)
            except InvalidTurnResponseError as error:
                _log_rejected_decision(request, error, repaired=repaired)
                if not repaired:
                    repair_prompt = _build_repair_prompt(
                        prompt,
                        error=error,
                        format_failure_code=None,
                        attempted_decision=decision,
                        child_asked_question=any(
                            mark in request.current_message.content
                            for mark in ("?", "？", "⁇", "⁈", "⁉")
                        ),
                    )
                    if (
                        measure_llm_request_bytes(repair_prompt)
                        <= self._limits.max_llm_request_bytes
                    ):
                        active_prompt = repair_prompt
                        repair_reason = "CONTRACT_VALIDATION_FAILED"
                        continue
                clarification_response = (
                    self._build_clarification_reply_recovery_response(
                        request,
                        decision,
                        response,
                        processed_at,
                        error,
                    )
                )
                if clarification_response is not None:
                    return await self._finalize_assistant_reply(
                        request, clarification_response
                    )
                recovery_response = self._build_rejected_output_recovery_response(
                    request,
                    processed_at,
                )
                if recovery_response is not None:
                    return recovery_response
                return self._recover_technical_output_or_fail(
                    request,
                    processed_at,
                    code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                    message=_REJECTED_AGENT_OUTPUT_MESSAGE,
                )
            drafted_response = self._accept_question_draft(request, decision, response)
            if drafted_response is not None:
                return drafted_response
            response, reply_written = await self._refine_next_question(
                request,
                decision,
                response,
                processed_at,
            )
            if reply_written:
                return response
            return await self._finalize_assistant_reply(request, response)

        raise RuntimeError("Недостижимое завершение цикла обработки LLM")

    def _accept_question_draft(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
    ) -> TurnSuccessResponse | None:
        """
        Принимает первый вариант реплики лишь для простого проверенного перехода.
        """
        from questionnaire_agent.prompts.assistant_reply import (
            compose_bridged_reply,
            normalize_repeated_opening,
        )

        draft: AgentQuestionDraft | None = decision.question_draft
        timing = _ACTIVE_TURN_TIMING.get()
        if timing is not None and draft is not None:
            timing.draft_proposed = True
        action = response.next_action
        active = request.active_question
        if (
            self._reply_writer_client is None
            or draft is None
            or request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
            or active is None
            or active.companion_question_code is not None
            or action.action != NextActionType.ASK_QUESTION
            or action.question_code is None
            or action.question_code == "E1"
            or action.companion_question_code is not None
            or draft.question_code != action.question_code
            or draft.asked_question_codes != (action.question_code,)
            or decision.preferred_next_question_code not in (None, action.question_code)
        ):
            return None
        if any(
            isinstance(change, SaveAnswerChange)
            and change.question_code != active.question_code
            for change in response.changes
        ):
            return None
        active_position = next(
            (
                index
                for index, question in enumerate(request.questionnaire.questions)
                if question.question_code == active.question_code
            ),
            None,
        )
        if (
            active_position is not None
            and len(request.dialog_history) >= 2
            and request.dialog_history[-2].role == DialogRole.USER
            and any(
                state.status == QuestionStatus.UNKNOWN
                for question in request.questionnaire.questions[:active_position]
                for state in request.state.questions
                if state.question_code == question.question_code
            )
            and any(isinstance(change, SaveAnswerChange) for change in response.changes)
        ):
            return None
        try:
            draft_reply = compose_bridged_reply(draft.bridge_text, draft.question_text)
        except ValueError:
            return None
        reply = normalize_repeated_opening(
            draft_reply,
            [
                message.content
                for message in request.dialog_history
                if message.role == DialogRole.ASSISTANT
            ],
        )
        proposed = response.model_copy(update={"assistant_reply": reply})
        try:
            validate_turn_response(request, proposed)
        except InvalidTurnResponseError:
            return None
        _LOGGER.info(
            "Первый вызов сформулировал следующий вопрос: correlation_id=%s "
            "question_code=%s",
            request.correlation_id,
            action.question_code,
        )
        if timing is not None:
            timing.draft_accepted = True
        return proposed

    async def _verify_handcraft_engagement(
        self,
        request: TurnRequest,
        decision: AgentDecision,
    ) -> AgentDecision:
        """
        Отклоняет неподтверждённый косвенный признак, сохраняя другие ответы.
        """
        proposed = next(
            (
                change
                for change in decision.changes
                if isinstance(change, AgentInterpretAnswerChange)
                and change.question_code == "X4"
                and isinstance(change.interpretation, AgentScaleInterpretation)
                and change.interpretation.direct_level is None
                and AgentIndirectSignal.SUSTAINED_HANDCRAFT_ENGAGEMENT
                in change.interpretation.indirect_signals
            ),
            None,
        )
        if proposed is None:
            return decision

        question = next(
            item for item in request.questionnaire.questions if item.question_code == "X4"
        )
        state = next(item for item in request.state.questions if item.question_code == "X4")
        needs_clarification = (
            request.active_question is not None
            and request.active_question.question_code == "X4"
            and state.clarification_attempts < _MAX_CLARIFICATION_ATTEMPTS
            and decision.clarification is None
            and decision.explanation is None
        )
        evidence = select_verified_evidence(
            request.current_message.content,
            proposed.evidence_candidate_id,
        )
        suggested_clarification: str | None = None
        try:
            prompt = build_handcraft_verification_prompt(
                request,
                evidence,
                needs_clarification=needs_clarification,
            )
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос проверки ручного творчества слишком велик")
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="handcraft_verification",
                reason="indirect_handcraft_signal",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ проверки ручного творчества слишком велик")
            verification = parse_handcraft_verification(
                completion.content,
                needs_clarification=needs_clarification,
            )
            if verification.supported:
                return decision
            suggested_clarification = verification.clarification_text
            _LOGGER.warning(
                "Косвенный признак ручного творчества отклонён: correlation_id=%s",
                request.correlation_id,
            )
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Проверка косвенного признака ручного творчества не завершилась: "
                "correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )

        filtered_changes = tuple(change for change in decision.changes if change is not proposed)
        clarification = decision.clarification
        if needs_clarification:
            previous_assistant_text = (
                request.dialog_history[-1].content if request.dialog_history else None
            )
            if suggested_clarification is None or not is_safe_contextual_clarification(
                question,
                suggested_clarification,
                previous_assistant_text=previous_assistant_text,
                context_texts=(request.current_message.content,),
            ):
                suggested_clarification = question.clarification_text
            assert suggested_clarification is not None
            clarification = AgentClarification(
                question_code="X4",
                reason="Неясен интерес к другим видам ручного творчества.",
                clarification_text=suggested_clarification,
            )
        return AgentDecision(
            changes=filtered_changes,
            clarification=clarification,
            transition_text=None if clarification is not None else decision.transition_text,
            explanation=decision.explanation,
            preferred_next_question_code=(
                None if clarification is not None else decision.preferred_next_question_code
            ),
        )

    async def _upgrade_saved_handcraft_engagement(
        self,
        request: TurnRequest,
        decision: AgentDecision,
    ) -> AgentDecision:
        """
        Обновляет прежний интерес к поделкам по новому ответу о том же занятии.
        """
        active = request.active_question
        if active is None or "X4" not in request.replaceable_question_codes:
            return decision
        active_question = next(
            (
                question
                for question in request.questionnaire.questions
                if question.question_code == active.question_code
            ),
            None,
        )
        saved_state = next(
            (state for state in request.state.questions if state.question_code == "X4"),
            None,
        )
        x4_question = next(
            (
                question
                for question in request.questionnaire.questions
                if question.question_code == "X4"
            ),
            None,
        )
        if (
            active_question is None
            or active_question.answer_meaning != "FOCUS_DURATION"
            or saved_state is None
            or saved_state.status != QuestionStatus.ANSWERED
            or saved_state.answer is None
            or x4_question is None
            or x4_question.answer_meaning != "HANDCRAFT_ATTITUDE"
            or not _prior_saved_answer_messages(request, "X4")
            or any(
                change.question_code == "X4"
                or getattr(change, "replace_existing", False)
                for change in decision.changes
            )
            or not any(
                change.question_code == active.question_code
                and isinstance(change, (AgentInterpretAnswerChange, AgentSaveAnswerChange))
                for change in decision.changes
            )
        ):
            return decision
        saved_level = next(
            (
                option.numeric_value
                for option in x4_question.options
                if option.option_code in saved_state.answer.option_codes
            ),
            None,
        )
        if saved_level is None or saved_level >= 3:
            return decision

        try:
            prompt = build_handcraft_verification_prompt(
                request,
                request.current_message.content,
                needs_clarification=False,
            )
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос проверки ручного творчества слишком велик")
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="handcraft_upgrade",
                reason="saved_handcraft_reassessment",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ проверки ручного творчества слишком велик")
            verification = parse_handcraft_verification(
                completion.content,
                needs_clarification=False,
            )
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Поздняя проверка ручного творчества не завершилась: "
                "correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return decision
        if not verification.supported:
            return decision
        upgraded_answer = AgentInterpretAnswerChange(
            operation="INTERPRET_ANSWER",
            question_code="X4",
            interpretation=AgentScaleInterpretation(
                kind="SCALE",
                direct_level=None,
                indirect_signals=(AgentIndirectSignal.SUSTAINED_HANDCRAFT_ENGAGEMENT,),
            ),
            evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
            replace_existing=True,
        )
        return decision.model_copy(
            update={"changes": (*decision.changes, upgraded_answer)}
        )

    async def _refine_next_question(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
    ) -> tuple[TurnSuccessResponse, bool]:
        """
        Уточняет тематический переход и отмечает готовую итоговую реплику.
        """
        from questionnaire_agent.prompts.next_question import (
            build_next_question_prompt,
            parse_next_question_choice,
        )

        if response.next_action.action != NextActionType.ASK_QUESTION:
            return response, False
        if response.next_action.companion_question_code is not None:
            return response, False
        if (
            request.active_question is not None
            and request.active_question.companion_question_code is not None
            and response.next_action.question_code
            in {
                request.active_question.question_code,
                request.active_question.companion_question_code,
            }
        ):
            return response, False
        open_questions = _open_next_question_candidates(request, response)
        if len(open_questions) < 2:
            return response, False
        saved_answers = tuple(
            change for change in response.changes if isinstance(change, SaveAnswerChange)
        )
        active_code = (
            None if request.active_question is None else request.active_question.question_code
        )
        active_position = next(
            (
                index
                for index, question in enumerate(request.questionnaire.questions)
                if question.question_code == active_code
            ),
            None,
        )
        statuses_by_code = {
            state.question_code: state.status for state in request.state.questions
        }
        continuing_earlier_topic = (
            active_position is not None
            and len(request.dialog_history) >= 2
            and request.dialog_history[-2].role == DialogRole.USER
            and any(
                statuses_by_code.get(question.question_code) == QuestionStatus.UNKNOWN
                for question in request.questionnaire.questions[:active_position]
            )
        )
        answer_crossed_topics = bool(saved_answers) and (
            request.answer_scope == TurnAnswerScope.ALL_UNANSWERED
            or any(change.question_code != active_code for change in saved_answers)
            or continuing_earlier_topic
        )
        preferred_was_rejected = (
            decision.preferred_next_question_code is not None
            and decision.preferred_next_question_code != response.next_action.question_code
        )
        if not preferred_was_rejected and not answer_crossed_topics:
            return response, False

        routing_reason = (
            "rejected_preference_and_cross_topic_answer"
            if preferred_was_rejected and answer_crossed_topics
            else "rejected_preference"
            if preferred_was_rejected
            else "cross_topic_answer"
        )
        if self._reply_writer_client is not None:
            return await self._route_and_write_next_question(
                request,
                decision,
                response,
                processed_at,
                open_questions,
                routing_reason,
            )

        prompt = build_next_question_prompt(
            request,
            response,
            open_questions,
            separate_reply=self._reply_writer_client is not None,
        )
        if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
            return response, False
        try:
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="next_question_selection",
                reason=routing_reason,
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return response, False
            choice = parse_next_question_choice(
                completion.content,
                frozenset(question.question_code for question in open_questions),
            )
        except (LLMClientError, ValueError, ValidationError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Тематический переход не уточнён: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return response, False
        if (
            choice.question_code == response.next_action.question_code
            and choice.transition_text is None
        ):
            return response, False

        transitions = (
            (None,)
            if self._reply_writer_client is not None or choice.transition_text is None
            else (choice.transition_text, None)
        )
        for transition_text in transitions:
            revised_decision = decision.model_copy(
                update={
                    "preferred_next_question_code": choice.question_code,
                    "transition_text": transition_text,
                }
            )
            try:
                revised_response = self._build_success_response(
                    request, revised_decision, processed_at
                )
                validate_turn_response(request, revised_response)
            except (InvalidTurnResponseError, ValidationError, SemanticMappingError):
                continue
            return revised_response, False
        return response, False

    async def _route_and_write_next_question(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
        open_questions: tuple[Question, ...],
        routing_reason: str,
    ) -> tuple[TurnSuccessResponse, bool]:
        """
        Совмещает тематический выбор с живой репликой для проверенной цели.
        """
        from questionnaire_agent.prompts.assistant_reply import normalize_repeated_opening
        from questionnaire_agent.prompts.routed_reply import (
            build_routed_reply_prompt,
            parse_routed_reply,
        )

        assert self._reply_writer_client is not None
        responses_by_code: dict[str, TurnSuccessResponse] = {}
        default_response: TurnSuccessResponse | None = None
        for route_code in (None, *(item.question_code for item in open_questions)):
            revised_decision = decision.model_copy(
                update={
                    "preferred_next_question_code": route_code,
                    "transition_text": None,
                }
            )
            try:
                candidate = self._build_success_response(
                    request,
                    revised_decision,
                    processed_at,
                    log_evidence_fallback=False,
                )
                validate_turn_response(request, candidate)
            except (InvalidTurnResponseError, ValidationError, SemanticMappingError):
                continue
            if candidate.next_action.action != NextActionType.ASK_QUESTION:
                continue
            if route_code is None:
                default_response = candidate
            else:
                responses_by_code[route_code] = candidate
        if default_response is None or not responses_by_code:
            return response, False

        try:
            prompt = build_routed_reply_prompt(
                request,
                response,
                open_questions,
                responses_by_code,
                default_response,
            )
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                return response, False
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._reply_writer_client,
                stage="next_question_reply",
                reason=routing_reason,
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return response, False
            choice = parse_routed_reply(
                completion.content, responses_by_code, default_response
            )
        except (LLMClientError, ValueError, ValidationError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Объединённый тематический вопрос не принят: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return response, False

        selected = (
            default_response
            if choice.question_code is None
            else responses_by_code[choice.question_code]
        )
        if choice.assistant_reply is None:
            return selected, False
        reply = normalize_repeated_opening(
            choice.assistant_reply,
            [
                message.content
                for message in request.dialog_history
                if message.role == DialogRole.ASSISTANT
            ],
        )
        natural_response = selected.model_copy(update={"assistant_reply": reply})
        try:
            validate_turn_response(request, natural_response)
        except InvalidTurnResponseError as error:
            _LOGGER.warning(
                "Реплика объединённого тематического вопроса заменена: "
                "correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return selected, False
        return natural_response, True

    async def _finalize_assistant_reply(
        self,
        request: TurnRequest,
        response: TurnSuccessResponse | TurnFailureResponse,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Формулирует проверенный следующий вопрос как цельную реплику ребёнку.
        """
        response = await self._naturalize_partial_choice_question(request, response)
        response = await self._write_clarification_reply(request, response)
        return await self._write_ask_question_reply(request, response)

    async def _write_clarification_reply(
        self,
        request: TurnRequest,
        response: TurnSuccessResponse | TurnFailureResponse,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Переформулирует смысловое уточнение, сохраняя проверенный запасной ход.
        """
        from questionnaire_agent.prompts.clarification_reply import (
            build_clarification_reply_prompt,
            parse_clarification_reply,
        )

        if (
            self._reply_writer_client is None
            or not isinstance(response, TurnSuccessResponse)
            or response.next_action.action != NextActionType.CLARIFY
            or response.next_action.question_code == "E1"
        ):
            return response

        question_code = response.next_action.question_code
        assert question_code is not None
        try:
            prompt = build_clarification_reply_prompt(request, response)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                return response
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._reply_writer_client,
                stage="clarification_writer",
                reason="clarify",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return response
            reply = parse_clarification_reply(completion.content, question_code)
            revised_changes = tuple(
                change.model_copy(update={"clarification_text": reply})
                if isinstance(change, RequestClarificationChange)
                else change
                for change in response.changes
            )
            natural_response = response.model_copy(
                update={"changes": revised_changes, "assistant_reply": reply}
            )
            validate_turn_response(request, natural_response)
        except (
            LLMClientError,
            InvalidTurnResponseError,
            ValidationError,
            ValueError,
            UnicodeEncodeError,
        ) as error:
            _LOGGER.warning(
                "Формулировка уточнения заменена исходной: "
                "correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return response
        return natural_response

    async def _write_ask_question_reply(
        self,
        request: TurnRequest,
        response: TurnSuccessResponse | TurnFailureResponse,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Принимает живую формулировку или безопасный вопрос из анкеты.
        """
        from questionnaire_agent.prompts.assistant_reply import (
            build_assistant_reply_prompt,
            compose_bridged_reply,
            normalize_repeated_opening,
            parse_assistant_reply,
        )

        if (
            self._reply_writer_client is None
            or not isinstance(response, TurnSuccessResponse)
            or response.next_action.action != NextActionType.ASK_QUESTION
            or response.next_action.question_code == "E1"
        ):
            return response

        next_action = response.next_action
        assert next_action.question_code is not None
        if next_action.companion_question_code is None:
            fallback_text = next(
                question.question_text
                for question in request.questionnaire.questions
                if question.question_code == next_action.question_code
            )
        else:
            fallback_text = next(
                composite.question_text
                for composite in request.questionnaire.composite_questions
                if composite.question_codes
                == (next_action.question_code, next_action.companion_question_code)
            )
        try:
            fallback_reply = compose_bridged_reply(
                _ASK_REPLY_BRIDGE_FALLBACK, fallback_text
            )
        except ValueError:
            fallback_reply = fallback_text
        fallback_response = response.model_copy(update={"assistant_reply": fallback_reply})
        try:
            validate_turn_response(request, fallback_response)
        except InvalidTurnResponseError:
            fallback_response = response.model_copy(update={"assistant_reply": fallback_text})

        try:
            prompt = build_assistant_reply_prompt(request, response)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                return fallback_response
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._reply_writer_client,
                stage="reply_writer",
                reason="ask_question",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return fallback_response
            expected_codes = (next_action.question_code,)
            if next_action.companion_question_code is not None:
                expected_codes = (*expected_codes, next_action.companion_question_code)
            reply = parse_assistant_reply(completion.content, expected_codes)
            reply = normalize_repeated_opening(
                reply,
                [
                    message.content
                    for message in request.dialog_history
                    if message.role == DialogRole.ASSISTANT
                ],
            )
            natural_response = response.model_copy(update={"assistant_reply": reply})
            validate_turn_response(request, natural_response)
        except (
            LLMClientError,
            InvalidTurnResponseError,
            ValidationError,
            ValueError,
            UnicodeEncodeError,
        ) as error:
            _LOGGER.warning(
                "Формулировка следующего вопроса заменена резервной: "
                "correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return fallback_response
        return natural_response

    async def _resolve_partial_multi_choice(
        self,
        request: TurnRequest,
        question: Question,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Завершает ранний выбор E1, учитывая только новые занятия в ответе.
        """
        processed_at = self._read_processed_at()
        try:
            prompt = build_partial_choice_verification_prompt(request, question)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос частичного выбора превышает допустимый размер")
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="partial_choice_verification",
                reason="active_partial_choice",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ частичного выбора превышает допустимый размер")
            verification = parse_choice_verification(completion.content, question)
            selected_codes = set(
                next(
                    state.answer.option_codes
                    for state in request.state.questions
                    if state.question_code == question.question_code and state.answer is not None
                )
            )
            selected_refs = {
                f"V{index}"
                for index, option in enumerate(question.options, start=1)
                if option.option_code in selected_codes
            }
            if selected_refs.intersection(verification.supported_option_refs):
                raise ValueError("проверка повторила уже выбранное занятие")
        except LLMClientError as error:
            return self._build_failure_response(
                request,
                processed_at=processed_at,
                code=error.code.value,
                message=str(error),
                retryable=error.retryable,
            )
        except (ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Проверка частичного выбора не завершилась: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return self._recover_technical_output_or_fail(
                request,
                processed_at,
                code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                message=_REJECTED_AGENT_OUTPUT_MESSAGE,
            )

        if verification.outcome == ChoiceVerificationOutcome.SELECTED:
            active_change: AgentStateChange = AgentInterpretAnswerChange(
                operation="INTERPRET_ANSWER",
                question_code=question.question_code,
                interpretation=AgentOptionSelectionInterpretation(
                    kind="OPTION_SELECTION",
                    option_refs=verification.supported_option_refs,
                ),
                evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
                replace_existing=False,
            )
            clarification = None
        elif verification.outcome == ChoiceVerificationOutcome.NONE:
            active_change = AgentConfirmAnswerChange(
                operation="CONFIRM_ANSWER",
                question_code=question.question_code,
                evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
            )
            clarification = None
        else:
            question_state = next(
                state
                for state in request.state.questions
                if state.question_code == question.question_code
            )
            if question_state.clarification_attempts >= _MAX_CLARIFICATION_ATTEMPTS:
                active_change = AgentConfirmAnswerChange(
                    operation="CONFIRM_ANSWER",
                    question_code=question.question_code,
                    evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
                )
                clarification = None
            else:
                active_change = None
                fallback_text = partial_multi_choice_followup(
                    question,
                    question_state.answer.option_codes,
                    question_state.clarification_attempts + 1,
                )
                clarification = AgentClarification(
                    question_code=question.question_code,
                    reason="Пока неясно, есть ли другие любимые занятия.",
                    clarification_text=await self._natural_partial_choice_clarification(
                        request,
                        question,
                        fallback_text,
                    ),
                )

        decision = AgentDecision(
            changes=() if active_change is None else (active_change,),
            clarification=clarification,
            transition_text=None,
        )
        try:
            response = self._build_success_response(request, decision, processed_at)
            validate_turn_response(request, response)
        except (InvalidTurnResponseError, ValidationError, SemanticMappingError) as error:
            if clarification is not None:
                fallback_text = partial_multi_choice_followup(
                    question,
                    next(
                        state.answer.option_codes
                        for state in request.state.questions
                        if state.question_code == question.question_code
                        and state.answer is not None
                    ),
                    next(
                        state.clarification_attempts
                        for state in request.state.questions
                        if state.question_code == question.question_code
                    )
                    + 1,
                )
                if clarification.clarification_text != fallback_text:
                    fallback_decision = decision.model_copy(
                        update={
                            "clarification": clarification.model_copy(
                                update={"clarification_text": fallback_text}
                            )
                        }
                    )
                    try:
                        fallback_response = self._build_success_response(
                            request,
                            fallback_decision,
                            processed_at,
                        )
                        validate_turn_response(request, fallback_response)
                    except (
                        InvalidTurnResponseError,
                        ValidationError,
                        SemanticMappingError,
                    ):
                        pass
                    else:
                        return fallback_response
            _LOGGER.warning(
                "Частичный выбор не прошёл контракт: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return self._recover_technical_output_or_fail(
                request,
                processed_at,
                code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                message=_REJECTED_AGENT_OUTPUT_MESSAGE,
            )
        return await self._naturalize_partial_choice_question(request, response)

    async def _natural_partial_choice_clarification(
        self,
        request: TurnRequest,
        question: Question,
        fallback_text: str,
    ) -> str:
        """
        Предпочитает живой вопрос модели и сохраняет безопасный запасной текст.
        """
        try:
            prompt = build_partial_choice_clarification_prompt(request, question)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                return fallback_text
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="partial_choice_clarification",
                reason="unverified_partial_choice",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return fallback_text
            return parse_partial_choice_clarification(completion.content)
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError):
            return fallback_text

    async def _naturalize_partial_choice_question(
        self,
        request: TurnRequest,
        response: TurnSuccessResponse | TurnFailureResponse,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Даёт модели сформулировать первый вопрос о других любимых занятиях.
        """
        if (
            not isinstance(response, TurnSuccessResponse)
            or response.next_action.action != NextActionType.ASK_QUESTION
            or response.next_action.question_code != "E1"
        ):
            return response
        question = next(
            (item for item in request.questionnaire.questions if item.question_code == "E1"),
            None,
        )
        if question is None:
            return response
        e1_state = next(
            (item for item in request.state.questions if item.question_code == "E1"),
            None,
        )
        selected_codes = (
            () if e1_state is None or e1_state.answer is None else e1_state.answer.option_codes
        )
        for change in response.changes:
            if isinstance(change, SaveAnswerChange) and change.question_code == "E1":
                selected_codes = tuple(dict.fromkeys((*selected_codes, *change.option_codes)))
        if not selected_codes or len(selected_codes) >= question.max_selections:
            return response

        try:
            prompt = build_partial_choice_question_prompt(request, question, selected_codes)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                return response
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="partial_choice_question",
                reason="ask_remaining_choices",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                return response
            question_text = parse_partial_choice_question(completion.content)
            natural_response = response.model_copy(
                update={"assistant_reply": question_text}
            )
            validate_turn_response(request, natural_response)
        except (
            LLMClientError,
            InvalidTurnResponseError,
            ValueError,
            TypeError,
            UnicodeEncodeError,
        ):
            return response
        return natural_response

    async def _reconcile_saved_multi_choice(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
    ) -> tuple[AgentDecision, TurnSuccessResponse]:
        """
        Пересматривает прежний E1 только после явного исправления другого ответа.
        """
        correction = request.answer_correction
        question = next(
            (item for item in request.questionnaire.questions if item.question_code == "E1"),
            None,
        )
        state = next(
            (item for item in request.state.questions if item.question_code == "E1"),
            None,
        )
        if (
            correction is None
            or correction.question_code != "E1"
            or correction.mode != AnswerCorrectionMode.ADDITIVE
            or question is None
            or state is None
            or state.answer is None
        ):
            return decision, response
        prior_messages = _prior_saved_answer_messages(request, "E1")
        if len(prior_messages) != len(state.answer.source_message_ids):
            return decision, response

        try:
            prompt = build_choice_reconciliation_prompt(request, question)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос пересмотра E1 превышает допустимый размер")
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="choice_reconciliation",
                reason="prior_answer_correction",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ пересмотра E1 превышает допустимый размер")
            verification = parse_choice_reconciliation(
                completion.content,
                question,
                state.answer.option_codes,
                tuple(
                    message.message_id
                    for message in request.dialog_history
                    if message.role == DialogRole.USER
                ),
            )
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Проверка пересмотра E1 не завершилась: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return decision, response

        removed_refs = set(verification.retracted_option_refs)
        added_refs = set(verification.new_option_refs)
        prior_codes = set(state.answer.option_codes)
        revised_codes = tuple(
            option.option_code
            for index, option in enumerate(question.options, start=1)
            if (
                (option.option_code in prior_codes and f"V{index}" not in removed_refs)
                or f"V{index}" in added_refs
            )
        )
        proposed_e1_change = any(change.question_code == "E1" for change in decision.changes)
        if set(revised_codes) == prior_codes and not proposed_e1_change:
            return decision, response

        filtered_decision = decision.model_copy(
            update={"changes": _other_question_changes(decision, "E1")}
        )
        support_ids = set(verification.supporting_prior_message_ids)
        support_messages = tuple(
            message
            for message in request.dialog_history
            if message.role == DialogRole.USER and message.message_id in support_ids
        )
        revision = (
            None
            if set(revised_codes) == prior_codes
            else ReviseMultiChoiceChange(
                question_code="E1",
                option_codes=revised_codes,
                source_message_ids=(
                    *(message.message_id for message in support_messages),
                    request.current_message.message_id,
                ),
                evidence="; ".join(
                    (*(message.content for message in support_messages), request.current_message.content)
                ),
            )
        )
        try:
            revised_response = self._build_success_response(
                request,
                filtered_decision,
                processed_at,
                supplemental_changes=() if revision is None else (revision,),
            )
            validate_turn_response(request, revised_response)
        except (InvalidTurnResponseError, ValidationError, SemanticMappingError) as error:
            _LOGGER.warning(
                "Пересмотр E1 не прошёл контракт: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return decision, response
        return filtered_decision, revised_response

    async def _verify_additive_multi_choice(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
    ) -> tuple[AgentDecision, TurnSuccessResponse] | None:
        """
        Проверяет новое любимое занятие и сохраняет открытый активный вопрос.
        """
        question = _proposed_additive_multi_choice_question(request, response)
        assert question is not None
        proposed_refs = _proposed_option_refs(question, response)
        verification_rejected = False
        try:
            prompt = build_additive_choice_verification_prompt(request, question)
            if measure_llm_request_bytes(prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос проверки дополнения превышает допустимый размер")
            completion = await self._complete_llm(
                request,
                prompt,
                client=self._llm_client,
                stage="additive_choice_verification",
                reason="new_favourite_choice",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ проверки дополнения превышает допустимый размер")
            verification = parse_choice_verification(completion.content, question)
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Проверка дополнения выбора не завершилась: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
        else:
            if (
                verification.outcome == ChoiceVerificationOutcome.SELECTED
                and set(verification.supported_option_refs) == set(proposed_refs)
            ):
                return decision, response
            _LOGGER.warning(
                "Дополнение выбора не подтверждено: correlation_id=%s code=REF_MISMATCH",
                request.correlation_id,
            )
            verification_rejected = True

        filtered_decision = decision.model_copy(
            update={
                "changes": _other_question_changes(decision, question.question_code),
            }
        )
        if (
            verification_rejected
            and not filtered_decision.changes
            and filtered_decision.clarification is None
            and filtered_decision.explanation is None
            and filtered_decision.transition_text is None
        ):
            recovery_decision = _build_rejected_output_recovery_decision(request)
            if recovery_decision is not None and recovery_decision.clarification is not None:
                filtered_decision = recovery_decision
        try:
            filtered_response = self._build_success_response(
                request,
                filtered_decision,
                processed_at,
            )
        except (ValidationError, SemanticMappingError):
            _LOGGER.warning(
                "Не удалось удалить неподтверждённое дополнение: correlation_id=%s",
                request.correlation_id,
            )
            return None
        return filtered_decision, filtered_response

    async def _verify_pending_multi_choice(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Независимо проверяет текущий выбор после уточнения списка вариантов.
        """
        question = _pending_multi_choice_question(request, response)
        assert question is not None
        proposed_option_refs = _proposed_option_refs(question, response)
        try:
            verification_prompt = build_choice_verification_prompt(
                request,
                question,
            )
            if measure_llm_request_bytes(verification_prompt) > self._limits.max_llm_request_bytes:
                raise ValueError("запрос проверки выбора превышает допустимый размер")
            completion = await self._complete_llm(
                request,
                verification_prompt,
                client=self._llm_client,
                stage="pending_choice_verification",
                reason="pending_choice_answer",
            )
            if measure_llm_output_bytes(completion.content) > self._limits.max_llm_output_bytes:
                raise ValueError("ответ проверки выбора превышает допустимый размер")
            verification = parse_choice_verification(completion.content, question)
        except (LLMClientError, ValueError, TypeError, UnicodeEncodeError) as error:
            _LOGGER.warning(
                "Проверка выбора не завершилась: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            if (
                decision.clarification is not None
                and decision.clarification.question_code == question.question_code
            ):
                return self._build_unverified_choice_response(
                    request,
                    decision,
                    processed_at,
                )
            return self._recover_technical_output_or_fail(
                request,
                processed_at,
                code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                message=_REJECTED_AGENT_OUTPUT_MESSAGE,
            )

        if verification.outcome == ChoiceVerificationOutcome.SELECTED:
            if proposed_option_refs and set(proposed_option_refs) != set(
                verification.supported_option_refs
            ):
                _LOGGER.warning(
                    "Выбор вариантов не подтверждён: correlation_id=%s code=REF_MISMATCH",
                    request.correlation_id,
                )
                return self._build_unverified_choice_response(request, decision, processed_at)
            if proposed_option_refs:
                try:
                    validate_turn_response(request, response)
                except InvalidTurnResponseError:
                    return self._build_unverified_choice_response(
                        request,
                        decision,
                        processed_at,
                    )
                return response
            active_change: AgentStateChange = AgentInterpretAnswerChange(
                operation="INTERPRET_ANSWER",
                question_code=question.question_code,
                interpretation=AgentOptionSelectionInterpretation(
                    kind="OPTION_SELECTION",
                    option_refs=verification.supported_option_refs,
                ),
                evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
                replace_existing=False,
            )
        elif verification.outcome == ChoiceVerificationOutcome.NONE:
            active_change = AgentSkipQuestionChange(
                operation="SKIP_QUESTION",
                question_code=question.question_code,
                reason=PolicySkipReason.NO_MATCHING_OPTIONS,
                evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
            )
        else:
            return self._build_unverified_choice_response(request, decision, processed_at)

        verified_decision = AgentDecision(
            changes=(
                *_other_question_changes(decision, question.question_code),
                active_change,
            ),
            clarification=None,
            transition_text=None,
        )
        try:
            verified_response = self._build_success_response(
                request,
                verified_decision,
                processed_at,
            )
            validate_turn_response(request, verified_response)
        except (InvalidTurnResponseError, ValidationError, SemanticMappingError):
            _LOGGER.warning(
                "Проверенный выбор не прошёл контракт: correlation_id=%s",
                request.correlation_id,
            )
            return self._build_unverified_choice_response(request, decision, processed_at)
        return verified_response

    def _build_unverified_choice_response(
        self,
        request: TurnRequest,
        decision: AgentDecision,
        processed_at: datetime,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Завершает непроверенный выбор уточнением или лимитным пропуском.
        """
        active_question = request.active_question
        assert active_question is not None
        question = next(
            item
            for item in request.questionnaire.questions
            if item.question_code == active_question.question_code
        )
        question_state = next(
            item
            for item in request.state.questions
            if item.question_code == active_question.question_code
        )
        if (
            question_state.clarification_attempts < _MAX_CLARIFICATION_ATTEMPTS
            and decision.clarification is not None
            and decision.clarification.question_code == question.question_code
        ):
            natural_decision = AgentDecision(
                changes=_other_question_changes(decision, question.question_code),
                clarification=decision.clarification,
                transition_text=None,
            )
            try:
                natural_response = self._build_success_response(
                    request,
                    natural_decision,
                    processed_at,
                )
                validate_turn_response(request, natural_response)
            except (InvalidTurnResponseError, ValidationError, SemanticMappingError):
                pass
            else:
                return natural_response
        clarification = (
            None
            if question_state.clarification_attempts >= _MAX_CLARIFICATION_ATTEMPTS
            else AgentClarification(
                question_code=question.question_code,
                reason="Выбор из показанных занятий пока неясен.",
                clarification_text=_fallback_clarification_text(
                    request,
                    question.question_code,
                ),
            )
        )
        fallback_decision = AgentDecision(
            changes=_other_question_changes(decision, question.question_code),
            clarification=clarification,
            transition_text=None,
        )
        try:
            fallback_response = self._build_success_response(
                request,
                fallback_decision,
                processed_at,
            )
            validate_turn_response(request, fallback_response)
        except (InvalidTurnResponseError, ValidationError, SemanticMappingError) as error:
            failure_code = (
                ",".join(item.code.value for item in error.violations)
                if isinstance(error, InvalidTurnResponseError)
                else type(error).__name__
            )
            _LOGGER.warning(
                "Безопасное уточнение выбора не прошло контракт: correlation_id=%s code=%s",
                request.correlation_id,
                failure_code,
            )
            return self._recover_technical_output_or_fail(
                request,
                processed_at,
                code=TurnProcessingErrorCode.REJECTED_AGENT_OUTPUT,
                message=_REJECTED_AGENT_OUTPUT_MESSAGE,
            )
        return fallback_response

    def _read_processed_at(self) -> datetime:
        """
        Получает время обработки и запрещает значение без часового пояса.
        """
        processed_at = self._clock()
        if processed_at.tzinfo is None or processed_at.utcoffset() is None:
            raise ValueError("Часы контроллера должны возвращать timezone-aware datetime")
        return processed_at.astimezone(timezone.utc)

    @staticmethod
    def _build_success_response(
        request: TurnRequest,
        decision: AgentDecision,
        processed_at: datetime,
        *,
        supplemental_changes: tuple[ReviseMultiChoiceChange, ...] = (),
        log_evidence_fallback: bool = True,
    ) -> TurnSuccessResponse:
        """
        Добавляет envelope и программно формирует следующую реплику.
        """
        policy_change = _build_clarification_limit_change(request, decision)
        clarification = None if policy_change is not None else _resolve_clarification(decision)
        explanation = None if policy_change is not None else decision.explanation
        effective_changes = [
            *_project_semantic_changes(request, decision.changes),
            *supplemental_changes,
        ]
        grounded_prior_context_codes = {
            change.question_code
            for change in decision.changes
            if isinstance(change, AgentInterpretAnswerChange)
            and change.replace_existing
            and isinstance(change.interpretation, AgentScaleInterpretation)
            and AgentIndirectSignal.SUSTAINED_HANDCRAFT_ENGAGEMENT
            in change.interpretation.indirect_signals
        }
        if policy_change is not None:
            effective_changes.append(policy_change)
        decision_payload = {
            "changes": [
                (
                    change.model_dump(mode="json", by_alias=True)
                    if isinstance(change, (ApplyDefaultAnswerChange, ReviseMultiChoiceChange))
                    else _build_external_change(
                        request,
                        change,
                        change_count=len(effective_changes),
                        include_prior_answer_context=(
                            change.question_code in grounded_prior_context_codes
                        ),
                        log_evidence_fallback=log_evidence_fallback,
                    )
                )
                for change in effective_changes
            ]
        }
        if clarification is not None:
            decision_payload["changes"].append(
                {
                    "operation": "REQUEST_CLARIFICATION",
                    "questionCode": clarification.question_code,
                    "reason": clarification.reason,
                    "clarificationText": clarification.clarification_text,
                }
            )
        next_action = _derive_next_action(
            request,
            tuple(effective_changes),
            clarification,
            explanation,
            preferred_next_question_code=decision.preferred_next_question_code,
        )
        next_action = _with_composite_question(
            request,
            tuple(effective_changes),
            next_action,
        )
        decision_payload["nextAction"] = next_action.model_dump(
            mode="json",
            by_alias=True,
        )
        canonical_reply = canonical_turn_reply(
            request.questionnaire,
            next_action.action,
            next_action.question_code,
            clarification_text=(
                None if clarification is None else clarification.clarification_text
            ),
            pending_clarification=request.state.pending_clarification is not None,
            corrected_answer=any(
                change.operation == "SAVE_ANSWER"
                and change.replace_existing
                and change.question_code in request.replaceable_question_codes
                for change in effective_changes
            ),
            recorded_other_answer=(
                request.active_question is not None
                and any(
                    change.operation == "SAVE_ANSWER"
                    and change.question_code
                    != request.active_question.question_code
                    for change in effective_changes
                )
            ),
        )
        if next_action.companion_question_code is not None:
            canonical_reply = next(
                composite.question_text
                for composite in request.questionnaire.composite_questions
                if composite.question_codes
                == (next_action.question_code, next_action.companion_question_code)
            )
        if (
            next_action.action == NextActionType.ASK_QUESTION
            and next_action.question_code == "E1"
            and _projected_partial_e1_options(request, tuple(effective_changes)) is not None
        ):
            e1_question = next(
                question
                for question in request.questionnaire.questions
                if question.question_code == "E1"
            )
            selected_codes = _projected_partial_e1_options(
                request,
                tuple(effective_changes),
            )
            assert selected_codes is not None
            canonical_reply = partial_multi_choice_prompt(e1_question, selected_codes)
        decision_payload["assistantReply"] = (
            _compose_explanatory_reask(
                request,
                explanation.explanation_text,
            )
            if explanation is not None
            else _compose_assistant_reply(
                canonical_reply,
                transition_text=(
                    None
                    if policy_change is not None
                    or (
                        request.active_question is not None
                        and request.active_question.companion_question_code is not None
                        and next_action.question_code
                        in {
                            request.active_question.question_code,
                            request.active_question.companion_question_code,
                        }
                    )
                    else (
                        decision.transition_text
                        if decision.preferred_next_question_code is None
                        or next_action.question_code == decision.preferred_next_question_code
                        else None
                    )
                    or _contextual_question_transition(request, next_action)
                ),
                next_action=next_action,
            )
        )
        return TurnSuccessResponse.model_validate(
            {
                **decision_payload,
                "status": "SUCCESS",
                **_build_response_envelope(request, processed_at),
            }
        )

    @staticmethod
    def _build_failure_response(
        request: TurnRequest,
        *,
        processed_at: datetime,
        code: str,
        message: str,
        retryable: bool,
    ) -> TurnFailureResponse:
        """
        Создаёт контролируемую ошибку без патча и ответа ассистента.
        """
        return TurnFailureResponse.model_validate(
            {
                **_build_response_envelope(request, processed_at),
                "status": "FAILED",
                "error": {
                    "code": code,
                    "message": message,
                    "retryable": retryable,
                },
            }
        )

    def _recover_technical_output_or_fail(
        self,
        request: TurnRequest,
        processed_at: datetime,
        *,
        code: TurnProcessingErrorCode,
        message: str,
    ) -> TurnSuccessResponse | TurnFailureResponse:
        """
        Продолжает анкету после повреждённого вывода без выдуманной оценки.
        """
        response = self._build_technical_output_recovery_response(request, processed_at)
        if response is not None:
            return response
        return self._build_failure_response(
            request,
            processed_at=processed_at,
            code=code.value,
            message=message,
            retryable=True,
        )

    @staticmethod
    def _build_technical_output_recovery_response(
        request: TurnRequest,
        processed_at: datetime,
    ) -> TurnSuccessResponse | None:
        """
        Повторяет активный вопрос либо начинает опрос после сбоя разбора рассказа.
        """
        try:
            if request.answer_scope == TurnAnswerScope.ALL_UNANSWERED:
                response = QuestionnaireTurnController._build_success_response(
                    request,
                    AgentDecision(changes=(), clarification=None, transition_text=None),
                    processed_at,
                )
            else:
                active_question = request.active_question
                if active_question is None:
                    return None
                next_action = NextAction(
                    action=NextActionType.REASK_QUESTION,
                    question_code=active_question.question_code,
                )
                active_state = next(
                    state
                    for state in request.state.questions
                    if state.question_code == active_question.question_code
                )
                reask_reply = canonical_turn_reply(
                    request.questionnaire,
                    next_action.action,
                    next_action.question_code,
                    pending_clarification=(
                        request.state.pending_clarification is not None
                    ),
                )
                if (
                    active_state.status == QuestionStatus.PARTIALLY_ANSWERED
                    and active_state.answer is not None
                ):
                    question = next(
                        item
                        for item in request.questionnaire.questions
                        if item.question_code == active_question.question_code
                    )
                    reask_reply = partial_multi_choice_prompt(
                        question,
                        active_state.answer.option_codes,
                    )
                response = TurnSuccessResponse.model_validate(
                    {
                        **_build_response_envelope(request, processed_at),
                        "status": "SUCCESS",
                        "changes": [],
                        "nextAction": next_action.model_dump(mode="json", by_alias=True),
                        "assistantReply": reask_reply,
                    }
                )
            validate_turn_response(request, response)
        except (InvalidTurnResponseError, ValidationError, SemanticMappingError) as error:
            _LOGGER.warning(
                "Технический повтор вопроса не прошёл контракт: correlation_id=%s code=%s",
                request.correlation_id,
                type(error).__name__,
            )
            return None

        _LOGGER.warning(
            "Повреждённый вывод LLM заменён безопасным ходом: correlation_id=%s",
            request.correlation_id,
        )
        return response

    @staticmethod
    def _build_clarification_reply_recovery_response(
        request: TurnRequest,
        decision: AgentDecision,
        response: TurnSuccessResponse,
        processed_at: datetime,
        error: InvalidTurnResponseError,
    ) -> TurnSuccessResponse | None:
        """
        Исправляет только небезопасный текст смыслового уточнения.
        """
        clarification = decision.clarification
        if clarification is None:
            return None
        if (
            response.next_action.action != NextActionType.CLARIFY
            or response.next_action.question_code != clarification.question_code
            or len(error.violations) != 1
            or error.violations[0].code
            != TurnResponseViolationCode.INVALID_ASSISTANT_REPLY
            or error.violations[0].path != "assistantReply"
        ):
            return None

        fallback_text = _fallback_clarification_text(
            request,
            clarification.question_code,
        )
        if fallback_text == clarification.clarification_text:
            return None
        corrected_decision = decision.model_copy(
            update={
                "clarification": clarification.model_copy(
                    update={"clarification_text": fallback_text}
                )
            }
        )
        try:
            corrected_response = QuestionnaireTurnController._build_success_response(
                request,
                corrected_decision,
                processed_at,
            )
            validate_turn_response(request, corrected_response)
        except (InvalidTurnResponseError, ValidationError):
            return None

        _LOGGER.warning(
            "Небезопасное уточнение LLM заменено каноническим: correlation_id=%s",
            request.correlation_id,
        )
        return corrected_response

    @staticmethod
    def _build_rejected_output_recovery_response(
        request: TurnRequest,
        processed_at: datetime,
    ) -> TurnSuccessResponse | None:
        """
        Заменяет повторно отклонённое решение безопасным программным ходом.
        """
        decision = _build_rejected_output_recovery_decision(request)
        if decision is None:
            return None
        try:
            response = QuestionnaireTurnController._build_success_response(
                request,
                decision,
                processed_at,
            )
            validate_turn_response(request, response)
        except (InvalidTurnResponseError, ValidationError):
            _LOGGER.exception(
                "Не удалось построить безопасный ход после отклонения LLM: "
                "correlation_id=%s",
                request.correlation_id,
            )
            return None

        _LOGGER.warning(
            "Отклонённое решение LLM заменено безопасным ходом: correlation_id=%s",
            request.correlation_id,
        )
        return response


def _proposed_additive_multi_choice_question(
    request: TurnRequest,
    response: TurnSuccessResponse,
) -> Question | None:
    """
    Находит дополнение E1, которое требуется проверить по текущей реплике.
    """
    correction = request.answer_correction
    if (
        correction is None
        or correction.mode != AnswerCorrectionMode.ADDITIVE
        or correction.question_code != "E1"
        or not any(
            isinstance(change, SaveAnswerChange)
            and change.question_code == correction.question_code
            for change in response.changes
        )
    ):
        return None
    return next(
        (
            question
            for question in request.questionnaire.questions
            if question.question_code == correction.question_code
            and question.question_type == QuestionType.MULTI_CHOICE
        ),
        None,
    )


def _pending_multi_choice_question(
    request: TurnRequest,
    response: TurnSuccessResponse,
) -> Question | None:
    """
    Находит активный выбор, требующий независимой проверки ответа.
    """
    pending = request.state.pending_clarification
    active_question = request.active_question
    if (
        request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or active_question is None
    ):
        return None
    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == active_question.question_code
        ),
        None,
    )
    if question is None or question.question_type != QuestionType.MULTI_CHOICE:
        return None
    if pending is None or pending.question_code != active_question.question_code:
        if question.question_code != "E1" or not any(
            (
                isinstance(change, SaveAnswerChange)
                and change.question_code == question.question_code
            )
            or (
                isinstance(change, SkipQuestionChange)
                and change.question_code == question.question_code
                and change.reason == PolicySkipReason.NO_MATCHING_OPTIONS
            )
            for change in response.changes
        ):
            return None
    if any(
        isinstance(change, SkipQuestionChange)
        and change.question_code == question.question_code
        and is_explicit_skip_evidence(change.evidence)
        for change in response.changes
    ):
        return None
    return question


def _active_partial_multi_choice_question(request: TurnRequest) -> Question | None:
    """
    Находит завершающий вопрос по уже частично выбранным занятиям.
    """
    active = request.active_question
    if request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION or active is None:
        return None
    question_state = next(
        (item for item in request.state.questions if item.question_code == active.question_code),
        None,
    )
    if question_state is None or question_state.status != QuestionStatus.PARTIALLY_ANSWERED:
        return None
    return next(
        (
            question
            for question in request.questionnaire.questions
            if question.question_code == active.question_code
            and question.question_code == "E1"
            and question.question_type == QuestionType.MULTI_CHOICE
        ),
        None,
    )


def _proposed_option_refs(
    question: Question,
    response: TurnSuccessResponse,
) -> tuple[str, ...]:
    """
    Возвращает позиции вариантов, которые основная модель хочет сохранить.
    """
    option_refs_by_code = {
        option.option_code: f"V{index}"
        for index, option in enumerate(question.options, start=1)
    }
    return tuple(
        option_refs_by_code[option_code]
        for change in response.changes
        if isinstance(change, SaveAnswerChange)
        and change.question_code == question.question_code
        for option_code in change.option_codes
    )


def _other_question_changes(
    decision: AgentDecision,
    question_code: str,
) -> tuple[AgentStateChange, ...]:
    """
    Сохраняет отдельно разрешённую коррекцию другого вопроса.
    """
    return tuple(
        change for change in decision.changes if change.question_code != question_code
    )


def _nonadditive_replacements(
    request: TurnRequest,
    decision: AgentDecision,
) -> tuple[AgentInterpretAnswerChange | AgentSaveAnswerChange, ...]:
    """
    Находит обычные замены прежних ответов для ограничения одного хода.
    """
    correction = request.answer_correction
    return tuple(
        change
        for change in decision.changes
        if isinstance(change, (AgentInterpretAnswerChange, AgentSaveAnswerChange))
        and change.replace_existing
        and not (
            correction is not None
            and correction.mode == AnswerCorrectionMode.ADDITIVE
            and correction.question_code == change.question_code
        )
    )


def _discard_conflicting_replacements(
    request: TurnRequest,
    decision: AgentDecision,
) -> AgentDecision:
    """
    После неудачного исправления убирает обычные замены, сохраняя дополнение E1.
    """
    conflicting_replacements = _nonadditive_replacements(request, decision)
    if len(conflicting_replacements) <= 1:
        return decision

    retained_changes = tuple(
        change
        for change in decision.changes
        if change not in conflicting_replacements
    )
    _LOGGER.warning(
        "Конфликтующие правки прежних ответов отброшены: "
        "correlation_id=%s count=%s",
        request.correlation_id,
        len(decision.changes) - len(retained_changes),
    )
    return decision.model_copy(
        update={
            "changes": retained_changes,
            "transition_text": None,
            "preferred_next_question_code": None,
        }
    )


def _project_semantic_changes(
    request: TurnRequest,
    changes: tuple[AgentStateChange, ...],
) -> tuple[AgentSaveAnswerChange | AgentSkipQuestionChange | AgentConfirmAnswerChange, ...]:
    """
    Переводит смысловые ответы модели в доверенные бизнес-коды вариантов.
    """
    questions_by_code = {
        question.question_code: question
        for question in request.questionnaire.questions
    }
    projected: list[
        AgentSaveAnswerChange | AgentSkipQuestionChange | AgentConfirmAnswerChange
    ] = []
    for change in changes:
        if not isinstance(change, AgentInterpretAnswerChange):
            projected.append(change)
            continue
        question = questions_by_code.get(change.question_code)
        if question is None:
            raise SemanticMappingError(
                SemanticMappingErrorCode.QUESTION_CODE_MISMATCH,
                "смысловая интерпретация ссылается на неизвестный вопрос",
            )
        projected.append(map_semantic_answer(question, change))
    return tuple(projected)


def _build_external_change(
    request: TurnRequest,
    change: AgentSaveAnswerChange | AgentSkipQuestionChange | AgentConfirmAnswerChange,
    *,
    change_count: int,
    include_prior_answer_context: bool = False,
    log_evidence_fallback: bool = True,
) -> dict[str, Any]:
    """
    Добавляет к смысловому изменению доверенную ссылку на текущую реплику.
    """
    payload = change.model_dump(mode="json", by_alias=True)
    evidence_candidate_id = payload.pop("evidenceCandidateId")
    if change.operation == "SAVE_ANSWER":
        source_message_ids = _answer_source_message_ids(
            request,
            change,
            include_prior_answer_context=include_prior_answer_context,
        )
        payload["sourceMessageIds"] = list(source_message_ids)
    else:
        payload["sourceMessageId"] = request.current_message.message_id
    current_evidence = select_verified_evidence(
        request.current_message.content,
        evidence_candidate_id,
        required_concepts=_required_evidence_concepts(request, change),
    )
    force_full_qualified_evidence = _requires_full_qualified_evidence(
        request,
        change,
        change_count=change_count,
    )
    if force_full_qualified_evidence:
        current_evidence = request.current_message.content
    payload["evidence"] = current_evidence
    if include_prior_answer_context and _prior_saved_answer_messages(
        request, change.question_code
    ):
        prior_state = next(
            state
            for state in request.state.questions
            if state.question_code == change.question_code
        )
        assert prior_state.answer is not None
        payload["evidence"] = f"{prior_state.answer.evidence}; {current_evidence}"
    pending = request.state.pending_clarification
    if (
        change.operation == "SAVE_ANSWER"
        and change.question_code == "E1"
        and pending is not None
        and pending.question_code == "E1"
    ):
        context_by_id = {
            message.message_id: message.content for message in request.dialog_history
        }
        payload["evidence"] = "; ".join(
            (
                *(context_by_id[message_id] for message_id in pending.context_message_ids),
                current_evidence,
            )
        )
    if (
        log_evidence_fallback
        and evidence_candidate_id != FULL_MESSAGE_EVIDENCE_ID
        and not force_full_qualified_evidence
        and current_evidence == request.current_message.content
    ):
        _LOGGER.warning(
            "Основание LLM заменено полной репликой: correlation_id=%s code=%s",
            request.correlation_id,
            "UNKNOWN_EVIDENCE_CANDIDATE",
        )
    return payload


def _requires_full_qualified_evidence(
    request: TurnRequest,
    change: AgentSaveAnswerChange | AgentSkipQuestionChange | AgentConfirmAnswerChange,
    *,
    change_count: int,
) -> bool:
    """
    Сохраняет смысловые оговорки в коротком ответе на активную шкалу.
    """
    if (
        change.operation != "SAVE_ANSWER"
        or request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or request.active_question is None
        or request.active_question.question_code != change.question_code
    ):
        return False

    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == change.question_code
        ),
        None,
    )
    if question is None:
        return False

    if (
        question.answer_meaning == "PERSISTENCE_ATTITUDE"
        and len(request.current_message.content)
        <= _MAX_SHORT_PERSISTENCE_EVIDENCE_CHARACTERS
        and any(
            option.option_code in change.option_codes and option.numeric_value == 1
            for option in question.options
        )
    ):
        return True

    return (
        change_count == 1
        and len(request.current_message.content)
        <= _MAX_SHORT_QUALIFIED_EVIDENCE_CHARACTERS
        and any(option.numeric_value is not None for option in question.options)
        and _QUALIFYING_CONNECTOR.search(request.current_message.content) is not None
    )


def _required_evidence_concepts(
    request: TurnRequest,
    change: AgentSaveAnswerChange | AgentSkipQuestionChange | AgentConfirmAnswerChange,
) -> tuple[str, ...]:
    """
    Возвращает понятия, которые обязано сохранять видимое основание ответа.
    """
    if change.operation != "SAVE_ANSWER":
        return ()

    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == change.question_code
        ),
        None,
    )
    if question is None:
        return ()

    if (
        question.answer_meaning == "PERFORMANCE_ATTITUDE"
        and len(change.option_codes) == 1
        and any(
            option.option_code == change.option_codes[0]
            and option.numeric_value == 1
            for option in question.options
        )
    ):
        return (
            _PERFORMANCE_POSITIVE_EVIDENCE_CONCEPT,
            _PERFORMANCE_INHIBITION_EVIDENCE_CONCEPT,
        )

    if question.question_type.value != "MULTI_CHOICE":
        return ()

    selected_codes = set(change.option_codes)
    return tuple(
        option.text
        for option in question.options
        if option.option_code in selected_codes
    )


def _resolve_clarification(decision: AgentDecision) -> AgentClarification | None:
    """
    Возвращает живое уточнение модели без шаблонной подмены.
    """
    return decision.clarification


def _build_clarification_limit_change(
    request: TurnRequest,
    decision: AgentDecision,
) -> ApplyDefaultAnswerChange | AgentSkipQuestionChange | None:
    """
    Завершает вопрос после двух уточнений согласно типу ответа.
    """
    if decision.explanation is not None:
        return None
    active_question = request.active_question
    pending = request.state.pending_clarification
    if active_question is None or pending is None:
        return None
    question_code = active_question.question_code
    if pending.question_code != question_code:
        return None
    if any(change.question_code == question_code for change in decision.changes):
        return None
    if any(
        isinstance(change, AgentInterpretAnswerChange)
        and change.replace_existing
        and change.question_code in request.replaceable_question_codes
        for change in decision.changes
    ):
        return None

    question_state = next(
        (
            item
            for item in request.state.questions
            if item.question_code == question_code
        ),
        None,
    )
    if (
        question_state is None
        or question_state.status != QuestionStatus.NEEDS_CLARIFICATION
        or question_state.clarification_attempts < _MAX_CLARIFICATION_ATTEMPTS
    ):
        return None
    question = next(
        item
        for item in request.questionnaire.questions
        if item.question_code == question_code
    )
    if question.question_type == QuestionType.SCALE_0_3:
        zero_code = next(
            option.option_code for option in question.options if option.numeric_value == 0
        )
        return ApplyDefaultAnswerChange(
            question_code=question_code,
            option_codes=(zero_code,),
            reason=AnswerDefaultReason.CLARIFICATION_LIMIT_REACHED,
        )
    return AgentSkipQuestionChange(
        operation="SKIP_QUESTION",
        question_code=question_code,
        reason=PolicySkipReason.NO_ANSWER_AFTER_RETRY,
        evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
    )


def _compose_explanatory_reask(request: TurnRequest, explanation_text: str) -> str:
    """
    Оставляет пояснение модели, но задаёт вопрос только об активном пункте.
    """
    active = request.active_question
    if active is None:
        return explanation_text
    question = next(
        item
        for item in request.questionnaire.questions
        if item.question_code == active.question_code
    )
    pending = request.state.pending_clarification
    trusted_question = (
        question.followup_clarification_text
        if pending is not None
        and pending.question_code == active.question_code
        and question.followup_clarification_text is not None
        else question.clarification_text or question.question_text
    )
    first_question_mark = next(
        (
            index
            for index, character in enumerate(explanation_text)
            if "QUESTION" in unicodedata.name(character, "")
        ),
        len(explanation_text),
    )
    introductory_text = explanation_text[:first_question_mark]
    last_sentence_end = max(
        (index for index, character in enumerate(introductory_text) if character in ".!"),
        default=-1,
    )
    if last_sentence_end < 0:
        return trusted_question
    introduction = introductory_text[: last_sentence_end + 1].strip()
    reply = f"{introduction} {trusted_question}"
    return reply if len(reply) <= 400 else trusted_question


def _compose_assistant_reply(
    canonical_reply: str,
    *,
    transition_text: str | None,
    next_action: NextAction,
) -> str:
    """
    Добавляет живую подводку только перед дословным очередным вопросом.
    """
    if (
        next_action.action != NextActionType.ASK_QUESTION
        or next_action.companion_question_code is not None
        or transition_text is None
        or _is_mechanical_transition(transition_text)
        or repeats_official_question(transition_text, canonical_reply)
    ):
        return canonical_reply
    transition_text = transition_text.rstrip(" ,;:")
    if not transition_text:
        return canonical_reply
    if transition_text[-1] not in ".!…":
        transition_text = f"{transition_text}."
    return f"{transition_text} {canonical_reply}"


def _contextual_question_transition(
    request: TurnRequest,
    next_action: NextAction,
) -> str | None:
    """
    Связывает следующий вопрос о способе рисования с подтверждённым интересом.
    """
    if next_action.action != NextActionType.ASK_QUESTION:
        return None
    if (
        request.active_question is not None
        and request.active_question.companion_question_code is not None
    ):
        return None
    next_question = next(
        (
            question
            for question in request.questionnaire.questions
            if question.question_code == next_action.question_code
        ),
        None,
    )
    if next_question is None or next_question.answer_meaning not in {
        "DIGITAL_CREATIVITY_INTEREST",
        "HANDCRAFT_ATTITUDE",
    }:
        return None
    drawing_interest = next(
        (
            state.answer
            for state in request.state.questions
            if state.question_code == "E1"
            and state.status == QuestionStatus.ANSWERED
            and state.answer is not None
            and "DRAW" in state.answer.option_codes
        ),
        None,
    )
    if drawing_interest is None:
        return None
    if next_question.answer_meaning == "DIGITAL_CREATIVITY_INTEREST":
        return "Вернёмся к рисованию."
    return "А теперь про краски и поделки."


def _is_mechanical_transition(value: str) -> bool:
    """
    Распознаёт пустое подтверждение ответа перед следующим вопросом.
    """
    normalized = normalize_dialog_text(value)
    first_token = normalized.split(maxsplit=1)[0]
    first_word = "".join(character for character in first_token if character.isalpha())
    return first_word in _MECHANICAL_TRANSITION_PREFIXES


def _answer_source_message_ids(
    request: TurnRequest,
    change: AgentSaveAnswerChange,
    *,
    include_prior_answer_context: bool = False,
) -> tuple[str, ...]:
    """
    Добавляет к ответу пользовательские реплики всей цепочки уточнения.
    """
    pending = request.state.pending_clarification
    context_ids = (
        pending.context_message_ids
        if pending is not None and pending.question_code == change.question_code
        else ()
    )
    prior_ids = (
        tuple(
            message.message_id
            for message in _prior_saved_answer_messages(request, change.question_code)
        )
        if include_prior_answer_context
        else ()
    )
    return tuple(
        dict.fromkeys((*prior_ids, *context_ids, request.current_message.message_id))
    )


def _prior_saved_answer_messages(
    request: TurnRequest,
    question_code: str,
) -> tuple[DialogMessage, ...]:
    """
    Берёт только детские сообщения, уже связанные с сохранённым ответом.
    """
    state = next(
        (item for item in request.state.questions if item.question_code == question_code),
        None,
    )
    if state is None or state.answer is None:
        return ()
    history_by_id = {message.message_id: message for message in request.dialog_history}
    messages = tuple(
        history_by_id.get(message_id)
        for message_id in state.answer.source_message_ids
    )
    if any(message is None or message.role != DialogRole.USER for message in messages):
        return ()
    return messages


def _build_grounded_explicit_skip(
    request: TurnRequest,
) -> AgentDecision | None:
    """
    Пропускает активный вопрос по точной команде интерфейса без обращения к LLM.
    """
    active_question = request.active_question
    if (
        request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or active_question is None
        or not _is_canonical_skip_command(request.current_message.content)
    ):
        return None

    return AgentDecision(
        changes=(
            AgentSkipQuestionChange(
                operation="SKIP_QUESTION",
                question_code=active_question.question_code,
                reason="Пользователь явно попросил пропустить вопрос.",
                evidence_candidate_id=FULL_MESSAGE_EVIDENCE_ID,
            ),
        ),
        clarification=None,
        transition_text=None,
    )


def _is_canonical_skip_command(value: str) -> bool:
    """
    Узнаёт только фиксированное значение кнопки пропуска, а не свободную речь.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold().strip(" .!?")
    return normalized == "пропустить"


def _build_rejected_output_recovery_decision(
    request: TurnRequest,
) -> AgentDecision | None:
    """
    Строит ограниченный ход, когда активный вопрос остался без ответа.

    До исчерпания двух уточнений возвращает только безопасное уточнение активного
    вопроса. После двух уточнений применяет правило для типа вопроса.
    """
    active_question = request.active_question
    if (
        request.answer_scope != TurnAnswerScope.ACTIVE_QUESTION
        or active_question is None
    ):
        return None

    question_code = active_question.question_code
    question_state = next(
        (
            item
            for item in request.state.questions
            if item.question_code == question_code
        ),
        None,
    )
    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == question_code
        ),
        None,
    )
    if question_state is None or question is None:
        return None
    if question_state.status not in {
        QuestionStatus.UNKNOWN,
        QuestionStatus.NEEDS_CLARIFICATION,
    }:
        return None

    if question_state.clarification_attempts >= _MAX_CLARIFICATION_ATTEMPTS:
        empty_decision = AgentDecision(
            changes=(),
            clarification=None,
            transition_text=None,
        )
        if _build_clarification_limit_change(request, empty_decision) is None:
            return None
        return empty_decision

    pending = request.state.pending_clarification
    if question_state.status == QuestionStatus.NEEDS_CLARIFICATION:
        if pending is None or pending.question_code != question_code:
            return None
    elif pending is not None:
        return None

    proposed_text = (
        _fallback_clarification_text(request, question_code)
        if pending is not None
        else question.clarification_text or question.question_text
    )
    return AgentDecision(
        changes=(),
        clarification=AgentClarification(
            question_code=question_code,
            reason="Ответ требует дополнительного уточнения.",
            clarification_text=_safe_clarification_text(
                request,
                question_code,
                proposed_text,
            ),
        ),
        transition_text=None,
    )


def _safe_clarification_text(
    request: TurnRequest,
    question_code: str,
    proposed_text: str,
) -> str:
    """
    Принимает безопасный по форме вопрос либо возвращает канонический fallback.
    """
    question = next(
        (
            item
            for item in request.questionnaire.questions
            if item.question_code == question_code
        ),
        None,
    )
    if question is None:
        return _UNKNOWN_QUESTION_FALLBACK

    previous_assistant_text = (
        request.dialog_history[-1].content
        if request.dialog_history and request.dialog_history[-1].role.value == "ASSISTANT"
        else None
    )
    if is_safe_contextual_clarification(
        question,
        proposed_text,
        previous_assistant_text=previous_assistant_text,
        context_texts=relevant_clarification_context_texts(request, question_code),
    ):
        return proposed_text

    _LOGGER.warning(
        "Уточнение LLM заменено каноническим: correlation_id=%s code=%s",
        request.correlation_id,
        "UNSAFE_OR_REPEATED_CLARIFICATION",
    )
    return _fallback_clarification_text(request, question_code)


def _fallback_clarification_text(request: TurnRequest, question_code: str) -> str:
    """
    Возвращает безопасное уточнение, избегая дословного повтора при возможности.
    """
    question = next(
        item
        for item in request.questionnaire.questions
        if item.question_code == question_code
    )
    configured = _short_single_question(question.clarification_text or question.question_text)
    option_clarification = build_option_clarification_text(question)
    original = _short_single_question(question.question_text)
    pending = request.state.pending_clarification
    if pending is not None and pending.question_code == question_code:
        followup = (
            (question.followup_clarification_text,)
            if question.followup_clarification_text is not None
            else ()
        )
        if question.answer_meaning == "FOCUS_DURATION":
            candidates = (configured, *followup, original)
        else:
            candidates = (*followup, option_clarification, configured, original)
    else:
        candidates = (configured, option_clarification, original)
    previous_text = request.dialog_history[-1].content if request.dialog_history else None
    for candidate in dict.fromkeys(candidates):
        if previous_text is not None and normalize_dialog_text(candidate) == normalize_dialog_text(
            previous_text
        ):
            continue
        if is_safe_contextual_clarification(
            question,
            candidate,
            previous_assistant_text=previous_text,
        ):
            return candidate
    return original


def _short_single_question(value: str) -> str:
    """
    Преобразует доверенный fallback в один вопрос допустимой длины.
    """
    without_question_marks = "".join(
        character
        for character in " ".join(value.split())
        if "QUESTION" not in unicodedata.name(character, "")
    ).rstrip(" .!;:")
    if len(without_question_marks) > 239:
        shortened = without_question_marks[:239].rsplit(" ", 1)[0].rstrip(" .!;:")
        without_question_marks = shortened or without_question_marks[:239].rstrip(" .!;:")
    return f"{without_question_marks}?"


def _build_response_envelope(
    request: TurnRequest,
    processed_at: datetime,
) -> dict[str, Any]:
    """
    Копирует служебные значения только из доверенного входного контракта.
    """
    return {
        "schemaVersion": request.schema_version,
        "correlationId": request.correlation_id,
        "sessionId": request.session_id,
        "sourceMessageId": request.current_message.message_id,
        "baseStateVersion": request.state.state_version,
        "questionnaireCode": request.questionnaire.questionnaire_code,
        "questionnaireVersion": request.questionnaire.questionnaire_version,
        "processedAt": processed_at,
    }


def _with_composite_question(
    request: TurnRequest,
    changes: tuple[Any, ...],
    next_action: NextAction,
) -> NextAction:
    """
    Выбирает общий вопрос только для двух ещё открытых способов рисования.
    """
    if next_action.action != NextActionType.ASK_QUESTION:
        return next_action

    e1_state = next(
        (state for state in request.state.questions if state.question_code == "E1"),
        None,
    )
    drawing_options = (
        set(e1_state.answer.option_codes)
        if e1_state is not None and e1_state.answer is not None
        else set()
    )
    for change in changes:
        if change.question_code == "E1" and change.operation == "SAVE_ANSWER":
            if change.replace_existing:
                drawing_options = set(change.option_codes)
            else:
                drawing_options.update(change.option_codes)
        elif isinstance(change, ReviseMultiChoiceChange):
            drawing_options = set(change.option_codes)
    if "DRAW" not in drawing_options:
        return next_action

    open_codes = {
        state.question_code
        for state in request.state.questions
        if state.status == QuestionStatus.UNKNOWN
    } - {change.question_code for change in changes}
    for composite in request.questionnaire.composite_questions:
        primary_code, companion_code = composite.question_codes
        if (
            composite.question_codes == ("X1", "X4")
            and next_action.question_code in composite.question_codes
            and primary_code in open_codes
            and companion_code in open_codes
        ):
            return NextAction(
                action=NextActionType.ASK_QUESTION,
                question_code=primary_code,
                companion_question_code=companion_code,
            )
    return next_action


def _derive_next_action(
    request: TurnRequest,
    changes: tuple[
        AgentSaveAnswerChange
        | AgentSkipQuestionChange
        | AgentConfirmAnswerChange
        | ApplyDefaultAnswerChange,
        ...,
    ],
    clarification: AgentClarification | None,
    explanation: AgentExplanation | None = None,
    *,
    preferred_next_question_code: str | None = None,
) -> NextAction:
    """
    Вычисляет следующий шаг по локальной проекции предложенных изменений.
    """
    projected_statuses = {
        question_state.question_code: question_state.status
        for question_state in request.state.questions
    }
    for change in changes:
        if change.question_code not in projected_statuses:
            continue
        if change.operation == "SAVE_ANSWER":
            e1_question = next(
                (
                    question
                    for question in request.questionnaire.questions
                    if question.question_code == change.question_code == "E1"
                ),
                None,
            )
            prior_e1_state = next(
                (
                    state
                    for state in request.state.questions
                    if state.question_code == "E1"
                ),
                None,
            )
            prior_codes = (
                ()
                if prior_e1_state is None or prior_e1_state.answer is None
                else prior_e1_state.answer.option_codes
            )
            selected_count = len(set((*prior_codes, *change.option_codes)))
            projected_statuses[change.question_code] = (
                QuestionStatus.PARTIALLY_ANSWERED
                if change.question_code == "E1"
                and (
                    request.active_question is None
                    or request.active_question.question_code != "E1"
                )
                and projected_statuses[change.question_code]
                != QuestionStatus.ANSWERED
                and e1_question is not None
                and selected_count < e1_question.max_selections
                else QuestionStatus.ANSWERED
            )
        elif isinstance(change, ReviseMultiChoiceChange):
            projected_statuses[change.question_code] = (
                QuestionStatus.UNKNOWN
                if not change.option_codes
                else projected_statuses[change.question_code]
            )
        elif change.operation in {"APPLY_DEFAULT_ANSWER", "CONFIRM_ANSWER"}:
            projected_statuses[change.question_code] = QuestionStatus.ANSWERED
        else:
            projected_statuses[change.question_code] = QuestionStatus.SKIPPED

    if clarification is not None:
        if clarification.question_code in projected_statuses:
            projected_statuses[clarification.question_code] = (
                QuestionStatus.PARTIALLY_ANSWERED
                if projected_statuses[clarification.question_code]
                == QuestionStatus.PARTIALLY_ANSWERED
                else QuestionStatus.NEEDS_CLARIFICATION
            )
        return NextAction(
            action=NextActionType.CLARIFY,
            question_code=clarification.question_code,
        )

    if explanation is not None:
        return NextAction(
            action=NextActionType.REASK_QUESTION,
            question_code=explanation.question_code,
        )

    active_question = request.active_question
    if (
        active_question is not None
        and active_question.companion_question_code is not None
        and projected_statuses[active_question.question_code] == QuestionStatus.UNKNOWN
        and projected_statuses[active_question.companion_question_code]
        == QuestionStatus.ANSWERED
        and any(
            change.question_code == active_question.companion_question_code
            and change.operation == "SAVE_ANSWER"
            for change in changes
        )
    ):
        return NextAction(
            action=NextActionType.ASK_QUESTION,
            question_code=active_question.question_code,
        )
    if (
        active_question is not None
        and projected_statuses[active_question.question_code]
        == next(
            state.status
            for state in request.state.questions
            if state.question_code == active_question.question_code
        )
        and changes
        and all(change.question_code != active_question.question_code for change in changes)
    ):
        return NextAction(
            action=NextActionType.REASK_QUESTION,
            question_code=active_question.question_code,
        )

    if (
        active_question is not None
        and active_question.companion_question_code is not None
        and projected_statuses[active_question.question_code]
        in {QuestionStatus.ANSWERED, QuestionStatus.SKIPPED}
        and projected_statuses[active_question.companion_question_code]
        == QuestionStatus.UNKNOWN
    ):
        return NextAction(
            action=NextActionType.ASK_QUESTION,
            question_code=active_question.companion_question_code,
        )

    unknown_code = _first_question_with_status(
        request,
        projected_statuses,
        QuestionStatus.UNKNOWN,
    )
    if unknown_code is not None:
        if (
            active_question is None
            or projected_statuses[active_question.question_code]
            not in {
                QuestionStatus.UNKNOWN,
                QuestionStatus.NEEDS_CLARIFICATION,
                QuestionStatus.PARTIALLY_ANSWERED,
            }
        ) and _is_eligible_preferred_question(
            projected_statuses, preferred_next_question_code
        ):
            unknown_code = preferred_next_question_code
        return NextAction(
            action=NextActionType.ASK_QUESTION,
            question_code=unknown_code,
        )

    unresolved_clarification_code = _first_question_with_status(
        request,
        projected_statuses,
        QuestionStatus.NEEDS_CLARIFICATION,
    )
    if unresolved_clarification_code is not None:
        return NextAction(
            action=NextActionType.ASK_QUESTION,
            question_code=unresolved_clarification_code,
        )

    partial_code = _first_question_with_status(
        request,
        projected_statuses,
        QuestionStatus.PARTIALLY_ANSWERED,
    )
    if partial_code is not None:
        return NextAction(
            action=NextActionType.ASK_QUESTION,
            question_code=partial_code,
        )

    has_skips = QuestionStatus.SKIPPED in projected_statuses.values()
    return NextAction(
        action=(
            NextActionType.COMPLETE_WITH_SKIPS if has_skips else NextActionType.COMPLETE
        ),
        question_code=None,
    )


def _is_eligible_preferred_question(
    projected_statuses: dict[str, QuestionStatus],
    question_code: str | None,
) -> bool:
    """
    Принимает рекомендацию модели только для ещё открытого вопроса.
    """
    if question_code is None or projected_statuses.get(question_code) != QuestionStatus.UNKNOWN:
        return False
    return question_code != "E1" or not any(
        code != "E1" and status == QuestionStatus.UNKNOWN
        for code, status in projected_statuses.items()
    )


def _open_next_question_candidates(
    request: TurnRequest,
    response: TurnSuccessResponse,
) -> tuple[Question, ...]:
    """
    Находит открытые вопросы после проверенного патча, оставляя E1 последним.
    """
    initial_statuses = {
        state.question_code: state.status for state in request.state.questions
    }
    closed_codes = {
        change.question_code
        for change in response.changes
        if isinstance(
            change,
            (SaveAnswerChange, SkipQuestionChange, ConfirmAnswerChange, ApplyDefaultAnswerChange),
        )
    }
    open_questions = tuple(
        question
        for question in request.questionnaire.questions
        if initial_statuses[question.question_code] == QuestionStatus.UNKNOWN
        and question.question_code not in closed_codes
    )
    if any(question.question_code != "E1" for question in open_questions):
        return tuple(
            question for question in open_questions if question.question_code != "E1"
        )
    return open_questions


def _first_question_with_status(
    request: TurnRequest,
    projected_statuses: dict[str, QuestionStatus],
    expected_status: QuestionStatus,
) -> str | None:
    """
    Находит первый вопрос с заданным статусом в порядке определения анкеты.
    """
    return next(
        (
            question.question_code
            for question in request.questionnaire.questions
            if projected_statuses.get(question.question_code) == expected_status
        ),
        None,
    )


def _projected_partial_e1_options(
    request: TurnRequest,
    changes: tuple[
        AgentSaveAnswerChange
        | AgentSkipQuestionChange
        | AgentConfirmAnswerChange
        | ApplyDefaultAnswerChange,
        ...,
    ],
) -> tuple[str, ...] | None:
    """
    Определяет уже выбранные занятия для вопроса о других интересах.
    """
    e1_state = next(
        (state for state in request.state.questions if state.question_code == "E1"),
        None,
    )
    selected = () if e1_state is None or e1_state.answer is None else e1_state.answer.option_codes
    partial = e1_state is not None and e1_state.status == QuestionStatus.PARTIALLY_ANSWERED
    for change in changes:
        if change.question_code != "E1":
            continue
        if isinstance(change, ReviseMultiChoiceChange):
            selected = change.option_codes
            partial = bool(selected) and partial
            continue
        if change.operation in {"SKIP_QUESTION", "CONFIRM_ANSWER"}:
            return None
        if change.operation == "SAVE_ANSWER":
            if (
                request.active_question is not None
                and request.active_question.question_code == "E1"
            ):
                return None
            selected = tuple(dict.fromkeys((*selected, *change.option_codes)))
            partial = True
    e1_question = next(
        (question for question in request.questionnaire.questions if question.question_code == "E1"),
        None,
    )
    if e1_question is not None and len(selected) >= e1_question.max_selections:
        return None
    return selected if partial else None


def _authorized_correction_targets(
    request: TurnRequest,
) -> tuple[frozenset[str], str | None]:
    """
    Вычисляет разрешённые цели обычной замены и дополнения из контракта хода.
    """
    prior_codes = set(request.replaceable_question_codes)
    correction = request.answer_correction
    favourite_code: str | None = None
    if correction is not None:
        if correction.mode == AnswerCorrectionMode.REPLACE:
            prior_codes.add(correction.question_code)
        elif correction.mode == AnswerCorrectionMode.ADDITIVE:
            favourite_code = correction.question_code
    return frozenset(prior_codes), favourite_code


def _parse_agent_decision(
    content: str,
    *,
    allowed_question_codes: frozenset[str],
    authorized_question_codes: frozenset[str],
    authorized_prior_correction_codes: frozenset[str],
    authorized_favourite_addition_code: str | None,
) -> AgentDecision:
    """
    Проверяет три области изменений и объединяет их во внутренний список.
    """
    raw_decision = json.loads(content, object_pairs_hook=_build_unique_json_object)
    expected_keys = {
        "changes",
        "priorCorrection",
        "favouriteAddition",
        "clarification",
        "transitionText",
        "explanation",
        "preferredNextQuestionCode",
        "questionDraft",
    }
    if not isinstance(raw_decision, dict) or not (
        set(raw_decision) == expected_keys
        or set(raw_decision) == expected_keys - {"questionDraft"}
    ):
        raise _InvalidDecisionShapeError
    change_slots = raw_decision["changes"]
    if not isinstance(change_slots, dict) or set(change_slots) != authorized_question_codes:
        raise _InvalidChangesByQuestionError
    changes: list[Any] = []
    for code, change in change_slots.items():
        if change is None:
            continue
        if not isinstance(change, dict) or change.get("questionCode") != code:
            raise _InvalidChangesByQuestionError
        changes.append(change)
    for field_name, authorized_codes in (
        ("priorCorrection", authorized_prior_correction_codes),
        (
            "favouriteAddition",
            frozenset(
                (authorized_favourite_addition_code,)
                if authorized_favourite_addition_code is not None
                else ()
            ),
        ),
    ):
        correction = raw_decision[field_name]
        if correction is None:
            continue
        if (
            not isinstance(correction, dict)
            or correction.get("operation") != "INTERPRET_ANSWER"
            or correction.get("replaceExisting") is not True
            or correction.get("questionCode") not in authorized_codes
        ):
            raise _InvalidDecisionShapeError
        changes.append(correction)
    raw_decision = {
        key: value
        for key, value in raw_decision.items()
        if key not in {"priorCorrection", "favouriteAddition"}
    }
    raw_decision["changes"] = changes
    content = json.dumps(raw_decision, ensure_ascii=False, separators=(",", ":"))
    repaired_decision = _discard_empty_scale_interpretation_for_clarification(raw_decision)
    if repaired_decision is not raw_decision:
        raw_decision = repaired_decision
        content = json.dumps(raw_decision, ensure_ascii=False, separators=(",", ":"))
    duplicate_codes = _duplicate_question_targets(raw_decision, allowed_question_codes)
    if duplicate_codes:
        raise _DuplicateQuestionTargetError(duplicate_codes)
    try:
        return AgentDecision.model_validate_json(content, strict=True)
    except ValidationError:
        if raw_decision.get("questionDraft") is None:
            raise
        # Черновик не меняет ответы. Его ошибка не должна запускать ещё один
        # дорогой вызов основной модели, если само решение пригодно.
        raw_decision["questionDraft"] = None
        decision_without_draft = AgentDecision.model_validate_json(
            json.dumps(raw_decision, ensure_ascii=False, separators=(",", ":")),
            strict=True,
        )
        _LOGGER.info("Неверный черновик вопроса отброшен без повтора решения")
        return decision_without_draft


def _discard_empty_scale_interpretation_for_clarification(
    raw_decision: Any,
) -> Any:
    """
    Удаляет пустую шкальную заглушку только рядом с уточнением того же вопроса.
    """
    if not isinstance(raw_decision, dict):
        return raw_decision
    clarification = raw_decision.get("clarification")
    changes = raw_decision.get("changes")
    if not isinstance(clarification, dict) or not isinstance(changes, list):
        return raw_decision
    question_code = clarification.get("questionCode")
    if not isinstance(question_code, str):
        return raw_decision
    matching_changes = [
        change
        for change in changes
        if isinstance(change, dict) and change.get("questionCode") == question_code
    ]
    if len(matching_changes) != 1:
        return raw_decision
    change = matching_changes[0]
    interpretation = change.get("interpretation")
    if (
        set(change)
        != {
            "operation",
            "questionCode",
            "interpretation",
            "evidenceCandidateId",
            "replaceExisting",
        }
        or change["operation"] != "INTERPRET_ANSWER"
        or not isinstance(interpretation, dict)
        or set(interpretation) != {"kind", "directLevel", "indirectSignals"}
        or interpretation["kind"] != "SCALE"
        or interpretation["directLevel"] is not None
        or interpretation["indirectSignals"] != []
    ):
        return raw_decision
    return {
        **raw_decision,
        "changes": [item for item in changes if item is not change],
    }


def _duplicate_question_targets(
    raw_decision: Any,
    allowed_question_codes: frozenset[str],
) -> tuple[str, ...]:
    """
    Возвращает повторённые доверенные коды из списка изменений модели.
    """
    if not isinstance(raw_decision, dict):
        return ()
    changes = raw_decision.get("changes")
    if not isinstance(changes, list):
        return ()

    seen: set[str] = set()
    duplicates: set[str] = set()
    for change in changes:
        if not isinstance(change, dict):
            continue
        question_code = change.get("questionCode")
        if not isinstance(question_code, str) or question_code not in allowed_question_codes:
            continue
        if question_code in seen:
            duplicates.add(question_code)
        seen.add(question_code)
    return tuple(sorted(duplicates))


def _build_unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """
    Собирает объект и отклоняет любое повторение имени свойства.
    """
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateJsonKeyError
        result[key] = value
    return result


def _build_repair_prompt(
    original_prompt: LLMCompletionRequest,
    error: InvalidTurnResponseError | None,
    format_failure_code: str | None,
    format_error: Exception | None = None,
    attempted_decision: AgentDecision | None = None,
    child_asked_question: bool = False,
) -> LLMCompletionRequest:
    """
    Добавляет один ограниченный запрос на исправление контекстных нарушений.
    """
    if isinstance(format_error, _DuplicateQuestionTargetError):
        violations = [
            {
                "code": "DUPLICATE_QUESTION_TARGET",
                "path": "$.changes",
                "message": (
                    "один questionCode может встречаться в changes не более одного раза; "
                    "оставь только ответы, подтверждённые currentMessage"
                ),
                "questionCodes": list(format_error.question_codes),
            }
        ]
    elif isinstance(format_error, _InvalidChangesByQuestionError):
        violations = [
            {
                "code": "INVALID_CHANGES_BY_QUESTION",
                "path": "$.changes",
                "message": (
                    "ключ changes должен быть разрешённым незаполненным вопросом "
                    "и совпадать с questionCode вложенного изменения"
                ),
            }
        ]
    elif isinstance(format_error, _InvalidDecisionShapeError):
        violations = [
            {
                "code": "INVALID_DECISION_SHAPE",
                "path": "$",
                "message": "области новых ответов и коррекций должны соответствовать JSON Schema",
            }
        ]
    elif format_failure_code == "MULTIPLE_REPLACEMENTS":
        violations = [
            {
                "code": "REPLACEMENT_NOT_AUTHORIZED",
                "path": "$.changes",
                "message": "за один ход допустима только одна обычная замена прежнего ответа",
            }
        ]
    else:
        violations = (
            [
                {
                    "code": format_failure_code or "SCHEMA_VALIDATION_FAILED",
                    "path": "$",
                    "message": "предыдущий ответ не соответствует обязательной JSON Schema",
                }
            ]
            if error is None
            else [
                {
                    "code": violation.code.value,
                    "path": violation.path,
                    "message": violation.message,
                }
                for violation in error.violations
            ]
        )
    keyed_changes = (
        original_prompt.response_schema.schema["properties"]["changes"].get("type")
        == "object"
    )
    if isinstance(format_error, _InvalidChangesByQuestionError) or (
        isinstance(format_error, _DuplicateQuestionTargetError) and keyed_changes
    ):
        repair_instruction = (
            "Сформируй весь JSON заново. changes должен быть объектом: каждый "
            "разрешённый незаполненный вопрос содержит null либо одно изменение "
            "с таким же questionCode. Обычную коррекцию сохрани в priorCorrection, "
            "дополнение E1 — в favouriteAddition. Не создавай неподтверждённых ответов."
        )
    elif isinstance(format_error, _InvalidDecisionShapeError):
        repair_instruction = (
            "Сформируй весь JSON заново по исходной JSON Schema. Верни changes "
            "для незаполненных вопросов, priorCorrection=null или одну явную "
            "обычную замену, favouriteAddition=null или одно разрешённое дополнение. "
            "Все три поля обязательны; не смешивай их значения."
        )
    elif isinstance(format_error, _DuplicateQuestionTargetError):
        repair_instruction = (
            "Сформируй весь JSON заново. В changes каждый questionCode укажи не более "
            "одного раза. maxItems — верхняя граница, а не требуемое число элементов. "
            "Если рассказ не отвечает на вопрос, не добавляй изменение для него. "
            "Несколько выбранных занятий объедини в одном OPTION_SELECTION."
        )
    elif format_failure_code == "MULTIPLE_REPLACEMENTS":
        repair_instruction = (
            "Сформируй весь JSON заново. Сохрани ответ на activeQuestion, только "
            "если currentMessage действительно отвечает на него; иначе оставь null. "
            "Сделай не более одной обычной замены прежнего ответа в priorCorrection, "
            "для которой currentMessage даёт отдельное ясное свидетельство. "
            "ADDITIVE-дополнение E1 можно сохранить отдельно в favouriteAddition, "
            "только если currentMessage называет новое любимое занятие."
        )
    elif error is not None and any(
        violation.code == TurnResponseViolationCode.INVALID_ASSISTANT_REPLY
        for violation in error.violations
    ):
        if attempted_decision is not None and (
            attempted_decision.explanation is not None or child_asked_question
        ):
            repair_instruction = (
                "Сформируй весь JSON заново. Ребёнок задал встречный вопрос. "
                "Если его личный ответ уже достаточен для оценки, сохрани ответ, "
                "кратко поясни его через transitionText и оставь explanation=null. "
                "Иначе верни explanation для activeQuestion и clarification=null: "
                "сначала ответь на вопрос ребёнка понятным утверждением, затем "
                "задай один связанный вопрос об измеряемом признаке и закончи "
                "знаком «?». Если перечисляешь варианты, в том же вопросе явно "
                "разреши ответить, что ничего не подходит. Не выставляй балл по одному условному примеру или "
                "действию другого человека. Не обрывай фразу или список вариантов."
            )
        else:
            repair_instruction = (
                "Сформируй весь JSON заново. Перепиши clarificationText одной строкой: "
                "сначала короткое утверждение, если нужно пояснение, затем ровно один "
                "вопрос с конечным знаком «?». Не повторяй вопрос ребёнка как вопрос "
                "и не переноси частные слова из currentMessage, если достаточно "
                "нейтральных слов questionText и options. Сохрани цель предыдущего "
                "уточнения: если ребёнок просит пример занятия, поясни занятие и "
                "спроси о нём и длительности; не заменяй это полной шкалой вариантов. "
                "Если выясняешь собственный опыт и качество лепки или вырезания, "
                "объедини их в одном вопросительном предложении."
            )
    else:
        repair_instruction = (
            "Исправь предыдущее решение и верни весь JSON заново. Соблюдай исходную "
            "JSON Schema и не добавляй отсутствующие в ней поля. Возвращай только "
            "смысловые изменения и одно разрешённое уточнение либо null."
        )
    feedback = {
        "repairInstruction": repair_instruction,
        "violations": violations,
    }
    return LLMCompletionRequest(
        messages=(
            *original_prompt.messages,
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(
                    feedback,
                    allow_nan=False,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ),
            ),
        ),
        response_schema=original_prompt.response_schema,
    )


def _log_rejected_decision(
    request: TurnRequest,
    error: InvalidTurnResponseError,
    *,
    repaired: bool,
) -> None:
    """
    Пишет безопасные коды нарушений без ответов пользователя и сырого JSON.
    """
    violations = ",".join(
        f"{violation.code.value}:{violation.path}" for violation in error.violations
    )
    _LOGGER.warning(
        "Решение LLM отклонено контекстной проверкой: correlation_id=%s repaired=%s violations=%s",
        request.correlation_id,
        repaired,
        violations,
    )


def _log_invalid_decision_format(
    request: TurnRequest,
    failure_code: str,
    *,
    repaired: bool,
    error: Exception | None = None,
) -> None:
    """
    Пишет код и безопасный путь ошибки без ответа и данных пользователя.
    """
    _LOGGER.warning(
        "Решение LLM не соответствует формату: "
        "correlation_id=%s repaired=%s code=%s validation=%s",
        request.correlation_id,
        repaired,
        failure_code,
        _safe_validation_error_details(error),
    )


def _safe_validation_error_details(error: Exception | None) -> str:
    """
    Оставляет от Pydantic-ошибки только тип и разрешённые части пути.
    """
    if not isinstance(error, ValidationError):
        return "-"
    details: list[tuple[int, str]] = []
    for item in error.errors(include_input=False, include_context=False):
        error_type = item.get("type")
        if not isinstance(error_type, str) or re.fullmatch(r"[a-z_]+", error_type) is None:
            error_type = "unknown"
        path_parts: list[str] = []
        known_parts = 0
        for part in item.get("loc", ()):
            if isinstance(part, str) and part in _SAFE_VALIDATION_PATH_PARTS:
                path_parts.append(part)
                known_parts += 1
            elif isinstance(part, int) and not isinstance(part, bool) and 0 <= part <= 100:
                path_parts.append(f"[{part}]")
            else:
                path_parts.append("*")
        details.append((known_parts, f"{error_type}:{'.'.join(path_parts) or '$'}"))
    details.sort(key=lambda item: -item[0])
    return ",".join(detail for _, detail in details[:3]) or "unknown:$"


def _decision_format_failure_code(
    error: (
        json.JSONDecodeError
        | ValidationError
        | SemanticMappingError
        | _DuplicateJsonKeyError
        | _DuplicateQuestionTargetError
        | _InvalidChangesByQuestionError
        | _InvalidDecisionShapeError
    ),
) -> str:
    """
    Преобразует внутренний тип ошибки в безопасный диагностический код.
    """
    if isinstance(error, json.JSONDecodeError):
        return "INVALID_JSON"
    if isinstance(error, _DuplicateJsonKeyError):
        return "DUPLICATE_JSON_KEY"
    if isinstance(error, _DuplicateQuestionTargetError):
        return "DUPLICATE_QUESTION_TARGET"
    if isinstance(error, _InvalidChangesByQuestionError):
        return "INVALID_CHANGES_BY_QUESTION"
    if isinstance(error, _InvalidDecisionShapeError):
        return "INVALID_DECISION_SHAPE"
    if isinstance(error, SemanticMappingError):
        return f"SEMANTIC_MAPPING_{error.code.value}"
    return "SCHEMA_VALIDATION_FAILED"
