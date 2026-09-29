"""
Проверяет смысл выбора из списка отдельно от основного решения модели.
"""

import json
import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from questionnaire_agent.llm import (
    LLMCompletionRequest,
    LLMMessage,
    LLMMessageRole,
    LLMResponseSchema,
)
from questionnaire_agent.prompts.assistant_reply import compose_bridged_reply
from questionnaire_contracts import (
    AnswerCorrectionMode,
    DialogRole,
    Question,
    QuestionStatus,
    TurnRequest,
)


class ChoiceVerificationOutcome(StrEnum):
    """
    Различает подтверждённый выбор, отказ от всего списка и неясный ответ.
    """

    SELECTED = "SELECTED"
    NONE = "NONE"
    UNCLEAR = "UNCLEAR"


@dataclass(frozen=True, slots=True)
class ChoiceVerification:
    """
    Хранит проверенный смысл текущей реплики и подтверждённые ссылки.
    """

    outcome: ChoiceVerificationOutcome
    supported_option_refs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ChoiceReconciliation:
    """
    Хранит смысловую отмену прежних и подтверждение новых любимых занятий.
    """

    retracted_option_refs: tuple[str, ...]
    new_option_refs: tuple[str, ...]
    supporting_prior_message_ids: tuple[str, ...]


def build_choice_reconciliation_prompt(
    request: TurnRequest,
    question: Question,
) -> LLMCompletionRequest:
    """
    Просит модель пересмотреть E1 после явного исправления другого ответа.
    """
    correction = request.answer_correction
    state = next(
        (item for item in request.state.questions if item.question_code == question.question_code),
        None,
    )
    if (
        question.question_code != "E1"
        or correction is None
        or correction.question_code != "E1"
        or correction.mode != AnswerCorrectionMode.ADDITIVE
        or state is None
        or state.answer is None
        or state.status not in {QuestionStatus.ANSWERED, QuestionStatus.PARTIALLY_ANSWERED}
    ):
        raise ValueError("пересмотр требует сохранённого и авторизованного E1")
    history_by_id = {message.message_id: message for message in request.dialog_history}
    if any(
        message_id not in history_by_id or history_by_id[message_id].role != DialogRole.USER
        for message_id in state.answer.source_message_ids
    ):
        raise ValueError("источники сохранённого E1 отсутствуют в истории")

    option_refs = tuple(f"V{index}" for index in range(1, len(question.options) + 1))
    selected_codes = set(state.answer.option_codes)
    selected_refs = tuple(
        ref
        for ref, option in zip(option_refs, question.options, strict=True)
        if option.option_code in selected_codes
    )
    remaining_refs = tuple(ref for ref in option_refs if ref not in selected_refs)
    prior_user_ids = tuple(
        message.message_id for message in request.dialog_history if message.role == DialogRole.USER
    )
    payload = {
        "previousQuestion": request.dialog_history[-1].content,
        "currentChildMessage": request.current_message.content,
        "options": [
            {"ref": ref, "text": option.text}
            for ref, option in zip(option_refs, question.options, strict=True)
        ],
        "alreadySelectedRefs": list(selected_refs),
        "savedAnswerSourceIds": list(state.answer.source_message_ids),
        "priorChildStatements": [
            {"messageId": message.message_id, "content": message.content}
            for message in request.dialog_history
            if message.role == DialogRole.USER
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "retractedOptionRefs": {
                "type": "array",
                "items": {"type": "string", "enum": list(selected_refs)},
            },
            "newOptionRefs": {
                "type": "array",
                "items": {"type": "string", "enum": list(remaining_refs)},
            },
            "supportingPriorMessageIds": {
                "type": "array",
                "items": {"type": "string", "enum": list(prior_user_ids)},
            },
        },
        "required": [
            "retractedOptionRefs",
            "newOptionRefs",
            "supportingPriorMessageIds",
        ],
        "additionalProperties": False,
    }
    system_text = (
        "Ребёнок исправил ранее сохранённый ответ на другой вопрос. Проверь, "
        "меняет ли его текущая реплика уже отмеченные любимые занятия E1, "
        "и верни только JSON по схеме. options задаёт занятия, "
        "alreadySelectedRefs — прежний выбор, а savedAnswerSourceIds показывает "
        "сообщения, на которых он был основан. Читай priorChildStatements по порядку "
        "и учитывай самостоятельные явные слова ребёнка об интересе к занятию, "
        "даже если они не были записаны как источник E1. "
        "В retractedOptionRefs включай прежний вариант, только если текущая "
        "реплика ясно отменяет единственное основание считать это занятие любимым "
        "или прямо отказывается от интереса к нему. Если остаётся независимое "
        "явное свидетельство личного интереса, сохрани вариант. Исправление "
        "частоты, умения или обстоятельств само по себе не отменяет предпочтение. "
        "Если смысл неясен, сохрани прежний вариант. Не выводи отмену варианта "
        "из числовой оценки другого вопроса. В newOptionRefs включай только "
        "ранее не выбранные занятия, к которым ребёнок в текущей реплике "
        "выразил ясный личный интерес. Не выбирай занятия по простому упоминанию, "
        "просмотру чужого результата или словам ассистента. В "
        "supportingPriorMessageIds перечисли только прежние сообщения ребёнка, "
        "которые действительно подтверждают оставшиеся выбранные занятия. "
        "Не включай источник только потому, что он раньше был связан с E1: "
        "если он говорит лишь об отменённом занятии, он больше не основание. "
        "Если прежних выбранных занятий не осталось, верни пустой список "
        "источников; новые занятия подтверждает currentChildMessage. "
        "При сохранении прежнего занятия укажи хотя бы один его источник."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_choice_reconciliation_v1",
            schema=schema,
        ),
    )


def parse_choice_reconciliation(
    content: str,
    question: Question,
    selected_option_codes: tuple[str, ...],
    prior_message_ids: tuple[str, ...],
) -> ChoiceReconciliation:
    """
    Принимает только уникальные ссылки на отменённые и новые варианты.
    """
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {
        "retractedOptionRefs",
        "newOptionRefs",
        "supportingPriorMessageIds",
    }:
        raise ValueError("пересмотр выбора вернул неподдерживаемую структуру")
    selected_codes = set(selected_option_codes)
    all_refs = tuple(f"V{index}" for index in range(1, len(question.options) + 1))
    selected_refs = {
        ref
        for ref, option in zip(all_refs, question.options, strict=True)
        if option.option_code in selected_codes
    }
    result: list[tuple[str, ...]] = []
    for key, allowed in (
        ("retractedOptionRefs", selected_refs),
        ("newOptionRefs", set(all_refs) - selected_refs),
    ):
        refs = raw[key]
        if (
            not isinstance(refs, list)
            or any(not isinstance(ref, str) or ref not in allowed for ref in refs)
            or len(refs) != len(set(refs))
        ):
            raise ValueError("пересмотр выбора вернул недопустимые ссылки")
        result.append(tuple(refs))
    sources = raw["supportingPriorMessageIds"]
    if (
        not isinstance(sources, list)
        or any(not isinstance(source, str) or source not in prior_message_ids for source in sources)
        or len(sources) != len(set(sources))
    ):
        raise ValueError("пересмотр выбора вернул недопустимые источники")
    retained_refs = selected_refs - set(result[0])
    if (retained_refs and not sources) or (not retained_refs and sources):
        raise ValueError("источники не соответствуют сохранённым прежним вариантам")
    return ChoiceReconciliation(
        retracted_option_refs=result[0],
        new_option_refs=result[1],
        supporting_prior_message_ids=tuple(sources),
    )


def build_choice_verification_prompt(
    request: TurnRequest,
    question: Question,
) -> LLMCompletionRequest:
    """
    Создаёт независимую проверку ответа на активный выбор.
    """
    pending = request.state.pending_clarification
    active = request.active_question
    if active is None or active.question_code != question.question_code:
        raise ValueError("проверка выбора требует активного вопроса")
    previous_question = next(
        (
            message.content
            for message in request.dialog_history
            if message.message_id == active.assistant_message_id
        ),
        None,
    )
    if previous_question is None:
        raise ValueError("в истории отсутствует последний вопрос о вариантах")

    context_ids = (
        set(pending.context_message_ids)
        if pending is not None and pending.question_code == question.question_code
        else set()
    )
    child_context = [
        message.content for message in request.dialog_history if message.message_id in context_ids
    ]

    option_refs = tuple(f"V{index}" for index in range(1, len(question.options) + 1))
    payload: dict[str, Any] = {
        "lastAssistantQuestion": previous_question,
        "priorChildContext": child_context,
        "currentChildMessage": request.current_message.content,
        "options": [
            {"ref": option_ref, "text": option.text}
            for option_ref, option in zip(option_refs, question.options, strict=True)
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": [item.value for item in ChoiceVerificationOutcome],
            },
            "supportedOptionRefs": {
                "type": "array",
                "items": {"type": "string", "enum": list(option_refs)},
            },
        },
        "required": ["outcome", "supportedOptionRefs"],
        "additionalProperties": False,
    }
    system_text = (
        "Проверь смысл связанной цепочки ответа ребёнка: priorChildContext и "
        "currentChildMessage после lastAssistantQuestion. Верни ровно JSON по схеме. "
        "lastAssistantQuestion нужен для понимания слов «это», «из них» и подобных "
        "ссылок, но названия вариантов в реплике ассистента сами по себе не "
        "являются свидетельством выбора ребёнка. Учитывай прежний контекст "
        "только если ребёнок уже ясно назвал любимое занятие из списка и текущая "
        "реплика не отменяет этот смысл либо конкретизирует его. Не выбирай "
        "варианты лишь за упоминание темы или по случайным старым репликам. "
        "SELECTED означает, что цепочка ясно подтверждает один или несколько "
        "вариантов; перечисли ровно все подтверждённые ссылки, включая уже "
        "ясный вариант из priorChildContext. NONE означает, что текущая реплика "
        "отвергла все варианты анкеты после полного списка; верни пустой "
        "supportedOptionRefs. Если в последнем вопросе названа только часть "
        "вариантов, отказ от «них» не доказывает отказа от остальных: верни "
        "UNCLEAR. UNCLEAR также означает, "
        "что выбор и отказ от всего списка не подтверждены; верни пустой список. "
        "Отрицание всего списка не является выбором всех перечисленных вариантов. "
        "Пользование устройством или просмотр контента сами по себе не означают "
        "цифровое творчество."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_choice_verification_v3",
            schema=schema,
        ),
    )


def build_partial_choice_verification_prompt(
    request: TurnRequest,
    question: Question,
) -> LLMCompletionRequest:
    """
    Проверяет, назвал ли ребёнок новые любимые занятия при частичном ответе.
    """
    active = request.active_question
    state = next(
        (item for item in request.state.questions if item.question_code == question.question_code),
        None,
    )
    if (
        active is None
        or active.question_code != question.question_code
        or state is None
        or state.answer is None
        or state.status != QuestionStatus.PARTIALLY_ANSWERED
    ):
        raise ValueError("проверка дополнения требует активного частичного ответа")

    already_selected_codes = set(state.answer.option_codes)
    remaining_refs = tuple(
        f"V{index}"
        for index, option in enumerate(question.options, start=1)
        if option.option_code not in already_selected_codes
    )
    last_question = next(
        (
            message.content
            for message in request.dialog_history
            if message.message_id == active.assistant_message_id
        ),
        None,
    )
    if last_question is None:
        raise ValueError("проверка дополнения требует последнего вопроса ассистента")
    payload: dict[str, Any] = {
        "lastQuestion": last_question,
        "currentChildMessage": request.current_message.content,
        "alreadySelected": [
            option.text
            for option in question.options
            if option.option_code in already_selected_codes
        ],
        "remainingOptions": [
            {"ref": f"V{index}", "text": option.text}
            for index, option in enumerate(question.options, start=1)
            if option.option_code not in already_selected_codes
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": [item.value for item in ChoiceVerificationOutcome],
            },
            "supportedOptionRefs": {
                "type": "array",
                "items": {"type": "string", "enum": list(remaining_refs)},
            },
        },
        "required": ["outcome", "supportedOptionRefs"],
        "additionalProperties": False,
    }
    system_text = (
        "Пойми смысл только текущего ответа ребёнка на вопрос, есть ли у него ещё "
        "любимые занятия. Верни только JSON по схеме. Уже выбранные занятия сохраняются "
        "в любом случае; не повторяй их в supportedOptionRefs. Названия вариантов "
        "в вопросе ассистента не доказывают, что ребёнок их выбрал. SELECTED — ребёнок "
        "явно назвал одно или несколько новых любимых занятий из remainingOptions; "
        "перечисли ровно их ссылки. NONE — ребёнок ясно говорит, что других любимых "
        "занятий из списка нет, либо просит закончить этот вопрос; пустой список. "
        "UNCLEAR — ответ не даёт понять, есть ли новые занятия; пустой список. "
        "Если ассистент назвал только примеры, а ребёнок отвергает лишь эти "
        "примеры, это ещё не отказ от всех остальных занятий: выбери UNCLEAR. "
        "Если ребёнок лишь повторил уже выбранное занятие, это не новое занятие. "
        "Игры и просмотр на устройстве не означают цифровое творчество."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_partial_choice_verification_v1",
            schema=schema,
        ),
    )


def build_partial_choice_clarification_prompt(
    request: TurnRequest,
    question: Question,
) -> LLMCompletionRequest:
    """
    Просит модель живо уточнить непонятный ответ о других занятиях.
    """
    state = next(
        item for item in request.state.questions if item.question_code == question.question_code
    )
    assert state.answer is not None
    selected_codes = set(state.answer.option_codes)
    payload = {
        "lastAssistantQuestion": request.dialog_history[-1].content,
        "currentChildMessage": request.current_message.content,
        "recentAssistantReplies": [
            message.content
            for message in request.dialog_history
            if message.role == DialogRole.ASSISTANT
        ][-4:],
        "alreadySelected": [
            option.text for option in question.options if option.option_code in selected_codes
        ],
        "remainingOptions": [
            option.text for option in question.options if option.option_code not in selected_codes
        ],
    }
    schema = {
        "type": "object",
        "properties": {
            "bridgeText": {"type": "string", "minLength": 1, "maxLength": 120},
            "questionText": {"type": "string", "minLength": 1, "maxLength": 140},
        },
        "required": ["bridgeText", "questionText"],
        "additionalProperties": False,
    }
    system_text = (
        "bridgeText — только ОДНО утверждение без вопросительного знака; "
        "спроси о других любимых занятиях только в questionText. Если "
        "последний ответ о другом деле, введи тему любимых занятий и не "
        "хвали прежнее дело ради перехода. "
        "Напиши короткую доброжелательную реплику по-русски для ребёнка. "
        "Уже выбранные любимые занятия сохранены; сейчас нужно понять, есть ли "
        "ещё что-нибудь любимое среди оставшихся вариантов. Перефразируй "
        "последний вопрос естественно и кратко. Верни два поля: bridgeText — "
        "одно короткое связующее утверждение до 120 символов с точкой или "
        "восклицательным знаком в конце, questionText — ровно один вопрос "
        "до 140 символов со знаком «?» в конце. Вместе не более 240 символов. "
        "Если ребёнок сообщил значимую новую деталь, откликнись на её смысл "
        "в bridgeText без дословного повтора; иначе сделай нейтральный "
        "переход к теме любимых занятий. Не повторяй подводки и зачины из "
        "recentAssistantReplies, не пиши дежурное «Понятно» и не связывай "
        "разные занятия ложным «раз ..., то ...». Не приписывай ребёнку "
        "неназванных предпочтений и чувств. В questionText отсылай "
        "к уже названным вариантам, не сужая выбор до одного-двух примеров; "
        "дай возможность ответить «больше ничего». Не предлагай новых действий, "
        "предметов или способов показа вне remainingOptions. Не повторяй "
        "lastAssistantQuestion дословно. Если род ребёнка не следует из его "
        "собственных слов, используй нейтральную форму. "
        "Верни только JSON по схеме."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_partial_choice_clarification_v3",
            schema=schema,
        ),
    )


def build_partial_choice_question_prompt(
    request: TurnRequest,
    question: Question,
    selected_option_codes: tuple[str, ...],
) -> LLMCompletionRequest:
    """
    Просит модель живо спросить о других занятиях после раннего выбора.
    """
    selected_codes = set(selected_option_codes)
    payload = {
        "alreadySelected": [
            option.text for option in question.options if option.option_code in selected_codes
        ],
        "remainingOptions": [
            option.text for option in question.options if option.option_code not in selected_codes
        ],
        "childStatements": [
            message.content for message in request.dialog_history if message.role == DialogRole.USER
        ]
        + [request.current_message.content],
        "recentAssistantReplies": [
            message.content
            for message in request.dialog_history
            if message.role == DialogRole.ASSISTANT
        ][-4:],
    }
    schema = {
        "type": "object",
        "properties": {
            "bridgeText": {"type": "string", "minLength": 1, "maxLength": 120},
            "questionText": {"type": "string", "minLength": 1, "maxLength": 200},
        },
        "required": ["bridgeText", "questionText"],
        "additionalProperties": False,
    }
    system_text = (
        "bridgeText — только ОДНО утверждение без вопросительного знака; "
        "спроси об остальных любимых занятиях только в questionText. Если "
        "последний ответ о другом деле, введи тему любимых занятий и не "
        "хвали прежнее дело ради перехода. "
        "Напиши по-русски одну короткую, живую и доброжелательную реплику "
        "для ребёнка восьми лет: переведи беседу к вопросу, нравятся ли ему "
        "ещё какие-то занятия из remainingOptions. Уже выбранные занятия "
        "сохранены в alreadySelected; называй прежним интересом только их. "
        "Ребёнок может добавить несколько или сказать, что больше "
        "ничего нет. Верни два отдельных поля: bridgeText — одно короткое "
        "связующее утверждение до 120 символов с точкой или восклицательным "
        "знаком в конце, questionText — ровно один вопрос с «?» в конце. "
        "Вся реплика должна быть не длиннее 240 символов. Прочитай "
        "childStatements по порядку, а по recentAssistantReplies пойми ритм "
        "беседы. В bridgeText откликнись на значимую новую деталь, если она "
        "уместна, не пересказывая выбранное занятие и слова ребёнка. "
        "Если уместной детали нет, "
        "сделай нейтральный переход к любимым занятиям. Не повторяй "
        "недавнюю подводку, дежурное «Понятно» и одну и ту же похвалу. "
        "Если последняя реплика о другом, не притягивай к ней вопрос и не "
        "отвечай на каждую её деталь. Не выдавай прежний интерес за причину "
        "интереса к другим занятиям через «раз» или «если ..., то». "
        "Не приписывай ребёнку неназванных чувств и умений. Сверь зачины "
        "последних вопросов: не продолжай цепочку однотипных «А ...», "
        "но не избегай союза «А», когда он звучит естественно. "
        "Если ребёнок прямо "
        "сказал, что занятие ему не "
        "нравится, не предлагай его снова как пример приятного занятия, даже "
        "если оно есть в remainingOptions; при явной перемене мнения учитывай "
        "более поздний ответ. Не превращай простое упоминание занятия в интерес "
        "к нему. Кратко назови все remainingOptions, кроме уже явно "
        "отвергнутых ребёнком, по смыслу, "
        "разговорными словами и без повторения alreadySelected. Не сужай "
        "выбор до нескольких примеров и не спрашивай, какой из них интереснее: "
        "ребёнок может назвать несколько или ни одного. Явно отвергнутое "
        "занятие не представляй как любимое. "
        "Поставь ровно один знак вопроса в конце questionText; не добавляй "
        "второй вопрос с примерами. questionText должен быть не длиннее 200 "
        "символов. Не употребляй "
        "коды, баллы, слово «анкета» и канцелярское «Я запомнил:». Не "
        "выдумывай интересы и не изменяй уже выбранные варианты. Род определяй "
        "только по собственным словам ребёнка, не по репликам ассистента; "
        "если он неизвестен, используй нейтральную форму. Верни только JSON по схеме."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_partial_choice_question_v3",
            schema=schema,
        ),
    )


def parse_partial_choice_question(content: str) -> str:
    """
    Соединяет короткую подводку и вопрос без усечения или изменения смысла.
    """
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {"bridgeText", "questionText"}:
        raise ValueError("вопрос о других занятиях вернул неподдерживаемую структуру")
    question = raw["questionText"]
    if not isinstance(question, str) or len(question) > 200:
        raise ValueError("вопрос о других занятиях содержит недопустимый текст")
    return _compose_single_e1_bridge(raw["bridgeText"], question)


def parse_partial_choice_clarification(content: str) -> str:
    """
    Соединяет подводку и короткий вопрос для безопасной проверки.
    """
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {"bridgeText", "questionText"}:
        raise ValueError("уточнение выбора вернуло неподдерживаемую структуру")
    question = raw["questionText"]
    if not isinstance(question, str) or len(question) > 140:
        raise ValueError("уточнение выбора содержит недопустимый текст")
    return _compose_single_e1_bridge(raw["bridgeText"], question)


def _compose_single_e1_bridge(bridge: Any, question: str) -> str:
    """
    При двух вступлениях сохраняет только ближайший к вопросу переход.
    """
    try:
        return compose_bridged_reply(bridge, question)
    except ValueError:
        if not isinstance(bridge, str) or any(mark in bridge for mark in "?？⁇⁈⁉"):
            raise
        sentences = re.split(r"(?<=[.!])\s+", bridge.strip())
        if len(sentences) < 2 or any(not part.endswith((".", "!")) for part in sentences):
            raise
        return compose_bridged_reply(sentences[-1], question)


def build_additive_choice_verification_prompt(
    request: TurnRequest,
    question: Question,
) -> LLMCompletionRequest:
    """
    Проверяет только новые любимые занятия в текущей реплике при ADDITIVE E1.
    """
    correction = request.answer_correction
    if (
        correction is None
        or correction.mode != AnswerCorrectionMode.ADDITIVE
        or correction.question_code != "E1"
        or question.question_code != correction.question_code
    ):
        raise ValueError("проверка добавления требует ADDITIVE-коррекции E1")

    question_state = next(
        (item for item in request.state.questions if item.question_code == "E1"),
        None,
    )
    if question_state is None or question_state.answer is None:
        raise ValueError("проверка добавления требует сохранённого ответа E1")

    option_refs = tuple(f"V{index}" for index in range(1, len(question.options) + 1))
    already_selected_codes = set(question_state.answer.option_codes)
    already_selected_refs = tuple(
        option_ref
        for option_ref, option in zip(option_refs, question.options, strict=True)
        if option.option_code in already_selected_codes
    )
    new_option_refs = tuple(
        option_ref for option_ref in option_refs if option_ref not in already_selected_refs
    )
    payload: dict[str, Any] = {
        "currentChildMessage": request.current_message.content,
        "options": [
            {"ref": option_ref, "text": option.text}
            for option_ref, option in zip(option_refs, question.options, strict=True)
        ],
        "alreadySelectedRefs": list(already_selected_refs),
    }
    schema = {
        "type": "object",
        "properties": {
            "outcome": {
                "type": "string",
                "enum": [item.value for item in ChoiceVerificationOutcome],
            },
            "supportedOptionRefs": {
                "type": "array",
                "items": {"type": "string", "enum": list(new_option_refs)},
            },
        },
        "required": ["outcome", "supportedOptionRefs"],
        "additionalProperties": False,
    }
    system_text = (
        "Проверь только текущую реплику ребёнка на новые занятия из списка, "
        "к которым он выразил ясный личный интерес. Это может быть прямое "
        "предпочтение, сильное желание самому заняться этим или добровольное "
        "повторяющееся действие, ясно связанное с занятием. Верни ровно JSON "
        "по схеме. "
        "Названия в options показывают варианты, но сами не доказывают выбор. "
        "Прежние реплики и уже сохранённые варианты не служат основанием для "
        "нового выбора. alreadySelectedRefs перечисляет ранее сохранённые "
        "варианты: не возвращай их в supportedOptionRefs. "
        "SELECTED означает, что текущая реплика ясно подтверждает один или "
        "несколько новых вариантов; перечисли ровно их ссылки. "
        "Тихое собственное напевание припева может подтверждать «Петь», а "
        "ритмичное движение под понравившуюся музыку — «Танцевать», даже если "
        "ребёнок стесняется делать это при других. Не требуй слова «люблю», "
        "но не считай интересом разовое случайное движение, упоминание песни "
        "или действие по принуждению. Умение обращаться с ножницами и награда "
        "за самодельную вещь без собственного изготовления и удовольствия от "
        "процесса не подтверждают «Мастерить». "
        "Общая настойчивость, желание награды или умение доводить дело до конца "
        "без названия любимого занятия не выбирают ни один вариант. "
        "Съёмка или публикация видео сама по себе относится к «Фото–видео», "
        "если ребёнок называет это любимым занятием; она не подтверждает "
        "«Компьютерное творчество» без явно описанного монтажа, эффектов "
        "или другого цифрового творчества. "
        "Один собственный творческий процесс может подтверждать несколько "
        "вариантов: явная любовь к созданию рисунков в цифровой программе "
        "подтверждает и «Рисовать», и «Компьютерное творчество». Для второго "
        "варианта не нужно отдельного цифрового действия сверх рисования "
        "в программе. Сильное личное желание самому начать создавать цифровой "
        "результат тоже подтверждает цифровое творчество, даже до первой "
        "попытки; если желанный результат — рисунок, оно подтверждает также "
        "«Рисовать». "
        "Простое любопытство или просмотр чужой работы — нет. "
        "NONE допустим только при явном отказе от всего списка в текущей "
        "реплике; верни пустой supportedOptionRefs. Во всех остальных случаях "
        "без подтверждённого нового варианта верни UNCLEAR и пустой список."
    )
    return LLMCompletionRequest(
        messages=(
            LLMMessage(role=LLMMessageRole.SYSTEM, content=system_text),
            LLMMessage(
                role=LLMMessageRole.USER,
                content=json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            ),
        ),
        response_schema=LLMResponseSchema(
            name="questionnaire_additive_choice_verification_v1",
            schema=schema,
        ),
    )


def parse_choice_verification(
    content: str,
    question: Question,
) -> ChoiceVerification:
    """
    Принимает только полное, непротиворечивое решение по показанным ссылкам.
    """
    raw = json.loads(content)
    if not isinstance(raw, dict) or set(raw) != {"outcome", "supportedOptionRefs"}:
        raise ValueError("проверка выбора вернула неподдерживаемую структуру")
    outcome = ChoiceVerificationOutcome(raw["outcome"])
    refs = raw["supportedOptionRefs"]
    if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
        raise ValueError("проверка выбора вернула недопустимые ссылки")
    allowed_refs = {f"V{index}" for index in range(1, len(question.options) + 1)}
    if len(refs) != len(set(refs)) or any(ref not in allowed_refs for ref in refs):
        raise ValueError("проверка выбора вернула повтор или неизвестную ссылку")
    if (outcome == ChoiceVerificationOutcome.SELECTED) != bool(refs):
        raise ValueError("исход проверки не согласован с выбранными ссылками")
    return ChoiceVerification(outcome=outcome, supported_option_refs=tuple(refs))
