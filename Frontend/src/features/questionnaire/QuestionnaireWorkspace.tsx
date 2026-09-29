import { type FormEvent, useEffect, useMemo, useRef, useState } from "react";

import {
  type DialogMode,
  type QuestionnaireQuestion,
  type QuestionnaireSnapshot,
} from "../../api/questionnaireApi";
import { QuestionnaireJsonViewer } from "./QuestionnaireJsonViewer";

type SelectedDialogMode = Exclude<DialogMode, "UNSELECTED">;

type QuestionnaireWorkspaceProps = {
  session: QuestionnaireSnapshot;
  isSending: boolean;
  isSelectingDialogMode: boolean;
  isResetting: boolean;
  error: string | null;
  onSend: (content: string) => Promise<boolean>;
  onSelectDialogMode: (dialogMode: SelectedDialogMode) => Promise<void>;
};

function SendIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="19" height="19">
      <path
        d="m5 12 14-7-4.75 14-2.5-5.25L5 12Zm6.75 1.75L19 5"
        fill="none"
        stroke="currentColor"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="1.8"
      />
    </svg>
  );
}

function answerLabels(question: QuestionnaireQuestion): string[] {
  if (question.answer === null) {
    return [];
  }

  const isScaleQuestion =
    question.options.length === 4 &&
    question.options.every((option, index) => option.numericValue === index);
  const labelsByCode = new Map(
    question.options.map((option) => [
      option.optionCode,
      isScaleQuestion && option.numericValue !== null && option.numericValue !== undefined
        ? `${option.numericValue}/3`
        : option.text,
    ]),
  );
  return question.answer.optionCodes.map((optionCode) => {
    return labelsByCode.get(optionCode) ?? optionCode;
  });
}

function isCurrentQuestion(
  question: QuestionnaireQuestion,
  activeQuestionCode: string | null,
  companionQuestionCode: string | null | undefined,
): boolean {
  return (
    question.questionCode === activeQuestionCode ||
    (companionQuestionCode != null && question.questionCode === companionQuestionCode)
  );
}

function questionClassName(
  question: QuestionnaireQuestion,
  activeQuestionCode: string | null,
  companionQuestionCode: string | null | undefined,
): string {
  const classes = ["answer-item"];
  if (
    isCurrentQuestion(question, activeQuestionCode, companionQuestionCode) ||
    question.status === "NEEDS_CLARIFICATION"
  ) {
    classes.push("current");
  }
  if (question.status === "SKIPPED") {
    classes.push("skipped");
  }
  if (question.status === "NOT_APPLICABLE") {
    classes.push("not-applicable");
  }
  return classes.join(" ");
}

function questionStatusText(
  question: QuestionnaireQuestion,
  activeQuestionCode: string | null,
  companionQuestionCode: string | null | undefined,
): string | null {
  const isActive = isCurrentQuestion(question, activeQuestionCode, companionQuestionCode);

  if (question.status === "NEEDS_CLARIFICATION") {
    return "Ассистент уточняет этот ответ";
  }
  if (question.status === "PARTIALLY_ANSWERED") {
    return isActive
      ? "Уточняем, нравятся ли тебе ещё какие-то занятия"
      : "Уже отмечено; в конце спросим про другие занятия";
  }
  if (question.status === "SKIPPED") {
    if (question.skipReason === "NO_ANSWER_AFTER_RETRY") {
      return "Ответ не удалось определить после двух уточнений";
    }
    return "Вопрос пропущен";
  }
  if (question.status === "NOT_APPLICABLE") {
    return "Вопрос не применяется";
  }
  if (isActive && question.status === "ANSWERED") {
    return "Сейчас уточняется сохранённый ответ";
  }
  if (isActive) {
    return "Сейчас задан этот вопрос";
  }
  if (question.status === "UNKNOWN") {
    return "Ещё не отвечено";
  }
  return null;
}

function completionText(session: QuestionnaireSnapshot): string {
  if (session.status === "COMPLETED_WITH_SKIPS") {
    return "Анкета завершена. Некоторые вопросы остались без ответа.";
  }
  return "Анкета завершена. Спасибо за ответы.";
}

export function QuestionnaireWorkspace({
  session,
  isSending,
  isSelectingDialogMode,
  isResetting,
  error,
  onSend,
  onSelectDialogMode,
}: QuestionnaireWorkspaceProps) {
  const [draft, setDraft] = useState("");
  const [pendingContent, setPendingContent] = useState<string | null>(null);
  const [needsMoreTime, setNeedsMoreTime] = useState(false);
  const headingRef = useRef<HTMLHeadingElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const isBusy = isSending || isSelectingDialogMode || isResetting;

  const resolvedCount = useMemo(
    () =>
      session.questions.filter((question) =>
        ["ANSWERED", "SKIPPED", "NOT_APPLICABLE"].includes(question.status),
      ).length,
    [session.questions],
  );

  useEffect(() => {
    headingRef.current?.focus({ preventScroll: true });
  }, []);

  useEffect(() => {
    const messagesElement = messagesRef.current;
    if (messagesElement !== null) {
      const scrollBehavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth";
      messagesElement.scrollTo({ top: messagesElement.scrollHeight, behavior: scrollBehavior });
    }
  }, [session.dialogHistory, pendingContent, isSending, isSelectingDialogMode, needsMoreTime]);

  useEffect(() => {
    if (!isSending) {
      return;
    }
    const timer = window.setTimeout(() => setNeedsMoreTime(true), 5000);
    return () => window.clearTimeout(timer);
  }, [isSending]);

  useEffect(() => {
    if (session.dialogMode !== "UNSELECTED" && !session.completed) {
      window.requestAnimationFrame(() => inputRef.current?.focus({ preventScroll: true }));
    }
  }, [session.dialogMode, session.completed]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const normalizedContent = draft.trim();
    if (normalizedContent.length === 0 || isBusy || session.completed) {
      return;
    }

    setNeedsMoreTime(false);
    setPendingContent(normalizedContent);
    const sent = await onSend(normalizedContent);
    setPendingContent(null);
    if (sent) {
      setDraft("");
    }
    window.requestAnimationFrame(() => inputRef.current?.focus({ preventScroll: true }));
  }

  return (
    <div className="workspace">
      <section className="chat-card card" aria-labelledby="dialog-title">
        <h2 className="sr-only" id="dialog-title" ref={headingRef} tabIndex={-1}>
          Разговор с ассистентом
        </h2>

        <div
          className="messages"
          ref={messagesRef}
          role="log"
          aria-live="polite"
          aria-label="История диалога"
          aria-busy={isSending || isSelectingDialogMode}
          tabIndex={-1}
        >
          {session.dialogHistory.map((message) => {
            const roleClass = message.role === "ASSISTANT" ? "assistant" : "user";
            return (
              <div className={`message-row ${roleClass}`} key={message.messageId}>
                <div className={`message-bubble ${roleClass}`}>
                  <span className="message-author">
                    {message.role === "ASSISTANT" ? "ИИ-АССИСТЕНТ" : "ВЫ"}
                  </span>
                  {message.content}
                </div>
              </div>
            );
          })}

          {pendingContent !== null && (
            <div className="message-row user pending" aria-hidden="true">
              <div className="message-bubble user">
                <span className="message-author">ВЫ</span>
                {pendingContent}
              </div>
            </div>
          )}

          {(isSending || isSelectingDialogMode) && (
            <div className="typing-indicator" role="status">
              <span aria-hidden="true" />
              <span aria-hidden="true" />
              <span aria-hidden="true" />
              {isSelectingDialogMode
                ? "Ассистент начинает диалог"
                : needsMoreTime
                  ? "Мне нужно ещё немного времени…"
                  : "Внимательно читаю твой ответ…"}
            </div>
          )}
        </div>

        {session.dialogMode === "UNSELECTED" ? (
          <div className="dialog-mode-selector" aria-labelledby="dialog-mode-title">
            {error !== null && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <div className="dialog-mode-heading">
              <span className="section-kicker">КАК НАЧНЁМ?</span>
              <h3 id="dialog-mode-title">Выбери удобный вариант</h3>
            </div>
            <div className="dialog-mode-options">
              <button
                className="dialog-mode-button"
                type="button"
                onClick={() => void onSelectDialogMode("FREE_STORY")}
                disabled={isBusy}
              >
                <span className="dialog-mode-icon" aria-hidden="true">✦</span>
                <span>
                  <strong>Сначала расскажу о себе</strong>
                  <small>Можно написать один небольшой рассказ в свободной форме</small>
                </span>
              </button>
              <button
                className="dialog-mode-button"
                type="button"
                onClick={() => void onSelectDialogMode("GUIDED")}
                disabled={isBusy}
              >
                <span className="dialog-mode-icon" aria-hidden="true">→</span>
                <span>
                  <strong>Задавай вопросы</strong>
                  <small>Ассистент будет задавать вопросы анкеты по одному</small>
                </span>
              </button>
            </div>
          </div>
        ) : (
          <div className="composer">
            {error !== null && (
              <p className="inline-error" role="alert">
                {error}
              </p>
            )}
            <form onSubmit={(event) => void handleSubmit(event)}>
              <label className="sr-only" htmlFor="message-input">
                Ваш ответ
              </label>
              <textarea
                id="message-input"
                ref={inputRef}
                value={draft}
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
                    event.preventDefault();
                    event.currentTarget.form?.requestSubmit();
                  }
                }}
                placeholder={
                  session.completed
                    ? "Анкета завершена"
                    : session.dialogMode === "FREE_STORY" && session.activeQuestionCode === null
                      ? "Напиши небольшой рассказ о себе и о том, что тебе нравится…"
                      : "Напишите ответ своими словами…"
                }
                rows={2}
                maxLength={4000}
                disabled={isBusy || session.completed}
              />
              <button
                className="send-button"
                type="submit"
                aria-label="Отправить сообщение"
                disabled={draft.trim().length === 0 || isBusy || session.completed}
              >
                <span className="send-label">Отправить</span>
                <SendIcon />
              </button>
            </form>
            <p>Enter — отправить · Shift + Enter — новая строка</p>
          </div>
        )}
      </section>

      <aside className="answers-card card" aria-labelledby="answers-title">
        <div className="panel-heading answers-heading">
          <div>
            <span className="section-kicker">ПРОГРЕСС</span>
            <h2 id="answers-title">Ответы анкеты</h2>
          </div>
          <div className="answers-header-actions">
            <QuestionnaireJsonViewer session={session} />
            <span className="progress-count">
              {resolvedCount} / {session.questions.length}
            </span>
          </div>
        </div>

        <div className="progress-track" aria-hidden="true">
          <span
            style={{
              width:
                session.questions.length === 0
                  ? "0%"
                  : `${(resolvedCount / session.questions.length) * 100}%`,
            }}
          />
        </div>
        <div className="answer-list" role="region" aria-label="Ответы анкеты" tabIndex={0}>
          {session.questions.map((question, index) => {
            const labels = answerLabels(question);
            const statusText = questionStatusText(
              question,
              session.activeQuestionCode,
              session.companionQuestionCode,
            );
            return (
              <article
                className={questionClassName(
                  question,
                  session.activeQuestionCode,
                  session.companionQuestionCode,
                )}
                key={question.questionCode}
              >
                <span className="question-number">
                  {String(index + 1).padStart(2, "0")}
                </span>
                <div>
                  <h3>{question.questionText}</h3>
                  {labels.length > 0 && (
                    <div className="answer-tags">
                      {labels.map((label, labelIndex) => (
                        <span className="answer-tag" key={`${label}-${labelIndex}`}>
                          {label}
                        </span>
                      ))}
                    </div>
                  )}
                  {question.answer?.origin === "POLICY_DEFAULT" && (
                    <p className="answer-evidence">
                      Автоматически выставлено 0/3 после двух уточнений без понятного ответа.
                    </p>
                  )}
                  {question.answer?.origin !== "POLICY_DEFAULT" && question.answer?.evidence && (
                    <p className="answer-evidence">Основание: «{question.answer.evidence}»</p>
                  )}
                  {statusText !== null && <p>{statusText}</p>}
                </div>
              </article>
            );
          })}
        </div>

        {session.completed && (
          <p className="completion-note" role="status">
            {completionText(session)}
          </p>
        )}

        <div className="privacy-note">
          <svg aria-hidden="true" viewBox="0 0 24 24" width="18" height="18">
            <rect
              x="5.5"
              y="10"
              width="13"
              height="9"
              rx="2"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
            />
            <path
              d="M8.5 10V7.75a3.5 3.5 0 0 1 7 0V10"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
            />
          </svg>
          <span>Ответы передаются сервису для обработки и сохраняются в активной сессии.</span>
        </div>
      </aside>
    </div>
  );
}
