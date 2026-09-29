import { useCallback, useEffect, useRef, useState } from "react";

import {
  QuestionnaireApiError,
  type DialogMode,
  type QuestionnaireAge,
  type QuestionnaireSnapshot,
  deleteSession,
  restoreSession,
  selectDialogMode,
  sendMessage,
  startSession,
} from "../api/questionnaireApi";
import { AppHeader } from "../components/AppHeader";
import { QuestionnaireSetup } from "../features/questionnaire/QuestionnaireSetup";
import { QuestionnaireWorkspace } from "../features/questionnaire/QuestionnaireWorkspace";

const SESSION_STORAGE_KEY = "questionnaire-agent-session";

type PendingTurn = {
  sessionId: string;
  stateVersion: number;
  messageId: string;
  content: string;
};

type SelectedDialogMode = Exclude<DialogMode, "UNSELECTED">;

type PendingDialogModeSelection = {
  sessionId: string;
  stateVersion: number;
  messageId: string;
  dialogMode: SelectedDialogMode;
};

function readStoredSessionId(): string | null {
  try {
    return window.sessionStorage.getItem(SESSION_STORAGE_KEY);
  } catch {
    return null;
  }
}

function storeSessionId(sessionId: string) {
  try {
    window.sessionStorage.setItem(SESSION_STORAGE_KEY, sessionId);
  } catch {
    // Работа анкеты не должна зависеть от доступности локального хранилища браузера.
  }
}

function removeStoredSessionId() {
  try {
    window.sessionStorage.removeItem(SESSION_STORAGE_KEY);
  } catch {
    // В закрытом хранилище достаточно сбросить состояние текущей вкладки.
  }
}

function readableError(error: unknown, fallback: string): string {
  return error instanceof QuestionnaireApiError ? error.message : fallback;
}

function isSessionNotFoundError(error: unknown): boolean {
  return (
    error instanceof QuestionnaireApiError &&
    (error.status === 404 || error.code === "SESSION_NOT_FOUND")
  );
}

export function App() {
  const storedSessionIdRef = useRef(readStoredSessionId());
  const pendingTurnRef = useRef<PendingTurn | null>(null);
  const pendingDialogModeRef = useRef<PendingDialogModeSelection | null>(null);
  const restoreAttemptRef = useRef(0);
  const hasOpenedWorkspace = useRef(false);
  const [session, setSession] = useState<QuestionnaireSnapshot | null>(null);
  const [isRestoring, setIsRestoring] = useState(storedSessionIdRef.current !== null);
  const [isStarting, setIsStarting] = useState(false);
  const [isSending, setIsSending] = useState(false);
  const [isSelectingDialogMode, setIsSelectingDialogMode] = useState(false);
  const [isResetting, setIsResetting] = useState(false);
  const [canRetryRestore, setCanRetryRestore] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const restoreKnownSession = useCallback(async (sessionId: string) => {
    const attempt = restoreAttemptRef.current + 1;
    restoreAttemptRef.current = attempt;
    setIsRestoring(true);
    setCanRetryRestore(false);
    setError(null);

    try {
      const restoredSession = await restoreSession(sessionId);
      if (restoreAttemptRef.current !== attempt) {
        return;
      }
      hasOpenedWorkspace.current = true;
      pendingTurnRef.current = null;
      pendingDialogModeRef.current = null;
      storedSessionIdRef.current = restoredSession.sessionId;
      setSession(restoredSession);
      storeSessionId(restoredSession.sessionId);
    } catch (restoreError: unknown) {
      if (restoreAttemptRef.current !== attempt) {
        return;
      }

      if (isSessionNotFoundError(restoreError)) {
        removeStoredSessionId();
        storedSessionIdRef.current = null;
      } else {
        setCanRetryRestore(true);
      }
      setError(
        readableError(
          restoreError,
          "Не удалось восстановить анкету. Проверьте подключение и повторите попытку.",
        ),
      );
    } finally {
      if (restoreAttemptRef.current === attempt) {
        setIsRestoring(false);
      }
    }
  }, []);

  useEffect(() => {
    const storedSessionId = storedSessionIdRef.current;
    if (storedSessionId === null) {
      return;
    }

    void restoreKnownSession(storedSessionId);

    return () => {
      restoreAttemptRef.current += 1;
    };
  }, [restoreKnownSession]);

  function retrySessionRestore() {
    const storedSessionId = storedSessionIdRef.current;
    if (storedSessionId !== null) {
      void restoreKnownSession(storedSessionId);
    }
  }

  async function startQuestionnaire(age: QuestionnaireAge) {
    if (isStarting || isRestoring || isSelectingDialogMode || isResetting) {
      return;
    }

    const canRestorePreviousSession = storedSessionIdRef.current !== null;
    setIsStarting(true);
    setCanRetryRestore(false);
    setError(null);
    try {
      const createdSession = await startSession(age);
      hasOpenedWorkspace.current = true;
      pendingTurnRef.current = null;
      pendingDialogModeRef.current = null;
      setSession(createdSession);
      storeSessionId(createdSession.sessionId);
      storedSessionIdRef.current = createdSession.sessionId;
    } catch (startError: unknown) {
      setCanRetryRestore(canRestorePreviousSession);
      setError(
        readableError(startError, "Не удалось начать анкету. Проверьте подключение к сервису."),
      );
    } finally {
      setIsStarting(false);
    }
  }

  async function submitMessage(content: string): Promise<boolean> {
    if (
      session === null ||
      isSending ||
      isSelectingDialogMode ||
      isResetting ||
      session.completed ||
      session.dialogMode === "UNSELECTED"
    ) {
      return false;
    }

    const currentPendingTurn = pendingTurnRef.current;
    const pendingTurn =
      currentPendingTurn !== null &&
      currentPendingTurn.sessionId === session.sessionId &&
      currentPendingTurn.stateVersion === session.stateVersion &&
      currentPendingTurn.content === content
        ? currentPendingTurn
        : {
            sessionId: session.sessionId,
            stateVersion: session.stateVersion,
            messageId: crypto.randomUUID(),
            content,
          };

    pendingTurnRef.current = pendingTurn;
    setIsSending(true);
    setError(null);
    try {
      const updatedSession = await sendMessage(session.sessionId, {
        messageId: pendingTurn.messageId,
        content: pendingTurn.content,
        stateVersion: pendingTurn.stateVersion,
      });
      pendingTurnRef.current = null;
      setSession(updatedSession);
      storeSessionId(updatedSession.sessionId);
      storedSessionIdRef.current = updatedSession.sessionId;
      return true;
    } catch (sendError: unknown) {
      if (sendError instanceof QuestionnaireApiError && sendError.status === 409) {
        pendingTurnRef.current = null;
        if (
          sendError.code === "STATE_VERSION_CONFLICT" ||
          sendError.code === "SESSION_COMPLETED"
        ) {
          try {
            const restoredSession = await restoreSession(session.sessionId);
            setSession(restoredSession);
            storeSessionId(restoredSession.sessionId);
            storedSessionIdRef.current = restoredSession.sessionId;
            setError(
              sendError.code === "STATE_VERSION_CONFLICT"
                ? `${sendError.message} Проверьте текущий вопрос перед повторной отправкой.`
                : sendError.message,
            );
          } catch (restoreError: unknown) {
            if (isSessionNotFoundError(restoreError)) {
              removeStoredSessionId();
              storedSessionIdRef.current = null;
              setSession(null);
            }
            setError(
              readableError(
                restoreError,
                "Состояние анкеты изменилось, но обновить его не удалось. Перезагрузите страницу.",
              ),
            );
          }
        } else if (sendError.code === "MESSAGE_ID_CONFLICT") {
          setError(`${sendError.message} Повторите отправку ответа.`);
        } else {
          setError(sendError.message);
        }
      } else {
        setError(
          readableError(sendError, "Не удалось отправить ответ. Проверьте подключение и повторите."),
        );
      }
      return false;
    } finally {
      setIsSending(false);
    }
  }

  async function chooseDialogMode(dialogMode: SelectedDialogMode): Promise<void> {
    if (
      session === null ||
      session.dialogMode !== "UNSELECTED" ||
      isSending ||
      isSelectingDialogMode ||
      isResetting
    ) {
      return;
    }

    const currentPendingSelection = pendingDialogModeRef.current;
    const pendingSelection =
      currentPendingSelection !== null &&
      currentPendingSelection.sessionId === session.sessionId &&
      currentPendingSelection.stateVersion === session.stateVersion &&
      currentPendingSelection.dialogMode === dialogMode
        ? currentPendingSelection
        : {
            sessionId: session.sessionId,
            stateVersion: session.stateVersion,
            messageId: crypto.randomUUID(),
            dialogMode,
          };

    pendingDialogModeRef.current = pendingSelection;
    setIsSelectingDialogMode(true);
    setError(null);
    try {
      const updatedSession = await selectDialogMode(session.sessionId, {
        messageId: pendingSelection.messageId,
        dialogMode: pendingSelection.dialogMode,
        stateVersion: pendingSelection.stateVersion,
      });
      pendingDialogModeRef.current = null;
      setSession(updatedSession);
      storeSessionId(updatedSession.sessionId);
      storedSessionIdRef.current = updatedSession.sessionId;
    } catch (selectionError: unknown) {
      if (selectionError instanceof QuestionnaireApiError && selectionError.status === 409) {
        pendingDialogModeRef.current = null;
        if (
          selectionError.code === "STATE_VERSION_CONFLICT" ||
          selectionError.code === "DIALOG_MODE_ALREADY_SELECTED"
        ) {
          try {
            const restoredSession = await restoreSession(session.sessionId);
            setSession(restoredSession);
            storeSessionId(restoredSession.sessionId);
            storedSessionIdRef.current = restoredSession.sessionId;
            if (restoredSession.dialogMode === "UNSELECTED") {
              setError(`${selectionError.message} Выберите вариант ещё раз.`);
            }
          } catch (restoreError: unknown) {
            if (isSessionNotFoundError(restoreError)) {
              removeStoredSessionId();
              storedSessionIdRef.current = null;
              setSession(null);
            }
            setError(
              readableError(
                restoreError,
                "Состояние анкеты изменилось, но обновить его не удалось. Перезагрузите страницу.",
              ),
            );
          }
        } else if (selectionError.code === "MESSAGE_ID_CONFLICT") {
          setError(`${selectionError.message} Выберите вариант ещё раз.`);
        } else {
          setError(selectionError.message);
        }
      } else {
        setError(
          readableError(
            selectionError,
            "Не удалось выбрать сценарий. Проверьте подключение и повторите попытку.",
          ),
        );
      }
    } finally {
      setIsSelectingDialogMode(false);
    }
  }

  async function resetQuestionnaire() {
    if (session === null || isSending || isSelectingDialogMode || isResetting) {
      return;
    }

    setIsResetting(true);
    setError(null);
    try {
      await deleteSession(session.sessionId);
    } catch (deleteError: unknown) {
      if (!isSessionNotFoundError(deleteError)) {
        setError(
          `${readableError(
            deleteError,
            "Не удалось начать новую анкету. Проверьте подключение и повторите попытку.",
          )} Текущая анкета сохранена.`,
        );
        setIsResetting(false);
        return;
      }
    }

    pendingTurnRef.current = null;
    pendingDialogModeRef.current = null;
    storedSessionIdRef.current = null;
    removeStoredSessionId();
    setSession(null);
    setCanRetryRestore(false);
    setError(null);
    setIsResetting(false);
  }

  return (
    <main className={`app-shell${session !== null ? " app-shell--workspace" : ""}`}>
      <AppHeader
        status={isRestoring ? "Восстановление анкеты" : "Готов к запуску"}
        onReset={session !== null ? () => void resetQuestionnaire() : null}
        isResetting={isResetting}
        resetDisabled={isSending || isSelectingDialogMode || isResetting}
      />

      {isRestoring ? (
        <section className="loading-card card" role="status" aria-live="polite">
          <span className="loading-spinner" aria-hidden="true" />
          <div>
            <h2>Восстанавливаем анкету</h2>
            <p>Загружаем историю диалога и сохранённые ответы.</p>
          </div>
        </section>
      ) : session !== null ? (
        <QuestionnaireWorkspace
          session={session}
          isSending={isSending}
          isSelectingDialogMode={isSelectingDialogMode}
          isResetting={isResetting}
          error={error}
          onSend={submitMessage}
          onSelectDialogMode={chooseDialogMode}
        />
      ) : (
        <>
          {error !== null && (
            <p className="app-error" role="alert">
              {error}
            </p>
          )}
          {canRetryRestore && storedSessionIdRef.current !== null && (
            <button
              className="secondary-button restore-button"
              type="button"
              onClick={retrySessionRestore}
            >
              Повторить восстановление
            </button>
          )}
          <QuestionnaireSetup
            focusOnMount={hasOpenedWorkspace.current}
            isStarting={isStarting}
            onStart={(age) => void startQuestionnaire(age)}
          />
        </>
      )}

      <footer className="app-footer">
        <span>Интерфейс анкетирования</span>
        <span aria-hidden="true">·</span>
        <span>Состояние сессии восстанавливается после обновления страницы</span>
      </footer>
    </main>
  );
}
