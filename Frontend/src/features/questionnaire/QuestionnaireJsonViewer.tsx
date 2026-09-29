import { useMemo, useRef, useState } from "react";

import { type QuestionnaireSnapshot } from "../../api/questionnaireApi";

import "./QuestionnaireJsonViewer.css";

type QuestionnaireJsonViewerProps = {
  session: QuestionnaireSnapshot;
};

export function QuestionnaireJsonViewer({ session }: QuestionnaireJsonViewerProps) {
  const dialogRef = useRef<HTMLDialogElement>(null);
  const openButtonRef = useRef<HTMLButtonElement>(null);
  const [actionMessage, setActionMessage] = useState("");
  const formattedJson = useMemo(() => JSON.stringify(session, null, 2), [session]);

  function openDialog() {
    setActionMessage("");
    dialogRef.current?.showModal();
  }

  function closeDialog() {
    dialogRef.current?.close();
  }

  async function copyJson() {
    try {
      await navigator.clipboard.writeText(formattedJson);
      setActionMessage("JSON скопирован.");
    } catch {
      setActionMessage("Не удалось скопировать JSON. Вы можете выделить текст и скопировать его вручную.");
    }
  }

  function downloadJson() {
    const blob = new Blob([formattedJson + "\n"], {
      type: "application/json;charset=utf-8",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    const safeSessionId = session.sessionId.replace(/[^a-zA-Z0-9_-]/g, "_");
    link.href = url;
    link.download = `questionnaire-${safeSessionId}.json`;
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 0);
    setActionMessage("JSON скачан.");
  }

  return (
    <>
      <button
        className="questionnaire-json-trigger"
        type="button"
        ref={openButtonRef}
        onClick={openDialog}
        aria-haspopup="dialog"
      >
        Посмотреть JSON
      </button>

      <dialog
        className="questionnaire-json-dialog"
        ref={dialogRef}
        aria-labelledby="questionnaire-json-title"
        aria-describedby="questionnaire-json-description"
        onClose={() => openButtonRef.current?.focus({ preventScroll: true })}
      >
        <div className="questionnaire-json-heading">
          <div>
            <h2 id="questionnaire-json-title">JSON анкеты</h2>
            <p id="questionnaire-json-description">Текущее состояние сессии после сохранённых ответов.</p>
          </div>
          <button
            className="questionnaire-json-close"
            type="button"
            onClick={closeDialog}
            aria-label="Закрыть JSON анкеты"
          >
            ×
          </button>
        </div>

        <div className="questionnaire-json-actions">
          <button className="questionnaire-json-action" type="button" onClick={() => void copyJson()}>
            Скопировать
          </button>
          <button className="questionnaire-json-action" type="button" onClick={downloadJson}>
            Скачать JSON
          </button>
          <span className="questionnaire-json-action-message" role="status" aria-live="polite">
            {actionMessage}
          </span>
        </div>

        <pre className="questionnaire-json-content" tabIndex={0}>{formattedJson}</pre>
      </dialog>
    </>
  );
}
