export type DialogRole = "ASSISTANT" | "USER";

export type DialogMode = "UNSELECTED" | "GUIDED" | "FREE_STORY";

export type QuestionnaireAge = 6 | 7 | 8 | 9;

export type QuestionStatus =
  | "UNKNOWN"
  | "NEEDS_CLARIFICATION"
  | "PARTIALLY_ANSWERED"
  | "ANSWERED"
  | "SKIPPED"
  | "NOT_APPLICABLE";

export type QuestionnaireStatus =
  | "IN_PROGRESS"
  | "COMPLETED"
  | "COMPLETED_WITH_SKIPS";

export type QuestionnaireOption = {
  optionCode: string;
  text: string;
  numericValue?: number | null;
};

export type QuestionnaireAnswer = {
  optionCodes: string[];
  evidence?: string | null;
  origin: "USER_EVIDENCE" | "POLICY_DEFAULT";
  defaultReason?: "CLARIFICATION_LIMIT_REACHED" | null;
};

export type QuestionnaireQuestion = {
  questionCode: string;
  questionText: string;
  status: QuestionStatus;
  options: QuestionnaireOption[];
  answer: QuestionnaireAnswer | null;
  skipReason?: string | null;
};

export type DialogMessage = {
  messageId: string;
  role: DialogRole;
  content: string;
  createdAt: string;
};

export type QuestionnaireSnapshot = {
  sessionId: string;
  stateVersion: number;
  status: QuestionnaireStatus;
  completed: boolean;
  dialogMode: DialogMode;
  activeQuestionCode: string | null;
  companionQuestionCode?: string | null;
  questionnaire: {
    questionnaireCode: string;
    questionnaireVersion: string;
  };
  questions: QuestionnaireQuestion[];
  dialogHistory: DialogMessage[];
};

type StartSessionRequest = {
  age: QuestionnaireAge;
  respondentMode: "CHILD";
};

type SelectDialogModeRequest = {
  messageId: string;
  dialogMode: Exclude<DialogMode, "UNSELECTED">;
  stateVersion: number;
};

type SendMessageRequest = {
  messageId: string;
  content: string;
  stateVersion: number;
};

type RequestOptions = {
  method?: "GET" | "POST" | "DELETE";
  body?: unknown;
  allowEmptyResponse?: boolean;
};

type ApiErrorDetails = {
  code: string | null;
  retryable: boolean | null;
  currentStateVersion: number | null;
};

const PUBLIC_SERVER_ERROR_CODES = new Set([
  "SESSION_CAPACITY_REACHED",
  "ML_SERVICE_UNAVAILABLE",
  "ML_RESPONSE_INVALID",
  "LLM_TIMEOUT",
  "LLM_CONNECTION",
  "LLM_RATE_LIMIT",
  "LLM_PROVIDER_UNAVAILABLE",
  "LLM_EMPTY_RESPONSE",
  "LLM_TRUNCATED_RESPONSE",
  "LLM_INVALID_RESPONSE",
  "AGENT_OUTPUT_TOO_LARGE",
  "AGENT_OUTPUT_INVALID",
  "AGENT_OUTPUT_REJECTED",
]);

export class QuestionnaireApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly retryable: boolean | null;
  readonly currentStateVersion: number | null;

  constructor(message: string, status: number, details: ApiErrorDetails = emptyErrorDetails()) {
    super(message);
    this.name = "QuestionnaireApiError";
    this.status = status;
    this.code = details.code;
    this.retryable = details.retryable;
    this.currentStateVersion = details.currentStateVersion;
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function validationMessage(value: unknown): string | null {
  if (!Array.isArray(value)) {
    return null;
  }

  const messages = value.flatMap((item) => {
    if (!isRecord(item)) {
      return [];
    }
    const message = item.msg ?? item.message;
    return typeof message === "string" ? [message] : [];
  });

  return messages.length > 0 ? messages.join(" ") : null;
}

function emptyErrorDetails(): ApiErrorDetails {
  return {
    code: null,
    retryable: null,
    currentStateVersion: null,
  };
}

function apiErrorDetails(payload: unknown): ApiErrorDetails {
  if (!isRecord(payload) || !isRecord(payload.detail)) {
    return emptyErrorDetails();
  }

  const detail = payload.detail;
  return {
    code: typeof detail.code === "string" ? detail.code : null,
    retryable: typeof detail.retryable === "boolean" ? detail.retryable : null,
    currentStateVersion:
      typeof detail.currentStateVersion === "number" &&
      Number.isInteger(detail.currentStateVersion) &&
      detail.currentStateVersion >= 0
        ? detail.currentStateVersion
        : null,
  };
}

function errorMessage(payload: unknown, status: number, details: ApiErrorDetails): string {
  if (
    status >= 500 &&
    (details.code === null || !PUBLIC_SERVER_ERROR_CODES.has(details.code))
  ) {
    return "Сервис временно недоступен. Попробуйте ещё раз.";
  }

  if (isRecord(payload)) {
    if (typeof payload.detail === "string") {
      return payload.detail;
    }

    const validationDetail = validationMessage(payload.detail);
    if (validationDetail !== null) {
      return validationDetail;
    }

    if (typeof payload.message === "string") {
      return payload.message;
    }
  }

  if (status >= 500) {
    return "Сервис временно недоступен. Попробуйте ещё раз.";
  }
  return "Не удалось выполнить запрос. Проверьте введённые данные.";
}

async function request<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const hasBody = options.body !== undefined;
  const response = await fetch(path, {
    method: options.method ?? "GET",
    headers: hasBody
      ? {
          Accept: "application/json",
          "Content-Type": "application/json",
        }
      : { Accept: "application/json" },
    body: hasBody ? JSON.stringify(options.body) : undefined,
    cache: "no-store",
    redirect: "error",
  });

  const rawBody = await response.text();
  let payload: unknown = null;
  if (rawBody.length > 0) {
    try {
      payload = JSON.parse(rawBody) as unknown;
    } catch {
      if (response.ok) {
        throw new QuestionnaireApiError(
          "Сервис вернул некорректный ответ.",
          response.status,
        );
      }
    }
  }

  if (!response.ok) {
    const details = apiErrorDetails(payload);
    throw new QuestionnaireApiError(
      errorMessage(payload, response.status, details),
      response.status,
      details,
    );
  }
  if (payload === null) {
    if (options.allowEmptyResponse) {
      return undefined as T;
    }
    throw new QuestionnaireApiError("Сервис вернул пустой ответ.", response.status);
  }

  return payload as T;
}

export function startSession(age: QuestionnaireAge): Promise<QuestionnaireSnapshot> {
  const body: StartSessionRequest = {
    age,
    respondentMode: "CHILD",
  };
  return request<QuestionnaireSnapshot>("/api/sessions", { method: "POST", body });
}

export function restoreSession(sessionId: string): Promise<QuestionnaireSnapshot> {
  return request<QuestionnaireSnapshot>(`/api/sessions/${encodeURIComponent(sessionId)}`);
}

export function sendMessage(
  sessionId: string,
  message: SendMessageRequest,
): Promise<QuestionnaireSnapshot> {
  return request<QuestionnaireSnapshot>(
    `/api/sessions/${encodeURIComponent(sessionId)}/messages`,
    { method: "POST", body: message },
  );
}

export function selectDialogMode(
  sessionId: string,
  selection: SelectDialogModeRequest,
): Promise<QuestionnaireSnapshot> {
  return request<QuestionnaireSnapshot>(
    `/api/sessions/${encodeURIComponent(sessionId)}/dialog-mode`,
    { method: "POST", body: selection },
  );
}

export function deleteSession(sessionId: string): Promise<void> {
  return request<void>(`/api/sessions/${encodeURIComponent(sessionId)}`, {
    method: "DELETE",
    allowEmptyResponse: true,
  });
}
