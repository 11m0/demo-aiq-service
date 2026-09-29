function SparkIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" width="24" height="24">
      <path
        d="M12 2.75c.42 4.94 3.31 7.83 8.25 8.25-4.94.42-7.83 3.31-8.25 8.25C11.58 14.31 8.69 11.42 3.75 11 8.69 10.58 11.58 7.69 12 2.75Z"
        fill="currentColor"
      />
    </svg>
  );
}

type AppHeaderProps = {
  status: string;
  onReset: (() => void) | null;
  isResetting: boolean;
  resetDisabled: boolean;
};

export function AppHeader({ status, onReset, isResetting, resetDisabled }: AppHeaderProps) {
  return (
    <header className="app-header">
      <div className="brand">
        <span className="brand-mark">
          <SparkIcon />
        </span>
        <div>
          <span className="eyebrow">СЕРВИС АНКЕТИРОВАНИЯ</span>
          <h1>ИИ-ассистент</h1>
        </div>
      </div>

      {onReset === null ? (
        <span className="status-badge">
          <span className="status-dot" aria-hidden="true" />
          {status}
        </span>
      ) : (
        <button
          className="secondary-button header-reset-button"
          type="button"
          onClick={onReset}
          disabled={resetDisabled}
        >
          {isResetting ? "Начинаем новую анкету…" : "Начать новую анкету"}
        </button>
      )}
    </header>
  );
}
