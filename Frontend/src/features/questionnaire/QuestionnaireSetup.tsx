import { type FormEvent, useEffect, useRef, useState } from "react";

import { type QuestionnaireAge } from "../../api/questionnaireApi";

type QuestionnaireSetupProps = {
  focusOnMount: boolean;
  isStarting: boolean;
  onStart: (age: QuestionnaireAge) => void;
};

const AVAILABLE_AGES: QuestionnaireAge[] = [6, 7, 8, 9];

export function QuestionnaireSetup({
  focusOnMount,
  isStarting,
  onStart,
}: QuestionnaireSetupProps) {
  const [age, setAge] = useState<QuestionnaireAge>(8);
  const startButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (focusOnMount) {
      startButtonRef.current?.focus({ preventScroll: true });
    }
  }, [focusOnMount]);

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    onStart(age);
  }

  return (
    <section className="setup-card card" aria-labelledby="setup-title">
      <form className="setup-form" onSubmit={handleSubmit}>
        <div className="setup-content">
          <span className="section-kicker">АНКЕТА 6–9 ЛЕТ</span>
          <h2 id="setup-title">Перед началом уточним два пункта</h2>
          <p>
            Это технические данные анкеты. После создания сессии можно будет выбрать,
            как начать разговор с ассистентом.
          </p>

          <div className="technical-fields">
            <fieldset className="technical-field age-field">
              <legend>
                <span className="question-code">D1</span>
                <span>Сколько тебе лет?</span>
              </legend>
              <div className="age-options">
                {AVAILABLE_AGES.map((availableAge) => (
                  <label key={availableAge}>
                    <input
                      type="radio"
                      name="age"
                      value={availableAge}
                      checked={age === availableAge}
                      onChange={() => setAge(availableAge)}
                      disabled={isStarting}
                    />
                    <span>{availableAge}</span>
                  </label>
                ))}
              </div>
            </fieldset>

            <div className="technical-field respondent-field" aria-labelledby="respondent-title">
              <div className="technical-question" id="respondent-title">
                <span className="question-code">D3</span>
                <span>Кто заполняет анкету?</span>
              </div>
              <span className="respondent-value">Ребёнок самостоятельно</span>
            </div>
          </div>
        </div>

        <button
          className="primary-button start-button"
          ref={startButtonRef}
          type="submit"
          disabled={isStarting}
        >
          {isStarting ? "Создаём сессию…" : "Продолжить"}
          <span aria-hidden="true">→</span>
        </button>
      </form>
    </section>
  );
}
