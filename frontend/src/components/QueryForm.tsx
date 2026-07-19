import { useState, type FormEvent, type KeyboardEvent } from "react";
import styles from "./QueryForm.module.css";

interface QueryFormProps {
  loading: boolean;
  onSubmit: (question: string) => void;
}

const EXAMPLE = "¿Cuál es el plazo mínimo de un contrato de alquiler de vivienda?";

/**
 * The accessible consultation form: a labelled textarea and a submit control.
 * Submits on the button or on Ctrl/Cmd+Enter, blocks empty and in-flight
 * submissions, and stays usable by keyboard alone.
 */
export function QueryForm({ loading, onSubmit }: QueryFormProps) {
  const [question, setQuestion] = useState("");
  const trimmed = question.trim();
  const canSubmit = trimmed.length > 0 && !loading;

  function submit(event: FormEvent) {
    event.preventDefault();
    if (canSubmit) {
      onSubmit(trimmed);
    }
  }

  function onKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && canSubmit) {
      onSubmit(trimmed);
    }
  }

  return (
    <form className={styles.form} onSubmit={submit}>
      <label className={styles.label} htmlFor="question">
        Tu consulta sobre alquiler de vivienda
      </label>
      <textarea
        id="question"
        className={styles.textarea}
        value={question}
        onChange={(event) => setQuestion(event.target.value)}
        onKeyDown={onKeyDown}
        placeholder={EXAMPLE}
        rows={3}
        disabled={loading}
        autoComplete="off"
      />
      <div className={styles.actions}>
        <span className={styles.hint}>
          Pulsa <kbd>Ctrl</kbd>+<kbd>Enter</kbd> para enviar
        </span>
        <button type="submit" className={styles.submit} disabled={!canSubmit}>
          {loading ? "Consultando…" : "Consultar"}
        </button>
      </div>
    </form>
  );
}
