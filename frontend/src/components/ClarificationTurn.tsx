import { useState, type FormEvent } from "react";
import type { Clarification } from "../types";
import { Turn } from "./Turn";
import styles from "./ClarificationTurn.module.css";

interface ClarificationTurnProps {
  clarification: Clarification;
  onAnswer: (answer: string) => void;
}

/**
 * The one question the assistant stops to ask, shown as its turn in the
 * consultation rather than as a modal: the consultation is paused, not
 * interrupted, and answering continues it in place. A date branch gets a date
 * field so the reply is unambiguous, and declining is a first-class button --
 * the run continues on a stated assumption either way.
 */
export function ClarificationTurn({ clarification, onAnswer }: ClarificationTurnProps) {
  const [answer, setAnswer] = useState("");
  const trimmed = answer.trim();
  const isDate = clarification.answer_kind === "fecha";

  function submit(event: FormEvent) {
    event.preventDefault();
    if (trimmed) {
      onAnswer(trimmed);
    }
  }

  return (
    <Turn speaker="assistant" label="Lexme necesita un dato">
      <p className={styles.question}>{clarification.question}</p>
      <form className={styles.form} onSubmit={submit}>
        <label className={styles.label} htmlFor="clarification-answer">
          {isDate ? "Fecha de firma del contrato" : "Tu respuesta"}
        </label>
        {isDate ? (
          <input
            id="clarification-answer"
            className={styles.input}
            type="date"
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
          />
        ) : (
          <textarea
            id="clarification-answer"
            className={styles.input}
            value={answer}
            onChange={(event) => setAnswer(event.target.value)}
            rows={2}
            autoComplete="off"
          />
        )}
        <div className={styles.actions}>
          <button type="button" className={styles.skip} onClick={() => onAnswer("")}>
            No lo sé / prefiero no decirlo
          </button>
          <button type="submit" className={styles.submit} disabled={!trimmed}>
            Continuar
          </button>
        </div>
      </form>
    </Turn>
  );
}
