import { useState } from "react";
import { submitFeedback } from "../api/client";
import type { FeedbackMode, FeedbackVote } from "../types";
import styles from "./FeedbackButtons.module.css";

interface FeedbackButtonsProps {
  mode: FeedbackMode;
  snapshot: object;
}

type VoteState = "idle" | "sending" | { sent: FeedbackVote } | "error";

/**
 * The thumbs up/down control shown under a terminal response. Casting a vote
 * sends the full response as its snapshot -- never a bare rating -- so it lands
 * as a case candidate for the eval harness rather than a metric. A vote is final:
 * once sent, the control shows which way it went instead of letting it be redone.
 */
export function FeedbackButtons({ mode, snapshot }: FeedbackButtonsProps) {
  const [state, setState] = useState<VoteState>("idle");

  async function vote(choice: FeedbackVote) {
    if (state !== "idle" && state !== "error") return;
    setState("sending");
    try {
      await submitFeedback(mode, choice, snapshot);
      setState({ sent: choice });
    } catch {
      setState("error");
    }
  }

  if (typeof state === "object") {
    return (
      <p className={styles.thanks} role="status">
        Gracias por tu valoración.
      </p>
    );
  }

  return (
    <div className={styles.feedback}>
      <span className={styles.prompt}>¿Te ha resultado útil?</span>
      <div className={styles.buttons}>
        <button
          type="button"
          className={styles.button}
          onClick={() => void vote("positivo")}
          disabled={state === "sending"}
          aria-label="Sí, me ha resultado útil"
        >
          👍
        </button>
        <button
          type="button"
          className={styles.button}
          onClick={() => void vote("negativo")}
          disabled={state === "sending"}
          aria-label="No, no me ha resultado útil"
        >
          👎
        </button>
      </div>
      {state === "error" && (
        <span className={styles.error}>No se pudo enviar. Inténtalo de nuevo.</span>
      )}
    </div>
  );
}
