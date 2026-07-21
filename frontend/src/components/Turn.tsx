import type { ReactNode } from "react";
import styles from "./Turn.module.css";

interface TurnProps {
  speaker: "user" | "assistant";
  label: string;
  children: ReactNode;
}

/**
 * One turn of the consultation, attributed to whoever said it. A consultation
 * that stops to ask something only reads as a conversation if both sides are
 * shown the same way, so the user's question and the assistant's are the same
 * component with a different speaker.
 */
export function Turn({ speaker, label, children }: TurnProps) {
  return (
    <div className={styles.turn} data-speaker={speaker}>
      <span className={styles.label}>{label}</span>
      <div className={styles.body}>{children}</div>
    </div>
  );
}
