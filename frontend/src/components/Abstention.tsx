import type { Abstention as AbstentionData } from "../types";
import styles from "./Abstention.module.css";

interface AbstentionProps {
  abstention: AbstentionData;
}

/**
 * The abstention state, presented as an honest non-answer rather than an error.
 * It shows the model's own message and the scope reminder, in calm notice
 * colours, so declining to answer reads as integrity, not failure.
 */
export function Abstention({ abstention }: AbstentionProps) {
  return (
    <section className={styles.abstention} aria-label="Sin respuesta">
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          ⓘ
        </span>
        <h2 className={styles.title}>Prefiero no responder</h2>
      </div>
      <p className={styles.message}>{abstention.message}</p>
      <p className={styles.scope}>{abstention.scope_reminder}</p>
    </section>
  );
}
