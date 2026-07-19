import type { Answer } from "../types";
import { AnswerView } from "./AnswerView";
import styles from "./PartialAnswer.module.css";

interface PartialAnswerProps {
  answer: Answer;
}

/**
 * The partial-answer state: a full answer for the parts that grounded, framed by
 * a banner that names the parts that did not. It reuses `AnswerView` for the
 * grounded content, so a partial answer reads as an honest "here is what I can
 * support, here is what I cannot", distinct from both a full answer and a refusal.
 */
export function PartialAnswer({ answer }: PartialAnswerProps) {
  return (
    <div className={styles.partial}>
      <section className={styles.banner} aria-label="Respuesta parcial">
        <div className={styles.header}>
          <span className={styles.icon} aria-hidden="true">
            ◑
          </span>
          <h2 className={styles.title}>Respuesta parcial</h2>
        </div>
        <p className={styles.lead}>
          He podido responder con base la parte principal, pero hay puntos que no he
          logrado fundamentar y prefiero no dar por respondidos:
        </p>
        {answer.huecos_declarados.length > 0 && (
          <ul className={styles.gaps}>
            {answer.huecos_declarados.map((gap, index) => (
              <li key={index} className={styles.gap}>
                {gap}
              </li>
            ))}
          </ul>
        )}
      </section>

      <AnswerView answer={answer} />
    </div>
  );
}
