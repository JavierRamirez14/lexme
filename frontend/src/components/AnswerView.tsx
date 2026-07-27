import type { Answer } from "../types";
import { Citation } from "./Citation";
import { InForceBanner } from "./InForceBanner";
import styles from "./AnswerView.module.css";

interface AnswerViewProps {
  answer: Answer;
}

/**
 * The three-layer answer, in descending authority: the cited foundation first
 * and most prominent, then the plain-language explanation, then the bounded
 * "what you can do" steps. The order and weight make the hierarchy legible --
 * everything downstream rests on the verified citations at the top. When the
 * answer is not situated in today's law, the in-force banner says so above all
 * of it, before the reader has taken any of it as current.
 */
export function AnswerView({ answer }: AnswerViewProps) {
  return (
    <article className={styles.answer}>
      <InForceBanner answer={answer} />

      <section className={styles.layer} aria-labelledby="fundamento-heading">
        <h2 className={styles.foundationHeading} id="fundamento-heading">
          Fundamento
        </h2>
        <p className={styles.layerNote}>Citas literales verificadas contra el texto vigente.</p>
        <div className={styles.citations}>
          {answer.fundamento.map((citation, index) => (
            <Citation key={`${citation.block_ref}-${index}`} citation={citation} />
          ))}
        </div>
      </section>

      <section className={styles.layer} aria-labelledby="explicacion-heading">
        <h2 className={styles.heading} id="explicacion-heading">
          En lenguaje llano
        </h2>
        <p className={styles.explanation}>{answer.explicacion}</p>
      </section>

      {answer.accion.length > 0 && (
        <section className={styles.layer} aria-labelledby="accion-heading">
          <h2 className={styles.heading} id="accion-heading">
            Qué puedes hacer
          </h2>
          <ul className={styles.actions}>
            {answer.accion.map((step, index) => (
              <li key={index} className={styles.action}>
                {step}
              </li>
            ))}
          </ul>
        </section>
      )}

      {answer.asunciones.length > 0 && (
        <section className={styles.assumptions} aria-labelledby="asunciones-heading">
          <h2 className={styles.assumptionsHeading} id="asunciones-heading">
            Asunciones que he hecho
          </h2>
          <ul className={styles.assumptionList}>
            {answer.asunciones.map((assumption, index) => (
              <li key={index} className={styles.assumption}>
                {assumption}
              </li>
            ))}
          </ul>
        </section>
      )}
    </article>
  );
}
