import styles from "./Disclaimer.module.css";

/**
 * The persistent legal footer: the "information, not legal advice" disclaimer and
 * the BOE source attribution the corpus licence requires. It is always present
 * but deliberately discreet; a later ticket enriches it with corpus freshness.
 */
export function Disclaimer() {
  return (
    <div className={styles.disclaimer}>
      <p className={styles.line}>
        Lexme ofrece <strong>información general, no asesoramiento jurídico</strong>. Para tu
        caso concreto, consulta con profesionales.
      </p>
      <p className={styles.line}>
        Textos legales consolidados de carácter meramente informativo, procedentes de la{" "}
        <a href="https://www.boe.es" target="_blank" rel="noreferrer">
          Agencia Estatal BOE
        </a>
        .
      </p>
    </div>
  );
}
