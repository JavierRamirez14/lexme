import { AssumptionList } from "./AssumptionList";
import type { ContractAnalysis } from "./types";
import styles from "./ContractResult.module.css";

interface ContractResultProps {
  analysis: ContractAnalysis;
}

/**
 * The good side of the gates: the contract's photo. The executive summary leads,
 * then the ficha as a field grid, then the segmented clauses as the scaffolding
 * the risk map (ticket 10) will populate. Assumptions the system made -- an
 * assumed use, an assumed signing date -- are stated at the end, never hidden.
 */
export function ContractResult({ analysis }: ContractResultProps) {
  const { sheet, summary, clauses, assumptions } = analysis;
  if (!sheet || !summary) {
    return null;
  }

  return (
    <article className={styles.result}>
      <section className={styles.summary} aria-labelledby="summary-heading">
        <p className={styles.badge}>Foto del documento</p>
        <h2 className={styles.headline} id="summary-heading">
          {summary.headline}
        </h2>
        <dl className={styles.grid}>
          {summary.items.map((item) => (
            <div key={item.label} className={styles.field}>
              <dt className={styles.term}>{item.label}</dt>
              <dd className={styles.value}>{item.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      <section className={styles.clauses} aria-labelledby="clauses-heading">
        <h3 className={styles.heading} id="clauses-heading">
          Cláusulas detectadas
          <span className={styles.count}>{clauses.length}</span>
        </h3>
        <p className={styles.note}>
          Cada cláusula está anclada por código a un fragmento literal de tu documento. El mapa de
          riesgo llegará sobre estas mismas cláusulas.
        </p>
        <ol className={styles.clauseList}>
          {clauses.map((clause) => (
            <li key={clause.id} className={styles.clause}>
              <p className={styles.clauseHeading}>{clause.heading}</p>
              <p className={styles.clauseText}>{clause.text}</p>
            </li>
          ))}
        </ol>
      </section>

      {assumptions.length > 0 && (
        <section className={styles.assumptions} aria-labelledby="assumptions-heading">
          <h3 className={styles.assumptionsHeading} id="assumptions-heading">
            Asunciones que he hecho
          </h3>
          <AssumptionList assumptions={assumptions} />
        </section>
      )}
    </article>
  );
}
