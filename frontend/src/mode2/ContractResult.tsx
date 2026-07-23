import { AssumptionList } from "./AssumptionList";
import { RiskMap } from "./RiskMap";
import type { ContractAnalysis } from "./types";
import styles from "./ContractResult.module.css";

interface ContractResultProps {
  analysis: ContractAnalysis;
}

/**
 * The good side of the gates: the contract's photo, then its risk map. The
 * executive summary leads, then the ficha as a field grid; the risk map places
 * every clause on the five-level spectrum and surfaces the protections the
 * contract omits. Assumptions the system made -- an assumed use, an assumed
 * signing date -- are stated at the end, never hidden.
 */
export function ContractResult({ analysis }: ContractResultProps) {
  const { sheet, summary, risk_map, assumptions } = analysis;
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

      {risk_map && <RiskMap riskMap={risk_map} />}

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
