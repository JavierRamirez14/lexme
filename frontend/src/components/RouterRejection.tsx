import type { RouterRejection as RouterRejectionData } from "../types";
import styles from "./RouterRejection.module.css";

interface RouterRejectionProps {
  rejection: RouterRejectionData;
}

/**
 * The out-of-scope state: the router declined before any retrieval because the
 * question is not about the state LAU. It is deliberately distinct from an
 * abstention -- nothing was searched -- so it reads as "wrong door", redirecting
 * the user to what this assistant does cover rather than as a failure to answer.
 */
export function RouterRejection({ rejection }: RouterRejectionProps) {
  return (
    <section className={styles.rejection} aria-label="Fuera de ámbito">
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          ⤺
        </span>
        <h2 className={styles.title}>Esto queda fuera de mi ámbito</h2>
      </div>
      <p className={styles.message}>{rejection.message}</p>
      <p className={styles.scope}>{rejection.scope_reminder}</p>
    </section>
  );
}
