import { AssumptionList } from "./AssumptionList";
import type { Rejection, RejectionReason } from "./types";
import styles from "./ContractStop.module.css";

interface ContractStopProps {
  rejection: Rejection;
  assumptions: string[];
}

/** The honest-stop copy per reason: a title and the kind of stop it is. */
const STOPS: Record<RejectionReason, { title: string; kind: "scope" | "unreadable" }> = {
  uso_fuera_de_ambito: { title: "Fuera del ámbito de la LAU", kind: "scope" },
  redaccion_anterior: { title: "Fuera del ámbito temporal", kind: "scope" },
  texto_no_extraible: { title: "No he podido leer el documento", kind: "unreadable" },
  formato_no_soportado: { title: "Formato no soportado", kind: "unreadable" },
  anclaje_roto: { title: "No he podido anclar las cláusulas", kind: "unreadable" },
};

/**
 * The two honest stops -- out of scope, and not analyzable -- rendered as
 * explained states, not errors. They wear calm notice colours and carry the
 * reason the pipeline gave, so declining reads as integrity rather than a fault,
 * distinct from the red banner reserved for a genuine request failure.
 */
export function ContractStop({ rejection, assumptions }: ContractStopProps) {
  const stop = STOPS[rejection.reason];
  return (
    <section className={styles.stop} aria-label={stop.title}>
      <div className={styles.header}>
        <span className={styles.icon} aria-hidden="true">
          {stop.kind === "scope" ? "⚖️" : "ⓘ"}
        </span>
        <h2 className={styles.title}>{stop.title}</h2>
      </div>
      <p className={styles.message}>{rejection.message}</p>
      <AssumptionList assumptions={assumptions} />
    </section>
  );
}
