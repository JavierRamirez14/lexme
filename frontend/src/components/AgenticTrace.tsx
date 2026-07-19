import type { AgenticTrace as AgenticTraceData, SubQueryReport } from "../types";
import styles from "./AgenticTrace.module.css";

interface AgenticTraceProps {
  trace: AgenticTraceData;
  step: string;
  running: boolean;
}

const MAX_PASSES = 3;

const STAGES = [
  { key: "enrutando", label: "Enrutar" },
  { key: "planificando", label: "Planificar" },
  { key: "loop", label: "Recuperar y criticar" },
  { key: "sintetizando", label: "Sintetizar" },
  { key: "decidiendo", label: "Decidir" },
] as const;

const STAGE_INDEX: Record<string, number> = {
  enrutando: 0,
  planificando: 1,
  recuperando: 2,
  iterando: 2,
  sintetizando: 3,
  decidiendo: 4,
};

const QUERY_TYPE_LABEL: Record<string, string> = {
  informativa: "Consulta informativa",
  situacional: "Consulta situacional",
  procedimental: "Consulta procedimental",
};

/**
 * The live view of the agentic run: the graph stages as a timeline, the sub-query
 * decomposition with each sub-query's verdict, and the agentic delta. It reads a
 * trace snapshot, so it renders identically whether fed live SSE steps or the
 * final response -- the demo of the agency, not a raw log.
 */
export function AgenticTrace({ trace, step, running }: AgenticTraceProps) {
  const current = running ? (STAGE_INDEX[step] ?? -1) : STAGES.length;
  const passCount = trace.passes.length;

  return (
    <section className={styles.trace} aria-label="Proceso del agente">
      <ol className={styles.timeline}>
        {STAGES.map((stage, index) => (
          <li
            key={stage.key}
            className={styles.stage}
            data-state={stageState(index, current)}
            aria-current={index === current ? "step" : undefined}
          >
            <span className={styles.dot} aria-hidden="true" />
            <span className={styles.stageLabel}>
              {stage.label}
              {stage.key === "loop" && passCount > 0 && (
                <span className={styles.passBadge}>
                  pasada {Math.min(passCount, MAX_PASSES)}/{MAX_PASSES}
                </span>
              )}
            </span>
          </li>
        ))}
      </ol>

      {trace.query_type && (
        <p className={styles.queryType}>
          {QUERY_TYPE_LABEL[trace.query_type] ?? trace.query_type}
        </p>
      )}

      {trace.subqueries.length > 0 && (
        <div className={styles.subqueries}>
          <h3 className={styles.subheading}>Descomposición en sub-consultas</h3>
          <ul className={styles.subqueryList}>
            {trace.subqueries.map((sub) => (
              <SubQueryCard key={sub.id} subquery={sub} />
            ))}
          </ul>
        </div>
      )}

      {passCount > 0 && <Delta trace={trace} />}
    </section>
  );
}

/** One decomposed sub-query: its intent, criticality and current verdict. */
function SubQueryCard({ subquery }: { subquery: SubQueryReport }) {
  return (
    <li className={styles.subquery} data-verdict={subquery.verdict ?? "pendiente"}>
      <div className={styles.subqueryHead}>
        <span className={styles.role}>{subquery.is_critical ? "nuclear" : "periférica"}</span>
        <span className={styles.verdict} data-verdict={subquery.verdict ?? "pendiente"}>
          {verdictLabel(subquery.verdict)}
        </span>
      </div>
      <p className={styles.subqueryText}>{subquery.text}</p>
      <p className={styles.purpose}>{subquery.purpose}</p>
      <p className={styles.evidence}>
        {subquery.evidence.length} bloque{subquery.evidence.length === 1 ? "" : "s"} de evidencia
      </p>
    </li>
  );
}

/** The agentic delta: grounded sub-queries on the first pass versus the last. */
function Delta({ trace }: { trace: AgenticTraceData }) {
  const gained = trace.agentic_delta;
  return (
    <div className={styles.delta}>
      <span className={styles.deltaLabel}>Aporte de la iteración</span>
      <span className={styles.deltaValue}>
        {trace.first_pass_sufficient} → {trace.final_sufficient} sub-consultas con base
        {gained > 0 && <strong className={styles.deltaGain}> (+{gained})</strong>}
      </span>
    </div>
  );
}

/** Whether a timeline stage is done, active or still pending. */
function stageState(index: number, current: number): "done" | "active" | "pending" {
  if (index < current) return "done";
  if (index === current) return "active";
  return "pending";
}

/** The human label for a sub-query verdict, including the not-yet-judged case. */
function verdictLabel(verdict: SubQueryReport["verdict"]): string {
  if (verdict === "suficiente") return "con base";
  if (verdict === "insuficiente") return "sin base";
  return "pendiente";
}
