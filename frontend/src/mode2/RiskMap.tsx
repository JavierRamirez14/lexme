import { Citation } from "../components/Citation";
import type {
  AbsenceFinding,
  ClauseFinding,
  CoverageStatus,
  RiskLevel,
  RiskMap as RiskMapData,
} from "./types";
import styles from "./RiskMap.module.css";

interface RiskMapProps {
  riskMap: RiskMapData;
}

const LEVEL_LABEL: Record<RiskLevel, string> = {
  ilegal: "Ilegal o abusiva",
  peor_que_default: "Peor que la ley",
  negociable: "Negociable",
  ausente: "Protección ausente",
  correcto: "Conforme a la ley",
};

const LEVEL_ORDER: RiskLevel[] = [
  "ilegal",
  "peor_que_default",
  "negociable",
  "ausente",
  "correcto",
];

const COVERAGE_LABEL: Record<CoverageStatus, string> = {
  evaluada: "Evaluada",
  informativa: "Informativa",
  fuera_de_ambito: "Fuera de ámbito",
  no_concluyente: "No concluyente",
};

const NON_EVALUATED: CoverageStatus[] = ["informativa", "fuera_de_ambito", "no_concluyente"];

const SEVERITY: Record<RiskLevel, number> = {
  ilegal: 0,
  peor_que_default: 1,
  negociable: 2,
  correcto: 3,
  ausente: 4,
};

/**
 * The risk map: the climax of Mode 2. A recount header states the spread without
 * ever folding it into a single verdict; the omitted-protection whites lead as the
 * differentiator a flat RAG cannot produce; the evaluated clauses follow, ordered
 * by severity, each with its highlighted contract snippet, verified ELI citation,
 * explanation and bounded action; and a coverage strip shows every clause with a
 * discrete status, so there are no silent holes.
 */
export function RiskMap({ riskMap }: RiskMapProps) {
  const { clause_findings, absence_findings, level_counts, coverage_counts } = riskMap;
  const evaluated = clause_findings
    .filter((finding) => finding.level !== null)
    .sort((a, b) => SEVERITY[a.level as RiskLevel] - SEVERITY[b.level as RiskLevel]);
  const unresolved = clause_findings.filter(
    (finding) => finding.level === null && finding.coverage !== "informativa",
  );

  return (
    <section className={styles.map} aria-labelledby="riskmap-heading">
      <header className={styles.header}>
        <p className={styles.badge}>Mapa de riesgo</p>
        <h2 className={styles.title} id="riskmap-heading">
          Cada cláusula, situada frente a la ley
        </h2>
        <Recount levelCounts={level_counts} coverageCounts={coverage_counts} />
        <p className={styles.noVerdict}>
          Lexme no resume tu contrato en una nota global: cada hallazgo se juzga por separado, con su
          propia base legal. Un semáforo único daría una falsa tranquilidad.
        </p>
      </header>

      {absence_findings.length > 0 && (
        <div className={styles.absences} aria-labelledby="absences-heading">
          <h3 className={styles.sectionHeading} id="absences-heading">
            <span className={styles.symbol} aria-hidden="true" />
            Protecciones que la ley te da y tu contrato no menciona
          </h3>
          <p className={styles.sectionNote}>
            Estos derechos son tuyos por ley aunque el contrato calle. No es que falte algo: es que
            conviene que lo sepas.
          </p>
          <ol className={styles.findingList}>
            {absence_findings.map((finding) => (
              <AbsenceCard key={finding.item_id} finding={finding} />
            ))}
          </ol>
        </div>
      )}

      <div className={styles.findings}>
        <h3 className={styles.sectionHeading}>Cláusulas evaluadas</h3>
        <ol className={styles.findingList}>
          {evaluated.map((finding) => (
            <ClauseCard key={finding.clause_id} finding={finding} />
          ))}
          {unresolved.map((finding) => (
            <ClauseCard key={finding.clause_id} finding={finding} />
          ))}
        </ol>
      </div>

      <CoverageStrip findings={clause_findings} />
    </section>
  );
}

/** The header recount: every level with its count, then any non-evaluated coverage. */
function Recount({
  levelCounts,
  coverageCounts,
}: {
  levelCounts: Record<string, number>;
  coverageCounts: Record<string, number>;
}) {
  return (
    <ul className={styles.recount}>
      {LEVEL_ORDER.map((level) => (
        <li key={level} className={styles.recountItem} data-level={level}>
          <span className={styles.recountDot} aria-hidden="true" />
          <span className={styles.recountCount}>{levelCounts[level] ?? 0}</span>
          <span className={styles.recountLabel}>{LEVEL_LABEL[level]}</span>
        </li>
      ))}
      {NON_EVALUATED.filter((status) => (coverageCounts[status] ?? 0) > 0).map((status) => (
        <li key={status} className={styles.recountItem} data-coverage={status}>
          <span className={styles.recountCount}>{coverageCounts[status]}</span>
          <span className={styles.recountLabel}>{COVERAGE_LABEL[status]}</span>
        </li>
      ))}
    </ul>
  );
}

/** One evaluated or unresolved clause: badge, snippet, citation, explanation, action. */
function ClauseCard({ finding }: { finding: ClauseFinding }) {
  const level = finding.level;
  return (
    <li className={styles.finding} data-level={level ?? undefined} data-coverage={finding.coverage}>
      <div className={styles.findingHead}>
        <span className={styles.levelBadge} data-level={level ?? undefined}>
          {level ? LEVEL_LABEL[level] : COVERAGE_LABEL[finding.coverage]}
        </span>
        <span className={styles.findingHeading}>{finding.heading}</span>
      </div>
      <blockquote className={styles.snippet} data-level={level ?? undefined}>
        {finding.snippet}
      </blockquote>
      {finding.explanation && <p className={styles.explanation}>{finding.explanation}</p>}
      {finding.coverage === "fuera_de_ambito" && finding.out_of_scope_matter && (
        <p className={styles.matter}>Se rige por: {finding.out_of_scope_matter}</p>
      )}
      {finding.citation && <Citation citation={finding.citation} />}
      <Actions actions={finding.what_you_can_do} />
    </li>
  );
}

/** One absence white: the right, the static explanation and the law that grants it. */
function AbsenceCard({ finding }: { finding: AbsenceFinding }) {
  return (
    <li className={styles.finding} data-level="ausente">
      <div className={styles.findingHead}>
        <span className={styles.levelBadge} data-level="ausente">
          {LEVEL_LABEL.ausente}
        </span>
        <span className={styles.findingHeading}>{finding.right}</span>
      </div>
      <p className={styles.explanation}>{finding.explanation}</p>
      {finding.citation && <Citation citation={finding.citation} />}
    </li>
  );
}

/** The bounded "what you can do" list, shown only when there is something to do. */
function Actions({ actions }: { actions: string[] }) {
  if (actions.length === 0) {
    return null;
  }
  return (
    <div className={styles.actions}>
      <p className={styles.actionsLabel}>Qué puedes hacer</p>
      <ul className={styles.actionsList}>
        {actions.map((action) => (
          <li key={action} className={styles.action}>
            {action}
          </li>
        ))}
      </ul>
    </div>
  );
}

/** Every clause with its discrete coverage status: the completeness proof, no gaps. */
function CoverageStrip({ findings }: { findings: ClauseFinding[] }) {
  return (
    <div className={styles.coverage} aria-labelledby="coverage-heading">
      <h3 className={styles.sectionHeading} id="coverage-heading">
        Cobertura completa
      </h3>
      <p className={styles.sectionNote}>
        Ninguna cláusula se queda sin revisar. Cada una aparece aquí con su estado.
      </p>
      <ul className={styles.coverageList}>
        {findings.map((finding) => (
          <li key={finding.clause_id} className={styles.coverageItem}>
            <span className={styles.coverageName}>{finding.heading}</span>
            <span
              className={styles.coverageStatus}
              data-level={finding.level ?? undefined}
              data-coverage={finding.coverage}
            >
              {finding.level ? LEVEL_LABEL[finding.level] : COVERAGE_LABEL[finding.coverage]}
            </span>
          </li>
        ))}
      </ul>
    </div>
  );
}
