import type { CitationVerdict, VerifiedCitation } from "../types";
import styles from "./Citation.module.css";

const VERDICT_LABELS: Record<CitationVerdict, string> = {
  verificada_directa: "Verificada",
  reparada_snap: "Verificada (ajustada)",
  reparada_anclaje: "Verificada (reubicada)",
  descartada: "Descartada",
};

function formatDate(iso: string): string {
  const date = new Date(`${iso}T00:00:00`);
  if (Number.isNaN(date.getTime())) {
    return iso;
  }
  return new Intl.DateTimeFormat("es-ES", { dateStyle: "long" }).format(date);
}

interface CitationProps {
  citation: VerifiedCitation;
}

/**
 * One verified citation: the literal quoted text set apart in a serif face, its
 * article title, the effective date of the cited redaction, and distinct links
 * to the ELI and the consolidated BOE text. The verified badge and the source
 * links make the guarantee -- this text really is in the law -- readable at a
 * glance.
 */
export function Citation({ citation }: CitationProps) {
  const { anchor, verdict, text } = citation;
  return (
    <figure className={styles.citation}>
      <div className={styles.head}>
        <span className={styles.article}>{anchor.title}</span>
        <span className={styles.badge} data-verdict={verdict}>
          {VERDICT_LABELS[verdict]}
        </span>
      </div>
      <blockquote className={styles.quote}>{text}</blockquote>
      <figcaption className={styles.meta}>
        <span className={styles.vigencia}>
          Vigente desde <time dateTime={anchor.effective_date}>{formatDate(anchor.effective_date)}</time>
        </span>
        <span className={styles.links}>
          <a className={styles.link} href={anchor.eli} target="_blank" rel="noreferrer">
            ELI
          </a>
          <a
            className={styles.link}
            href={anchor.consolidated_html_url}
            target="_blank"
            rel="noreferrer"
          >
            Texto consolidado
          </a>
        </span>
      </figcaption>
    </figure>
  );
}
