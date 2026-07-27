import { formatDate, formatYear } from "../format";
import type { Answer } from "../types";
import styles from "./InForceBanner.module.css";

interface InForceBannerProps {
  answer: Answer;
}

const HISTORICAL_CODE = "fecha_objetivo_pasada";

/**
 * The date the answer is situated at, stated before the answer itself, plus the
 * code-derived warnings about how far that is from today's law. The single worst
 * misreading of a point-in-time answer is taking the law of then for the law of
 * now, so an answer anchored in the past is marked loudly ("Derecho vigente en
 * 2017") rather than footnoted. Whether an answer counts as historical is the
 * API's ruling, read off the notices, never recomputed here against the browser
 * clock.
 */
export function InForceBanner({ answer }: InForceBannerProps) {
  const notices = answer.avisos_vigencia;
  const historical = notices.some((notice) => notice.code === HISTORICAL_CODE);
  if (!historical && notices.length === 0) {
    return null;
  }

  return (
    <section
      className={styles.banner}
      data-historical={historical}
      aria-label="Vigencia de la respuesta"
    >
      <div className={styles.head}>
        <span className={styles.badge}>
          {historical
            ? `Derecho vigente en ${formatYear(answer.fecha_objetivo)}`
            : "Derecho vigente hoy"}
        </span>
        <span className={styles.date}>
          Resuelto a{" "}
          <time dateTime={answer.fecha_objetivo}>{formatDate(answer.fecha_objetivo)}</time>
        </span>
      </div>
      {notices.length > 0 && (
        <ul className={styles.notices}>
          {notices.map((notice, index) => (
            <li key={`${notice.code}-${notice.block_ref ?? index}`} className={styles.notice}>
              {notice.message}
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
