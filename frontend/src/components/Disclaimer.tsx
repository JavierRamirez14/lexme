import { useEffect, useState } from "react";
import { fetchCorpusStatus } from "../api/client";
import { formatDate } from "../format";
import type { CorpusStatus } from "../types";
import styles from "./Disclaimer.module.css";

/**
 * The persistent legal footer: the "information, not legal advice" disclaimer,
 * the BOE source attribution the corpus licence requires, and the corpus itself --
 * every norm it holds, with how many articles and a link to that norm's own
 * consolidated text, plus the freshness date. The list is read from the API rather
 * than written here, so the scope shown is the scope actually ingested. It is
 * always present but deliberately discreet, and present in both modes since every
 * screen here shows legal text.
 */
export function Disclaimer() {
  const [status, setStatus] = useState<CorpusStatus | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchCorpusStatus(controller.signal).then(setStatus);
    return () => controller.abort();
  }, []);

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
      {status && status.norms.length > 0 && (
        <p className={styles.line}>
          Corpus
          {status.updated_at &&
            ` actualizado a fecha de ${formatDate(status.updated_at.slice(0, 10))}`}
          :{" "}
          <span className={styles.norms}>
            {status.norms.map((norm) => (
              <a
                key={norm.norm_id}
                className={styles.norm}
                href={norm.consolidated_html_url}
                target="_blank"
                rel="noreferrer"
                title={norm.title}
              >
                {norm.label} <span className={styles.count}>{norm.blocks}</span>
              </a>
            ))}
          </span>
        </p>
      )}
    </div>
  );
}
