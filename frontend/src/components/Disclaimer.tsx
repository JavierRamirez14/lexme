import { useEffect, useState } from "react";
import { fetchCorpusStatus } from "../api/client";
import { formatDate } from "../format";
import styles from "./Disclaimer.module.css";

/**
 * The persistent legal footer: the "information, not legal advice" disclaimer,
 * the BOE source attribution the corpus licence requires, and the corpus's own
 * freshness date. It is always present but deliberately discreet, and present in
 * both modes since every screen here shows legal text.
 */
export function Disclaimer() {
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    void fetchCorpusStatus(controller.signal).then((status) => {
      if (status?.updated_at) {
        setUpdatedAt(status.updated_at);
      }
    });
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
      {updatedAt && (
        <p className={styles.line}>Corpus actualizado a fecha de {formatDate(updatedAt.slice(0, 10))}.</p>
      )}
    </div>
  );
}
