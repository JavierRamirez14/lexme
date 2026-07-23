import styles from "./RetentionNotice.module.css";

/**
 * The no-retention, no-training statement that accompanies the upload flow. It is
 * always visible next to the drop zone so the reader sees, before uploading, that
 * the document is analyzed in the moment and never stored or used to train.
 */
export function RetentionNotice() {
  return (
    <p className={styles.notice}>
      <span className={styles.icon} aria-hidden="true">
        🔒
      </span>
      Tu contrato se analiza en el momento y no se guarda: no queda en disco ni en base de datos,
      y no se usa para entrenar ningún modelo.
    </p>
  );
}
