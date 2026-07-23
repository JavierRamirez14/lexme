import { useId, useRef, useState, type ChangeEvent, type DragEvent } from "react";
import styles from "./UploadZone.module.css";

interface UploadZoneProps {
  busy: boolean;
  onFile: (file: File) => void;
}

const ACCEPT = ".pdf,.docx";

/**
 * The document drop zone: drag-and-drop over a large target, or the same file
 * picker behind a button, accepting a PDF or a Word document. While an analysis
 * is in flight it shows a progress state and rejects further drops, so a second
 * file cannot race the first.
 */
export function UploadZone({ busy, onFile }: UploadZoneProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const inputId = useId();

  function pick(file: File | undefined) {
    if (file && !busy) {
      onFile(file);
    }
  }

  function onDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragging(false);
    pick(event.dataTransfer.files[0]);
  }

  function onDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!busy) setDragging(true);
  }

  function onChange(event: ChangeEvent<HTMLInputElement>) {
    pick(event.target.files?.[0]);
    event.target.value = "";
  }

  return (
    <div
      className={`${styles.zone} ${dragging ? styles.dragging : ""} ${busy ? styles.busy : ""}`}
      onDrop={busy ? undefined : onDrop}
      onDragOver={onDragOver}
      onDragLeave={() => setDragging(false)}
    >
      <input
        ref={inputRef}
        id={inputId}
        className={styles.input}
        type="file"
        accept={ACCEPT}
        onChange={onChange}
        disabled={busy}
      />
      {busy ? (
        <p className={styles.progress}>
          <span className={styles.spinner} aria-hidden="true" />
          Analizando tu contrato…
        </p>
      ) : (
        <>
          <p className={styles.headline}>Arrastra aquí tu contrato</p>
          <p className={styles.hint}>PDF con texto o Word (.docx)</p>
          <label className={styles.button} htmlFor={inputId}>
            Selecciona un archivo
          </label>
        </>
      )}
    </div>
  );
}
