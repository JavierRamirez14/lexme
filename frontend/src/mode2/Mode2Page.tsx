import { useRef, useState } from "react";
import { analyzeContract, AskError } from "../api/client";
import { FeedbackButtons } from "../components/FeedbackButtons";
import { ContractResult } from "./ContractResult";
import { ContractStop } from "./ContractStop";
import { RetentionNotice } from "./RetentionNotice";
import { UploadZone } from "./UploadZone";
import type { ContractAnalysis } from "./types";
import styles from "./Mode2Page.module.css";

type PageState =
  | { phase: "idle" }
  | { phase: "uploading" }
  | { phase: "done"; analysis: ContractAnalysis }
  | { phase: "error"; message: string };

/**
 * Mode 2: the contract surface. A tenant uploads a lease and gets one honest
 * verdict back -- the ficha, executive summary and code-anchored clauses when the
 * document is readable and in scope, or an explained stop (out of scope, or not
 * analyzable) that is deliberately not a red error. The no-retention statement
 * sits with the upload the whole time. Only the latest upload wins, so a second
 * file cancels the one in flight.
 */
export function Mode2Page() {
  const [state, setState] = useState<PageState>({ phase: "idle" });
  const controllerRef = useRef<AbortController | null>(null);

  async function upload(file: File) {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ phase: "uploading" });
    try {
      const analysis = await analyzeContract(file, controller.signal);
      if (!controller.signal.aborted) {
        setState({ phase: "done", analysis });
      }
    } catch (error) {
      if (controller.signal.aborted) {
        return;
      }
      const message =
        error instanceof AskError ? error.message : "Se produjo un error inesperado.";
      setState({ phase: "error", message });
    }
  }

  return (
    <div className={styles.page}>
      <section className={styles.intro}>
        <h1 className={styles.title}>Revisa tu contrato de alquiler</h1>
        <p className={styles.lede}>
          Sube tu contrato y Lexme extrae su ficha, ancla cada cláusula al texto literal del
          documento y comprueba que está dentro del ámbito de la Ley de Arrendamientos Urbanos antes
          de analizarlo.
        </p>
      </section>

      <div className={styles.upload}>
        <UploadZone busy={state.phase === "uploading"} onFile={upload} />
        <RetentionNotice />
      </div>

      {state.phase === "error" && (
        <div className={styles.error} role="alert">
          <strong>No se pudo completar el análisis.</strong>
          <span>{state.message}</span>
        </div>
      )}

      {state.phase === "done" && <Outcome analysis={state.analysis} />}
    </div>
  );
}

/** Route a finished analysis to its view: the contract's photo, or an honest stop. */
function Outcome({ analysis }: { analysis: ContractAnalysis }) {
  return (
    <>
      <Verdict analysis={analysis} />
      <FeedbackButtons mode="contrato" snapshot={analysis} />
    </>
  );
}

/** The analyzed ficha and risk map, or an honest stop -- never both. */
function Verdict({ analysis }: { analysis: ContractAnalysis }) {
  if (analysis.outcome === "analizado") {
    return <ContractResult analysis={analysis} />;
  }
  if (analysis.rejection) {
    return <ContractStop rejection={analysis.rejection} assumptions={analysis.assumptions} />;
  }
  return null;
}
