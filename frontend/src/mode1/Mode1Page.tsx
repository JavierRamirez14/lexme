import { useRef, useState } from "react";
import { askQuestion, AskError } from "../api/client";
import { AnswerView } from "../components/AnswerView";
import { Abstention } from "../components/Abstention";
import { QueryForm } from "../components/QueryForm";
import type { AskResponse } from "../types";
import styles from "./Mode1Page.module.css";

type RequestState =
  | { phase: "idle" }
  | { phase: "loading" }
  | { phase: "error"; message: string }
  | { phase: "done"; response: AskResponse };

/**
 * Mode 1: the consultation surface. Owns the request lifecycle -- idle, loading,
 * error, done -- and renders the matching state, delegating the answer and the
 * abstention to their own components. Only the latest request wins, so a fast
 * follow-up cancels the one in flight.
 */
export function Mode1Page() {
  const [state, setState] = useState<RequestState>({ phase: "idle" });
  const controllerRef = useRef<AbortController | null>(null);

  async function submit(question: string) {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ phase: "loading" });
    try {
      const response = await askQuestion(question, controller.signal);
      if (!controller.signal.aborted) {
        setState({ phase: "done", response });
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
        <h1 className={styles.title}>Consulta la ley de alquiler, con citas verificables</h1>
        <p className={styles.lede}>
          Pregunta en lenguaje natural. Cada respuesta se apoya en el texto vigente de la Ley
          de Arrendamientos Urbanos, con su cita literal y su fecha de vigencia.
        </p>
      </section>

      <QueryForm loading={state.phase === "loading"} onSubmit={submit} />

      <div className={styles.result} aria-live="polite">
        {state.phase === "loading" && (
          <div className={styles.loading}>
            <span className={styles.spinner} aria-hidden="true" />
            <span>Buscando en la ley y verificando las citas…</span>
          </div>
        )}

        {state.phase === "error" && (
          <div className={styles.error} role="alert">
            <strong>No se pudo completar la consulta.</strong>
            <span>{state.message}</span>
          </div>
        )}

        {state.phase === "done" && state.response.outcome === "respuesta" && state.response.answer && (
          <AnswerView answer={state.response.answer} />
        )}

        {state.phase === "done" &&
          state.response.outcome === "abstencion" &&
          state.response.abstention && <Abstention abstention={state.response.abstention} />}
      </div>
    </div>
  );
}
