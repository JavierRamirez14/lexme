import { useRef, useState } from "react";
import { askQuestionStream, AskError } from "../api/client";
import { AgenticTrace } from "../components/AgenticTrace";
import { AnswerView } from "../components/AnswerView";
import { Abstention } from "../components/Abstention";
import { PartialAnswer } from "../components/PartialAnswer";
import { QueryForm } from "../components/QueryForm";
import { RouterRejection } from "../components/RouterRejection";
import type { AgenticTrace as AgenticTraceData, AskResponse } from "../types";
import styles from "./Mode1Page.module.css";

type RequestState =
  | { phase: "idle" }
  | { phase: "streaming"; step: string; trace: AgenticTraceData }
  | { phase: "done"; response: AskResponse; trace: AgenticTraceData | null }
  | { phase: "error"; message: string };

const EMPTY_TRACE: AgenticTraceData = {
  query_type: null,
  subqueries: [],
  passes: [],
  first_pass_sufficient: 0,
  final_sufficient: 0,
  agentic_delta: 0,
};

/**
 * Mode 1: the consultation surface. Streams the agentic run over SSE, showing the
 * graph work live, then renders the terminal state -- a cited answer, a partial
 * answer, an honest abstention or an out-of-scope rejection -- each in its own
 * component. Only the latest request wins, so a fast follow-up cancels the one in
 * flight.
 */
export function Mode1Page() {
  const [state, setState] = useState<RequestState>({ phase: "idle" });
  const controllerRef = useRef<AbortController | null>(null);

  async function submit(question: string) {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ phase: "streaming", step: "inicio", trace: EMPTY_TRACE });

    let latestTrace: AgenticTraceData = EMPTY_TRACE;
    try {
      await askQuestionStream(
        question,
        {
          onStep: (step, agentic) => {
            latestTrace = agentic;
            if (!controller.signal.aborted) {
              setState({ phase: "streaming", step, trace: agentic });
            }
          },
          onResult: (response) => {
            if (!controller.signal.aborted) {
              setState({ phase: "done", response, trace: response.agentic ?? latestTrace });
            }
          },
        },
        controller.signal,
      );
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
          Pregunta en lenguaje natural. El asistente descompone tu consulta, busca en el texto
          vigente de la Ley de Arrendamientos Urbanos y verifica cada cita antes de responder.
        </p>
      </section>

      <QueryForm loading={state.phase === "streaming"} onSubmit={submit} />

      <div className={styles.result} aria-live="polite">
        {state.phase === "streaming" && (
          <div className={styles.running}>
            <p className={styles.runningCaption}>
              <span className={styles.spinner} aria-hidden="true" />
              Procesando tu consulta…
            </p>
            <AgenticTrace trace={state.trace} step={state.step} running />
          </div>
        )}

        {state.phase === "error" && (
          <div className={styles.error} role="alert">
            <strong>No se pudo completar la consulta.</strong>
            <span>{state.message}</span>
          </div>
        )}

        {state.phase === "done" && <Result response={state.response} trace={state.trace} />}
      </div>
    </div>
  );
}

/** Render a finished run: the process trace (when any) above its terminal state. */
function Result({ response, trace }: { response: AskResponse; trace: AgenticTraceData | null }) {
  return (
    <div className={styles.done}>
      {trace && <AgenticTrace trace={trace} step="done" running={false} />}
      <Terminal response={response} />
    </div>
  );
}

/** Pick the component for the run's terminal outcome. */
function Terminal({ response }: { response: AskResponse }) {
  if (response.outcome === "respuesta" && response.answer) {
    return <AnswerView answer={response.answer} />;
  }
  if (response.outcome === "respuesta_parcial" && response.answer) {
    return <PartialAnswer answer={response.answer} />;
  }
  if (response.outcome === "abstencion" && response.abstention) {
    return <Abstention abstention={response.abstention} />;
  }
  if (response.outcome === "rechazo_router" && response.rejection) {
    return <RouterRejection rejection={response.rejection} />;
  }
  return null;
}
