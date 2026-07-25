import { useRef, useState } from "react";
import { askQuestionStream, resumeQuestionStream, AskError } from "../api/client";
import type { StreamHandlers } from "../api/client";
import { AgenticTrace } from "../components/AgenticTrace";
import { AnswerView } from "../components/AnswerView";
import { Abstention } from "../components/Abstention";
import { ClarificationTurn } from "../components/ClarificationTurn";
import { FeedbackButtons } from "../components/FeedbackButtons";
import { PartialAnswer } from "../components/PartialAnswer";
import { QueryForm } from "../components/QueryForm";
import { RouterRejection } from "../components/RouterRejection";
import { Turn } from "../components/Turn";
import type { AgenticTrace as AgenticTraceData, AskResponse, Clarification } from "../types";
import styles from "./Mode1Page.module.css";

type RequestState =
  | { phase: "idle" }
  | { phase: "streaming"; step: string; trace: AgenticTraceData }
  | { phase: "asking"; clarification: Clarification; threadId: string; trace: AgenticTraceData }
  | { phase: "done"; response: AskResponse; trace: AgenticTraceData | null }
  | { phase: "error"; message: string };

/** The disambiguating question and the reply to it, kept once it has been answered. */
interface Exchange {
  clarification: Clarification;
  answer: string;
}

const EMPTY_TRACE: AgenticTraceData = {
  query_type: null,
  subqueries: [],
  passes: [],
  first_pass_sufficient: 0,
  final_sufficient: 0,
  agentic_delta: 0,
};

/**
 * Mode 1: the consultation surface, read as a conversation. It streams the
 * agentic run over SSE, showing the graph work live, and then either renders the
 * terminal state -- a cited answer, a partial answer, an honest abstention or an
 * out-of-scope rejection -- or, when the run paused to ask one disambiguating
 * question, shows that question as the assistant's turn and resumes the very same
 * run in place once it is answered. The exchange stays on screen after the reply,
 * so the answer is read against the case it was narrowed to. Only the latest
 * request wins, so a fast follow-up cancels the one in flight.
 */
export function Mode1Page() {
  const [question, setQuestion] = useState<string | null>(null);
  const [exchange, setExchange] = useState<Exchange | null>(null);
  const [state, setState] = useState<RequestState>({ phase: "idle" });
  const controllerRef = useRef<AbortController | null>(null);

  /** Stream one leg of a consultation and route wherever it ends. */
  async function run(leg: (handlers: StreamHandlers, signal: AbortSignal) => Promise<void>) {
    controllerRef.current?.abort();
    const controller = new AbortController();
    controllerRef.current = controller;
    setState({ phase: "streaming", step: "inicio", trace: EMPTY_TRACE });

    let latestTrace: AgenticTraceData = EMPTY_TRACE;
    try {
      await leg(
        {
          onStep: (step, agentic) => {
            latestTrace = agentic;
            if (!controller.signal.aborted) {
              setState({ phase: "streaming", step, trace: agentic });
            }
          },
          onResult: (response) => {
            if (!controller.signal.aborted) {
              setState(resultState(response, latestTrace));
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

  async function submit(asked: string) {
    setQuestion(asked);
    setExchange(null);
    await run((handlers, signal) => askQuestionStream(asked, handlers, signal));
  }

  async function answerClarification(
    clarification: Clarification,
    threadId: string,
    answer: string,
  ) {
    setExchange({ clarification, answer });
    await run((handlers, signal) => resumeQuestionStream(threadId, answer, handlers, signal));
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

      <div className={styles.conversation} aria-live="polite">
        {question !== null && state.phase !== "idle" && (
          <Turn speaker="user" label="Tu consulta">
            {question}
          </Turn>
        )}

        {exchange && (
          <>
            <Turn speaker="assistant" label="Lexme necesita un dato">
              {exchange.clarification.question}
            </Turn>
            <Turn speaker="user" label="Tu respuesta">
              {exchange.answer || "Prefiero no decirlo."}
            </Turn>
          </>
        )}

        {state.phase === "streaming" && (
          <div className={styles.running}>
            <p className={styles.runningCaption}>
              <span className={styles.spinner} aria-hidden="true" />
              Procesando tu consulta…
            </p>
            <AgenticTrace trace={state.trace} step={state.step} running />
          </div>
        )}

        {state.phase === "asking" && (
          <div className={styles.done}>
            <AgenticTrace trace={state.trace} step="done" running={false} />
            <ClarificationTurn
              clarification={state.clarification}
              onAnswer={(answer) =>
                answerClarification(state.clarification, state.threadId, answer)
              }
            />
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

/** Map a finished stream to the state that renders it: paused to ask, or done. */
function resultState(response: AskResponse, trace: AgenticTraceData): RequestState {
  if (response.outcome === "desambiguacion" && response.clarification) {
    return {
      phase: "asking",
      clarification: response.clarification,
      threadId: response.thread_id,
      trace: response.agentic ?? trace,
    };
  }
  return { phase: "done", response, trace: response.agentic ?? trace };
}

/** Render a finished run: the process trace (when any) above its terminal state. */
function Result({ response, trace }: { response: AskResponse; trace: AgenticTraceData | null }) {
  return (
    <div className={styles.done}>
      {trace && <AgenticTrace trace={trace} step="done" running={false} />}
      <Terminal response={response} />
      <FeedbackButtons mode="consulta" snapshot={response} />
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
