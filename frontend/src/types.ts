/**
 * The Mode 1 answer contract, mirroring the API's `AskResponse`. Kept in sync by
 * hand with `lexme.mode1.models`; the field names and enum string values match
 * the JSON the endpoint returns, for both the blocking `/ask` and the streamed
 * `/ask/stream` (whose `step` events carry an `AgenticTrace` snapshot).
 */

export type Outcome =
  | "respuesta"
  | "respuesta_parcial"
  | "abstencion"
  | "rechazo_router"
  | "desambiguacion";

export type QueryType = "informativa" | "situacional" | "procedimental";

export type SubQueryVerdict = "suficiente" | "insuficiente";

export type CitationVerdict =
  | "verificada_directa"
  | "reparada_snap"
  | "reparada_anclaje"
  | "descartada";

export interface VerifiedAnchor {
  eli: string;
  consolidated_html_url: string;
  block_id: string;
  title: string;
  effective_date: string;
}

export interface VerifiedCitation {
  block_id: string;
  text: string;
  verdict: CitationVerdict;
  anchor: VerifiedAnchor;
}

export type NoticeCode =
  | "fecha_objetivo_pasada"
  | "redaccion_superada"
  | "modificacion_reciente";

export interface InForceNotice {
  code: NoticeCode;
  message: string;
  block_id: string | null;
}

export interface Answer {
  fundamento: VerifiedCitation[];
  explicacion: string;
  accion: string[];
  fecha_objetivo: string;
  avisos_vigencia: InForceNotice[];
  asunciones: string[];
  huecos_declarados: string[];
}

export type AnswerKind = "fecha" | "texto";

export interface Clarification {
  branch_id: string;
  question: string;
  answer_kind: AnswerKind;
}

export type AbstentionReason =
  | "sin_evidencia"
  | "sin_cita_verificable"
  | "sin_base_nuclear";

export interface Abstention {
  reason: AbstentionReason;
  message: string;
  scope_reminder: string;
}

export interface RouterRejection {
  message: string;
  scope_reminder: string;
}

export interface RankedBlockRef {
  norm_id: string;
  block_id: string;
}

export interface RetrievalTrace {
  dense: RankedBlockRef[];
  lexical: RankedBlockRef[];
  fused: RankedBlockRef[];
}

export interface SubQueryReport {
  id: string;
  text: string;
  purpose: string;
  is_critical: boolean;
  verdict: SubQueryVerdict | null;
  evidence: RankedBlockRef[];
  retrieval: RetrievalTrace | null;
}

export interface PassReport {
  pass_number: number;
  sufficient_ids: string[];
  insufficient_ids: string[];
  evidence_count: number;
}

export interface AgenticTrace {
  query_type: QueryType | null;
  subqueries: SubQueryReport[];
  passes: PassReport[];
  first_pass_sufficient: number;
  final_sufficient: number;
  agentic_delta: number;
}

export interface AskResponse {
  outcome: Outcome;
  thread_id: string;
  answer: Answer | null;
  abstention: Abstention | null;
  rejection: RouterRejection | null;
  clarification: Clarification | null;
  agentic: AgenticTrace | null;
  citation_verdicts: Record<string, number>;
}
