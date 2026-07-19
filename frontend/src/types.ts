/**
 * The Mode 1 answer contract, mirroring the API's `AskResponse`. Kept in sync by
 * hand with `lexme.mode1.models`; the field names and enum string values match
 * the JSON the endpoint returns.
 */

export type Outcome = "respuesta" | "abstencion";

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

export interface Answer {
  fundamento: VerifiedCitation[];
  explicacion: string;
  accion: string[];
}

export type AbstentionReason = "sin_evidencia" | "sin_cita_verificable";

export interface Abstention {
  reason: AbstentionReason;
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

export interface AskResponse {
  outcome: Outcome;
  answer: Answer | null;
  abstention: Abstention | null;
  retrieval: RetrievalTrace;
  citation_verdicts: Record<string, number>;
}
