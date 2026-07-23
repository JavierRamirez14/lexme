/**
 * The Mode 2 contract-analysis contract, mirroring the API's `ContractAnalysis`.
 * Kept in sync by hand with `lexme.mode2.models`; the field names and enum string
 * values match the JSON `/contract/analyze` returns.
 */

import type { VerifiedCitation } from "../types";

export type Mode2Outcome = "analizado" | "fuera_de_ambito" | "no_analizable";

export type RejectionReason =
  | "texto_no_extraible"
  | "anclaje_roto"
  | "formato_no_soportado"
  | "uso_fuera_de_ambito"
  | "redaccion_anterior";

export interface ContractSheet {
  arrendador: string;
  arrendatario: string;
  inmueble: string;
  renta: string;
  duracion: string;
  fianza: string;
  fecha_firma: string;
}

export interface Clause {
  id: string;
  heading: string;
  text: string;
  start: number;
  end: number;
}

export interface SummaryItem {
  label: string;
  value: string;
}

export interface ExecutiveSummary {
  headline: string;
  items: SummaryItem[];
  clause_count: number;
}

export interface Rejection {
  reason: RejectionReason;
  message: string;
}

/** The five-colour risk spectrum. `ausente` is the white carried only by absences. */
export type RiskLevel =
  | "ilegal"
  | "peor_que_default"
  | "negociable"
  | "ausente"
  | "correcto";

/** Why a clause sits where it does, so coverage is never a silent hole. */
export type CoverageStatus =
  | "evaluada"
  | "informativa"
  | "fuera_de_ambito"
  | "no_concluyente";

export interface ClauseFinding {
  clause_id: string;
  heading: string;
  snippet: string;
  start: number;
  end: number;
  coverage: CoverageStatus;
  level: RiskLevel | null;
  explanation: string;
  what_you_can_do: string[];
  citation: VerifiedCitation | null;
  out_of_scope_matter: string;
  chk_ids: string[];
  related_clause_ids: string[];
}

export interface AbsenceFinding {
  item_id: string;
  right: string;
  level: RiskLevel;
  silence_tone: string;
  explanation: string;
  citation: VerifiedCitation | null;
}

export interface RiskMap {
  clause_findings: ClauseFinding[];
  absence_findings: AbsenceFinding[];
  level_counts: Record<string, number>;
  coverage_counts: Record<string, number>;
}

export interface ContractAnalysis {
  outcome: Mode2Outcome;
  sheet: ContractSheet | null;
  summary: ExecutiveSummary | null;
  clauses: Clause[];
  risk_map: RiskMap | null;
  rejection: Rejection | null;
  assumptions: string[];
}
