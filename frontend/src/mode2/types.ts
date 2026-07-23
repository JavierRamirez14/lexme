/**
 * The Mode 2 contract-analysis contract, mirroring the API's `ContractAnalysis`.
 * Kept in sync by hand with `lexme.mode2.models`; the field names and enum string
 * values match the JSON `/contract/analyze` returns.
 */

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

export interface ContractAnalysis {
  outcome: Mode2Outcome;
  sheet: ContractSheet | null;
  summary: ExecutiveSummary | null;
  clauses: Clause[];
  rejection: Rejection | null;
  assumptions: string[];
}
