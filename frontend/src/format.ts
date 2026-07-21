/**
 * Rendering of the API's ISO dates for a Spanish-reading user. The effective date
 * of a citation and the date an answer is situated at are the same kind of fact,
 * so they are formatted the same way wherever they appear.
 */

/** An ISO `YYYY-MM-DD` as a long Spanish date, or unchanged if it is not one. */
export function formatDate(iso: string): string {
  const date = parseIso(iso);
  if (!date) {
    return iso;
  }
  return new Intl.DateTimeFormat("es-ES", { dateStyle: "long" }).format(date);
}

/** The year of an ISO `YYYY-MM-DD`, or unchanged if it is not one. */
export function formatYear(iso: string): string {
  const date = parseIso(iso);
  return date ? String(date.getFullYear()) : iso;
}

/** Parse an ISO date at local midnight, or `null` when it is not a date. */
function parseIso(iso: string): Date | null {
  const date = new Date(`${iso}T00:00:00`);
  return Number.isNaN(date.getTime()) ? null : date;
}
