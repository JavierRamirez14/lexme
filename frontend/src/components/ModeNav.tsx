import styles from "./ModeNav.module.css";

export type ModeId = "consulta" | "contrato";

interface Mode {
  id: ModeId;
  label: string;
  hint: string;
}

const MODES: Mode[] = [
  { id: "consulta", label: "Consulta", hint: "Pregunta en lenguaje natural" },
  { id: "contrato", label: "Revisar contrato", hint: "Sube tu contrato y obtén su ficha" },
];

interface ModeNavProps {
  active: ModeId;
  onSelect: (mode: ModeId) => void;
}

/**
 * The switch between the two product modes. Both are live now; selecting one
 * swaps the surface rendered inside the shell. The active tab is marked for
 * assistive tech with `aria-current`.
 */
export function ModeNav({ active, onSelect }: ModeNavProps) {
  return (
    <nav className={styles.nav} aria-label="Modos">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          type="button"
          className={styles.tab}
          aria-current={mode.id === active ? "page" : undefined}
          title={mode.hint}
          onClick={() => onSelect(mode.id)}
        >
          <span className={styles.label}>{mode.label}</span>
        </button>
      ))}
    </nav>
  );
}
