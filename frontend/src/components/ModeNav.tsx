import styles from "./ModeNav.module.css";

export type ModeId = "consulta" | "contrato";

interface Mode {
  id: ModeId;
  label: string;
  hint: string;
  enabled: boolean;
}

const MODES: Mode[] = [
  { id: "consulta", label: "Consulta", hint: "Pregunta en lenguaje natural", enabled: true },
  {
    id: "contrato",
    label: "Revisar contrato",
    hint: "Disponible próximamente",
    enabled: false,
  },
];

interface ModeNavProps {
  active: ModeId;
}

/**
 * The switch between the two product modes. Mode 2 is present but disabled until
 * its ticket lands, so the shell already shows where it will live without
 * pretending it works.
 */
export function ModeNav({ active }: ModeNavProps) {
  return (
    <nav className={styles.nav} aria-label="Modos">
      {MODES.map((mode) => (
        <button
          key={mode.id}
          type="button"
          className={styles.tab}
          aria-current={mode.id === active ? "page" : undefined}
          disabled={!mode.enabled}
          title={mode.hint}
        >
          <span className={styles.label}>{mode.label}</span>
          {!mode.enabled && <span className={styles.badge}>pronto</span>}
        </button>
      ))}
    </nav>
  );
}
