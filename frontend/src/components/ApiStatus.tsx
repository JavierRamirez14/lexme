import { useEffect, useState } from "react";
import { fetchHealth, type HealthStatus } from "../api/client";
import styles from "./ApiStatus.module.css";

const LABELS: Record<HealthStatus, string> = {
  checking: "Comprobando servicio",
  ok: "Servicio operativo",
  degraded: "Servicio degradado",
  unreachable: "Servicio no disponible",
};

const POLL_INTERVAL_MS = 20_000;

/**
 * A subtle, self-refreshing indicator of API reachability. It is deliberately
 * quiet: a coloured dot and a short label, never a raw status string, so a
 * healthy backend recedes and only a problem draws the eye.
 */
export function ApiStatus() {
  const [status, setStatus] = useState<HealthStatus>("checking");

  useEffect(() => {
    const controller = new AbortController();
    let active = true;

    async function probe() {
      const next = await fetchHealth(controller.signal);
      if (active) {
        setStatus(next);
      }
    }

    void probe();
    const timer = window.setInterval(() => void probe(), POLL_INTERVAL_MS);
    return () => {
      active = false;
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  return (
    <span className={styles.status} data-status={status} title={LABELS[status]}>
      <span className={styles.dot} aria-hidden="true" />
      <span className={styles.label}>{LABELS[status]}</span>
    </span>
  );
}
