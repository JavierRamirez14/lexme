import type { ReactNode } from "react";
import { ApiStatus } from "./ApiStatus";
import { Disclaimer } from "./Disclaimer";
import { ModeNav, type ModeId } from "./ModeNav";
import styles from "./AppShell.module.css";

interface AppShellProps {
  activeMode: ModeId;
  children: ReactNode;
}

/**
 * The application frame every mode renders inside: a header carrying the brand,
 * the mode switch and the API status, a centred content column, and the
 * persistent legal footer. This is the reusable layout later modes plug into.
 */
export function AppShell({ activeMode, children }: AppShellProps) {
  return (
    <div className={styles.shell}>
      <header className={styles.header}>
        <div className={styles.headerInner}>
          <div className={styles.brand}>
            <span className={styles.wordmark}>Lexme</span>
            <span className={styles.tagline}>Arrendamientos urbanos</span>
          </div>
          <ModeNav active={activeMode} />
          <ApiStatus />
        </div>
      </header>
      <main className={styles.main}>{children}</main>
      <footer className={styles.footer}>
        <div className={styles.footerInner}>
          <Disclaimer />
        </div>
      </footer>
    </div>
  );
}
