import { AppShell } from "./components/AppShell";
import { Mode1Page } from "./mode1/Mode1Page";

/**
 * The application root: the reusable shell wrapping the active mode. Mode 1 is
 * the only surface for now; later modes render inside the same shell.
 */
export function App() {
  return (
    <AppShell activeMode="consulta">
      <Mode1Page />
    </AppShell>
  );
}
