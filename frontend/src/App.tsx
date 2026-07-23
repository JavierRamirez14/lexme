import { useState } from "react";
import { AppShell } from "./components/AppShell";
import type { ModeId } from "./components/ModeNav";
import { Mode1Page } from "./mode1/Mode1Page";
import { Mode2Page } from "./mode2/Mode2Page";

/**
 * The application root: the reusable shell wrapping the active mode. It owns which
 * mode is selected and renders the matching surface inside the shell.
 */
export function App() {
  const [mode, setMode] = useState<ModeId>("consulta");
  return (
    <AppShell activeMode={mode} onSelectMode={setMode}>
      {mode === "consulta" ? <Mode1Page /> : <Mode2Page />}
    </AppShell>
  );
}
