# Lexme frontend

React + Vite SPA. This is the **first user surface**, so it also fixes the design
foundation that later mode tickets (06/07/09/10/11) reuse rather than reinvent.

## Styling approach

Deliberately light on dependencies: **no UI framework, no CSS-in-JS runtime**.
Just two layers, both zero extra runtime cost.

1. **Design tokens** — `src/styles/tokens.css` defines every colour, type step
   and spacing value as CSS custom properties on `:root`, with a
   `prefers-color-scheme: dark` block that re-points the same variables. Dark mode
   therefore needs no per-component code.
2. **CSS Modules** — each component owns a `Component.module.css` next to it.
   Class names are locally scoped by Vite (built in, no plugin), so there are no
   global collisions and no naming convention to police. Modules **only read
   tokens**; they never hardcode a colour or a raw pixel value.

`src/styles/global.css` holds the light reset plus document-wide typography and
focus styles. It is imported once in `main.tsx`.

### Rules for new components

- Read tokens; do not introduce new literal colours or ad-hoc spacing. If a value
  is genuinely missing, add a token.
- One CSS Module per component, colocated.
- Keep legal text legible: quoted law is set in the serif token
  (`--font-serif`); UI chrome uses the sans token.
- Style for both themes by using tokens; avoid `prefers-color-scheme` in
  component modules.

## Structure

```
src/
  api/client.ts        API access (the only place that calls fetch)
  types.ts             TS mirror of the API's AskResponse contract
  components/          shared, mode-agnostic UI (shell, nav, status, disclaimer,
                       citation, answer, abstention, query form)
  mode1/               the Mode 1 (Consulta) surface and its request lifecycle
  styles/              tokens + global base
```

The `AppShell` (header with mode nav + API status, centred content, persistent
legal footer) is the frame every mode renders inside.

## Develop

```
npm install
npm run dev        # http://localhost:5173, expects the API on :8000
npm run build      # type-check (tsc -b) + production build
```

`VITE_API_URL` overrides the API base URL (defaults to `http://localhost:8000`).
