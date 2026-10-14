import { useEffect, useState } from "react";

/** Runs an async loader on mount; returns { data, error, loading, reload }. */
export function useLoad(loader, deps = []) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const [tick, setTick] = useState(0);
  useEffect(() => {
    let alive = true;
    setState((s) => ({ ...s, loading: true, error: null }));
    loader()
      .then((data) => alive && setState({ data, error: null, loading: false }))
      .catch((error) => alive && setState({ data: null, error, loading: false }));
    return () => { alive = false; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, tick]);
  return { ...state, reload: () => setTick((t) => t + 1) };
}

export function Loading({ children = "Loading…" }) {
  return <div className="loading" role="status"><span className="spinner" />{children}</div>;
}

export function ErrorNote({ error }) {
  if (!error) return null;
  return <div className="alert error" role="alert">{error.message || String(error)}</div>;
}

const PRIORITY_STATUS = { critical: "critical", high: "serious", medium: "warning", low: "good" };

/** Priority shown as a coloured dot + text label (never colour alone). */
export function PriorityTag({ value, confidence }) {
  if (!value) return <span className="muted">—</span>;
  const status = PRIORITY_STATUS[value] || "warning";
  return (
    <span>
      <span className="tag"><span className="dot" style={{ background: `var(--st-${status})` }} />{cap(value)}</span>
      {confidence != null && <span className="conf">{pct(confidence)} confidence</span>}
    </span>
  );
}

export function Tag({ value, confidence, note }) {
  if (!value) return <span className="muted">—</span>;
  return (
    <span>
      <span className="tag">{cap(value)}</span>
      {confidence != null && <span className="conf">{pct(confidence)} confidence</span>}
      {note && <span className="conf">{note}</span>}
    </span>
  );
}

export const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
export const pct = (v, digits = 0) => (v == null ? "—" : `${(v * 100).toFixed(digits)}%`);
export const fmt = (v, digits = 3) => (v == null || Number.isNaN(Number(v)) ? "—" : Number(v).toFixed(digits));
export const secs = (ms) => (ms == null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${ms} ms`);
export const when = (iso) =>
  new Date(iso).toLocaleString(undefined, { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
