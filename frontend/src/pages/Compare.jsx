import { useState } from "react";
import { api } from "../api.js";
import { ErrorNote, Loading, fmt, secs, useLoad } from "../components/ui.jsx";

const SAMPLE = {
  subject: "Refund not received",
  body: "I requested my refund almost two weeks ago. The amount was deducted from my account but I still haven't received the refund. I already contacted support twice.",
};

const STATUS_TEXT = { not_configured: "Not set up", unavailable: "Model unavailable", error: "Request failed" };

function ResultColumn({ r }) {
  return (
    <div className={`compare-col${r.provider === "finetuned" ? " ours" : ""}`}>
      <div>
        <h3>{r.label}</h3>
        <span className="small muted">
          {r.status === "ok" ? `${r.model}, ${secs(r.latency_ms)}` : STATUS_TEXT[r.status] || r.status}
        </span>
      </div>
      {r.status === "ok"
        ? <div className="reply">{r.text || <span className="muted">(empty output)</span>}</div>
        : <p className="small muted">{r.message}</p>}
    </div>
  );
}

function BenchmarkTable() {
  const { data } = useLoad(() => api.evaluation());
  const cmp = data?.comparison;
  if (!cmp?.metrics) return null;
  const systems = Object.keys(cmp.metrics);
  const rows = [
    ["ROUGE-1", "rouge1"], ["ROUGE-L", "rougeL"], ["BLEU", "bleu", 2],
    ["BERTScore F1 (rescaled)", "bertscore_f1_rescaled"], ["Semantic cosine", "semantic_cosine"],
    ["Average words per reply", "avg_pred_words", 1],
  ];
  return (
    <div className="panel section">
      <h2>Offline benchmark on {cmp.n_tickets} test tickets</h2>
      <p className="small muted" style={{ margin: "4px 0 12px" }}>
        Computed in Colab (notebook 06) against the real agent answers.
        {cmp.gemini ? "" : " Gemini was not configured for that run, so it is not listed."}
      </p>
      <div className="table-wrap">
        <table>
          <thead><tr><th>Metric</th>{systems.map((s) => <th key={s} className="n">{s}</th>)}</tr></thead>
          <tbody>
            {rows.map(([label, key, digits = 3]) => (
              <tr key={key}><td>{label}</td>
                {systems.map((s) => <td key={s} className="n">{fmt(cmp.metrics[s][key], digits)}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {cmp.gemini && (
        <p className="small" style={{ marginTop: 10 }}>
          Gemini: mean latency {fmt(cmp.gemini.mean_latency_s, 2)} s, {cmp.gemini.errors} failed requests
          {cmp.gemini.estimated_cost_usd != null && `, estimated cost $${fmt(cmp.gemini.estimated_cost_usd, 4)}`}.
        </p>
      )}
    </div>
  );
}

export default function Compare() {
  const [subject, setSubject] = useState(SAMPLE.subject);
  const [body, setBody] = useState(SAMPLE.body);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const run = async (e) => {
    e.preventDefault();
    setBusy(true); setError(null); setResult(null);
    try { setResult(await api.compare(subject, body)); }
    catch (err) { setError(err); }
    finally { setBusy(false); }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model comparison</h1>
          <p>The same ticket, the same instruction and the same decoding settings, sent to the original
            FLAN-T5-small, our fine-tuned version and (if configured) Gemini.</p>
        </div>
      </div>

      <form className="panel" onSubmit={run}>
        <div className="field">
          <label htmlFor="c-subject">Subject</label>
          <input id="c-subject" type="text" value={subject} onChange={(e) => setSubject(e.target.value)} />
        </div>
        <div className="field">
          <label htmlFor="c-body">Customer ticket</label>
          <textarea id="c-body" style={{ minHeight: 120 }} value={body} onChange={(e) => setBody(e.target.value)} />
        </div>
        <button className="btn" type="submit" disabled={busy}>{busy ? "Generating…" : "Compare models"}</button>
        {error && <div style={{ marginTop: 14 }}><ErrorNote error={error} /></div>}
      </form>

      <div className="section">
        {busy && <Loading>Generating with each model. The base model downloads on first use.</Loading>}
        {result && <div className="compare-cols">{result.results.map((r) => <ResultColumn key={r.provider} r={r} />)}</div>}
      </div>

      <BenchmarkTable />
    </>
  );
}
