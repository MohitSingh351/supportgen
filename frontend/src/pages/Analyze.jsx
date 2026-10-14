import { useState } from "react";
import { api } from "../api.js";
import TicketResult from "../components/TicketResult.jsx";
import { ErrorNote, Loading } from "../components/ui.jsx";

const MAX_CHARS = 5000;

const SAMPLES = [
  {
    label: "Refund delay",
    subject: "Refund not received",
    body: "I requested my refund almost two weeks ago. The amount was deducted from my account but I still haven't received the refund. I already contacted support twice.",
  },
  {
    label: "Login / OTP",
    subject: "Cannot log in after password reset",
    body: "Hello, since I reset my password yesterday I cannot log in to my account. The OTP code never arrives on my phone and the app says verification failed. I need access urgently for a client meeting.",
  },
  {
    label: "Damaged product",
    subject: "Laptop arrived with a cracked screen",
    body: "The laptop I ordered last week arrived today with a cracked screen. The box looked fine from the outside. I would like to return it and get a replacement as soon as possible. What do I need to do?",
  },
];

export default function Analyze() {
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const tooLong = body.length > MAX_CHARS;
  const submit = async (e) => {
    e.preventDefault();
    if (!body.trim()) { setError(new Error("Enter the customer's ticket text first.")); return; }
    setBusy(true); setError(null);
    try {
      setResult(await api.analyze(subject, body));
    } catch (err) {
      setError(err);
    } finally {
      setBusy(false);
    }
  };

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Analyze a ticket</h1>
          <p>Paste a customer email. Our fine-tuned FLAN-T5 drafts a reply, and supporting models
            predict the category, priority, type and sentiment.</p>
        </div>
      </div>

      <div className="grid-2" style={{ alignItems: "start" }}>
        <form className="panel" onSubmit={submit}>
          <div className="samples" aria-label="Sample tickets">
            {SAMPLES.map((s) => (
              <button type="button" key={s.label} className="chip-btn"
                      onClick={() => { setSubject(s.subject); setBody(s.body); setError(null); }}>
                {s.label}
              </button>
            ))}
          </div>
          <div className="field">
            <label htmlFor="subject">Subject</label>
            <input id="subject" type="text" maxLength={300} value={subject}
                   onChange={(e) => setSubject(e.target.value)} placeholder="e.g. Refund not received" />
          </div>
          <div className="field">
            <label htmlFor="body">Customer ticket</label>
            <textarea id="body" value={body} onChange={(e) => setBody(e.target.value)}
                      placeholder="The customer's message…" />
            <div className="field-foot">
              <span>At least 10 characters</span>
              <span className={tooLong ? "over num" : "num"}>{body.length.toLocaleString()} / {MAX_CHARS.toLocaleString()}</span>
            </div>
          </div>
          <div className="btn-row">
            <button className="btn" type="submit" disabled={busy || tooLong}>
              {busy ? "Analyzing…" : "Analyze ticket"}
            </button>
            {(subject || body) && !busy && (
              <button type="button" className="btn ghost" onClick={() => { setSubject(""); setBody(""); setResult(null); setError(null); }}>
                Clear
              </button>
            )}
          </div>
          {busy && (
            <p className="small muted" style={{ marginTop: 12 }}>
              Usually about 2 seconds. The very first request after starting the server also
              downloads the summary and sentiment models, which can take several minutes.
            </p>
          )}
          {error && <div style={{ marginTop: 14 }}><ErrorNote error={error} /></div>}
        </form>

        <div>
          {busy && <div className="panel"><Loading>Generating the reply and running the classifiers…</Loading></div>}
          {!busy && result && <TicketResult ticket={result} />}
          {!busy && !result && (
            <div className="panel empty">
              <p>The suggested reply and ticket insights will appear here.</p>
              <p className="small">Try one of the sample tickets to see it in action.</p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
