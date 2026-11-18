import { Link } from "react-router-dom";
import { api } from "../api.js";
import { fmt, useLoad } from "../components/ui.jsx";

const A = "A_base_flan-t5-small";
const B = "B_ft_flan-t5-small";

const BIG_PICTURE = [
  { title: "Data", where: "Hugging Face", text: "Thousands of example support tickets, each with the reply a real agent wrote." },
  { title: "Training", where: "Google Colab (free GPU)", text: "We fine-tune FLAN-T5 so it learns to write support replies." },
  { title: "Backend", where: "Django on a laptop", text: "Loads the trained models and runs them on every new ticket." },
  { title: "Website", where: "React", text: "The agent pastes a ticket and gets a draft reply and insights." },
];

const PIPELINE = [
  ["Draft reply", "FLAN-T5-small, fine-tuned by us", "Yes, the main GenAI part"],
  ["Category, priority, type", "TF-IDF + Linear SVM", "Yes, trained by us"],
  ["Sentiment", "Pretrained RoBERTa sentiment model", "No, used as-is"],
  ["Summary", "FLAN-T5-base (zero-shot), key sentences as a fallback", "No"],
  ["Recommended action", "Simple rules based on category and priority", "No, rules"],
];

const PAGES = [
  ["/analyze", "Analyze ticket", "Paste a ticket and get the draft reply, summary, action and labels."],
  ["/compare", "Model comparison", "The same ticket through the original and our fine-tuned model, side by side."],
  ["/evaluation", "Model evaluation", "All scores from training and testing, explained."],
  ["/history", "Ticket history", "Every ticket analysed so far, saved in the database."],
  ["/", "Dashboard", "Charts of all analysed tickets: categories, priority, sentiment."],
];

function Step({ n, title, children }) {
  return (
    <div className="section">
      <h2><span className="step-n">{n}</span>{title}</h2>
      {children}
    </div>
  );
}

export default function HowItWorks() {
  const { data } = useLoad(() => api.evaluation());
  const gen = data?.generation?.models;
  const a = gen?.[A], b = gen?.[B];
  const judge = data?.llm_judge;
  const tc = data?.training_config || {};
  const split = data?.dataset?.split_sizes;
  const raw = data?.dataset?.eda?.cleaning_log;
  const n = (v) => (v == null ? "…" : Number(v).toLocaleString());

  return (
    <>
      <div className="page-head">
        <div>
          <h1>How SupportGen works</h1>
          <p>The whole project on one page, from the data to the website you are using now.</p>
        </div>
      </div>

      <div className="panel idea">
        <p><b>The problem:</b> support agents answer hundreds of similar emails every day, and writing each
          reply from scratch is slow.</p>
        <p><b>Our idea:</b> a copilot that reads the customer's ticket and writes a <b>draft reply</b>,
          using a language model that <b>we trained ourselves</b>. It also says what kind of ticket it is,
          how urgent it is and how the customer feels. A human agent always checks the draft before sending it.</p>
      </div>

      <Step n="1" title="The big picture">
        <div className="flow">
          {BIG_PICTURE.map((s, i) => (
            <div className="flow-step" key={s.title}>
              <div className="flow-card">
                <div className="flow-n">{i + 1}</div>
                <b>{s.title}</b>
                <span className="small muted">{s.where}</span>
                <p className="small">{s.text}</p>
              </div>
              {i < BIG_PICTURE.length - 1 && <div className="flow-arrow" aria-hidden="true">→</div>}
            </div>
          ))}
        </div>
      </Step>

      <Step n="2" title="The data">
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div className="panel">
            <p style={{ marginBottom: 10 }}>We used the public <b>customer-support-tickets</b> dataset from Hugging
              Face. Each row has the customer's subject and message, the agent's reply, and labels for
              category, priority and type.</p>
            <p className="small muted">The dataset is synthetic (AI-generated), so results show how the model
              does on realistic, but not real, tickets.</p>
          </div>
          <div className="panel">
            <dl className="facts">
              <dt>Raw rows</dt><dd className="num">{n(raw?.[0]?.rows)}</dd>
              <dt>English only</dt><dd className="num">{n(raw?.[1]?.rows)}</dd>
              <dt>After cleaning</dt><dd className="num">{n(raw?.[(raw?.length || 1) - 1]?.rows)} <span className="muted small">(empty and duplicate tickets removed)</span></dd>
              <dt>Split</dt><dd className="num">{n(split?.train)} to learn from, {n(split?.val)} to check during training, {n(split?.test)} for the final test</dd>
            </dl>
            <p className="small muted" style={{ marginTop: 10 }}>Similar tickets were kept in the same group, so the
              model is never tested on a ticket it has already seen.</p>
          </div>
        </div>
      </Step>

      <Step n="3" title="Training the model (the main GenAI part)">
        <div className="grid-2" style={{ alignItems: "start" }}>
          <div className="panel">
            <p style={{ marginBottom: 10 }}>We started from <b>FLAN-T5-small</b>, a free language model from Google
              with about 77 million parameters. It reads a text and writes a new text.</p>
            <p style={{ marginBottom: 10 }}>Then we <b>fine-tuned</b> it: we showed it {n(tc.train_examples)} examples of
              "customer ticket → agent reply", and it adjusted its weights to write replies like a support agent.</p>
            <dl className="facts">
              <dt>Training time</dt><dd>{tc.train_minutes ?? "…"} minutes on Google Colab's free {tc.gpu || "GPU"}</dd>
              <dt>Rounds over the data</dt><dd className="num">{tc.epochs_completed ?? "…"} epochs</dd>
              <dt>Runs on</dt><dd>A normal laptop CPU, about 1 second per reply. No internet or paid API needed.</dd>
            </dl>
          </div>
          <div className="panel">
            <h3 style={{ marginBottom: 10 }}>Before vs after training (real output)</h3>
            <p className="small muted" style={{ marginBottom: 6 }}>Ticket: "Since installing the latest update yesterday,
              our server crashes every few hours…"</p>
            <div className="before-after">
              <div>
                <span className="tag">Before: original model</span>
                <p className="reply-text">Sorry, I can't help you with this problem. Please help me fix it.</p>
              </div>
              <div className="after">
                <span className="tag">After: our fine-tuned model</span>
                <p className="reply-text">Dear [Customer Name], we apologize for the inconvenience caused by the server
                  crashing after the recent update… could you please provide us with more details about the error
                  message you are receiving and the steps you have taken so far?…</p>
              </div>
            </div>
          </div>
        </div>
      </Step>

      <Step n="4" title="What happens when you click “Analyze ticket”">
        <p className="lede">The backend runs five small AI steps on the ticket, saves the result in the database and
          sends it back to the website. It takes about 1–2 seconds.</p>
        <div className="panel table-wrap">
          <table>
            <thead><tr><th>What you see</th><th>How it is made</th><th>Trained by us?</th></tr></thead>
            <tbody>
              {PIPELINE.map(([what, how, ours], i) => (
                <tr key={what}><td><b>{what}</b></td><td>{how}</td><td className={i < 2 ? "better" : "muted"}>{ours}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Step>

      <Step n="5" title="How we know it works">
        <p className="lede">We tested both models on tickets they had never seen and compared their replies with the
          real agent replies.</p>
        <div className="stats">
          <div className="stat">
            <div className="value num">{fmt(a?.rougeL, 2)} → {fmt(b?.rougeL, 2)}</div>
            <div className="label">ROUGE-L: word overlap with the real reply (more than doubled)</div>
          </div>
          <div className="stat">
            <div className="value num">{fmt(a?.semantic_cosine, 2)} → {fmt(b?.semantic_cosine, 2)}</div>
            <div className="label">Meaning similarity with the real reply</div>
          </div>
          <div className="stat">
            <div className="value num">{judge ? `${fmt(judge.mean_scores[Object.keys(judge.mean_scores)[0]].overall, 1)} → ${fmt(judge.mean_scores[Object.keys(judge.mean_scores)[1]].overall, 1)}` : "…"}</div>
            <div className="label">AI-judge quality score (out of 5)</div>
          </div>
          <div className="stat">
            <div className="value num">{judge ? `${judge.pairwise.finetuned_better} / ${judge.items}` : "…"}</div>
            <div className="label">Tickets where our reply was judged better</div>
          </div>
        </div>
        <p className="small muted" style={{ marginTop: 8 }}>Details on the <Link to="/evaluation">Model evaluation</Link> page.</p>
      </Step>

      <Step n="6" title="The website">
        <div className="panel table-wrap">
          <table>
            <tbody>
              {PAGES.map(([to, name, text]) => (
                <tr key={to}><td style={{ whiteSpace: "nowrap" }}><Link to={to}><b>{name}</b></Link></td><td>{text}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Step>

      <Step n="7" title="Limits, and why a human stays in charge">
        <div className="panel">
          <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 6 }}>
            <li>Replies can sound repetitive: many of them offer a phone call.</li>
            <li>The model can get a fact wrong, for example saying a refund was received when it was not.</li>
            <li>The category and priority labels are right about 6 times out of 10, so they come with a confidence score.</li>
          </ul>
          <p style={{ marginTop: 12 }}><b>That is why every reply is a draft.</b> The agent reads it, fixes it if
            needed, and then sends it.</p>
        </div>
      </Step>

      <Step n="8" title="Built with">
        <div className="btn-row">
          {["Python", "PyTorch", "Hugging Face Transformers", "scikit-learn", "Google Colab", "Django REST Framework",
            "PostgreSQL", "React", "Vite"].map((t) => <span className="tag" key={t}>{t}</span>)}
        </div>
      </Step>
    </>
  );
}
