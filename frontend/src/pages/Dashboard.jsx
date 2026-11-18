import { Link, useNavigate } from "react-router-dom";
import { IS_DEMO, api } from "../api.js";
import { DistributionBars, Lines } from "../components/charts.jsx";
import { ErrorNote, Loading, PriorityTag, Tag, fmt, pct, useLoad, when } from "../components/ui.jsx";

const PRIORITY_ORDER = ["critical", "high", "medium", "low"];
const share = (dist, keys, total) => (total ? keys.reduce((n, k) => n + (dist?.[k] || 0), 0) / total : null);
const ordered = (dist, order) => Object.fromEntries(order.filter((k) => dist?.[k]).map((k) => [k, dist[k]]));

function ModelHeadline() {
  const { data } = useLoad(() => api.evaluation());
  const models = data?.generation?.models;
  if (!models) return null;
  const a = models["A_base_flan-t5-small"];
  const b = models["B_ft_flan-t5-small"];
  if (!a || !b) return null;
  const tc = data.training_config || {};
  return (
    <div className="panel">
      <div className="panel-head">
        <h2>Response model</h2>
        <Link to="/evaluation" className="small">Full evaluation</Link>
      </div>
      <dl className="facts">
        <dt>Model</dt><dd>FLAN-T5-small fine-tuned on {tc.train_examples?.toLocaleString()} support conversations</dd>
        <dt>ROUGE-L on test set</dt><dd className="num">{fmt(a.rougeL)} before fine-tuning, <b>{fmt(b.rougeL)}</b> after</dd>
        <dt>BERTScore F1 (rescaled)</dt><dd className="num">{fmt(a.bertscore_f1_rescaled)} before, <b>{fmt(b.bertscore_f1_rescaled)}</b> after</dd>
        <dt>Test tickets</dt><dd className="num">{data.generation.eval_subset_size}</dd>
      </dl>
    </div>
  );
}

export default function Dashboard() {
  const { data, error, loading } = useLoad(() => api.analytics());
  const navigate = useNavigate();

  if (loading) return <Loading />;
  if (error) return <ErrorNote error={error} />;
  const total = data.total_tickets;
  const topCategory = Object.entries(data.category || {})[0];

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <p>{IS_DEMO
            ? "Demo data: the 10 sample tickets, analysed by the real models. In the full app this shows every ticket analysed."
            : "Every ticket analyzed in SupportGen, with the predicted category, priority, type and sentiment."}</p>
        </div>
        <Link to="/analyze" className="btn" style={{ textDecoration: "none" }}>Analyze a ticket</Link>
      </div>

      {total === 0 ? (
        <div className="panel empty">
          <p>No tickets analyzed yet. Analyze your first ticket to fill this dashboard.</p>
          <Link to="/analyze" className="btn" style={{ textDecoration: "none" }}>Analyze a ticket</Link>
        </div>
      ) : (
        <>
          <div className="stats">
            <div className="stat"><div className="value num">{total.toLocaleString()}</div><div className="label">Tickets analyzed</div></div>
            <div className="stat"><div className="value num">{pct(share(data.priority, ["critical", "high"], total))}</div><div className="label">High or critical priority</div></div>
            <div className="stat"><div className="value num">{pct(share(data.sentiment, ["negative"], total))}</div><div className="label">Negative sentiment (inferred)</div></div>
            <div className="stat"><div className="value" style={{ fontSize: "1.2rem", paddingTop: 8 }}>{topCategory?.[0] || "—"}</div><div className="label">Most common category</div></div>
          </div>

          <div className="grid-2 section">
            <div className="panel"><h2>Categories</h2><DistributionBars data={data.category} /></div>
            <div className="panel"><h2>Priority</h2><DistributionBars data={ordered(data.priority, PRIORITY_ORDER)} /></div>
            <div className="panel"><h2>Ticket types</h2><DistributionBars data={data.ticket_type} /></div>
            <div className="panel">
              <h2>Sentiment</h2>
              <DistributionBars data={ordered(data.sentiment, ["negative", "neutral", "positive"])} />
              <p className="small muted">Inferred by a pretrained model; the dataset has no sentiment labels.</p>
            </div>
          </div>

          {data.tickets_per_day?.length > 1 && (
            <div className="panel section">
              <h2>Tickets per day (last 30 days)</h2>
              <Lines rows={data.tickets_per_day} xKey="day" xType="category"
                     series={[{ key: "count", name: "Tickets", color: "var(--seq)", dots: true }]}
                     format={(v) => v} height={200} />
            </div>
          )}

          <div className="grid-2 section" style={{ alignItems: "start" }}>
            <div className="panel">
              <div className="panel-head"><h2>Recent tickets</h2><Link to="/history" className="small">All tickets</Link></div>
              <div className="table-wrap">
                <table>
                  <thead><tr><th>Ticket</th><th>Category</th><th>Priority</th></tr></thead>
                  <tbody>
                    {data.recent.map((t) => (
                      <tr key={t.id} className="clickable" onClick={() => navigate(`/tickets/${t.id}`)}>
                        <td><div className="clip">{t.subject || t.body}</div><span className="small muted">{when(t.created_at)}</span></td>
                        <td><Tag value={t.category} /></td>
                        <td><PriorityTag value={t.priority} /></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
            <ModelHeadline />
          </div>
        </>
      )}
    </>
  );
}
