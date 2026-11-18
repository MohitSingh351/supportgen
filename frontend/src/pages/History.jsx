import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { IS_DEMO, api } from "../api.js";
import { ErrorNote, Loading, PriorityTag, Tag, cap, useLoad, when } from "../components/ui.jsx";

export default function History() {
  const [page, setPage] = useState(1);
  const { data, error, loading } = useLoad(() => api.tickets(page), [page]);
  const navigate = useNavigate();
  const pages = data ? Math.max(1, Math.ceil(data.count / 20)) : 1;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>Ticket history</h1>
          <p>{IS_DEMO ? "Demo data: the 10 sample tickets, with real saved model output. Open one to see the full reply and insights." : "Every analyzed ticket, newest first. Open one to see the full reply and insights."}</p>
        </div>
      </div>
      {loading && <Loading />}
      <ErrorNote error={error} />
      {data && data.count === 0 && (
        <div className="panel empty">
          <p>No tickets yet.</p>
          <Link to="/analyze" className="btn" style={{ textDecoration: "none" }}>Analyze a ticket</Link>
        </div>
      )}
      {data && data.count > 0 && (
        <div className="panel">
          <div className="table-wrap">
            <table>
              <thead>
                <tr><th>#</th><th>Ticket</th><th>Category</th><th>Priority</th><th>Type</th><th>Sentiment</th><th>Analyzed</th></tr>
              </thead>
              <tbody>
                {data.results.map((t) => (
                  <tr key={t.id} className="clickable" tabIndex={0}
                      onClick={() => navigate(`/tickets/${t.id}`)}
                      onKeyDown={(e) => e.key === "Enter" && navigate(`/tickets/${t.id}`)}>
                    <td className="num muted">{t.id}</td>
                    <td style={{ minWidth: 200, maxWidth: 380 }}>
                      <b>{t.subject || "(no subject)"}</b>
                      <div className="clip small muted">{t.body}</div>
                    </td>
                    <td><Tag value={t.category} /></td>
                    <td><PriorityTag value={t.priority} /></td>
                    <td>{t.ticket_type || "—"}</td>
                    <td>{cap(t.sentiment) || "—"}</td>
                    <td className="small muted" style={{ whiteSpace: "nowrap" }}>{when(t.created_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="pager">
            <span className="small muted num">{data.count} tickets, page {page} of {pages}</span>
            <div className="btn-row">
              <button className="btn ghost small" disabled={!data.previous} onClick={() => setPage((p) => p - 1)}>Previous</button>
              <button className="btn ghost small" disabled={!data.next} onClick={() => setPage((p) => p + 1)}>Next</button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
