import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import TicketResult from "../components/TicketResult.jsx";
import { ErrorNote, Loading, useLoad, when } from "../components/ui.jsx";

export default function TicketDetail() {
  const { id } = useParams();
  const { data, error, loading } = useLoad(() => api.ticket(id), [id]);
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Ticket #{id}</h1>
          {data && <p>Analyzed {when(data.created_at)}</p>}
        </div>
        <Link to="/history" className="btn ghost small" style={{ textDecoration: "none" }}>Back to history</Link>
      </div>
      {loading && <Loading />}
      <ErrorNote error={error} />
      {data && <div style={{ maxWidth: 820 }}><TicketResult ticket={data} /></div>}
    </>
  );
}
