import { NavLink, Route, Routes } from "react-router-dom";
import { IS_DEMO, api } from "./api.js";
import { useLoad } from "./components/ui.jsx";
import Analyze from "./pages/Analyze.jsx";
import Compare from "./pages/Compare.jsx";
import Dashboard from "./pages/Dashboard.jsx";
import Evaluation from "./pages/Evaluation.jsx";
import History from "./pages/History.jsx";
import HowItWorks from "./pages/HowItWorks.jsx";
import TicketDetail from "./pages/TicketDetail.jsx";

function SystemStatus() {
  const { data, error } = useLoad(() => api.health());
  if (error) return <div className="status"><div className="status-row"><span className="dot off" />Backend offline</div></div>;
  if (!data) return null;
  if (data.demo) {
    return (
      <div className="status" aria-label="System status">
        <div className="status-row"><span className="dot warn" />Read-only demo</div>
        <div>Saved output from the real models. Live analysis runs locally.</div>
      </div>
    );
  }
  const rows = [
    ["Response model", data.response_model?.available],
    ["Classifiers", data.classifiers?.available],
    ["Database", data.database?.ok],
    ["Gemini benchmark", data.gemini_configured],
  ];
  return (
    <div className="status" aria-label="System status">
      {rows.map(([label, ok]) => (
        <div className="status-row" key={label}>
          <span className={`dot ${ok ? "ok" : label === "Gemini benchmark" ? "" : "off"}`} />
          {label}{ok ? "" : label === "Gemini benchmark" ? " not set up" : " unavailable"}
        </div>
      ))}
      <div>Inference on {data.device?.toUpperCase()}</div>
    </div>
  );
}

const NAV = [
  ["/", "Dashboard"],
  ["/how-it-works", "How it works"],
  ["/analyze", "Analyze ticket"],
  ["/history", "Ticket history"],
  ["/compare", "Model comparison"],
  ["/evaluation", "Model evaluation"],
];

export default function App() {
  return (
    <div className="shell">
      <aside className="rail">
        <div className="brand">
          <strong>SupportGen</strong>
          <span>Support copilot with fine-tuned FLAN-T5</span>
        </div>
        <nav className="nav">
          {NAV.map(([to, label]) => (
            <NavLink key={to} to={to} end={to === "/"}>{label}</NavLink>
          ))}
        </nav>
        <SystemStatus />
      </aside>
      <main className="main">
        {IS_DEMO && (
          <div className="demo-banner" role="note">
            <b>Read-only demo.</b> The AI models are too large for this free host, so the sample tickets show
            real, saved output from our models. The full app runs locally:{" "}
            <a href="https://github.com/MohitSingh351/supportgen" target="_blank" rel="noreferrer">source and setup on GitHub</a>.
          </div>
        )}
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/how-it-works" element={<HowItWorks />} />
          <Route path="/analyze" element={<Analyze />} />
          <Route path="/history" element={<History />} />
          <Route path="/tickets/:id" element={<TicketDetail />} />
          <Route path="/compare" element={<Compare />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="*" element={<p>Page not found.</p>} />
        </Routes>
      </main>
    </div>
  );
}
