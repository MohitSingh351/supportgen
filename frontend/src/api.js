// Thin wrapper around the Django REST API (proxied to :8000 by Vite in development).
const BASE = import.meta.env.VITE_API_BASE || "/api";

function messageFrom(data, status) {
  if (!data) return `Request failed (HTTP ${status}).`;
  if (typeof data.error === "string") return data.error;
  if (typeof data.detail === "string") return data.detail;
  // DRF validation errors: { field: ["message", ...] }
  const parts = Object.entries(data).map(([field, msgs]) =>
    `${field === "body" ? "Ticket" : field}: ${[].concat(msgs).join(" ")}`);
  return parts.join(" ") || `Request failed (HTTP ${status}).`;
}

async function request(path, options = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch {
    throw new Error("Cannot reach the SupportGen server. Start the Django backend on port 8000.");
  }
  const data = await res.json().catch(() => null);
  if (!res.ok) {
    const err = new Error(messageFrom(data, res.status));
    err.status = res.status;
    throw err;
  }
  return data;
}

export const api = {
  health: () => request("/health/"),
  analyze: (subject, body) =>
    request("/tickets/analyze/", { method: "POST", body: JSON.stringify({ subject, body }) }),
  tickets: (page = 1) => request(`/tickets/?page=${page}`),
  ticket: (id) => request(`/tickets/${id}/`),
  analytics: () => request("/analytics/"),
  evaluation: () => request("/evaluation/"),
  compare: (subject, body, providers) =>
    request("/compare/", { method: "POST", body: JSON.stringify({ subject, body, providers }) }),
  plotUrl: (key) => `${BASE}/evaluation/plots/${key}/`,
};
