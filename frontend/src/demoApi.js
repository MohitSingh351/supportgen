// Read-only demo backend used when the site is built with VITE_DEMO=1 (the Vercel deployment).
// It answers the same calls as the Django API, but from static files in public/demo/:
//   evaluation.json - the exact payload of GET /api/evaluation/ (scores exported from Colab)
//   samples.json    - 10 test tickets with the real, saved output of the models
// Nothing is generated in the browser; new tickets need the real backend.

export const DEMO_MESSAGE =
  "This online version is a read-only demo: the AI models need a server with more memory than this host offers. " +
  "Pick one of the sample tickets above to see real, saved model output, or run the full project locally (see the GitHub README).";

const cache = {};
async function load(name) {
  if (!cache[name]) {
    cache[name] = fetch(`/demo/${name}.json`).then((r) => {
      if (!r.ok) throw new Error(`Could not load demo data (${name}).`);
      return r.json();
    });
  }
  return cache[name];
}

const norm = (s) => (s || "").replace(/\s+/g, " ").trim().toLowerCase();

async function findSample(subject, body) {
  const samples = await load("samples");
  return samples.find((s) => norm(s.body) === norm(body) && norm(s.subject) === norm(subject))
    || samples.find((s) => norm(s.body) === norm(body));
}

const LIST_FIELDS = ["id", "subject", "body", "category", "priority", "ticket_type", "sentiment", "summary", "model_used", "created_at"];
const pick = (t) => Object.fromEntries(LIST_FIELDS.map((k) => [k, t[k]]));

function distribution(samples, field) {
  const out = {};
  samples.forEach((t) => { if (t[field]) out[t[field]] = (out[t[field]] || 0) + 1; });
  return Object.fromEntries(Object.entries(out).sort((a, b) => b[1] - a[1]));
}

export const demoApi = {
  isDemo: true,
  samples: () => load("samples"),

  health: async () => ({ demo: true }),

  evaluation: () => load("evaluation"),

  analyze: async (subject, body) => {
    const s = await findSample(subject, body);
    if (!s) throw new Error(DEMO_MESSAGE);
    return { ...s, saved_demo_output: true };
  },

  compare: async (subject, body) => {
    const s = await findSample(subject, body);
    if (!s) throw new Error(DEMO_MESSAGE);
    return {
      subject, body, saved_demo_output: true,
      results: [
        { provider: "base", label: "Base FLAN-T5-small (no fine-tuning)", status: "ok", text: s.base_response, model: s.base_model, latency_ms: s.base_latency_ms },
        { provider: "finetuned", label: "Fine-tuned FLAN-T5-small (ours)", status: "ok", text: s.generated_response, model: s.model_used, latency_ms: s.timings_ms?.response },
        { provider: "gemini", label: "Gemini (optional benchmark)", status: "not_configured", message: "Gemini is not set up in this demo." },
      ],
    };
  },

  tickets: async () => {
    const samples = [...(await load("samples"))].reverse();
    return { count: samples.length, next: null, previous: null, results: samples.map(pick) };
  },

  ticket: async (id) => {
    const s = (await load("samples")).find((t) => String(t.id) === String(id));
    if (!s) throw new Error("Ticket not found in the demo data.");
    return s;
  },

  analytics: async () => {
    const samples = await load("samples");
    const high = samples.filter((t) => ["high", "critical"].includes(t.priority));
    return {
      total_tickets: samples.length,
      priority: distribution(samples, "priority"),
      category: distribution(samples, "category"),
      ticket_type: distribution(samples, "ticket_type"),
      sentiment: distribution(samples, "sentiment"),
      high_priority_by_category: distribution(high, "category"),
      tickets_per_day: [],
      recent: [...samples].reverse().slice(0, 5).map(pick),
    };
  },

  plotUrl: (key) => `/demo/plots/${key}.png`,
};
