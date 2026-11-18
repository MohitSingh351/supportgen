import { api } from "../api.js";
import { GroupedBars, Lines } from "../components/charts.jsx";
import { ErrorNote, Loading, fmt, pct, useLoad } from "../components/ui.jsx";

const A = "A_base_flan-t5-small";
const B = "B_ft_flan-t5-small";
const num = (v) => (v === "" || v == null ? null : Number(v));
const stripHtml = (s = "") => s.replace(/<br\s*\/?>/gi, "\n").replace(/<[^>]+>/g, "");

function Training({ tc, log }) {
  const hp = tc.hyperparameters || {};
  const parsed = (log || []).map((r) => ({ epoch: num(r.epoch), loss: num(r.loss), eval_loss: num(r.eval_loss) }));
  const trainRows = parsed.filter((r) => r.epoch != null && r.loss != null);
  const valRows = parsed.filter((r) => r.epoch != null && r.eval_loss != null);
  return (
    <div className="section">
      <h2>Fine-tuning</h2>
      <p className="lede">All weights of the pretrained encoder-decoder were updated on ticket → agent-answer
        pairs with teacher forcing and token-level cross-entropy loss.</p>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <div className="panel">
          <dl className="facts">
            <dt>Base model</dt><dd>{tc.base_model} ({(tc.parameters / 1e6).toFixed(1)}M parameters, all trainable)</dd>
            <dt>Data</dt><dd className="num">{tc.train_examples?.toLocaleString()} train, {tc.val_examples?.toLocaleString()} validation, {tc.test_examples?.toLocaleString()} test</dd>
            <dt>Epochs</dt><dd className="num">{tc.epochs_completed} (all epochs run, validation loss checked after each)</dd>
            <dt>Learning rate</dt><dd className="num">{hp.learning_rate}, linear schedule, {pct(hp.warmup_fraction)} warm-up</dd>
            <dt>Batch size</dt><dd className="num">{tc.effective_batch_size}</dd>
            <dt>Max lengths</dt><dd className="num">{tc.max_input_len} input tokens, {tc.max_target_len} output tokens</dd>
            <dt>Decoding</dt><dd>Beam search ({tc.generation?.num_beams} beams), no repeated 3-grams</dd>
            <dt>Hardware</dt><dd>{tc.gpu}, {tc.train_minutes} minutes</dd>
            <dt>Best validation loss</dt><dd className="num">{fmt(tc.best_val_loss)} (perplexity {fmt(Math.exp(tc.best_val_loss), 2)})</dd>
            <dt>Test loss</dt><dd className="num">{fmt(tc.test_loss)} (perplexity {fmt(Math.exp(tc.test_loss), 2)})</dd>
          </dl>
        </div>
        <div className="panel">
          <h3 style={{ marginBottom: 10 }}>Loss curves</h3>
          <Lines rows={trainRows} xKey="epoch" xLabel="Epoch" xTicks={[1, 2, 3, 4].filter((e) => e <= (tc.epochs_completed || 4))}
                 series={[{ key: "loss", name: "Training loss", color: "var(--series-1)", data: trainRows },
                          { key: "eval_loss", name: "Validation loss (end of each epoch)", color: "var(--series-2)", dots: true, data: valRows }]} />
          <p className="small muted">Validation loss falls every epoch, so the model was not overfitting.</p>
        </div>
      </div>
    </div>
  );
}

function GenerationQuality({ gen }) {
  const a = gen.models[A], b = gen.models[B];
  const chartRows = [
    ["ROUGE-1", "rouge1"], ["ROUGE-2", "rouge2"], ["ROUGE-L", "rougeL"],
    ["BERTScore", "bertscore_f1_rescaled"], ["Cosine", "semantic_cosine"],
  ].map(([metric, k]) => ({ metric, base: a[k], ft: b[k] }));
  const table = [
    ["ROUGE-1", "rouge1"], ["ROUGE-2", "rouge2"], ["ROUGE-L", "rougeL"], ["BLEU (0–100)", "bleu", 2],
    ["BERTScore F1", "bertscore_f1"], ["BERTScore F1 (rescaled)", "bertscore_f1_rescaled"],
    ["Semantic cosine", "semantic_cosine"], ["Distinct-2 (variety)", "distinct_2"],
    ["Average words per reply", "avg_pred_words", 1],
  ];
  return (
    <div className="section">
      <h2>Response quality: before vs after fine-tuning</h2>
      <p className="lede">Both models answered the same {gen.eval_subset_size} held-out test tickets with the same prompt
        and decoding. Scores compare each reply with the real agent answer. Reference answers average {fmt(a.avg_ref_words, 0)} words.</p>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <div className="panel">
          <GroupedBars rows={chartRows} series={[
            { key: "base", name: "Base FLAN-T5-small", color: "var(--series-2)" },
            { key: "ft", name: "Fine-tuned (ours)", color: "var(--series-1)" }]} />
          <p className="small muted">BERTScore is the baseline-rescaled F1; cosine is sentence-embedding similarity. Higher is better for all five.</p>
        </div>
        <div className="panel table-wrap">
          <table>
            <thead><tr><th>Metric</th><th className="n">Base</th><th className="n">Fine-tuned</th></tr></thead>
            <tbody>
              {table.map(([label, k, d = 3]) => (
                <tr key={k}><td>{label}</td><td className="n">{fmt(a[k], d)}</td><td className="n">{fmt(b[k], d)}</td></tr>
              ))}
            </tbody>
          </table>
          <p className="small muted" style={{ marginTop: 10 }}>BERTScore model: {a.bertscore_model}. BLEU rewards exact n-gram
            matches, so it is harsh on open-ended replies where many wordings are correct.</p>
        </div>
      </div>
    </div>
  );
}

const CRITERIA = [["relevance", "Relevance"], ["correctness", "Correctness"], ["fluency", "Fluency"],
                  ["professional_tone", "Professional tone"], ["helpfulness", "Helpfulness"], ["overall", "Overall (mean)"]];

function JudgeEvaluation({ j }) {
  if (!j) return null;
  const [baseName, ftName] = Object.keys(j.mean_scores);
  const base = j.mean_scores[baseName], ft = j.mean_scores[ftName];
  const fails = j.failure_counts;
  return (
    <div className="section">
      <h2>AI-judge evaluation (LLM-as-a-judge)</h2>
      <p className="lede">An AI model (Claude) rated {j.items} blinded ticket pairs from 1 to 5, using a rubric written
        before rating. This is not a human evaluation. It checks meaning and correctness, which word-overlap metrics cannot.</p>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <div className="panel">
          <GroupedBars rows={CRITERIA.map(([k, label]) => ({ metric: label.replace(" (mean)", "").replace("Professional tone", "Tone"), base: base[k], ft: ft[k] }))}
                       series={[{ key: "base", name: "Base FLAN-T5-small", color: "var(--series-2)" },
                                { key: "ft", name: "Fine-tuned (ours)", color: "var(--series-1)" }]}
                       format={(v) => v.toFixed(2)} yMax={5} />
          <p className="small muted">Mean rating on a 1–5 scale.</p>
        </div>
        <div className="panel table-wrap">
          <table>
            <thead><tr><th>Criterion</th><th className="n">Base</th><th className="n">Fine-tuned</th></tr></thead>
            <tbody>
              {CRITERIA.map(([k, label]) => (
                <tr key={k}><td>{label}</td><td className="n">{fmt(base[k], 2)}</td><td className="n">{fmt(ft[k], 2)}</td></tr>
              ))}
            </tbody>
          </table>
          <dl className="facts" style={{ marginTop: 14 }}>
            <dt>Head to head</dt>
            <dd className="num">Fine-tuned rated higher on {j.pairwise.finetuned_better} of {j.items} tickets,
              base on {j.pairwise.base_better}, {j.pairwise.ties} ties (Wilcoxon p = {Number(j.pairwise.wilcoxon_p_overall).toExponential(1)})</dd>
            <dt>Not a support reply</dt>
            <dd className="num">Base {fails[baseName]["not_a_support_reply (tone = 1)"]} of {j.items}, fine-tuned {fails[ftName]["not_a_support_reply (tone = 1)"]} of {j.items}</dd>
            <dt>Correctness ≤ 2</dt>
            <dd className="num">Base {fails[baseName]["correctness <= 2 (wrong, invented or no reply)"]}, fine-tuned {fails[ftName]["correctness <= 2 (wrong, invented or no reply)"]}</dd>
          </dl>
        </div>
      </div>
      {j.finetuned_low_correctness_examples?.length > 0 && (
        <div className="panel" style={{ marginTop: 16 }}>
          <h3 style={{ marginBottom: 8 }}>Where the fine-tuned model went wrong</h3>
          <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 4 }}>
            {j.finetuned_low_correctness_examples.map((e) => <li key={e.item} className="small">{e.comment}</li>)}
          </ul>
        </div>
      )}
      <p className="small muted" style={{ marginTop: 10 }}>Caveats: a single AI rater; it may favour longer or more polite
        replies; it helped build this project; and blinding is imperfect because the fine-tuned style is recognisable.
        Rubric: results/llm_judge_rubric.md.</p>
    </div>
  );
}

/** Number of test tickets per class, from the confusion matrix (rows = true class). */
function supports(m) {
  const out = {};
  (m.labels || []).forEach((label, i) => { out[label] = (m.confusion_matrix?.[i] || []).reduce((s, v) => s + v, 0); });
  return out;
}

function MacroF1Explained({ cls }) {
  const q = cls.queue_group, p = cls.priority, t = cls.type;
  if (!q || !p || !t) return null;
  const rows = [["Category", q], ["Priority", p], ["Ticket type", t]].map(([metric, m]) => ({
    metric,
    majority: m.baseline_majority?.macro_f1 ?? 0,
    random: m.baseline_stratified_random?.macro_f1 ?? 0,
    ours: m.macro_f1,
  }));
  const qs = supports(q);
  const total = Object.values(qs).reduce((s, v) => s + v, 0);
  const sorted = Object.entries(qs).sort((x, y) => y[1] - x[1]);
  const [bigName, bigN] = sorted[0] || [];
  const [smallName, smallN] = sorted[sorted.length - 1] || [];
  const f1 = (m, k) => fmt(m.per_class_f1?.[k], 2);

  return (
    <div className="panel" style={{ marginTop: 16 }}>
      <h3 style={{ marginBottom: 6 }}>Is a macro-F1 of {fmt(q.macro_f1, 2)} too low?</h3>
      <p style={{ marginBottom: 14, maxWidth: "75ch" }}>
        <b>Short answer: it is moderate, not bad.</b> Ticket type ({fmt(t.macro_f1, 2)}) is good. Category
        ({fmt(q.macro_f1, 2)}) and priority ({fmt(p.macro_f1, 2)}) are clearly better than guessing, but
        there is room to improve. The chart compares our classifiers with two "dumb" guessers.
      </p>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <div>
          <GroupedBars rows={rows} yMax={1} yTicks={[0, 0.25, 0.5, 0.75, 1]} format={(v) => v.toFixed(2)} height={240}
                       series={[{ key: "majority", name: "Always guess the most common class", color: "var(--series-4)" },
                                { key: "random", name: "Random guessing", color: "var(--series-2)" },
                                { key: "ours", name: "Our classifier", color: "var(--series-1)" }]} />
          <p className="small muted">Macro-F1 on {total.toLocaleString()} test tickets (0 = useless, 1 = perfect).</p>
        </div>
        <div>
          <h3 style={{ marginBottom: 8 }}>Why the score looks low</h3>
          <ol style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 8 }}>
            <li><b>Macro-F1 is a strict score.</b> Every class counts equally, so a small class
              like {smallName} ({smallN} test tickets) counts as much as {bigName} ({bigN}).
              Mistakes on the small classes pull the average down. Plain accuracy is {pct(q.accuracy)}.</li>
            <li><b>The 10 categories overlap.</b> Technical Support, IT Support, Product Support and
              Customer Service often describe the same kind of problem. Even people would disagree on some tickets.</li>
            <li><b>Priority is subjective.</b> The same words can be "medium" or "low" urgency. The
              "low" class is the hardest (F1 {f1(p, "low")}).</li>
            <li><b>The classifier only counts words</b> (TF-IDF). It does not understand meaning,
              so "return my laptop" can look like an IT ticket.</li>
            <li><b>The labels are synthetic</b>, so some of them are partly random.</li>
          </ol>
        </div>
      </div>
      <div className="grid-2" style={{ marginTop: 16, alignItems: "start" }}>
        <div className="alert warn">
          <b>How the app handles it:</b> every prediction shows a confidence score, so the agent knows
          when to double-check. These labels only help the agent. The main GenAI part is the reply,
          which is evaluated above.
        </div>
        <div className="alert warn">
          <b>How to improve it:</b> use a Transformer classifier (e.g. DistilBERT) that understands
          meaning, add more examples of the small classes, and merge categories that overlap.
        </div>
      </div>
    </div>
  );
}

function Classification({ cls }) {
  const targets = [["queue_group", "Category (queue)"], ["priority", "Priority"], ["type", "Ticket type"]];
  return (
    <div className="section">
      <h2>Supporting classifiers</h2>
      <p className="lede">TF-IDF features with a linear SVM, trained on the same split. Classes are imbalanced, so
        macro-F1 (every class counts equally) is the main score, compared with always predicting the most common class.</p>
      <div className="panel table-wrap">
        <table>
          <thead><tr><th>Target</th><th className="n">Accuracy</th><th className="n">Macro-F1</th><th className="n">Weighted F1</th><th className="n">Majority-class macro-F1</th><th>Weakest class</th></tr></thead>
          <tbody>
            {targets.filter(([k]) => cls[k]).map(([k, label]) => {
              const m = cls[k];
              const weakest = Object.entries(m.per_class_f1 || {}).sort((x, y) => x[1] - y[1])[0];
              return (
                <tr key={k}>
                  <td>{label}</td>
                  <td className="n">{fmt(m.accuracy)}</td>
                  <td className="n better">{fmt(m.macro_f1)}</td>
                  <td className="n">{fmt(m.weighted_f1)}</td>
                  <td className="n">{fmt(m.baseline_majority?.macro_f1)}</td>
                  <td>{weakest ? `${weakest[0]} (F1 ${fmt(weakest[1], 2)})` : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <MacroF1Explained cls={cls} />
      {cls.sentiment_inferred && (
        <p className="small muted" style={{ marginTop: 12 }}>
          Sentiment uses the pretrained {cls.sentiment_inferred.model}. The dataset has no sentiment labels, so it is
          not evaluated. On {cls.sentiment_inferred.n_sampled} test tickets it predicted{" "}
          {Object.entries(cls.sentiment_inferred.distribution).map(([k, v]) => `${pct(v)} ${k}`).join(", ")}.
        </p>
      )}
    </div>
  );
}

function Dataset({ ds }) {
  const eda = ds.eda || {};
  return (
    <div className="section">
      <h2>Dataset preparation</h2>
      <p className="lede">Tobi-Bueck/customer-support-tickets (CC BY-NC 4.0, synthetically generated), English subset.</p>
      <div className="grid-2" style={{ alignItems: "start" }}>
        <div className="panel table-wrap">
          <table>
            <thead><tr><th>Cleaning step</th><th className="n">Rows left</th></tr></thead>
            <tbody>
              {(eda.cleaning_log || []).map((s) => <tr key={s.step}><td>{s.step}</td><td className="n">{s.rows.toLocaleString()}</td></tr>)}
            </tbody>
          </table>
        </div>
        <div className="panel">
          <dl className="facts">
            <dt>Final split</dt><dd className="num">{Object.entries(ds.split_sizes || {}).map(([k, v]) => `${v.toLocaleString()} ${k}`).join(", ")}</dd>
            <dt>Near-duplicate groups</dt><dd className="num">{eda.near_duplicates?.tickets_in_multi_ticket_groups} tickets kept together in one split (cosine ≥ {eda.near_duplicates?.threshold_cosine})</dd>
            <dt>Leakage audit</dt><dd className="num">{eda.leakage_audit?.val?.removed_near_duplicates} validation and {eda.leakage_audit?.test?.removed_near_duplicates} test tickets removed for being too close to a training ticket</dd>
            <dt>Truncation</dt><dd className="num">{fmt(eda.max_lengths?.pct_inputs_truncated, 1)}% of inputs and {fmt(eda.max_lengths?.pct_targets_truncated, 1)}% of answers exceed the max length</dd>
          </dl>
        </div>
      </div>
    </div>
  );
}

function Examples({ rows }) {
  if (!rows?.length) return null;
  return (
    <div className="section">
      <h2>Example replies</h2>
      <p className="lede">Random held-out test tickets. Reference is the real agent answer.</p>
      <div className="panel table-wrap">
        <table className="examples">
          <thead><tr><th>Ticket</th><th>Reference</th><th>Base FLAN-T5</th><th>Fine-tuned (ours)</th></tr></thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i}>
                <td style={{ whiteSpace: "pre-wrap" }}>{stripHtml(r.ticket)}</td>
                <td>{r.reference}</td><td>{r["A: base"]}</td><td>{r["B: fine-tuned"]}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p className="small muted" style={{ marginTop: 8 }}>Raw model output: placeholders such as "tel_num&gt;" appear here because
        T5's vocabulary has no "&lt;". The app repairs them before showing a reply.</p>
    </div>
  );
}

function Limitations({ a, b, human }) {
  return (
    <div className="section">
      <h2>Limitations</h2>
      <div className="panel">
        <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 6 }}>
          <li>Replies are formulaic: distinct-2 falls from {fmt(a.distinct_2, 2)} to {fmt(b.distinct_2, 2)}. Many replies offer a call, copying the dataset's style.</li>
          <li>Part of the ROUGE gain reflects length: base replies average {fmt(a.avg_pred_words, 0)} words, fine-tuned {fmt(b.avg_pred_words, 0)}, references {fmt(a.avg_ref_words, 0)}.</li>
          <li>The model can state facts that contradict the ticket (hallucination), so every reply is a draft for a human agent.</li>
          <li>The dataset is synthetic, so results show adaptation to its domain and style rather than to a real company.</li>
          <li>Summaries and recommended actions have no ground-truth labels. Zero-shot FLAN-T5 output was poor, so the app uses fallbacks and labels the source.</li>
          <li>{human ? "Human ratings are available in results/human_eval_summary.json." : "No human evaluation was carried out; quality beyond automatic metrics was judged by an AI rater (see above)."}</li>
        </ul>
      </div>
    </div>
  );
}

export default function Evaluation() {
  const { data, error, loading } = useLoad(() => api.evaluation());
  if (loading) return <Loading />;
  if (error) return <ErrorNote error={error} />;
  if (!data.generation) {
    return (
      <div className="panel empty">
        <p>No evaluation results found. Unzip the Colab export so that <code>results/gen_metrics.json</code> exists.</p>
      </div>
    );
  }
  const a = data.generation.models[A], b = data.generation.models[B];
  return (
    <>
      <div className="page-head">
        <div>
          <h1>Model evaluation</h1>
          <p>Results from the Colab notebooks. Nothing on this page is recomputed or edited by the app.</p>
        </div>
      </div>
      {data.training_config && <Training tc={data.training_config} log={data.train_log} />}
      <GenerationQuality gen={data.generation} />
      <JudgeEvaluation j={data.llm_judge} />
      {data.classification && <Classification cls={data.classification} />}
      {data.dataset?.eda && <Dataset ds={data.dataset} />}
      <Examples rows={data.qualitative_examples} />
      <Limitations a={a} b={b} human={data.human_eval} />
    </>
  );
}
