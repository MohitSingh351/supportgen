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

const PHASES = [
  ["phase-a", "A", "Prepare the data"],
  ["phase-b", "B", "Build the models"],
  ["phase-c", "C", "Test the models"],
  ["phase-d", "D", "Run the app"],
];

const PARTS = [
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

function Section({ n, title, id, children }) {
  return (
    <div className="section" id={id}>
      <h2><span className="step-n">{n}</span>{title}</h2>
      {children}
    </div>
  );
}

function Phase({ id, letter, title, where, intro, children }) {
  return (
    <div className="phase" id={id}>
      <div className="phase-head">
        <span className="phase-letter">{letter}</span>
        <div>
          <h3>{title}</h3>
          <span className="small muted">{where}</span>
        </div>
      </div>
      {intro && <p className="phase-intro">{intro}</p>}
      <div className="stages">{children}</div>
    </div>
  );
}

function Stage({ n, title, where, result, children }) {
  return (
    <div className="stage">
      <div className="stage-n">{n}</div>
      <div className="stage-body">
        <div className="stage-title"><b>{title}</b>{where && <span className="stage-where">{where}</span>}</div>
        <div className="stage-text">{children}</div>
        {result && <div className="stage-result"><span>Result</span>{result}</div>}
      </div>
    </div>
  );
}

export default function HowItWorks() {
  const { data } = useLoad(() => api.evaluation());
  const gen = data?.generation?.models;
  const a = gen?.[A], b = gen?.[B];
  const judge = data?.llm_judge;
  const judgeNames = judge ? Object.keys(judge.mean_scores) : [];
  const tc = data?.training_config || {};
  const hp = tc.hyperparameters || {};
  const eda = data?.dataset?.eda || {};
  const log = eda.cleaning_log || [];
  const rows = (i) => log[i]?.rows;
  const split = data?.dataset?.split_sizes || {};
  const nd = eda.near_duplicates || {};
  const leak = eda.leakage_audit || {};
  const ml = eda.max_lengths || {};
  const cls = data?.classification || {};
  const n = (v) => (v == null ? "…" : Number(v).toLocaleString());
  const f = (v, d = 3) => (v == null ? "…" : fmt(v, d));
  const firstValLoss = (data?.train_log || []).map((r) => r.eval_loss).find((v) => v !== "" && v != null);
  const stepsPerEpoch = tc.train_examples && tc.effective_batch_size ? Math.ceil(tc.train_examples / tc.effective_batch_size) : null;

  return (
    <>
      <div className="page-head">
        <div>
          <h1>How SupportGen works</h1>
          <p>The whole project on one page, from the raw data to the website you are using now.</p>
        </div>
      </div>

      <div className="panel idea">
        <p><b>The problem:</b> support agents answer hundreds of similar emails every day, and writing each
          reply from scratch is slow.</p>
        <p><b>Our idea:</b> a copilot that reads the customer's ticket and writes a <b>draft reply</b>,
          using a language model that <b>we trained ourselves</b>. It also says what kind of ticket it is,
          how urgent it is and how the customer feels. A human agent always checks the draft before sending it.</p>
      </div>

      <Section n="1" title="The big picture">
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
      </Section>

      <Section n="2" title="The full pipeline, step by step">
        <p className="lede">Phases A to C run in six Google Colab notebooks. Phase D runs on a normal laptop.
          Jump to a phase:</p>
        <div className="btn-row" style={{ marginBottom: 18 }}>
          {PHASES.map(([id, letter, title]) => (
            <a key={id} href={`#${id}`} className="chip-btn" style={{ textDecoration: "none" }}>{letter}. {title}</a>
          ))}
        </div>

        <Phase id="phase-a" letter="A" title="Prepare the data" where="Notebook 01_dataset_preparation"
               intro="Goal: turn a raw public dataset into clean, fair training and test sets.">
          <Stage n="A1" title="Download the dataset" where="Hugging Face"
                 result={`${n(rows(0))} rows`}>
            We load the public <b>Tobi-Bueck/customer-support-tickets</b> dataset (several CSV files). Each row has
            the customer's <b>subject</b> and <b>message</b>, the agent's <b>answer</b>, and labels for
            <b> queue</b> (category), <b>priority</b>, <b>type</b> and <b>language</b>. The data is synthetic
            (AI-generated), in English and German.
          </Stage>
          <Stage n="A2" title="Clean the text"
                 result={`${n(rows(1))} English → ${n(rows(2))} non-empty → ${n(rows(3))} long enough → ${n(rows(4))} unique tickets`}>
            <ul>
              <li>Keep only <b>English</b> tickets.</li>
              <li>Fix the text: turn HTML codes like <code>&amp;amp;</code> back into characters, normalise
                special characters and remove extra spaces.</li>
              <li>Drop tickets with an <b>empty</b> message or answer, or one shorter than <b>20 characters</b>.</li>
              <li>Drop <b>exact duplicates</b>: the same message appearing twice, often across two CSV files.</li>
            </ul>
          </Stage>
          <Stage n="A3" title="Group look-alike tickets"
                 result={`${n(nd.tickets_in_multi_ticket_groups)} tickets are in look-alike groups (largest group: ${nd.largest_group ?? "…"})`}>
            Some tickets are almost copies of each other. We turn every ticket into a word-statistics vector
            (TF-IDF) and measure how similar each pair is (cosine similarity). Tickets that are at least
            <b> {Math.round((nd.threshold_cosine ?? 0.9) * 100)}% similar</b> are put in the same group.
            <p className="why">Why: if one copy is in the training data and the other in the test data, the
              model would be tested on something it has already seen, and the scores would look better than
              they really are. This is called <b>data leakage</b>.</p>
          </Stage>
          <Stage n="A4" title="Split into train, validation and test"
                 result="About 80% train, 10% validation, 10% test">
            Each whole group goes into <b>one</b> split only, and every split keeps the same mix of categories
            (a <i>stratified group split</i>, with a fixed random seed of 42 so it can be repeated).
            <ul>
              <li><b>Train:</b> the model learns from these.</li>
              <li><b>Validation:</b> used during training to check progress.</li>
              <li><b>Test:</b> kept aside and used only once at the end, for the final score.</li>
            </ul>
          </Stage>
          <Stage n="A5" title="Leakage check"
                 result={`Final split: ${n(split.train)} train / ${n(split.val)} validation / ${n(split.test)} test`}>
            Every validation and test ticket is compared with its closest training ticket. Those that are still
            too similar are removed: <b>{leak.val?.removed_near_duplicates ?? "…"}</b> validation and
            <b> {leak.test?.removed_near_duplicates ?? "…"}</b> test tickets. On average a test ticket is only
            {" "}{leak.test ? Math.round(leak.test.mean_similarity_to_train * 100) : "…"}% similar to its closest
            training ticket, so the test set is genuinely new.
          </Stage>
          <Stage n="A6" title="Prepare the input for the model"
                 result={`Max ${ml.max_input_len ?? "…"} input tokens and ${ml.max_target_len ?? "…"} output tokens`}>
            Every ticket is written as the same instruction:
            <pre className="prompt">{"Generate a professional customer support response.\nSubject: {subject}\nCustomer ticket: {body}"}</pre>
            The model reads <b>tokens</b> (pieces of words), not letters. We set the length limits so that 95%
            of tickets fit: only {f(ml.pct_inputs_truncated, 1)}% of inputs and {f(ml.pct_targets_truncated, 1)}%
            of answers are cut short. We also fix a random set of <b>500 test tickets</b> that every model is
            scored on, so the comparison is fair.
          </Stage>
        </Phase>

        <Phase id="phase-b" letter="B" title="Build the models" where="Notebooks 02, 03 and 05"
               intro="Goal: train the reply generator (the main GenAI part) and the three classifiers.">
          <Stage n="B1" title="Baseline: the original model (Model A)" where="Notebook 02"
                 result="A reference point to measure our training against">
            We take <b>FLAN-T5-small</b> from Google: an encoder-decoder Transformer with
            {" "}{tc.parameters ? (tc.parameters / 1e6).toFixed(0) : "77"} million parameters that reads a text
            and writes a new one. We give it the same instruction <b>without any training</b> and save its
            replies to the test tickets.
          </Stage>
          <Stage n="B2" title="Fine-tune the model (Model B)" where="Notebook 03, Tesla T4 GPU"
                 result={`Validation loss ${f(firstValLoss)} after epoch 1 → ${f(tc.best_val_loss)} after epoch ${tc.epochs_completed ?? 4}; test loss ${f(tc.test_loss)}`}>
            <p>Fine-tuning means continuing to train the model on our own examples. In simple words, one
              training step works like this:</p>
            <ol>
              <li>The model reads a ticket and tries to write the agent's reply, one token at a time.</li>
              <li>We compare its guess with the real reply. The size of the mistake is the <b>loss</b>.</li>
              <li>All {tc.parameters ? (tc.parameters / 1e6).toFixed(0) : "77"} million weights are adjusted a
                little so the mistake gets smaller next time.</li>
            </ol>
            <p>This repeats for {n(tc.train_examples)} examples in batches of {tc.effective_batch_size ?? 16},
              {" "}{stepsPerEpoch ? n(stepsPerEpoch) : "…"} steps per pass, {tc.epochs_completed ?? 4} passes
              (epochs), {stepsPerEpoch && tc.epochs_completed ? n(stepsPerEpoch * tc.epochs_completed) : "…"} steps
              in total. It took {tc.train_minutes ?? 30} minutes on Colab's free {tc.gpu || "GPU"}.</p>
            <dl className="facts small">
              <dt>Optimizer</dt><dd>AdamW, learning rate {hp.learning_rate ?? "3e-4"}, slow start over the first {hp.warmup_fraction != null ? Math.round(hp.warmup_fraction * 100) : 5}% of steps, then slowly lowered</dd>
              <dt>Weight decay</dt><dd>{hp.weight_decay ?? 0.01} (keeps weights small, reduces overfitting)</dd>
              <dt>Checking</dt><dd>Validation loss after every epoch: it went down every time, so no overfitting</dd>
            </dl>
            <div className="before-after" style={{ marginTop: 12 }}>
              <div>
                <span className="tag">Before (Model A)</span>
                <p className="reply-text">Sorry, I can't help you with this problem. Please help me fix it.</p>
              </div>
              <div className="after">
                <span className="tag">After (Model B, ours)</span>
                <p className="reply-text">Dear [Customer Name], we apologize for the inconvenience caused by the server
                  crashing after the recent update… could you please provide us with more details about the error
                  message you are receiving and the steps you have taken so far?…</p>
              </div>
            </div>
            <p className="small muted" style={{ marginTop: 6 }}>Real output for the ticket "Server keeps crashing after update".</p>
          </Stage>
          <Stage n="B3" title="Train the classifiers" where="Notebook 05"
                 result={`Linear SVM won for all three: category, priority, type`}>
            <ul>
              <li><b>Turn text into numbers</b> with TF-IDF, using whole words (1–2 word phrases) and
                3–5 letter pieces, so typos still match.</li>
              <li><b>Try two models</b>, Logistic Regression and Linear SVM, each with four settings
                (C = 0.1, 0.5, 1, 4). Small classes get extra weight so they are not ignored.</li>
              <li><b>Pick the best</b> on the validation set using macro-F1. Linear SVM with C = 4 won every time.</li>
              <li><b>Add confidence scores:</b> the SVM is wrapped in a calibrator, so the app can show "74% confidence".</li>
            </ul>
            <p style={{ marginTop: 8 }}>Three separate models: <b>category</b> (10 classes), <b>priority</b> (3) and <b>type</b> (4).</p>
          </Stage>
          <Stage n="B4" title="Ready-made models (no training)" result="Sentiment and summary come from existing models">
            <ul>
              <li><b>Sentiment:</b> a pretrained RoBERTa model (trained on tweets) labels the ticket negative,
                neutral or positive. The dataset has no sentiment labels, so it is shown as <i>inferred</i>.</li>
              <li><b>Summary:</b> the bigger FLAN-T5-base, asked to "summarize in one sentence" (zero-shot).</li>
            </ul>
          </Stage>
          <div className="panel table-wrap" style={{ marginTop: 4 }}>
            <table>
              <thead><tr><th>Part of the app</th><th>How it is made</th><th>Trained by us?</th></tr></thead>
              <tbody>
                {PARTS.map(([what, how, ours], i) => (
                  <tr key={what}><td><b>{what}</b></td><td>{how}</td><td className={i < 2 ? "better" : "muted"}>{ours}</td></tr>
                ))}
              </tbody>
            </table>
          </div>
        </Phase>

        <Phase id="phase-c" letter="C" title="Test the models" where="Notebooks 04 and 06"
               intro="Goal: prove that training helped, using tickets the models have never seen.">
          <Stage n="C1" title="Generate replies" result="1,000 replies: 500 from each model">
            Both models answer the same <b>500 fixed test tickets</b> with the same settings: <b>beam search</b>
            {" "}with 4 beams (it keeps the 4 best partial replies at each step and picks the best full one) and
            no repeated 3-word phrases.
          </Stage>
          <Stage n="C2" title="Score the replies against the real agent replies"
                 result={`ROUGE-L ${f(a?.rougeL)} → ${f(b?.rougeL)}, BLEU ${f(a?.bleu, 2)} → ${f(b?.bleu, 2)}, meaning similarity ${f(a?.semantic_cosine)} → ${f(b?.semantic_cosine)}`}>
            <ul>
              <li><b>ROUGE and BLEU:</b> how many words and phrases match the real reply.</li>
              <li><b>BERTScore and cosine similarity:</b> whether the <i>meaning</i> matches, even with different words.</li>
              <li><b>Statistical tests</b> (Wilcoxon test and bootstrap) check that the improvement is not luck.
                It is significant for every metric.</li>
            </ul>
          </Stage>
          <Stage n="C3" title="AI judge"
                 result={judge ? `Quality ${fmt(judge.mean_scores[judgeNames[0]].overall, 2)} → ${fmt(judge.mean_scores[judgeNames[1]].overall, 2)} out of 5; ours better on ${judge.pairwise.finetuned_better} of ${judge.items}` : "…"}>
            Word-matching scores can't see tone or mistakes, so an AI model (Claude) rated {judge?.items ?? 50} pairs
            of replies from 1 to 5 on relevance, correctness, fluency, professional tone and helpfulness. The replies
            were shuffled and unlabelled, so the judge didn't know which model wrote which.
          </Stage>
          <Stage n="C4" title="Test the classifiers"
                 result={`Accuracy: type ${f(cls.type?.accuracy, 2)}, priority ${f(cls.priority?.accuracy, 2)}, category ${f(cls.queue_group?.accuracy, 2)}`}>
            All {n(split.test)} test tickets, compared with two "dumb" guessers (always the most common class,
            and random guessing). Ours beat both on every label.
          </Stage>
          <Stage n="C5" title="Compare and export" where="Notebook 06" result="One ZIP file with everything the app needs">
            A final side-by-side comparison on 100 tickets. Then everything is packed into
            <code> supportgen_models.zip</code>: the fine-tuned model, the three classifiers, and all scores
            and charts (<code>results/</code>).
          </Stage>
        </Phase>

        <Phase id="phase-d" letter="D" title="Run the app" where="Django backend + React website on a laptop"
               intro="Goal: let a support agent use the models through a simple website.">
          <Stage n="D1" title="Set up" result="The models stay in memory, so each ticket takes about 1–2 seconds">
            The ZIP is unzipped into the project. When the Django server gets its first request, it loads every
            model <b>once</b> and keeps it in memory. No internet or paid API is needed, and customer data never
            leaves the machine.
          </Stage>
          <Stage n="D2" title="What happens when you click “Analyze ticket”"
                 result="Draft reply + summary + action + category, priority, type and sentiment, saved in the database">
            <ol className="req-flow">
              <li><b>Send:</b> the website sends the subject and message to the backend
                (<code>POST /api/tickets/analyze/</code>).</li>
              <li><b>Check:</b> the message must be 10–5,000 characters, or an error is shown.</li>
              <li><b>Write the reply:</b> build the same instruction as in training → split into tokens →
                beam search with the fine-tuned model → turn tokens back into text.</li>
              <li><b>Tidy the reply:</b> fix placeholders the model writes oddly, e.g. <code>tel_num&gt;</code> →
                <code> [Phone]</code> (T5's vocabulary has no "&lt;" character).</li>
              <li><b>Label the ticket:</b> the three classifiers give category, priority and type, each with a
                confidence score.</li>
              <li><b>Sentiment:</b> the RoBERTa model gives negative, neutral or positive.</li>
              <li><b>Summary:</b> FLAN-T5-base writes one sentence. If it only repeats the subject, the app uses the
                first useful sentences of the ticket instead.</li>
              <li><b>Recommended action:</b> a rule picks the next step for the predicted category, and adds
                "Handle with high priority" for urgent tickets.</li>
              <li><b>Save and return:</b> everything is stored in PostgreSQL and sent back to the website with the
                time each step took.</li>
            </ol>
          </Stage>
          <Stage n="D3" title="The other pages" result="Every score on the website comes from the Colab results; none is typed in">
            <ul>
              <li><b>History and Dashboard</b> read the saved tickets from the database.</li>
              <li><b>Model comparison</b> runs the original and the fine-tuned model live on the same ticket.</li>
              <li><b>Model evaluation</b> reads the score files exported in step C5.</li>
            </ul>
          </Stage>
        </Phase>
      </Section>

      <Section n="3" title="How we know it works">
        <p className="lede">Scores on test tickets the models had never seen, compared with the real agent replies.</p>
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
            <div className="value num">{judge ? `${fmt(judge.mean_scores[judgeNames[0]].overall, 1)} → ${fmt(judge.mean_scores[judgeNames[1]].overall, 1)}` : "…"}</div>
            <div className="label">AI-judge quality score (out of 5)</div>
          </div>
          <div className="stat">
            <div className="value num">{judge ? `${judge.pairwise.finetuned_better} / ${judge.items}` : "…"}</div>
            <div className="label">Tickets where our reply was judged better</div>
          </div>
        </div>
        <p className="small muted" style={{ marginTop: 8 }}>Details on the <Link to="/evaluation">Model evaluation</Link> page.</p>
      </Section>

      <Section n="4" title="The website">
        <div className="panel table-wrap">
          <table>
            <tbody>
              {PAGES.map(([to, name, text]) => (
                <tr key={to}><td style={{ whiteSpace: "nowrap" }}><Link to={to}><b>{name}</b></Link></td><td>{text}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section n="5" title="Limits, and why a human stays in charge">
        <div className="panel">
          <ul style={{ margin: 0, paddingLeft: 20, display: "grid", gap: 6 }}>
            <li>Replies can sound repetitive: many of them offer a phone call.</li>
            <li>The model can get a fact wrong, for example saying a refund was received when it was not.</li>
            <li>The category and priority labels are right about 6 times out of 10, so they come with a confidence score.</li>
            <li>The data is synthetic, so real company tickets may be harder.</li>
          </ul>
          <p style={{ marginTop: 12 }}><b>That is why every reply is a draft.</b> The agent reads it, fixes it if
            needed, and then sends it.</p>
        </div>
      </Section>

      <Section n="6" title="Built with">
        <div className="btn-row">
          {["Python", "PyTorch", "Hugging Face Transformers", "scikit-learn", "Google Colab", "Django REST Framework",
            "PostgreSQL", "React", "Vite"].map((t) => <span className="tag" key={t}>{t}</span>)}
        </div>
      </Section>
    </>
  );
}
