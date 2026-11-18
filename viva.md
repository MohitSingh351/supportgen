# SupportGen: Viva Questions and Answers

Short, direct answers to the questions an examiner is likely to ask. All numbers come from `results/` (the Colab export). For the full project description, see [`README.md`](README.md).

**Numbers to remember**

| | |
|---|---|
| Base model | FLAN-T5-small, ~77M parameters, encoder-decoder |
| Data split | 18,848 train / 2,354 val / 2,339 test (English, synthetic) |
| Training | 4 epochs, LR 3e-4, batch 16, Colab T4, ~30 min |
| Loss | best validation 1.122, test 1.111 (perplexity 3.04) |
| ROUGE-L | 0.158 → **0.343** |
| BLEU | 1.17 → **19.57** |
| BERTScore F1 | 0.867 → **0.899** (rescaled 0.210 → 0.403) |
| LLM judge overall | 1.59 → **4.03** out of 5; fine-tuned preferred on 50 of 50 |
| Classifier accuracy | type 0.88 · priority 0.64 · queue 0.62 |
| Model size / speed | 77.0M parameters, 310 MB on disk · 1.60 s per reply on a T4 GPU · ~1 s on an M4 Mac CPU |

---

## Contents
1. [Project overview](#1-project-overview)
2. [Generative AI and Transformer theory](#2-generative-ai-and-transformer-theory)
3. [Dataset and preprocessing](#3-dataset-and-preprocessing)
4. [Fine-tuning](#4-fine-tuning)
5. [Evaluation](#5-evaluation)
6. [Classifiers, sentiment, summary and action](#6-classifiers-sentiment-summary-and-action)
7. [Backend, frontend and system design](#7-backend-frontend-and-system-design)
8. [Limitations, ethics and future work](#8-limitations-ethics-and-future-work)
9. [Tricky questions](#9-tricky-questions)
10. [Model choice: why FLAN-T5-small, and what else was tried](#10-model-choice-why-flan-t5-small-and-what-else-was-tried)
11. [Explaining the Model evaluation page, section by section](#11-explaining-the-model-evaluation-page-section-by-section)
12. [Explaining failure cases from the live demo](#12-explaining-failure-cases-from-the-live-demo)
13. [More questions examiners often ask](#13-more-questions-examiners-often-ask)

---

## 1. Project overview

**Q1. What is your project in one sentence?**
SupportGen is a copilot for customer-support agents: it reads a ticket and generates a draft reply using a FLAN-T5 model that we fine-tuned on support conversations, and it also predicts the ticket's category, priority, type and sentiment, writes a summary and suggests the next action.

**Q2. What problem does it solve?**
Support agents answer many similar tickets, and writing every reply from scratch is slow. SupportGen drafts the reply and triages the ticket, so the agent only reviews and edits. It saves time and keeps the tone consistent.

**Q3. What is the main contribution? Isn't this just calling an API?**
No. The core contribution is our own pipeline: we prepared a dataset, fine-tuned a Transformer, evaluated it against the untrained version with several metrics and significance tests, and deployed it in a web app that runs locally. Gemini is only an optional comparison point; the app works fully without it.

**Q4. Why is this a Generative AI project?**
The central component *generates* new text: the reply is produced token by token by a sequence-to-sequence language model. It is not picked from a list of templates or retrieved from a database.

**Q5. Does the system reply to customers automatically?**
No. It is a human-in-the-loop design. The agent sees a draft and decides whether to send, edit or discard it. This matters because the model can make factual mistakes.

**Q6. What are the main parts of the system?**
Four parts: (1) data preparation, (2) model training in Google Colab, (3) a Django REST backend that runs the models and stores results, and (4) a React frontend for agents.

---

## 2. Generative AI and Transformer theory

**Q7. What is a Transformer?**
A neural network architecture built on **self-attention**, introduced in "Attention Is All You Need" (2017). Instead of reading text word by word like an RNN, it looks at all tokens at once and learns how much each token should attend to every other token. This makes it parallelisable and good at long-range context.

**Q8. Explain self-attention simply.**
Each token creates three vectors: a **query** ("what am I looking for?"), a **key** ("what do I contain?") and a **value** ("what do I pass on?"). A token's query is compared with every key using a dot product. The scores are scaled by √d and passed through softmax to get weights, and the output is the weighted sum of the values. **Multi-head attention** does this several times in parallel so different heads can learn different relationships.

**Q9. What are the three Transformer families? Which one is FLAN-T5?**
- **Encoder-only** (BERT, RoBERTa): understanding tasks such as classification.
- **Decoder-only** (GPT, Llama): open-ended text generation.
- **Encoder-decoder** (T5, BART): the encoder reads the input, and the decoder generates an output conditioned on it.

FLAN-T5 is **encoder-decoder**. That fits our task: read a ticket, write a reply.

**Q10. What is T5, and what does FLAN add?**
**T5** (Text-to-Text Transfer Transformer) treats every NLP task as text in → text out. **FLAN** is instruction tuning: Google further trained T5 on more than 1,000 tasks phrased as instructions. That is why FLAN-T5 can follow prompts like "Generate a professional customer support response" even before we fine-tune it.

**Q11. Why FLAN-T5-small instead of a bigger model?**
- It trains in about 30 minutes on Colab's free T4 GPU.
- It runs on a laptop CPU in about 2 seconds per ticket, with no API cost.
- It is small enough that the improvement from fine-tuning is clearly visible.

The trade-off is lower quality than larger models, which we document in the limitations.

**Q12. What is fine-tuning? How is it different from pre-training?**
**Pre-training** teaches a model general language from huge text corpora. **Fine-tuning** continues training an already pre-trained model on a smaller, task-specific dataset, here ticket → reply pairs, so it adapts to our domain and style. It needs far less data and compute than pre-training because the model already knows the language.

**Q13. What is zero-shot? What is your zero-shot baseline?**
Zero-shot means using a model on a task without any task-specific training, only an instruction. Our baseline (Model A) is the original FLAN-T5-small given the same prompt but with no fine-tuning. Comparing A with B isolates the effect of our training.

**Q14. What is tokenisation? Which tokenizer does T5 use?**
Tokenisation splits text into units the model understands. T5 uses a **SentencePiece** sub-word tokenizer with a 32k vocabulary, so rare words are split into smaller pieces. This tokenizer is also the reason for one of our bugs (see Q48).

**Q15. How does the model generate text? What decoding did you use?**
The decoder produces one token at a time, each conditioned on the input and the tokens already generated. We used **beam search with 4 beams**, which keeps the 4 most likely partial sequences at each step instead of only the single best (greedy). We also set `no_repeat_ngram_size=3`, which forbids repeating any 3-word sequence, a common fix for repetitive output.

**Q16. Greedy vs beam search vs sampling?**
- **Greedy:** always pick the most likely next token. Fast, but can miss better overall sentences.
- **Beam search:** keeps several candidates and picks the best complete sequence. More reliable, so it suits professional replies.
- **Sampling** (temperature, top-k, top-p): picks tokens at random according to their probabilities. More creative and varied, but less predictable.

We chose beam search because consistency matters more than creativity in support replies.

**Q17. What is the loss function during fine-tuning?**
**Cross-entropy loss** over the target tokens: at each position, how much probability the model gave to the correct next token of the real reply. Training uses **teacher forcing**, meaning the decoder is fed the real previous tokens rather than its own predictions.

**Q18. What is perplexity? What is yours?**
Perplexity = e^loss. Roughly, it is how many tokens the model is "choosing between" on average. Our test loss of 1.111 gives a perplexity of **3.04**, which means the model is fairly confident about the next token in a support reply.

---

## 3. Dataset and preprocessing

**Q19. Which dataset did you use?**
`Tobi-Bueck/customer-support-tickets` from Hugging Face. Each row has a subject, the ticket body, the agent's answer, and labels for queue, priority and type. The license is CC BY-NC 4.0, which allows non-commercial use with attribution.

**Q20. Is the data real?**
No. The dataset card says it is **synthetically generated**. We state this clearly: our results show performance on synthetic support conversations, and real tickets would likely be messier.

**Q21. How did you clean the data?**
1. Kept only English rows (28,261 of 61,765).
2. Removed empty or broken rows and normalised whitespace.
3. Removed exact duplicates (4,597 rows; 4,592 of them were tickets shared between two of the dataset's CSV files).
4. Grouped near-duplicates so similar tickets stay in the same split.

This left 23,557 unique English tickets.

**Q22. What is data leakage, and how did you prevent it?**
Leakage means test data, or something very similar to it, is seen during training. The model then looks better on the test set than it really is. We prevented it by:
- splitting by **near-duplicate groups**, not by individual rows, and
- running a **leakage audit** that compared val and test tickets against training and removed 8 val and 8 test tickets that were too similar.

Final split: **18,848 train / 2,354 val / 2,339 test**.

**Q23. Why do you need three splits?**
- **Train:** the model learns from it.
- **Validation:** used during training to pick the best checkpoint and tune settings.
- **Test:** used only once at the end, for an unbiased final score.

**Q24. What were the input and output of the model?**
- **Input:** `Generate a professional customer support response.\nSubject: {subject}\nCustomer ticket: {body}`
- **Output (target):** the real agent reply.

The maximum lengths are 160 input tokens and 128 output tokens, chosen from the token-length distribution plot (`results/plots/token_lengths.png`) to cover most tickets while keeping training fast.

**Q25. Why did you drop the German data?**
To keep the task focused and the evaluation clean: the English metrics (BERTScore with roberta-large, an English sentence-embedding model) and the English demo all assume English. Multilingual support is listed as future work.

---

## 4. Fine-tuning

**Q26. What were your hyperparameters?**
Full fine-tuning of FLAN-T5-small with AdamW, learning rate 3e-4, batch size 16, 4 epochs (4,712 steps), max input 160 and max target 128 tokens, on a Colab Tesla T4 in about 30 minutes.

**Q27. Why such a high learning rate (3e-4)?**
T5 models are commonly fine-tuned with larger learning rates (1e-4 to 1e-3) than BERT-style models (around 2e-5). T5 was pre-trained with Adafactor at high learning rates, and 3e-4 is a standard, stable choice for it. Our loss curves show smooth convergence with no divergence.

**Q28. Full fine-tuning or LoRA? Why?**
Full fine-tuning, meaning all ~77M weights are updated. The model is small enough to fit easily on a T4, so we did not need a parameter-efficient method. For a larger model (FLAN-T5-large, XL), **LoRA** would be the better choice: it freezes the original weights and trains small low-rank adapter matrices, which uses far less memory.

**Q29. How do you know the model did not overfit?**
- The loss curves (`results/plots/loss_curves_B_ft_flan-t5-small.png`) show validation loss falling at every epoch (1.272 → 1.178 → 1.137 → 1.122). It never started rising, which is the usual sign of overfitting.
- Best validation loss was 1.122 and test loss was 1.111, almost identical, so the model generalises to unseen tickets.
- Training loss (1.351, averaged over training) is not far below validation loss.

**Q30. Why only 4 epochs?**
The gains were getting small: validation loss went 1.272 → 1.178 → 1.137 → 1.122 over the four epochs, so the last epoch improved it by only 0.015. More epochs would give little benefit while risking overfitting and even more formulaic replies, and 4 epochs fit comfortably within Colab's session limits. Training longer is a reasonable experiment for future work.

*Be precise if asked about early stopping:* early stopping (patience 2) was configured, but the training log shows the Trainer disabled it because `metric_for_best_model` was not set. All 4 epochs ran. Validation loss improved every epoch, so early stopping would not have stopped training anyway, and the result is the same.

**Q31. Why did you use FP32 and not mixed precision?**
The T4 GPU does not support bf16, and fp16 is known to be unstable for T5 (it can produce NaN losses). So training ran in fp32. The notebook enables bf16 automatically on newer GPUs such as the A100 or L4.

---

## 5. Evaluation

**Q32. How did you evaluate the generated replies?**
On 500 held-out test tickets, we compared the base model (A) and the fine-tuned model (B) against the real agent replies using:
- **Lexical metrics:** ROUGE-1/2/L and BLEU (word overlap)
- **Semantic metrics:** BERTScore and embedding cosine similarity (meaning)
- **Statistics:** paired Wilcoxon tests and bootstrap 95% confidence intervals
- **LLM-as-a-judge:** Claude rated 50 anonymised replies on five criteria
- **Qualitative review:** example replies and error analysis

**Q33. What is ROUGE?**
Recall-Oriented Understudy for Gisting Evaluation. It measures overlap between generated and reference text:
- **ROUGE-1:** overlapping single words
- **ROUGE-2:** overlapping word pairs
- **ROUGE-L:** the longest common subsequence, which rewards correct word order

Our ROUGE-L went from 0.158 to **0.343**.

**Q34. What is BLEU?**
Bilingual Evaluation Understudy, originally built for machine translation. It measures n-gram *precision* (n = 1 to 4) with a **brevity penalty** for outputs that are too short. Ours went from 1.17 to **19.57**. The base model's BLEU is very low partly because its replies are short (18 words vs 60 in the references), so the brevity penalty hits hard.

**Q35. Why not rely only on ROUGE and BLEU?**
They only count matching words. Two replies can mean the same thing with different words and still score low. So we added:
- **BERTScore:** matches tokens using contextual embeddings from roberta-large, so synonyms count as matches. F1 went from 0.867 to **0.899**.
- **Embedding cosine similarity:** both texts are encoded with `all-MiniLM-L6-v2`, and we measure the angle between the vectors. It went from 0.580 to **0.781**.

**Q36. BERTScore went up "only" from 0.867 to 0.899. Is that a small improvement?**
No. Raw BERTScore values are squeezed into a narrow, high range because any two English sentences already share some similarity. The **rescaled** BERTScore (adjusted against a baseline) went from 0.210 to **0.403**, almost double. The increase is also highly significant (p ≈ 3e-63).

**Q37. How do you know the improvement is not luck?**
- **Paired Wilcoxon signed-rank test:** compares A and B on the *same* tickets, without assuming a normal distribution. Every metric gives p < 1e-50.
- **Bootstrap 95% confidence intervals:** we resampled the test tickets many times. The CI for the difference never includes 0; for ROUGE-1 it is [+0.253, +0.284].
- **Per ticket:** B has higher ROUGE-1 than A on 92% of tickets.

**Q38. What is LLM-as-a-judge, and what did it show?**
A strong LLM (here Claude, `claude-opus-5-5`) is given a rubric and rates outputs, a scalable stand-in for human rating. We gave it 50 tickets with the A and B replies anonymised and shuffled, so it could not tell which model wrote which. It rated each reply 1–5 on relevance, correctness, fluency, professional tone and helpfulness.
- Overall: A = 1.59, B = **4.03**.
- B was preferred on **50 of 50** tickets (Wilcoxon p = 7.2e-10).
- 46 of 50 base replies were not support replies at all.

**Q39. Is LLM-as-a-judge the same as human evaluation?**
No, and we label it clearly as a model's rating. LLM judges can have biases, such as preferring longer or more fluent answers. We reduced these by blinding and shuffling the replies and using a fixed rubric (`results/llm_judge_rubric.md`). A human evaluation sheet (`results/human_eval_sheet.csv`, with a separate answer key) is prepared for a proper human study.

**Q40. Which metric do you trust most?**
No single metric. Lexical metrics (ROUGE, BLEU) show the model learned the domain's wording. Semantic metrics (BERTScore, cosine) show the meaning moved closer to real replies. The LLM judge checks qualities that metrics miss, such as tone and correctness. All of them point the same way, which is what makes the result convincing.

**Q41. What is "distinct-2", and why did it drop?**
Distinct-n is the share of unique n-grams across all outputs, a measure of diversity. Distinct-2 dropped from 0.55 (base) to 0.16 (fine-tuned). The fine-tuned model learned a common reply template, which makes it more relevant but more repetitive. For example, 85% of its replies offer a phone call. This is a known side effect of fine-tuning a small model.

---

## 6. Classifiers, sentiment, summary and action

**Q42. How do you predict category, priority and type?**
Three separate **TF-IDF + Linear SVM** classifiers (scikit-learn), trained on `subject + " . " + body`:
- **TF-IDF** turns text into a vector of word weights. A word gets more weight when it is frequent in this ticket but rare across all tickets.
- **Linear SVM** finds the boundary that best separates the classes.

The regularisation parameter C was chosen on the validation set using macro F1.

**Q43. Why classic ML and not a Transformer for classification?**
TF-IDF + SVM is fast (milliseconds), small, easy to explain and a strong baseline for short-text classification. The project's GenAI focus is the generator; the classifiers are supporting components. Fine-tuning a BERT-style classifier is listed as future work.

**Q44. What are the classifier results?**

| Target | Accuracy | Macro F1 | Majority baseline |
|---|---|---|---|
| Type (4 classes) | 0.881 | 0.871 | 0.391 |
| Priority (3 classes) | 0.643 | 0.580 | 0.413 |
| Queue (10 classes) | 0.620 | 0.593 | 0.279 |

All beat the majority-class and random baselines by a wide margin.

**Q45. Why are queue and priority harder?**
- **Queue:** several queues overlap in meaning ("Technical Support", "IT Support", "Product Support"), and the confusion matrix shows most errors between those similar queues.
- **Priority:** urgency is subjective, and similar tickets can carry different priorities in a synthetic dataset.

**Q46. Why macro F1 and not only accuracy?**
The classes are imbalanced. Accuracy can look good while the model ignores small classes; macro F1 averages F1 over classes equally, so small classes count as much as large ones.

**Q47. How do you get sentiment? Is it evaluated?**
With the pretrained `cardiffnlp/twitter-roberta-base-sentiment-latest`, used as-is. The dataset has no sentiment labels, so we cannot measure its accuracy. That is why the app labels it **inferred**. On 500 sampled tickets it gave 55% negative, 28% neutral and 17% positive, which is plausible for support tickets.

**Q48. How are the summary and recommended action produced?**
The dataset has no reference summaries or actions, so we could not train for them.
- **Summary:** zero-shot FLAN-T5-base. In evaluation it often just repeated the subject line, so the code detects degenerate output and falls back to an **extractive summary** (the first informative sentences, skipping greetings).
- **Action:** zero-shot actions were unusable, so by default we use **rules** based on the predicted queue and priority, e.g. Billing → "Check the customer's payment and refund records…", with "Handle with high priority." added in front for high-priority tickets. Rules are predictable and never invent facts.

The app always shows which method was used (`summary_source`, `action_source`).

---

## 7. Backend, frontend and system design

**Q49. Describe the architecture.**
React frontend → Django REST API → AI services (`backend/ai/`) → models on disk, with results stored in PostgreSQL or SQLite. The AI code has no Django imports, so it can be tested and reused on its own.

**Q50. What happens when an agent clicks "Analyze"?**
1. The frontend sends `POST /api/tickets/analyze/` with the subject and body.
2. The backend validates the input (not empty, at least 10 and at most 5,000 characters).
3. `TicketAnalyzer` runs the five steps: reply, classifiers, sentiment, summary, action.
4. The reply text is cleaned (placeholder repair).
5. The result is saved to the database and returned as JSON, including per-step timings.
6. The frontend displays it.

**Q51. How do you keep responses fast?**
Models are loaded **once per process** in `ai/registry.py` and kept in memory, not loaded per request. Loading takes seconds; generating takes milliseconds to a couple of seconds. After warm-up, a full analysis takes about 2 s on an M4 laptop CPU.

**Q52. Explain the `<tel_num>` placeholder bug.**
The dataset anonymises personal data with placeholders such as `<tel_num>` and `<name>`. T5's SentencePiece vocabulary has **no `<` character**, so the fine-tuned model learned to output `tel_num>`. `ai/postprocess.py` uses regular expressions to turn every variant (`tel_num>`, `<tel_num>`, `[Tel Num]`) into a readable form such as `[Phone]`, which the agent then fills in.

**Q53. How do you handle errors?**
- Missing fine-tuned model → **503 Service Unavailable**.
- Bad input (empty, too short, too long) → **400 Bad Request**.
- Optional parts (classifiers, sentiment, summary model) fail → the reply is still returned with a warning.
- Database failure → the analysis is still returned with a warning.

**Q54. How did you test the backend?**
32 tests with Django's test framework:
- unit tests for the AI helpers (placeholder repair, degenerate-output detection, rules),
- API tests with fake models (validation, error codes, saving),
- optional tests that load the real models.

**Q55. Why Django and React?**
Django + Django REST Framework give us an ORM, migrations, an admin panel and a clean REST API with little code, and the ML ecosystem is in Python anyway. React gives an interactive single-page app for the dashboard, charts and comparison views. Vite provides fast development builds.

**Q56. How does the comparison page work?**
`POST /api/compare/` sends the same ticket through every available **provider**: base FLAN-T5, fine-tuned FLAN-T5 and, if `GEMINI_API_KEY` is set, Gemini. All providers implement the same interface (`ai/providers.py`), so adding a new model means writing one class.

**Q57. Why compare with Gemini at all?**
To position our small model against a large commercial LLM. Gemini will usually write more fluent replies, but our model runs **locally**: no API key, no per-request cost, and customer data never leaves the company. That trade-off is the business case for a small fine-tuned model.

**Q58. Are the numbers on the Evaluation page hard-coded?**
No. The page reads them from the JSON, CSV and PNG files in `results/` through `/api/evaluation/`. If the notebooks are re-run and exported again, the page updates automatically.

---

## 8. Limitations, ethics and future work

**Q59. What are the main limitations?**
1. **Formulaic replies:** 85% offer a phone call, and distinct-2 is 0.16.
2. **Factual slips:** e.g. "the refund you received" when the refund was *not* received. Small models handle negation poorly.
3. **Synthetic data:** may not reflect real tickets.
4. **Moderate classifiers** for queue and priority (~62–64%).
5. **No knowledge of company policy:** the model does not know real prices, order status or policies.
6. **Unevaluated parts:** sentiment, summary and action have no ground truth.

**Q60. How would you fix the factual errors?**
- **RAG (retrieval-augmented generation):** retrieve relevant policy documents or similar resolved tickets and add them to the prompt, so the model grounds its reply in facts.
- A **larger model** (FLAN-T5-base or large with LoRA) that handles negation better.
- Keep the **human-in-the-loop** review, which is already part of the design.

**Q61. How would you make replies less repetitive?**
Sampling with a moderate temperature or top-p instead of pure beam search, a repetition or diversity penalty, de-duplicating repetitive training replies, or training on more varied data.

**Q62. What ethical issues did you consider?**
- **Privacy:** the dataset is anonymised, and the model runs locally, so ticket data is not sent to third parties (unless the optional Gemini comparison is used).
- **Human oversight:** replies are drafts only, never sent automatically.
- **Transparency:** the app shows confidence scores and marks sentiment as "inferred" and the action as "rule-based".
- **Licensing:** CC BY-NC 4.0 means non-commercial use only, with attribution.
- **Honest reporting:** we report limitations and label LLM-judge results as a model's rating, not human evaluation.

**Q63. What would you do with more time?**
- Fine-tune a larger model with LoRA.
- Add RAG over a knowledge base.
- Run a proper human evaluation (the sheet is already prepared).
- Replace TF-IDF with a fine-tuned BERT/DeBERTa classifier.
- Collect agents' edits as feedback for further training.
- Support German and other languages.
- Deploy with Docker and a GPU inference server.

---

## 9. Tricky questions

**Q64. If BLEU is only 19.57, isn't the model bad?**
No. BLEU compares against a *single* reference reply, but there are many valid ways to answer a ticket. For open-ended generation, scores around 20 are reasonable. What matters is the jump from 1.17 to 19.57 on the same tickets, confirmed by semantic metrics and the LLM judge.

**Q65. The fine-tuned model writes longer replies. Is the score higher just because of length?**
Partly length is involved: the base model's replies are much shorter (18 vs 60 words), which hurts its recall and BLEU. But the fine-tuned replies (68 words) are close to the reference length (60), which is the correct behaviour. The length-independent signals, such as BERTScore precision, cosine similarity and the judge's relevance and tone scores, improve too.

**Q66. Could the model have memorised the test set?**
Very unlikely. We split by near-duplicate groups and audited val and test against training for leakage. Also, test loss (1.111) is close to validation loss (1.122), which is consistent with generalisation rather than memorisation.

**Q67. Why did you use Claude as a judge when you compare against Gemini?**
The judge must be independent of the models being compared. Claude did not generate any of the A or B replies, the replies were blinded and shuffled, and a fixed rubric was written before rating. Gemini was not part of the judged set.

Be honest about the caveats, which the Evaluation page also lists: there is a single AI rater, it may favour longer or more polite replies, Claude was also used as an assistant while building this project, and blinding is imperfect because the fine-tuned model's style is easy to recognise. That is why it complements the automatic metrics rather than replacing a human study.

**Q68. Why not just use ChatGPT or Gemini for everything?**
Cost per request, latency, dependence on an internet connection and a third-party service, and privacy, because customer data would leave the company. A small fine-tuned model runs on a normal laptop, costs nothing per request and can be fully controlled and audited. The project also demonstrates the GenAI pipeline end to end instead of wrapping an API.

**Q69. What happens if a customer writes in German or Hindi?**
The model was trained only on English, so quality would drop sharply. It might reply in English or produce poor text. The classifiers would also be unreliable. Language detection with a "not supported" warning, or multilingual training, would fix this.

**Q70. What did you personally learn from this project?**
The full GenAI lifecycle: careful data preparation (especially avoiding leakage), fine-tuning a Transformer, evaluating it with several complementary metrics and significance tests, being honest about failure modes, and turning a model into a usable application with sensible engineering such as model caching, fallbacks and error handling.

---

## 10. Model choice: why FLAN-T5-small, and what else was tried

**Q71. Why did you use only FLAN-T5-small? Why not a bigger or different model?**
It was a deliberate choice that fits the task, the hardware and the goal:
1. **The right architecture.** Reply generation is "read an input, write an output", which is exactly what an encoder-decoder model is built for.
2. **Already instruction-tuned.** FLAN-T5 already follows instructions, so a small dataset and a short training run are enough to adapt it.
3. **Fits the free hardware.** Full fine-tuning of 77M parameters takes about 30 minutes on Colab's free T4 GPU and needs only about 441 MB of GPU memory to generate.
4. **Runs on a laptop.** About 1 second per reply on a Mac CPU, with no GPU, no API key and no cost per request. Customer data stays on the machine.
5. **A clean experiment.** Using the same model before and after fine-tuning (A vs B) means any improvement comes from our training, not from a different model.

The trade-off is quality: a larger model would make fewer factual mistakes. We accept that and list it as a limitation.

**Q72. Did you try other models?**
Be precise. This is what was actually done:

| Component | What was compared |
|---|---|
| Reply generation | **Base FLAN-T5-small (A)** vs **fine-tuned FLAN-T5-small (B)** on the same 500 test tickets. Notebook 03 is also set up to fine-tune **FLAN-T5-base (~248M, "Model D")**, and notebook 06 can add **Gemini** as a benchmark, but neither was run, because of Colab GPU time limits and to keep the project free of paid APIs. |
| Classifiers | **Logistic Regression vs Linear SVM**, each with several values of the regularisation strength C, chosen by validation macro-F1. Linear SVM with C = 4 won for all three targets (validation macro-F1: queue 0.666 vs 0.587 for logistic regression; priority 0.677 vs 0.637; type 0.884 vs 0.869). |
| Summary and action | **Zero-shot FLAN-T5-base** was tested in notebook 04. It mostly repeated the subject line, so we added an extractive fallback for the summary and switched the action to rules. |
| Sentiment | A pretrained RoBERTa sentiment model, used as-is (there are no labels to train or compare against). |

So the honest answer is: "We compared base vs fine-tuned FLAN-T5-small, and two classifier families with hyperparameter search. FLAN-T5-base and Gemini are prepared in the pipeline but were not run; they are the next experiments."

**Q73. Why not GPT-2 or another decoder-only model?**
GPT-2 is not instruction-tuned and, at a similar size (124M), it tends to wander off topic and is harder to control. Decoder-only models also treat the ticket and the reply as one long sequence, whereas an encoder-decoder model reads the whole ticket in both directions first and then generates. For short input → output tasks at small sizes, T5-style models are a strong, standard choice.

**Q74. Why not BART?**
BART is also a good encoder-decoder model and would be a fair alternative. FLAN-T5 has the advantage of instruction tuning on more than 1,000 tasks, so it understands a prompt such as "Generate a professional customer support response" from the start. BART-base is not instruction-tuned.

**Q75. Why not fine-tune Llama, Mistral or another modern LLM?**
A 7B-parameter model is about 90 times larger than FLAN-T5-small. Even with LoRA and 4-bit quantisation, it needs a bigger GPU, takes hours to train, and is slow on a laptop CPU. That would also make the before/after effect of fine-tuning harder to show within the course's time and budget. It is a natural next step with more compute.

**Q76. Why not just use Gemini or GPT with a good prompt (prompt engineering) instead of fine-tuning?**
1. The course goal is to build and evaluate the GenAI pipeline ourselves, not to call an API.
2. A large API model costs money for every request, needs the internet and sends customer data to a third party.
3. A prompted general model does not know a company's reply style. Fine-tuning teaches it the style and structure from real examples.

We kept Gemini as an *optional* benchmark so a comparison is possible without the app depending on it.

**Q77. Why FLAN-T5-*base* for the summary but *small* for the reply?**
The summary task was not trained (there are no reference summaries), so it relies entirely on the model's existing ability, and the larger base model is better at zero-shot tasks. The reply task *was* trained, so the small model is good enough after fine-tuning, and it is faster.

**Q78. Why TF-IDF + SVM and not BERT for classification?**
It is a strong and fast baseline for short texts. It trains in seconds, predicts in about 3 ms and is easy to explain. It also beats the majority baseline by a wide margin on every target. A fine-tuned DistilBERT or DeBERTa would probably help most where meaning matters more than keywords (the laptop/returns and pricing/sales errors), and it is listed as future work.

**Q79. Why `twitter-roberta-base-sentiment-latest` for sentiment?**
It is a widely used, well-documented RoBERTa model trained on about 124M tweets, with three classes (negative, neutral, positive). Tweets, like support tickets, are short, informal texts written by customers. The dataset has no sentiment labels, so we use it as an *inferred* signal only.

**Q80. Why roberta-large for BERTScore and MiniLM for cosine similarity?**
These are the standard defaults. `roberta-large` is the default BERTScore model for English and correlates well with human judgement. `all-MiniLM-L6-v2` is a fast, popular sentence-embedding model for measuring similarity. Using standard choices makes the results comparable with other work.

**Q81. If you trained FLAN-T5-base, what would you expect?**
Better grammar, better handling of negation and fewer invented facts, probably with higher ROUGE and BERTScore. Training would take roughly 3–4 times longer (it has about 3 times the parameters), and generation would also be slower. The notebook already supports this: set `MODEL_NAME = "google/flan-t5-base"` and re-run notebook 03. This is a prediction, not a measured result.

---

## 11. Explaining the Model evaluation page, section by section

Use this as a script when the examiner opens the **Model evaluation** page. Every number on the page is read from `results/` (the Colab export); the app does not compute or edit any of them.

### 11.1 Fine-tuning (top of the page)

**What it shows:** the training setup on the left and the loss curves on the right.

| Item | Value | How to explain it |
|---|---|---|
| Base model | google/flan-t5-small, 77.0M parameters, all trainable | Full fine-tuning: every weight was updated |
| Data | 18,848 train / 2,354 val / 2,339 test | Leakage-safe split from notebook 01 |
| Epochs | 4 (all epochs run, validation loss checked after each) | Early stopping was configured, but the Trainer disabled it, so all 4 epochs ran (see Q30) |
| Learning rate | 3e-4, linear schedule, 5% warm-up | Warm-up: the learning rate rises slowly for the first 5% of steps, so the large early updates do not damage the pretrained weights. After that it decreases linearly to 0. |
| Batch size | 16 | 16 ticket-reply pairs per step |
| Max lengths | 160 input / 128 output tokens | 4.1% of inputs and 5.0% of answers are cut at these limits |
| Decoding | Beam search (4 beams), no repeated 3-grams | See Q15 |
| Hardware | Tesla T4, 30 minutes | Free Colab GPU |
| Best validation loss | 1.122 (perplexity 3.07) | Perplexity = e^loss |
| Test loss | 1.111 (perplexity 3.04) | Almost equal to validation loss, so the model generalises |

**Loss curves:**
- The training loss (the continuous line) starts at about 2.9 and falls quickly in the first few hundred steps as the model learns the reply format, then slowly to about 1.3.
- The validation loss (dots, one per epoch) falls every epoch: 1.272 → 1.178 → 1.137 → 1.122.
- **What to say:** "Validation loss never goes up, so there is no overfitting. The gap between training and validation is small. The curve is flattening, so more epochs would give only small gains."

### 11.2 Response quality: before vs after fine-tuning

**What it shows:** a bar chart and a table comparing base (A) and fine-tuned (B) on the same 500 test tickets.

| Metric | Base | Fine-tuned | Meaning |
|---|---|---|---|
| ROUGE-1 | 0.205 | **0.474** | Word overlap with the real reply more than doubles |
| ROUGE-2 | 0.080 | **0.240** | Word-pair overlap triples, so phrases match, not just single words |
| ROUGE-L | 0.158 | **0.343** | Longest matching word sequence: the structure is closer to real replies |
| BLEU (0–100) | 1.17 | **19.57** | Exact n-gram precision. Harsh for open-ended text; about 20 is reasonable when many answers are correct. |
| BERTScore F1 | 0.867 | **0.899** | Meaning similarity. The raw scale is compressed, so the change looks small. |
| BERTScore F1 (rescaled) | 0.210 | **0.403** | The same score, spread out against a baseline: almost doubled |
| Semantic cosine | 0.580 | **0.781** | Whole-sentence meaning similarity |
| Distinct-2 (variety) | 0.554 | **0.162** | *Lower* after fine-tuning: replies are more repetitive (a limitation, see Q41) |
| Average words per reply | 18.5 | **67.8** | References average 60 words, so the fine-tuned length is right; the base model is far too short |

**What to say:** "Every quality metric improves, both word-based and meaning-based. The one that gets worse is variety, which is the cost of learning one company's reply style."

### 11.3 Is the improvement statistically significant? (table)

For each metric the table shows the mean improvement, its 95% confidence interval, the p-value and the share of tickets where B beat A:

| Metric | Mean difference | 95% CI | p-value | B better on |
|---|---|---|---|---|
| ROUGE-1 | +0.269 | 0.253 to 0.284 | 2.7e-76 | 92.0% of tickets |
| ROUGE-2 | +0.160 | 0.147 to 0.172 | 3.2e-69 | 88.4% |
| ROUGE-L | +0.185 | 0.171 to 0.198 | 1.1e-68 | 87.8% |
| BERTScore F1 | +0.033 | 0.030 to 0.035 | 3.0e-63 | 85.4% |
| Semantic cosine | +0.202 | 0.180 to 0.224 | 1.2e-56 | 82.2% |

How to explain the columns:
- **Mean difference:** the average of (B's score − A's score) over the 500 tickets.
- **95% CI (bootstrap, 2,000 resamples):** we resampled the 500 tickets with replacement 2,000 times and recomputed the mean difference each time. 95% of those means fall in this range. **The range never includes 0**, so the improvement is real.
- **p-value (paired Wilcoxon signed-rank test):** the probability of seeing a difference this large if fine-tuning had no effect. A value like 1e-76 is effectively zero. It is *paired* because both models answered the same tickets, and *non-parametric* because the scores are not normally distributed.
- **B better on:** fine-tuning is not only better on average; it is better on most individual tickets.

### 11.4 AI-judge evaluation (LLM-as-a-judge)

**What it shows:** Claude rated 50 blinded ticket pairs from 1 to 5 on five criteria.

| Criterion | Base | Fine-tuned |
|---|---|---|
| Relevance | 1.48 | **4.34** |
| Correctness | 1.10 | **3.58** |
| Fluency | 3.30 | **4.68** |
| Professional tone | 1.08 | **4.74** |
| Helpfulness | 1.00 | **2.82** |
| Overall (mean) | 1.59 | **4.03** |

- **Head to head:** fine-tuned rated higher on 50 of 50 tickets, base on 0, no ties (Wilcoxon p = 7.2e-10).
- **Not a support reply:** base 46 of 50, fine-tuned 0 of 50.
- **Correctness ≤ 2:** base 49, fine-tuned 5.

**What to say:**
- "The base model's fluency (3.30) is not bad. It writes proper English, but it doesn't write *support replies*: tone 1.08, helpfulness 1.00. Fine-tuning taught it the task, not the language."
- "Helpfulness (2.82) is the weakest score for our model, because without company data it can only ask for details or offer a call. RAG would fix this."

**"Where the fine-tuned model went wrong":** the 5 low-correctness cases fall into two types:
- **Role confusion (2 cases):** the model asks the customer for information the customer had asked support for, such as the API version or the update details.
- **Hallucination (3 cases):** it asserts things it cannot know, such as a HIPAA certification, a list of services, or payment gateway support and discounts.

**Caveats shown on the page:** a single AI rater; it may prefer longer or more polite replies; Claude also helped build the project; blinding is imperfect. See Q39 and Q67.

### 11.5 Size and speed

| Item | Value | How to explain it |
|---|---|---|
| Parameters | 77.0M | Small: about 1/90 of a 7B LLM |
| Size on disk | 310 MB | Stored as fp32: 77M × 4 bytes ≈ 308 MB |
| Per reply on a T4 GPU | 1.60 s (p95: 2.12 s) | One ticket at a time, beam search with 4 beams |
| Per reply on a Colab CPU | 4.96 s | Colab's CPU is slow. On an M4 Mac the app takes about 0.5–1.5 s. |
| Peak GPU memory | 441 MB | Fits on almost any GPU |

**What to say:** "It is small and fast enough to run on an ordinary laptop with no API, which is the practical advantage over a large hosted LLM."

### 11.6 Supporting classifiers

| Target | Accuracy | Macro-F1 | Weighted F1 | Majority-class macro-F1 | Weakest class |
|---|---|---|---|---|---|
| Category (queue) | 0.620 | **0.593** | 0.611 | 0.044 | General Inquiry (F1 0.45) |
| Priority | 0.643 | **0.580** | 0.622 | 0.195 | low (F1 0.37) |
| Ticket type | 0.881 | **0.871** | 0.876 | 0.141 | Problem (F1 0.67) |

How to explain the columns:
- **Macro-F1** (highlighted) is the main score: the average F1 over all classes, each counted equally, so small classes matter.
- **Weighted F1** weights each class by how many tickets it has, so it is closer to accuracy.
- **Majority-class macro-F1** is the score of a "model" that always predicts the most common class. Ours is 4 to 13 times higher.
- **Weakest class** shows where to improve.

Key observations:
- **Queue: high precision, low recall.** Macro precision is 0.81 but macro recall only 0.52. When the classifier predicts a small queue it is usually right (General Inquiry precision 1.00), but it often misses those tickets and puts them in large queues (General Inquiry recall 0.29). It is cautious because the large queues dominate.
- **Priority "low" recall is 0.24.** Most low-priority tickets are predicted as medium. Urgency is weakly visible in the words, and the synthetic labels are partly arbitrary.
- **Type works well** (0.88): words like "please change", "request" and "error" signal the type directly. *Problem* vs *Incident* is the main confusion, because both describe something broken.

**Confusion matrices (three images):**
- Rows are the **true** class and columns the **predicted** class. They are row-normalised, so each row adds up to 100%.
- The **diagonal** is the share predicted correctly; a darker diagonal is better.
- Off-diagonal cells show which classes get mixed up. For queue, these are mostly the overlapping support queues (Technical / IT / Product Support, Customer Service).

**Sentiment line under the matrices:** on 500 test tickets the pretrained model predicted 55% negative, 28% neutral and 17% positive. This is not evaluated (there are no labels); it only shows that the predictions are plausible for complaint tickets.

### 11.7 Dataset preparation

| Cleaning step | Rows left |
|---|---|
| Raw (all files, all languages) | 61,765 |
| English only | 28,261 |
| Non-empty body and answer | 28,254 |
| Body and answer ≥ 20 characters | 28,154 |
| Drop exact duplicate bodies | 23,557 |

- **Final split:** 18,848 train, 2,354 val, 2,339 test.
- **Near-duplicate groups:** tickets with cosine similarity ≥ 0.9 were grouped. 770 tickets were in groups of two or more (the largest group had 5), and each group was kept within one split.
- **Leakage audit:** 8 validation and 8 test tickets were removed for being too close to a training ticket. After removal, the mean similarity of test tickets to their closest training ticket is 0.54, so the test set is genuinely new.
- **Truncation:** 4.1% of inputs and 5.0% of answers are longer than the token limits and get cut. That is a small share, chosen as a trade-off for training speed.

### 11.8 Example replies

Random test tickets shown in four columns: ticket, reference (the real agent answer), base output and fine-tuned output.
- **What to point out:** the base output is short, often repeats the ticket or speaks *as the customer*. The fine-tuned output has the structure of a real support reply: greeting, empathy, a request for details, next steps.
- **Note on placeholders:** this table shows *raw* model output, so you will see `tel_num>` instead of `<tel_num>`. T5's vocabulary has no `<` character. The live app repairs these to `[Phone]` and similar (see Q52).

### 11.9 Limitations (bottom of the page)

The six points listed on the page, each with a one-line explanation:
1. **Formulaic replies:** distinct-2 falls from 0.55 to 0.16, and many replies offer a call. The model copies the dataset's most common pattern.
2. **Part of the ROUGE gain comes from length:** the base model writes 18 words, the fine-tuned 68 and the references 60. Meaning-based metrics and the AI judge confirm the gain is not *only* length (see Q65).
3. **Hallucination:** the model can state facts that contradict the ticket, so every reply is a draft.
4. **Synthetic data:** the results show adaptation to this dataset's domain and style, not to a real company.
5. **Summary and action have no ground truth.** Zero-shot output was poor, so the app uses fallbacks and labels the source.
6. **No human evaluation yet:** quality beyond automatic metrics was judged by an AI rater. A human rating sheet is prepared.

### 11.10 The Model comparison page (offline benchmark table)

At the bottom of **Model comparison**, the same metrics are shown on a separate 100-ticket comparison set (notebook 06):

| Metric | Base | Fine-tuned |
|---|---|---|
| ROUGE-1 | 0.223 | **0.483** |
| ROUGE-L | 0.177 | **0.351** |
| BLEU | 1.28 | **20.15** |
| BERTScore F1 (rescaled) | 0.225 | **0.408** |
| Semantic cosine | 0.613 | **0.787** |
| Average words per reply | 19.4 | **66.0** |

**Q82. Why are these numbers slightly different from the Evaluation page?**
They come from a different sample: 100 comparison tickets instead of 500 test tickets. The values are very close (ROUGE-L 0.351 vs 0.343), which shows the result is stable across samples. There is no Gemini column because no API key was configured.

---

## 12. Explaining failure cases from the live demo

These come from `TESTING.md` (cases marked ⚠). If the examiner tries their own ticket and gets a strange output, explain it calmly with the same reasoning.

**Q83. The reply to "Refund not received" says "the refund you received". Why?**
Small models handle negation poorly. "Refund" + "received" appears often in the training replies, and the "not" in the ticket loses out. This is a known weakness of small seq2seq models. Fixes: a larger model, training examples that contain negation, and an automatic check that warns the agent when the ticket says "not received" but the reply says "received".

**Q84. A damaged laptop was classified as "IT Support", not "Returns and Exchanges". Why?**
TF-IDF only counts words; it doesn't understand meaning. "Laptop" appears mostly in IT tickets, and the classifier cannot tell that "return it and get a replacement" is the real intent. Its confidence was only 28%, which is a sign of uncertainty. Fixes: a Transformer classifier that understands context, and a rule that marks low-confidence predictions for manual routing.

**Q85. A pricing question was put in "Billing" and the reply mentioned plans that don't exist. Why?**
- **Classification:** "pricing" and "plan" are typical Billing words in this dataset.
- **Reply:** the model has no company information, so it fills the gap with plausible text. That is *hallucination*.

Fix: **RAG**, which retrieves the real pricing page and gives it to the model, so the reply is grounded in facts. This is exactly why the app shows every reply as a draft for human review.

**Q86. A thank-you message got an apology reply and "High" priority. Why?**
This is **training-data bias**: almost every ticket in the dataset is a problem, so the model has never learned to answer a compliment. The sentiment model correctly said *Positive*, which shows the components disagree. Fixes: add non-complaint examples to the training data, or use a separate "thank you" reply when sentiment is strongly positive.

**Q87. An HR request was classified as "Technical Support". Why?**
Human Resources is a small class: 46 of 2,339 test tickets (2%). The classifier finds only 43% of HR tickets (recall 0.43), although it is almost always right when it does predict HR (precision 0.95). Small classes are hard even with class balancing. Fixes: more HR data, data augmentation, a Transformer classifier, and low-confidence routing (the confidence here was only 32%).

**Q88. Why are some confidence scores so low (20–30%)?**
With 10 queue classes, the probability is shared among them, so even the best class can be at 20–30%. The scores come from `CalibratedClassifierCV`, which turns SVM scores into calibrated probabilities. A low score is useful: it tells the agent "check this one". A future version could highlight predictions below a threshold.

---

## 13. More questions examiners often ask

**Q89. What is the difference between training, validation and test loss?**
All three are the same cross-entropy loss computed on different data. Training loss is on the data the model learns from. Validation loss is on held-out data used during training to monitor progress. Test loss is on data used only once at the end. Ours: training about 1.35 (averaged over all training steps), validation 1.122, test 1.111.

**Q90. Why is the training loss *higher* than the validation loss?**
The reported training loss (1.351) is an *average over the whole run*, including the early steps when the loss was about 2.9. The validation loss is measured at the end of each epoch, after learning. Training loss also includes dropout, which is switched off during evaluation. At the end of training, the two are close.

**Q91. What is weight decay? What did you use?**
A regulariser that slightly shrinks the weights at every step so they don't grow too large, which reduces overfitting. We used 0.01, the standard value for AdamW.

**Q92. What is AdamW?**
Adam is an optimiser that adapts the learning rate of each weight using running averages of its gradients and squared gradients. AdamW applies weight decay *separately* from the gradient update, which works better than mixing it into the gradient as plain Adam does. It is the standard optimiser for fine-tuning Transformers.

**Q93. What is the random seed for, and what was yours?**
It fixes the random choices (data shuffling, weight initialisation of new layers, dropout) so the experiment can be repeated with the same result. We used seed 42.

**Q94. Why evaluate generation on 500 test tickets and not all 2,339?**
Generating with beam search and computing BERTScore with roberta-large is slow. 500 tickets is enough for a tight estimate: the 95% confidence interval for the ROUGE-1 improvement is only ±0.016 wide. The classifiers, which are fast, were evaluated on all 2,339 test tickets.

**Q95. Why only 50 tickets for the AI judge?**
Each rating requires reading a ticket and two replies against a detailed rubric. 50 pairs is enough for a clear result: the fine-tuned model won all 50, and p = 7.2e-10.

**Q96. What is the difference between precision and recall?**
- **Precision:** of the tickets we *predicted* as class X, how many really are X.
- **Recall:** of the tickets that *really are* class X, how many we found.
- **F1:** the harmonic mean of the two, which is high only when both are high.

Example: General Inquiry has precision 1.00 but recall 0.29, so it is always right when it predicts General Inquiry but finds only 29% of them.

**Q97. What is hallucination in LLMs?**
When a model generates fluent text that is not supported by the input or by facts, such as an invented policy or a wrong detail. It happens because the model learns to produce *likely* text, not *true* text. Ways to reduce it: RAG (grounding in documents), larger models, constrained decoding, and human review.

**Q98. What is RAG, and how would it fit into SupportGen?**
Retrieval-Augmented Generation. Before generating, the system searches a knowledge base (policies, FAQs, past resolved tickets) for passages relevant to the ticket and adds them to the prompt. The model then writes its reply from real information. In SupportGen this would mean: embed the ticket → retrieve the top 3 similar resolved tickets or policy passages → put them in the prompt → generate.

**Q99. What is LoRA?**
Low-Rank Adaptation. It freezes the original weights and adds small trainable matrices to the attention layers: a weight update is expressed as the product of two thin matrices (rank r, e.g. 8). Only about 0.1–1% of the parameters are trained, which saves a lot of memory and makes large models trainable on a small GPU. We didn't need it, because full fine-tuning of 77M parameters fits easily.

**Q100. What is catastrophic forgetting? Did it happen here?**
When fine-tuning on a narrow task makes a model lose general abilities it had before. Our fine-tuned model is used only for support replies, so losing other abilities doesn't matter here. The lower variety (distinct-2) is a mild sign that the model has narrowed to one style. That's also why the summary uses the separate, untouched FLAN-T5-base.

**Q101. What is temperature, and why didn't you use it?**
Temperature scales the probabilities before sampling. Low values (< 1) make output more predictable; high values (> 1) make it more random. It only matters when *sampling*. We used beam search, which is deterministic, because support replies should be consistent and the same ticket should always give the same draft.

**Q102. What is the difference between instruction tuning and fine-tuning?**
Instruction tuning (what Google did to make FLAN-T5) trains a model on many different tasks written as instructions, so it becomes good at *following instructions in general*. Our fine-tuning trains it on *one* task (support replies) so it becomes good at that specific job and style.

**Q103. What is RLHF, and why didn't you use it?**
Reinforcement Learning from Human Feedback: humans rank model outputs, a reward model learns those preferences, and the language model is optimised against it. It is how chat assistants are aligned. It needs large amounts of human preference data and much more compute. Supervised fine-tuning on good example replies is the right first step for a project of this size. In a real deployment, agents' edits to the drafts could become preference data later.

**Q104. How would you deploy this for a real company?**
- Package the backend and models in Docker and serve the model with a GPU inference server (or ONNX/quantised on CPU).
- Put Django behind Gunicorn and Nginx, and serve the built React app as static files.
- Add authentication for agents and HTTPS.
- Integrate with the helpdesk (Zendesk, Freshdesk) via its API so drafts appear inside the agent's tool.
- Log agents' edits to measure quality and collect training data, and monitor latency and errors.

**Q105. How would you know whether it works in real use?**
Track online metrics: the share of drafts sent with few or no edits (edit distance between draft and sent reply), time saved per ticket, customer satisfaction (CSAT) scores, and the share of routing predictions agents had to change. Run an A/B test with and without drafts.

**Q106. What if the data changes over time (new products, new policies)?**
That is *data drift*. Monitor the edit rate and classifier confidence; when they get worse, retrain on recent tickets. RAG helps a lot here, because updating the knowledge base updates the facts immediately without retraining.

**Q107. How much training data would you need for a new company?**
We used about 19k pairs. Because FLAN-T5 is already instruction-tuned, a few thousand good ticket-reply pairs would likely be enough to learn the style. Quality matters more than quantity: clean, correct, varied replies. This is an estimate; it should be checked with a learning curve (train on 1k, 5k, 10k and compare).

**Q108. Is the project reproducible?**
Yes. The notebooks run top to bottom on Colab with a fixed seed (42), pinned library versions (`transformers` 5.18.0, `scikit-learn` 1.6.1), a recorded configuration (`training_config.json`) and all metrics exported to `results/`. The web app reads those files directly, so the numbers shown are exactly what the notebooks produced.

**Q109. What was the hardest part of the project?**
A good honest answer: making the evaluation trustworthy. That meant finding the exact and near-duplicate tickets that would have leaked into the test set, choosing metrics that capture meaning and not only word overlap, adding significance tests, and documenting where the model fails instead of hiding it. A technical surprise was the `<tel_num>` placeholder issue caused by T5's vocabulary having no `<` character.

**Q110. What makes your project different from a ChatGPT wrapper?**
We own every step: data cleaning and leakage-safe splitting, fine-tuning the model weights, evaluation with five metrics plus significance tests and an AI judge, and a local deployment where the model runs on our machine. A wrapper only sends a prompt to someone else's model and has nothing to measure or improve.
