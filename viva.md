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
The judge must be independent of the models being compared. Claude did not generate any of the A or B replies, the replies were blinded and shuffled, and a fixed rubric was used. Gemini was not part of the judged set.

**Q68. Why not just use ChatGPT or Gemini for everything?**
Cost per request, latency, dependence on an internet connection and a third-party service, and privacy, because customer data would leave the company. A small fine-tuned model runs on a normal laptop, costs nothing per request and can be fully controlled and audited. The project also demonstrates the GenAI pipeline end to end instead of wrapping an API.

**Q69. What happens if a customer writes in German or Hindi?**
The model was trained only on English, so quality would drop sharply. It might reply in English or produce poor text. The classifiers would also be unreliable. Language detection with a "not supported" warning, or multilingual training, would fix this.

**Q70. What did you personally learn from this project?**
The full GenAI lifecycle: careful data preparation (especially avoiding leakage), fine-tuning a Transformer, evaluating it with several complementary metrics and significance tests, being honest about failure modes, and turning a model into a usable application with sensible engineering such as model caching, fallbacks and error handling.
