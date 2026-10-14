# LLM-as-a-judge evaluation: protocol and rubric

**Judge:** Claude (Anthropic, model `claude-opus-5-5`). This is an AI judge, **not** a human evaluation.

**Material:** `human_eval_sheet.csv`, produced by notebook 04.
- 50 held-out test tickets, chosen with seed 42.
- Each ticket has 2 replies, labelled X and Y in random order.
- One reply comes from base FLAN-T5-small (Model A), the other from fine-tuned FLAN-T5-small (Model B).

**Blinding:**
- The judge rated from the sheet only. It did not read `human_eval_key.csv`.
- The key was joined by a script only after all ratings were written.
- Blinding is imperfect: the fine-tuned model has a recognisable long, templated style.

**Procedure:**
- Each reply was rated on its own, against the customer's ticket.
- No reference answer was used, so a reply is not rewarded for copying the dataset's wording.
- Ratings were fixed before unblinding and never changed afterwards.

## Criteria (1–5)

| Criterion | 1 | 3 | 5 |
|---|---|---|---|
| **Relevance** | Ignores the ticket, or repeats/echoes it | Addresses the general topic but misses the main request | Directly addresses the customer's specific issue |
| **Correctness** | Contradicts the ticket or invents wrong facts; or is not a reply at all | No clear errors, but vague; or minor inaccuracies | Everything stated is consistent with the ticket; no invented facts |
| **Fluency** | Broken or ungrammatical; fragments | Understandable, with awkward or repeated phrases | Natural, grammatical, well-formed |
| **Professional tone** | Not written as a support agent (e.g. written as the customer); rude or inappropriate | Neutral, but lacks the courtesy expected of support | Polite, empathetic, appropriate for customer support |
| **Helpfulness** | Gives the customer nothing to act on | Generic next step (e.g. "we will look into it") | Concrete, sensible next steps or the information requested |

**Placeholders:**
- Anonymisation placeholders such as `name>` and `tel_num>` are an artefact of the dataset and tokenizer, and the app repairs them. They are **not** penalised under fluency.
- They are penalised only where the reply depends on a missing fact. For example, "call us at tel_num>" is still treated as a valid next step.

**Known limitations of this method:**
1. An AI judge may systematically prefer certain styles, such as longer or more polite replies.
2. The judge is the same assistant that helped build the project, so there is a potential for bias.
3. There is a single rater, so inter-rater agreement cannot be measured.
4. Blinding is imperfect, as noted above.
