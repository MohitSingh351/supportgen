"""Ticket summary and recommended next action.

The dataset has no reference summaries or actions, so neither task was trained.
In the Colab evaluation (notebook 04) zero-shot FLAN-T5-base mostly echoed the subject
line for both tasks. Therefore:

* Summary: zero-shot FLAN-T5 generation, with an extractive fallback (key sentences from
  the ticket) whenever the generated text is degenerate. The response says which was used.
* Recommended action: by default a rule-based suggestion derived from the predicted queue
  and priority (predictable, never invents facts). ``SUPPORTGEN_ACTION_MODE=generate``
  tries zero-shot generation first, with the same rules as fallback.

Suggestions only assist a human agent; nothing is executed automatically.
"""
import re

SUMMARY_TEMPLATE = (
    "Summarize the following customer support ticket in one sentence.\n"
    "Subject: {subject}\n"
    "Ticket: {body}"
)
ACTION_TEMPLATE = (
    "You are helping a human customer support agent. In one short sentence, "
    "recommend the next action the agent should take for this ticket.\n"
    "Subject: {subject}\n"
    "Ticket: {body}"
)

QUEUE_ACTIONS = {
    "Billing and Payments": "Check the customer's payment and refund records with the billing team and confirm the transaction status.",
    "Customer Service": "Review the customer's account history and respond with the requested information or next steps.",
    "General Inquiry": "Answer the question directly or route the ticket to the team that owns the topic.",
    "Human Resources": "Forward the request to the HR team and confirm receipt to the requester.",
    "IT Support": "Collect system details and error messages, then follow the internal IT troubleshooting checklist.",
    "Product Support": "Ask for product model, version and steps to reproduce, then check known issues for the product.",
    "Returns and Exchanges": "Verify the order and return eligibility under the return policy, then share the return instructions.",
    "Sales and Pre-Sales": "Pass the inquiry to the sales team with the customer's requirements and offer a follow-up call.",
    "Service Outages and Maintenance": "Check the service status dashboard for an ongoing incident and link the ticket to it if one exists.",
    "Technical Support": "Gather error details and reproduction steps, check the knowledge base, and escalate to engineering if unresolved.",
}
PRIORITY_PREFIX = {
    "critical": "Escalate immediately to a senior agent. ",
    "high": "Handle with high priority. ",
}
DEFAULT_ACTION = "Review the ticket details and route it to the appropriate support team."

_GREETING = re.compile(r"^(dear|hello|hi|hey|greetings|good (morning|afternoon|evening)|to whom)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def is_degenerate(generated: str, subject: str, min_words: int = 5) -> bool:
    """Detects the failure modes seen in evaluation: echoing the subject, 'Subject: ...',
    questions back to the user, or too-short output."""
    g = generated.strip()
    if len(g.split()) < min_words:
        return True
    if g.lower().startswith(("subject:", "ticket:")) or g.endswith("?"):
        return True
    ng, ns = _norm(g), _norm(subject)
    return bool(ns) and (ng == ns or ng in ns or (ns in ng and len(ng) < len(ns) + 15))


def extractive_summary(body: str, max_words: int = 40) -> str:
    """First informative sentences of the ticket (greetings skipped)."""
    sentences = [s.strip() for s in _SENTENCE.split(body.strip()) if s.strip()]
    informative = [s for s in sentences if not _GREETING.match(s)] or sentences
    words: list[str] = []
    for sentence in informative:
        words.extend(sentence.split())
        if len(words) >= 15:
            break
    summary = " ".join(words[:max_words])
    return summary if len(words) <= max_words else summary.rstrip(",;:") + "…"


def rule_based_action(queue: str | None, priority: str | None) -> str:
    action = QUEUE_ACTIONS.get(queue or "", DEFAULT_ACTION)
    return PRIORITY_PREFIX.get((priority or "").lower(), "") + action
