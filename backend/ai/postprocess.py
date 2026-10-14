"""Clean-up of generated text before it is shown to a support agent.

The training data uses anonymisation placeholders such as ``<tel_num>`` and ``<name>``.
FLAN-T5's SentencePiece vocabulary has no ``<`` character, so the fine-tuned model emits
them as ``tel_num>`` / ``name>``. The dataset also mixes styles (``[Tel Num]``, ``[name]``).
Here every variant is normalised to one readable bracket form, e.g. ``[Phone]``.
"""
import re

_CANONICAL = {
    "name": "[Customer Name]",
    "customer name": "[Customer Name]",
    "your name": "[Agent Name]",
    "tel_num": "[Phone]",
    "tel num": "[Phone]",
    "tel number": "[Phone]",
    "phone number": "[Phone]",
    "acc_num": "[Account Number]",
    "acc num": "[Account Number]",
    "account number": "[Account Number]",
    "email": "[Email]",
}

# "<tel_num>" or the broken "tel_num>" produced by the model
_ANGLE = re.compile(r"<?\b(tel_num|acc_num|name|email)>", re.IGNORECASE)
# "[Tel Num]", "[name]", "[Phone Number]", ...
_BRACKET = re.compile(r"\[(" + "|".join(re.escape(k) for k in _CANONICAL) + r")\]", re.IGNORECASE)
_LINE_BREAK = re.compile(r"<?\bbr\s*/?>", re.IGNORECASE)
_SPACES = re.compile(r"[ \t]{2,}")


def clean_generated_text(text: str) -> str:
    text = _LINE_BREAK.sub(" ", text)
    text = _ANGLE.sub(lambda m: _CANONICAL[m.group(1).lower()], text)
    text = _BRACKET.sub(lambda m: _CANONICAL[m.group(1).lower()], text)
    return _SPACES.sub(" ", text).strip()
