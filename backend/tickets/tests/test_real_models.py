"""Integration tests against the real exported model files (skipped if they are absent)."""
from unittest import TestCase, skipUnless

from ai.classify import TicketClassifier
from ai.config import load_config
from ai.providers import LocalTransformerProvider

CFG = load_config()
HAS_MODEL = (CFG.model_dir / "config.json").exists()
HAS_CLASSIFIERS = (CFG.classifier_dir / "meta.json").exists()


@skipUnless(HAS_CLASSIFIERS, "classifiers not exported yet")
class RealClassifierTests(TestCase):
    def test_predictions_have_valid_labels_and_confidences(self):
        clf = TicketClassifier(CFG.classifier_dir)
        preds = clf.predict("Refund not received", "I requested a refund two weeks ago and it has not arrived.")
        self.assertEqual(set(preds), {"queue_group", "priority", "type"})
        for p in preds.values():
            self.assertTrue(p.label)
            self.assertTrue(0.0 <= p.confidence <= 1.0)


@skipUnless(HAS_MODEL, "fine-tuned model not exported yet")
class RealGeneratorTests(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.provider = LocalTransformerProvider("finetuned", CFG.model_dir, device="cpu")

    def test_uses_training_config(self):
        self.assertEqual(self.provider.max_input_len, 160)
        self.assertIn("{subject}", self.provider.prompt_template)

    def test_generates_clean_non_empty_reply(self):
        r = self.provider.generate("Refund not received",
                                   "I requested my refund two weeks ago, the amount was deducted but I have "
                                   "not received it yet. I already contacted support twice.")
        self.assertGreater(len(r.text.split()), 5)
        self.assertNotRegex(r.text, r"\b(tel_num|name|acc_num)>")   # placeholders cleaned
        self.assertEqual(r.model, "flan-t5-supportgen")

    def test_lm_head_not_tied_to_embeddings(self):
        import torch
        self.provider._ensure_loaded()
        self.assertFalse(torch.equal(self.provider._model.lm_head.weight, self.provider._model.shared.weight))
