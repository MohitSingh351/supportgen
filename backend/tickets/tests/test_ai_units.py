"""Unit tests for the AI layer that need no trained model."""
from pathlib import Path
from unittest import TestCase

from ai import assist
from ai.classify import Prediction
from ai.postprocess import clean_generated_text
from ai.providers import (GeminiProvider, GenerationResult, LocalTransformerProvider,
                          ProviderUnavailable)
from ai.services import TicketAnalyzer


class PostprocessTests(TestCase):
    def test_broken_angle_placeholders_are_restored(self):
        text = "name>, thank you. Please call us at tel_num> about account acc_num>."
        self.assertEqual(clean_generated_text(text),
                         "[Customer Name], thank you. Please call us at [Phone] about account [Account Number].")

    def test_intact_and_bracket_variants_are_normalised(self):
        self.assertEqual(clean_generated_text("Call <tel_num> or [Tel Num]. [Your Name]"),
                         "Call [Phone] or [Phone]. [Agent Name]")

    def test_ordinary_text_is_untouched(self):
        text = "Your refund of 20 EUR was issued on Monday."
        self.assertEqual(clean_generated_text(text), text)


class AssistTests(TestCase):
    def test_degenerate_outputs_are_detected(self):
        subject = "Critical Data Analytics Tool Crash"
        self.assertTrue(assist.is_degenerate("Critical Data Analytics Tool Crash", subject))
        self.assertTrue(assist.is_degenerate("Subject: System Security Ticket", subject))
        self.assertTrue(assist.is_degenerate("Is this what you need?", subject))
        self.assertFalse(assist.is_degenerate(
            "The customer's analytics tool crashes on startup after the latest update.", subject))

    def test_extractive_summary_skips_greeting(self):
        body = "Dear Support, I hope you are well. My refund has not arrived after two weeks. Please help."
        summary = assist.extractive_summary(body)
        self.assertNotIn("Dear Support", summary)
        self.assertIn("refund has not arrived", summary)

    def test_rule_based_action_uses_queue_and_priority(self):
        action = assist.rule_based_action("Billing and Payments", "high")
        self.assertTrue(action.startswith("Handle with high priority."))
        self.assertIn("billing team", action)
        self.assertEqual(assist.rule_based_action(None, None), assist.DEFAULT_ACTION)


class ProviderTests(TestCase):
    def test_missing_local_model_is_unavailable(self):
        provider = LocalTransformerProvider("finetuned", Path("/nonexistent/model-dir"))
        self.assertFalse(provider.is_available())
        with self.assertRaises(ProviderUnavailable):
            provider.generate("Subject", "Body text of a ticket")

    def test_gemini_without_key_is_not_configured(self):
        provider = GeminiProvider(api_key="", model="gemini-test")
        self.assertFalse(provider.is_available())
        with self.assertRaises(ProviderUnavailable):
            provider.generate("Subject", "Body")


# ── Fakes for analyzer tests ──
class FakeGenerator:
    model_id, source = "fake-model", "/fake"

    def is_available(self):
        return True

    def generate(self, subject, body):
        return GenerationResult("Generated reply.", "finetuned", self.model_id, 5)


class FakeAux:
    model_id = "fake-aux"

    def __init__(self, output):
        self.output = output

    def generate_from_prompt(self, prompt, max_new_tokens=None):
        return self.output


class FakeClassifier:
    classifier_dir = Path("/fake")

    def __init__(self, available=True):
        self.available = available

    def is_available(self):
        return self.available

    def predict(self, subject, body):
        return {"queue_group": Prediction("Billing and Payments", 0.8),
                "priority": Prediction("high", 0.7), "type": Prediction("Incident", 0.9)}


class AnalyzerTests(TestCase):
    SUBJECT, BODY = "Refund not received", "Hello, I requested my refund two weeks ago and it has not arrived yet."

    def test_full_result_with_generated_summary(self):
        aux = FakeAux("The customer has been waiting two weeks for a refund that has not arrived.")
        r = TicketAnalyzer(FakeGenerator(), FakeClassifier(), None, aux).analyze(self.SUBJECT, self.BODY)
        self.assertEqual(r.generated_response, "Generated reply.")
        self.assertEqual((r.category, r.priority, r.ticket_type), ("Billing and Payments", "high", "Incident"))
        self.assertEqual(r.summary_source, "generated")
        self.assertEqual(r.action_source, "rule_based")
        self.assertIn("billing team", r.recommended_action)

    def test_degenerate_summary_falls_back_to_extractive(self):
        r = TicketAnalyzer(FakeGenerator(), FakeClassifier(), None, FakeAux("Refund not received")).analyze(
            self.SUBJECT, self.BODY)
        self.assertEqual(r.summary_source, "extractive_fallback")
        self.assertIn("refund", r.summary)

    def test_missing_classifiers_still_returns_response(self):
        r = TicketAnalyzer(FakeGenerator(), FakeClassifier(available=False), None, None).analyze(
            self.SUBJECT, self.BODY)
        self.assertIsNone(r.category)
        self.assertEqual(r.recommended_action, assist.DEFAULT_ACTION)
        self.assertTrue(any("Classifiers not found" in w for w in r.warnings))
