"""API tests. The AI layer is replaced by fakes so these run fast and without model files."""
from unittest.mock import patch

from django.db import DatabaseError
from rest_framework.test import APITestCase

from ai.providers import GeminiProvider, GenerationError, GenerationResult, ProviderUnavailable
from ai.services import AnalysisResult
from tickets.models import Ticket

RESPONSE_FIELDS = {"id", "subject", "body", "summary", "category", "priority", "ticket_type", "sentiment",
                   "recommended_action", "generated_response", "model_used", "created_at"}
VALID = {"subject": "Refund not received",
         "body": "I requested my refund almost two weeks ago and still have not received it."}


def fake_result(**overrides):
    data = dict(summary="Customer waiting two weeks for a refund.", summary_source="generated",
                category="Billing and Payments", category_confidence=0.8, priority="high",
                priority_confidence=0.7, ticket_type="Incident", type_confidence=0.9,
                sentiment="negative", sentiment_score=0.9,
                recommended_action="Check the refund status.", action_source="rule_based",
                generated_response="We are sorry for the delay...", model_used="flan-t5-supportgen",
                timings_ms={"response": 900}, warnings=[])
    data.update(overrides)
    return AnalysisResult(**data)


class FakeAnalyzer:
    def __init__(self, exc=None):
        self.exc = exc

    def analyze(self, subject, body):
        if self.exc:
            raise self.exc
        return fake_result()


class FakeProvider:
    def __init__(self, name, available=True):
        self.name, self.available, self.source = name, available, "/fake"

    def is_available(self):
        return self.available

    def generate(self, subject, body):
        return GenerationResult(f"{self.name} reply", self.name, f"{self.name}-model", 10)


@patch("tickets.views.get_analyzer", return_value=FakeAnalyzer())
class AnalyzeEndpointTests(APITestCase):
    url = "/api/tickets/analyze/"

    def test_valid_ticket_returns_full_schema_and_is_saved(self, _):
        r = self.client.post(self.url, VALID, format="json")
        self.assertEqual(r.status_code, 201)
        self.assertTrue(RESPONSE_FIELDS <= set(r.data))
        self.assertEqual(r.data["model_used"], "flan-t5-supportgen")
        self.assertEqual(Ticket.objects.count(), 1)
        self.assertEqual(Ticket.objects.get().priority, "high")

    def test_subject_is_optional(self, _):
        r = self.client.post(self.url, {"body": VALID["body"]}, format="json")
        self.assertEqual(r.status_code, 201)

    def test_empty_or_whitespace_body_is_rejected(self, _):
        for body in ["", "    "]:
            r = self.client.post(self.url, {"subject": "x", "body": body}, format="json")
            self.assertEqual(r.status_code, 400)
            self.assertIn("body", r.data)
        self.assertEqual(Ticket.objects.count(), 0)

    def test_missing_body_is_rejected(self, _):
        self.assertEqual(self.client.post(self.url, {"subject": "x"}, format="json").status_code, 400)

    def test_too_short_body_is_rejected(self, _):
        self.assertEqual(self.client.post(self.url, {"body": "help"}, format="json").status_code, 400)

    def test_oversized_ticket_is_rejected(self, _):
        r = self.client.post(self.url, {"body": "a" * 20000}, format="json")
        self.assertEqual(r.status_code, 400)
        self.assertIn("too long", str(r.data["body"]))

    def test_malformed_json_is_rejected(self, _):
        r = self.client.post(self.url, "{not json", content_type="application/json")
        self.assertEqual(r.status_code, 400)

    def test_model_unavailable_returns_503(self, mock):
        mock.return_value = FakeAnalyzer(ProviderUnavailable("Model not found at models/flan-t5-supportgen"))
        r = self.client.post(self.url, VALID, format="json")
        self.assertEqual(r.status_code, 503)
        self.assertIn("Model not found", r.data["error"])

    def test_inference_error_returns_500(self, mock):
        mock.return_value = FakeAnalyzer(GenerationError("boom"))
        self.assertEqual(self.client.post(self.url, VALID, format="json").status_code, 500)

    def test_database_error_still_returns_analysis(self, _):
        with patch("tickets.views.Ticket.objects.create", side_effect=DatabaseError("db down")):
            r = self.client.post(self.url, VALID, format="json")
        self.assertEqual(r.status_code, 200)
        self.assertIsNone(r.data["id"])
        self.assertEqual(r.data["generated_response"], "We are sorry for the delay...")
        self.assertTrue(any("not saved" in w for w in r.data["warnings"]))


class HistoryAndAnalyticsTests(APITestCase):
    def setUp(self):
        for priority in ["high", "high", "low"]:
            Ticket.objects.create(subject="s", body="body text", **fake_result(priority=priority).to_dict())

    def test_ticket_list_is_paginated_newest_first(self):
        r = self.client.get("/api/tickets/")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.data["count"], 3)
        self.assertEqual(r.data["results"][0]["id"], Ticket.objects.latest("created_at").id)

    def test_ticket_detail(self):
        t = Ticket.objects.first()
        self.assertEqual(self.client.get(f"/api/tickets/{t.id}/").data["generated_response"], t.generated_response)
        self.assertEqual(self.client.get("/api/tickets/999999/").status_code, 404)

    def test_analytics_counts(self):
        r = self.client.get("/api/analytics/")
        self.assertEqual(r.data["total_tickets"], 3)
        self.assertEqual(r.data["priority"], {"high": 2, "low": 1})
        self.assertEqual(r.data["high_priority_by_category"], {"Billing and Payments": 2})
        self.assertEqual(len(r.data["recent"]), 3)


class CompareEndpointTests(APITestCase):
    url = "/api/compare/"

    @patch("tickets.views.get_gemini_provider", return_value=GeminiProvider(api_key="", model="gemini-test"))
    @patch("tickets.views.get_finetuned_provider", return_value=FakeProvider("finetuned"))
    @patch("tickets.views.get_base_provider", return_value=FakeProvider("base"))
    def test_gemini_not_configured_does_not_break_comparison(self, *_):
        r = self.client.post(self.url, VALID, format="json")
        self.assertEqual(r.status_code, 200)
        by_key = {x["provider"]: x for x in r.data["results"]}
        self.assertEqual(by_key["base"]["status"], "ok")
        self.assertEqual(by_key["finetuned"]["text"], "finetuned reply")
        self.assertEqual(by_key["gemini"]["status"], "not_configured")

    def test_unknown_provider_is_rejected(self):
        r = self.client.post(self.url, {**VALID, "providers": ["gpt"]}, format="json")
        self.assertEqual(r.status_code, 400)


class EvaluationEndpointTests(APITestCase):
    def test_evaluation_payload_shape(self):
        r = self.client.get("/api/evaluation/")
        self.assertEqual(r.status_code, 200)
        for key in ["generation", "classification", "experiments", "plots", "training_config", "dataset"]:
            self.assertIn(key, r.data)

    def test_unknown_plot_is_404(self):
        self.assertEqual(self.client.get("/api/evaluation/plots/../../etc/").status_code, 404)
        self.assertEqual(self.client.get("/api/evaluation/plots/not_a_plot/").status_code, 404)
