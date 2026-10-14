import logging
from datetime import timedelta

from django.db import DatabaseError, connection
from django.db.models import Count
from django.db.models.functions import TruncDate
from django.http import FileResponse, Http404
from django.utils import timezone
from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView

from ai.providers import GenerationError, ProviderUnavailable
from ai.registry import (get_analyzer, get_base_provider, get_config, get_finetuned_provider,
                         get_gemini_provider)

from .evaluation import PLOT_FILES, load_evaluation
from .models import Ticket
from .serializers import (CompareInputSerializer, TicketInputSerializer, TicketListSerializer,
                          TicketSerializer)

logger = logging.getLogger(__name__)


def _error(message: str, code: int) -> Response:
    return Response({"error": message}, status=code)


class AnalyzeTicketView(APIView):
    """POST /api/tickets/analyze/ — run every AI component on one ticket and store the result."""

    def post(self, request):
        data = TicketInputSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        subject, body = data.validated_data["subject"], data.validated_data["body"]

        try:
            result = get_analyzer().analyze(subject, body).to_dict()
        except ProviderUnavailable as exc:
            return _error(str(exc), status.HTTP_503_SERVICE_UNAVAILABLE)
        except GenerationError as exc:
            logger.exception("Inference failed")
            return _error(f"Inference failed: {exc}", status.HTTP_500_INTERNAL_SERVER_ERROR)

        try:
            ticket = Ticket.objects.create(subject=subject, body=body, **result)
            payload, code = TicketSerializer(ticket).data, status.HTTP_201_CREATED
        except DatabaseError:
            # The analysis is still useful to the agent even if it could not be stored.
            logger.exception("Could not save ticket")
            payload = {"id": None, "subject": subject, "body": body, **result}
            payload["warnings"] = result["warnings"] + ["Database unavailable: this analysis was not saved."]
            code = status.HTTP_200_OK
        return Response(payload, status=code)


class TicketListView(generics.ListAPIView):
    """GET /api/tickets/ — analysed ticket history (paginated, newest first)."""
    queryset = Ticket.objects.all()
    serializer_class = TicketListSerializer


class TicketDetailView(generics.RetrieveAPIView):
    queryset = Ticket.objects.all()
    serializer_class = TicketSerializer


def _distribution(field: str) -> dict:
    rows = Ticket.objects.exclude(**{f"{field}__isnull": True}).values(field).annotate(n=Count("id")).order_by("-n")
    return {r[field]: r["n"] for r in rows}


class AnalyticsView(APIView):
    """GET /api/analytics/ — dashboard statistics over analysed tickets."""

    def get(self, request):
        since = timezone.now() - timedelta(days=30)
        per_day = (Ticket.objects.filter(created_at__gte=since).annotate(day=TruncDate("created_at"))
                   .values("day").annotate(n=Count("id")).order_by("day"))
        high = Ticket.objects.filter(priority__in=["high", "critical"])
        return Response({
            "total_tickets": Ticket.objects.count(),
            "priority": _distribution("priority"),
            "category": _distribution("category"),
            "ticket_type": _distribution("ticket_type"),
            "sentiment": _distribution("sentiment"),
            "high_priority_by_category": {r["category"]: r["n"] for r in
                                          high.values("category").annotate(n=Count("id")).order_by("-n")},
            "tickets_per_day": [{"day": r["day"].isoformat(), "count": r["n"]} for r in per_day],
            "recent": TicketListSerializer(Ticket.objects.all()[:5], many=True).data,
        })


class EvaluationView(APIView):
    """GET /api/evaluation/ — metrics produced by the Colab notebooks."""

    def get(self, request):
        cfg = get_config()
        return Response(load_evaluation(cfg.results_dir, cfg.model_dir))


class EvaluationPlotView(APIView):
    """GET /api/evaluation/plots/<key>/ — one of the known result plots (whitelisted names only)."""

    def get(self, request, key):
        name = PLOT_FILES.get(key)
        path = get_config().results_dir / "plots" / name if name else None
        if not path or not path.exists():
            raise Http404("Unknown plot")
        return FileResponse(path.open("rb"), content_type="image/png")


class CompareView(APIView):
    """POST /api/compare/ — the same ticket through base FLAN-T5, fine-tuned FLAN-T5 and (optionally) Gemini."""

    LABELS = {"base": "Base FLAN-T5-small (no fine-tuning)",
              "finetuned": "Fine-tuned FLAN-T5-small (ours)",
              "gemini": "Gemini (optional benchmark)"}

    def post(self, request):
        data = CompareInputSerializer(data=request.data)
        data.is_valid(raise_exception=True)
        subject, body = data.validated_data["subject"], data.validated_data["body"]
        providers = {"base": get_base_provider, "finetuned": get_finetuned_provider, "gemini": get_gemini_provider}

        results = []
        for key in data.validated_data["providers"]:
            provider = providers[key]()
            item = {"provider": key, "label": self.LABELS[key], "text": None, "latency_ms": None, "model": None}
            if not provider.is_available():
                item["status"] = "not_configured" if key == "gemini" else "unavailable"
                item["message"] = ("Set GEMINI_API_KEY in .env to enable the Gemini benchmark."
                                   if key == "gemini" else f"Model not found at {getattr(provider, 'source', '')}")
            else:
                try:
                    r = provider.generate(subject, body)
                    item.update(status="ok", text=r.text, latency_ms=r.latency_ms, model=r.model)
                except (ProviderUnavailable, GenerationError) as exc:
                    item.update(status="error", message=str(exc))
            results.append(item)
        return Response({"subject": subject, "body": body, "results": results})


class HealthView(APIView):
    """GET /api/health/ — which components are available."""

    def get(self, request):
        try:
            connection.ensure_connection()
            db = {"ok": True, "engine": connection.vendor}
        except DatabaseError as exc:
            db = {"ok": False, "error": str(exc)}
        info = get_analyzer().status()
        info.update(database=db, gemini_configured=get_gemini_provider().is_available(),
                    device=get_finetuned_provider().device)
        return Response(info)
