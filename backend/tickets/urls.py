from django.urls import path

from . import views

urlpatterns = [
    path("tickets/analyze/", views.AnalyzeTicketView.as_view(), name="ticket-analyze"),
    path("tickets/", views.TicketListView.as_view(), name="ticket-list"),
    path("tickets/<int:pk>/", views.TicketDetailView.as_view(), name="ticket-detail"),
    path("analytics/", views.AnalyticsView.as_view(), name="analytics"),
    path("evaluation/", views.EvaluationView.as_view(), name="evaluation"),
    path("evaluation/plots/<str:key>/", views.EvaluationPlotView.as_view(), name="evaluation-plot"),
    path("compare/", views.CompareView.as_view(), name="compare"),
    path("health/", views.HealthView.as_view(), name="health"),
]
