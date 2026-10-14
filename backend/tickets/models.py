from django.db import models


class Ticket(models.Model):
    """One analysed support ticket and everything the AI produced for it."""

    subject = models.CharField(max_length=300, blank=True)
    body = models.TextField()

    generated_response = models.TextField()
    model_used = models.CharField(max_length=100)

    summary = models.TextField()
    summary_source = models.CharField(max_length=30)

    category = models.CharField(max_length=100, null=True, blank=True)
    category_confidence = models.FloatField(null=True, blank=True)
    priority = models.CharField(max_length=30, null=True, blank=True)
    priority_confidence = models.FloatField(null=True, blank=True)
    ticket_type = models.CharField(max_length=30, null=True, blank=True)
    type_confidence = models.FloatField(null=True, blank=True)
    sentiment = models.CharField(max_length=20, null=True, blank=True)
    sentiment_score = models.FloatField(null=True, blank=True)

    recommended_action = models.TextField()
    action_source = models.CharField(max_length=30)

    timings_ms = models.JSONField(default=dict)
    warnings = models.JSONField(default=list)
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.pk} {self.subject or self.body[:40]}"
