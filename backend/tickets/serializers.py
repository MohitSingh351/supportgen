from rest_framework import serializers

from ai.registry import get_config

from .models import Ticket

MIN_BODY_CHARS = 10


class TicketInputSerializer(serializers.Serializer):
    subject = serializers.CharField(max_length=300, required=False, allow_blank=True, default="")
    body = serializers.CharField(trim_whitespace=True, allow_blank=False,
                                 error_messages={"blank": "The customer ticket cannot be empty."})

    def validate_body(self, value):
        max_chars = get_config().max_ticket_chars
        if len(value) < MIN_BODY_CHARS:
            raise serializers.ValidationError(f"The ticket is too short (minimum {MIN_BODY_CHARS} characters).")
        if len(value) > max_chars:
            raise serializers.ValidationError(
                f"The ticket is too long ({len(value)} characters, maximum {max_chars}).")
        return value


class CompareInputSerializer(TicketInputSerializer):
    PROVIDERS = ("base", "finetuned", "gemini")
    providers = serializers.ListField(child=serializers.ChoiceField(choices=PROVIDERS),
                                      required=False, default=list(PROVIDERS))


class TicketSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ticket
        fields = "__all__"


class TicketListSerializer(serializers.ModelSerializer):
    class Meta:
        model = Ticket
        fields = ["id", "subject", "body", "category", "priority", "ticket_type", "sentiment",
                  "summary", "model_used", "created_at"]

