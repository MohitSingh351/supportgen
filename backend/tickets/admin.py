from django.contrib import admin

from .models import Ticket


@admin.register(Ticket)
class TicketAdmin(admin.ModelAdmin):
    list_display = ("id", "subject", "category", "priority", "ticket_type", "sentiment", "created_at")
    list_filter = ("category", "priority", "ticket_type", "sentiment")
    search_fields = ("subject", "body")
