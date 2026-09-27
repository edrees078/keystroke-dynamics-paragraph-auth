from django.contrib import admin
from .models import Participant, Session, Attempt, KeystrokeEvent

@admin.register(Participant)
class ParticipantAdmin(admin.ModelAdmin):
    list_display = ("id", "alias")
    search_fields = ("alias",)

@admin.register(Session)
class SessionAdmin(admin.ModelAdmin):
    list_display = ("id", "participant", "word", "index")
    list_filter = ("index", "word")

@admin.register(Attempt)
class AttemptAdmin(admin.ModelAdmin):
    list_display = ("id", "session", "attempt_no", "total_time_ms")
    list_filter = ("session__index",)
    readonly_fields = ("features_json",)

@admin.register(KeystrokeEvent)
class KeystrokeEventAdmin(admin.ModelAdmin):
    list_display = ("id", "attempt", "type", "key", "code", "t")
    list_filter = ("type", "key")
    search_fields = ("key", "code")
