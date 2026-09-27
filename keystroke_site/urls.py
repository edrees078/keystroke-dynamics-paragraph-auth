from django.contrib import admin
from django.urls import path


from collector.views import (
    home, typing_page, typing_paragraph_page, start_attempt, post_batch, finish_attempt, export_processed, export_aggregated,export_aggregated_paragraph, api_authenticate, auth_paragraph_page, auth_paragraph_verify_page
)

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", home, name="home"),
    path("type/", typing_page, name="typing"),
    path("typeParagraf/", typing_paragraph_page, name="typing_paragraph"),
    path("api/attempt/start/", start_attempt),
    path("api/keystrokes/batch/", post_batch),
    path("api/attempt/finish/", finish_attempt),
    path("api/export/processed/", export_processed),
    path("api/export/aggregated/", export_aggregated),
    path("api/export/aggregated_paragraph/", export_aggregated_paragraph),
    path("api/authenticate/", api_authenticate, name="api_authenticate"),
    path("auth/paragraph/", auth_paragraph_page, name="auth_paragraph"),
    path("auth/paragraph/verify/", auth_paragraph_verify_page, name="auth_paragraph_verify"),
]
