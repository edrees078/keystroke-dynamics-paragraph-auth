# collector/management/commands/purge_paragraph_user.py
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from ...models import Participant, Session, Attempt, KeystrokeEvent


PARAGRAPH_WORD = "TR_PARAGRAPH_1"  # نفس القيمة التي تستخدمها في جلسة الفقرة


class Command(BaseCommand):
    help = "Delete ONLY paragraph-test data for a given user alias (sessions with word=TR_PARAGRAPH_1)."

    def add_arguments(self, parser):
        parser.add_argument(
            "--alias",
            required=True,
            help="User alias to delete paragraph-session data for (without touching word-test data)."
        )

    @transaction.atomic
    def handle(self, *args, **opts):
        alias = opts["alias"].strip()

        try:
            p = Participant.objects.get(alias=alias)
        except Participant.DoesNotExist:
            raise CommandError(f"user '{alias}' not found")

        # كل الجلسات الخاصة بالفقرة لهذا المستخدم فقط
        paragraph_sessions = Session.objects.filter(participant=p, word=PARAGRAPH_WORD)

        if not paragraph_sessions.exists():
            raise CommandError(
                f"user '{alias}' has no paragraph sessions (word={PARAGRAPH_WORD}). "
                "Nothing to delete."
            )

        # (اختياري) حساب الأعداد للتقرير قبل الحذف
        attempts_qs = Attempt.objects.filter(session__in=paragraph_sessions)
        events_qs = KeystrokeEvent.objects.filter(attempt__in=attempts_qs)

        n_sessions = paragraph_sessions.count()
        n_attempts = attempts_qs.count()
        n_events = events_qs.count()

        self.stdout.write(self.style.WARNING(
            f"Deleting paragraph data for user '{alias}': "
            f"{n_sessions} sessions, {n_attempts} attempts, {n_events} keystroke events..."
        ))

        # بفضل CASCADE يكفي حذف الـ Sessions
        paragraph_sessions.delete()

        self.stdout.write(self.style.SUCCESS(
            "Done. Word-test sessions (.tie5Roanl) for this user were NOT touched."
        ))
