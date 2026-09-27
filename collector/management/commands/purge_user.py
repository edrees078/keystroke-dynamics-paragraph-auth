# collector/management/commands/purge_user.py
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

# استيراد نسبي من نفس التطبيق الذي يحوي models.py
from ...models import Participant  # CASCADE سيحذف الجلسات والمحاولات والأحداث

class Command(BaseCommand):
    help = "Delete a user and all related records by alias"

    def add_arguments(self, parser):
        parser.add_argument("--alias", required=True, help="User alias to delete")

    @transaction.atomic
    def handle(self, *args, **opts):
        alias = opts["alias"].strip()
        try:
            p = Participant.objects.get(alias=alias)
        except Participant.DoesNotExist:
            raise CommandError(f"user '{alias}' not found")

        self.stdout.write(self.style.WARNING(f"Deleting user '{alias}' ..."))
        # بفضل on_delete=CASCADE في models، حذف Participant يكفي
        p.delete()
        self.stdout.write(self.style.SUCCESS("Done."))
