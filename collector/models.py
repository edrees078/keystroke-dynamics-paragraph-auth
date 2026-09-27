from django.db import models

class Participant(models.Model):
    alias = models.CharField(max_length=64, unique=True)

class Session(models.Model):
    participant = models.ForeignKey(Participant, on_delete=models.CASCADE)
    word = models.CharField(max_length=32, default=".tie5Roanl")
    index = models.IntegerField()  # 1..8

class Attempt(models.Model):
    session = models.ForeignKey(Session, on_delete=models.CASCADE)
    attempt_no = models.IntegerField(default=1)
    started_at = models.FloatField(default=0.0)    # performance.now() ms
    finished_at = models.FloatField(default=0.0)
    total_time_ms = models.FloatField(default=0.0)
    features_json = models.JSONField(default=dict)

class KeystrokeEvent(models.Model):
    attempt = models.ForeignKey(Attempt, on_delete=models.CASCADE, related_name="events")
    type = models.CharField(max_length=4)  # "down" / "up"
    key = models.CharField(max_length=32)
    code = models.CharField(max_length=32)
    t = models.FloatField()  # ms


# Create your models here.
