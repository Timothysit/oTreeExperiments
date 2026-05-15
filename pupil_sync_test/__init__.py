from otree.api import *
import json
import time


class C(BaseConstants):
    NAME_IN_URL = "pupil_sync_test"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    # Durations in milliseconds
    STIM_SEQUENCE = [
        {"color": "black", "duration_ms": 2000},
        {"color": "white", "duration_ms": 1000},
        {"color": "black", "duration_ms": 3000},
        {"color": "white", "duration_ms": 1500},
        {"color": "black", "duration_ms": 1000},
        {"color": "white", "duration_ms": 3000},
    ]


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    sync_log_json = models.LongStringField(blank=True)


class Stimulus(Page):
    live_method = "live_stimulus"

    @staticmethod
    def js_vars(player):
        return dict(
            stim_sequence=C.STIM_SEQUENCE,
            pupil_annotation_url="http://127.0.0.1:5000/annotation",
        )

    @staticmethod
    def live_stimulus(player, data):
        if data.get("type") == "sync_log":
            player.sync_log_json = json.dumps(data.get("log", []))

        return {player.id_in_group: dict(status="ok", server_ts=time.time())}


page_sequence = [Stimulus]