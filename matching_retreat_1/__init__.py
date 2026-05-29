from otree.api import *
import random
import time
import json

from .pupil_sync import send_pupil_annotation


class C(BaseConstants):
    NAME_IN_URL = "matching_retreat_1"
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1
    NUM_TRIALS_MULTI = 400
    REWARD_WIN = 10
    REWARD_LOSS = 0
    TRIAL_TIMER_MIN_MS = 2000
    TRIAL_TIMER_MAX_MS = 5000


def num_trials_multi(player):
    g = player.group
    if getattr(g, "num_trials_multi", None):
        return int(g.num_trials_multi)
    return int(player.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))


def _overall_done(player):
    return player.current_trial >= num_trials_multi(player)


def _new_trial_timer_ms():
    return random.randint(C.TRIAL_TIMER_MIN_MS, C.TRIAL_TIMER_MAX_MS)


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    started = models.BooleanField(initial=False)
    p1_started = models.BooleanField(initial=False)
    p2_started = models.BooleanField(initial=False)

    # Trial count set in Setup, per group.
    num_trials_multi = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_MULTI)

    # Temporary storage for each multiplayer trial.
    # Choice can be "L", "R", or "NONE" if the player was outside both zones at deadline.
    p1_choice = models.StringField(blank=True)
    p2_choice = models.StringField(blank=True)
    p1_rt_ms = models.IntegerField(initial=0)
    p2_rt_ms = models.IntegerField(initial=0)

    # Same timer is sent to both players for the current trial.
    trial_timer_ms = models.IntegerField(initial=0)

    # One row per completed multiplayer trial.
    trial_log_json = models.LongStringField(initial="[]")

    def append_trial(self, row: dict):
        log = json.loads(self.trial_log_json or "[]")
        log.append(row)
        self.trial_log_json = json.dumps(log)
    
    # for saving cursor positions 
    cursor_log_json = models.LongStringField(initial="[]")


class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)
    last_rt_ms = models.IntegerField(initial=0)


def _ready_payload(player: Player):
    return dict(
        type="ready",
        phase="multi",
        trial=player.current_trial + 1,
        trial_total=num_trials_multi(player),
        total_points=player.total_points,
        timer_ms=player.group.trial_timer_ms,
    )


def live_game(player: Player, data):
    """
    Handles messages from the browser.

    Expected messages:
        {type: "start"}
        {type: "cursor", x: 0.5, y: 0.5}
        {type: "choice", choice: "L" or "R" or "NONE", rt_ms: 3000}
    """
    print("LIVE_GAME", player.participant.code, player.id_in_group, data, flush=True)

    msg_type = data.get("type")
    g = player.group

    # ---------------------------------------------------------------------
    # Real-time cursor relay.
    # ---------------------------------------------------------------------
    if msg_type == "cursor":
        try:
            x = float(data.get("x"))
            y = float(data.get("y"))
        except (TypeError, ValueError):
            return

        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        others = player.get_others_in_group()
        if not others:
            return

        return {
            others[0].id_in_group: dict(
                type="opponent_cursor",
                x=x,
                y=y,
                player_id=player.id_in_group,
            )
        }

    if msg_type == "cursor_trace":
        g = player.group

        trace_log = json.loads(g.cursor_log_json or "[]")
        trace_log.append(dict(
            player_id=player.id_in_group,
            participant_code=player.participant.code,
            trial=player.current_trial + 1,
            samples=data.get("samples", []),
            server_ts=time.time(),
        ))
        g.cursor_log_json = json.dumps(trace_log)

        return

    # ---------------------------------------------------------------------
    # First click after intro. Wait until both players have clicked before
    # starting the first timer, so the trial begins together.
    # ---------------------------------------------------------------------
    if msg_type == "start":
        if not g.started:
            g.started = True
            g.trial_log_json = "[]"
            g.p1_choice = ""
            g.p2_choice = ""
            g.p1_rt_ms = 0
            g.p2_rt_ms = 0
            g.trial_timer_ms = _new_trial_timer_ms()

            for p in g.get_players():
                p.current_trial = 0
                p.total_points = 0
                p.last_choice = ""
                p.last_reward = 0
                p.last_rt_ms = 0

        if player.id_in_group == 1:
            g.p1_started = True
        else:
            g.p2_started = True

        send_pupil_annotation(
            "task_start",
            participant_code=player.participant.code,
            player_id=player.id_in_group,
            phase="multi",
            trial=player.current_trial + 1,
        )

        if not (g.p1_started and g.p2_started):
            return {
                player.id_in_group: dict(
                    type="waiting_start",
                    message="Waiting for the other player to start...",
                    trial=player.current_trial + 1,
                    total_points=player.total_points,
                )
            }

        p1 = g.get_player_by_id(1)
        p2 = g.get_player_by_id(2)
        return {
            1: _ready_payload(p1),
            2: _ready_payload(p2),
        }

    if msg_type != "choice":
        return

    if _overall_done(player):
        return

    choice = data.get("choice")
    if choice not in ["L", "R", "NONE"]:
        choice = "NONE"

    rt_ms = int(data.get("rt_ms", 0) or 0)

    player.last_choice = choice
    player.last_rt_ms = rt_ms

    if player.id_in_group == 1:
        g.p1_choice = choice
        g.p1_rt_ms = rt_ms
    else:
        g.p2_choice = choice
        g.p2_rt_ms = rt_ms

    send_pupil_annotation(
        "multi_deadline_choice",
        participant_code=player.participant.code,
        player_id=player.id_in_group,
        overall_trial=player.current_trial + 1,
        block="multi",
        block_trial=player.current_trial + 1,
        choice=choice,
        rt_ms=rt_ms,
        timer_ms=g.trial_timer_ms,
    )

    # If the other player has not reported their deadline choice yet, wait.
    if not g.p1_choice or not g.p2_choice:
        return {
            player.id_in_group: dict(
                type="wait_opponent",
                phase="multi",
                trial=player.current_trial + 1,
                trial_total=num_trials_multi(player),
                total_points=player.total_points,
            )
        }

    # Both deadline choices are in -> resolve the trial for BOTH players.
    p1 = g.get_player_by_id(1)
    p2 = g.get_player_by_id(2)
    c1, c2 = g.p1_choice, g.p2_choice
    rt1, rt2 = g.p1_rt_ms, g.p2_rt_ms

    p1_valid = c1 in ["L", "R"]
    p2_valid = c2 in ["L", "R"]

    outcome_reason = ""
    winner = 0

    if p1_valid and not p2_valid:
        r1, r2 = C.REWARD_WIN, C.REWARD_LOSS
        winner = 1
        outcome_reason = "p2_no_choice"
    elif p2_valid and not p1_valid:
        r1, r2 = C.REWARD_LOSS, C.REWARD_WIN
        winner = 2
        outcome_reason = "p1_no_choice"
    elif not p1_valid and not p2_valid:
        r1, r2 = C.REWARD_LOSS, C.REWARD_LOSS
        winner = 0
        outcome_reason = "both_no_choice"
    else:
        # Player 1 is the matcher; Player 2 is the mismatcher.
        if c1 == c2:
            r1, r2 = C.REWARD_WIN, C.REWARD_LOSS
            winner = 1
            outcome_reason = "match"
        else:
            r1, r2 = C.REWARD_LOSS, C.REWARD_WIN
            winner = 2
            outcome_reason = "mismatch"

    p1.last_reward = r1
    p2.last_reward = r2
    p1.total_points += r1
    p2.total_points += r2

    current_trial = p1.current_trial + 1

    pupil_sync = send_pupil_annotation(
        "multi_trial_outcome",
        overall_trial=current_trial,
        block="multi",
        block_trial=current_trial,
        p1_code=p1.participant.code,
        p2_code=p2.participant.code,
        p1_choice=c1,
        p2_choice=c2,
        p1_rt_ms=rt1,
        p2_rt_ms=rt2,
        p1_reward=r1,
        p2_reward=r2,
        winner=winner,
        outcome_reason=outcome_reason,
        timer_ms=g.trial_timer_ms,
    )

    g.append_trial(
        dict(
            overall_trial=current_trial,
            block="multi",
            block_trial=current_trial,
            opponent_type="human",
            p1_code=p1.participant.code,
            p2_code=p2.participant.code,
            p1_choice=c1,
            p2_choice=c2,
            p1_valid_choice=p1_valid,
            p2_valid_choice=p2_valid,
            p1_rt_ms=rt1,
            p2_rt_ms=rt2,
            timer_ms=g.trial_timer_ms,
            p1_reward=r1,
            p2_reward=r2,
            winner=winner,
            outcome_reason=outcome_reason,
            p1_total_points_after=p1.total_points,
            p2_total_points_after=p2.total_points,
            server_ts=time.time(),
            pupil_multi_outcome_sync=pupil_sync,
        )
    )

    p1.current_trial += 1
    p2.current_trial += 1

    is_last = _overall_done(p1) or _overall_done(p2)

    # Prepare next trial's random deadline before sending feedback.
    next_timer_ms = 0 if is_last else _new_trial_timer_ms()
    g.trial_timer_ms = next_timer_ms

    g.p1_choice = ""
    g.p2_choice = ""
    g.p1_rt_ms = 0
    g.p2_rt_ms = 0

    return {
        1: dict(
            type="feedback",
            phase="multi",
            trial=current_trial,
            trial_total=num_trials_multi(p1),
            your_choice=c1,
            other_choice=c2,
            reward=r1,
            total_points=p1.total_points,
            winner=winner,
            outcome_reason=outcome_reason,
            is_last=is_last,
            next_trial=p1.current_trial + 1,
            next_timer_ms=next_timer_ms,
        ),
        2: dict(
            type="feedback",
            phase="multi",
            trial=current_trial,
            trial_total=num_trials_multi(p2),
            your_choice=c2,
            other_choice=c1,
            reward=r2,
            total_points=p2.total_points,
            winner=winner,
            outcome_reason=outcome_reason,
            is_last=is_last,
            next_trial=p2.current_trial + 1,
            next_timer_ms=next_timer_ms,
        ),
    }


class Setup(Page):
    form_model = "group"
    form_fields = ["num_trials_multi"]

    @staticmethod
    def is_displayed(player):
        return player.id_in_group == 1


class WaitAfterSetup(WaitPage):
    pass


class Game(Page):
    live_method = live_game

    @staticmethod
    def js_vars(player: Player):
        return dict(
            num_trials_multi=num_trials_multi(player),
            player_id=player.id_in_group,
        )


class End(Page):
    @staticmethod
    def is_displayed(player: Player):
        return _overall_done(player)

    @staticmethod
    def vars_for_template(player: Player):
        return dict(
            total_points=player.total_points,
            participant_code=player.participant.code,
            survey_url="https://forms.gle/KR7CJY4MENZ5dtX98",
        )


page_sequence = [Setup, WaitAfterSetup, Game, End]
