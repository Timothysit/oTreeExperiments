from otree.api import *
import random
import time
import json

from .pupil_sync import send_pupil_annotation


class C(BaseConstants):
    NAME_IN_URL = "matching_retreat_1"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1
    NUM_TRIALS_SINGLE = 400
    NUM_TRIALS_MULTI = 400
    NUM_CALIBRATION_TRIALS = 10
    ANTICIPATE_WARMUP_TRIALS = 5
    REWARD_WIN = 10
    REWARD_LOSS = 0
    TRIAL_TIMER_MIN_MS = 2000
    TRIAL_TIMER_MAX_MS = 5000


def num_trials_multi(player):
    g = player.group
    if getattr(g, "num_trials_multi", None):
        return int(g.num_trials_multi)
    return int(player.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))


def num_trials_single(player):
    g = player.group
    if getattr(g, "num_trials_single", None):
        return int(g.num_trials_single)
    return int(player.session.config.get("num_trials_single", C.NUM_TRIALS_SINGLE))


def is_single_player_mode(player):
    if player.session.config.get("game_mode") == "single":
        return True

    if player.group.game_mode == "single":
        return True

    try:
        if player.session.num_participants == 1:
            return True
    except AttributeError:
        pass

    return len(player.group.get_players()) == 1


def selected_game_mode(player):
    return "single" if is_single_player_mode(player) else "multi"


def is_calibration_trial(player):
    return is_single_player_mode(player) and player.current_trial < C.NUM_CALIBRATION_TRIALS


def num_trials_total(player):
    if is_single_player_mode(player):
        return C.NUM_CALIBRATION_TRIALS + num_trials_single(player)
    return num_trials_multi(player)


def _overall_done(player):
    return player.current_trial >= num_trials_total(player)


def _new_trial_timer_ms():
    return random.randint(C.TRIAL_TIMER_MIN_MS, C.TRIAL_TIMER_MAX_MS)


def _median(values):
    clean = sorted(float(v) for v in values if v is not None)
    n = len(clean)
    if n == 0:
        return None
    mid = n // 2
    if n % 2:
        return clean[mid]
    return (clean[mid - 1] + clean[mid]) / 2


def _calibration_metrics(player):
    try:
        metrics = json.loads(player.calibration_metrics_json or "[]")
    except json.JSONDecodeError:
        return []
    return metrics if isinstance(metrics, list) else []


def _calibration_medians(player):
    metrics = _calibration_metrics(player)
    return dict(
        rt_ms=_median(row.get("movement_rt_ms") for row in metrics),
        top_speed_px_s=_median(row.get("top_speed_px_s") for row in metrics),
    )


def _single_block_trial(player):
    if is_calibration_trial(player):
        return player.current_trial + 1
    return player.current_trial - C.NUM_CALIBRATION_TRIALS + 1


def _single_block_total(player):
    return C.NUM_CALIBRATION_TRIALS if is_calibration_trial(player) else num_trials_single(player)


class Subsession(BaseSubsession):
    def creating_session(self):
        default_game_mode = "single" if self.session.num_participants == 1 else "multi"
        game_mode = self.session.config.get("game_mode", default_game_mode)
        single_opponent = self.session.config.get("single_opponent", "follow")
        num_single = int(self.session.config.get("num_trials_single", C.NUM_TRIALS_SINGLE))
        num_multi = int(self.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))
        follow_reaction_time_pct = int(self.session.config.get("follow_reaction_time_pct", 100))
        follow_speed_pct = int(self.session.config.get("follow_speed_pct", 80))
        anticipate_warmup_trials = int(
            self.session.config.get("anticipate_warmup_trials", C.ANTICIPATE_WARMUP_TRIALS)
        )

        for g in self.get_groups():
            g.game_mode = game_mode
            g.single_opponent = single_opponent
            g.single_opponent_final = single_opponent
            g.num_trials_single = num_single
            g.num_trials_multi = num_multi
            g.follow_reaction_time_pct = follow_reaction_time_pct
            g.follow_speed_pct = follow_speed_pct
            g.anticipate_warmup_trials = anticipate_warmup_trials


def get_single_opponent_for_player(player):
    opp = player.group.single_opponent_final or "follow"
    return opp, f"{opp}_v1"


class Group(BaseGroup):
    game_mode = models.StringField(
        choices=[["multi", "Two players"], ["single", "Single player"]],
        initial="multi",
        blank=False,
        widget=widgets.RadioSelect,
    )

    started = models.BooleanField(initial=False)
    p1_started = models.BooleanField(initial=False)
    p2_started = models.BooleanField(initial=False)

    # Trial count set in Setup, per group.
    num_trials_single = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_SINGLE)
    num_trials_multi = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_MULTI)

    single_opponent = models.StringField(
        choices=[["follow", "Follow"], ["anticipate", "Anticipate"]],
        initial="follow",
        blank=False,
        widget=widgets.RadioSelect,
    )
    single_opponent_final = models.StringField(blank=True)
    follow_reaction_time_pct = models.IntegerField(min=0, max=300, initial=100)
    follow_speed_pct = models.IntegerField(min=1, max=300, initial=80)
    anticipate_warmup_trials = models.IntegerField(min=0, max=100, initial=C.ANTICIPATE_WARMUP_TRIALS)

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
    algo_state_json = models.LongStringField(initial="{}")

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
    trial_timer_ms = models.IntegerField(initial=0)
    instructed_choice = models.StringField(blank=True)
    calibration_rt_sum_ms = models.FloatField(initial=0)
    calibration_top_speed_sum_px_s = models.FloatField(initial=0)
    calibration_count = models.IntegerField(initial=0)
    calibration_metrics_json = models.LongStringField(initial="[]")


def _new_instruction():
    return random.choice(["L", "R"])


def _ready_payload(player: Player):
    phase = "single" if is_single_player_mode(player) else "multi"
    timer_ms = player.trial_timer_ms if phase == "single" else player.group.trial_timer_ms
    if phase == "single" and is_calibration_trial(player) and player.instructed_choice not in ["L", "R"]:
        player.instructed_choice = _new_instruction()

    calibration_medians = _calibration_medians(player)
    trial = _single_block_trial(player) if phase == "single" else player.current_trial + 1
    trial_total = _single_block_total(player) if phase == "single" else num_trials_multi(player)

    return dict(
        type="ready",
        phase=phase,
        block="calibration" if is_calibration_trial(player) else phase,
        trial=trial,
        trial_total=trial_total,
        total_points=player.total_points,
        timer_ms=timer_ms,
        calibration_trials=C.NUM_CALIBRATION_TRIALS,
        instructed_choice=player.instructed_choice if is_calibration_trial(player) else "",
        single_opponent_type=player.group.single_opponent_final or player.group.single_opponent,
        follow_reaction_time_pct=player.group.follow_reaction_time_pct,
        follow_speed_pct=player.group.follow_speed_pct,
        anticipate_warmup_trials=player.group.anticipate_warmup_trials,
        calibration_avg_rt_ms=calibration_medians["rt_ms"],
        calibration_avg_top_speed_px_s=calibration_medians["top_speed_px_s"],
        calibration_median_rt_ms=calibration_medians["rt_ms"],
        calibration_median_top_speed_px_s=calibration_medians["top_speed_px_s"],
    )


def _single_choice(player: Player, data, choice: str, rt_ms: int):
    g = player.group

    opponent_type, opponent_id = get_single_opponent_for_player(player)
    current_trial = player.current_trial + 1
    block_trial = _single_block_trial(player)
    block_total = _single_block_total(player)
    calibration = is_calibration_trial(player)

    opponent_choice = data.get("opponent_choice")
    if opponent_choice not in ["L", "R", "NONE"]:
        opponent_choice = "NONE" if calibration else random.choice(["L", "R"])

    instructed_choice = data.get("instructed_choice") or player.instructed_choice
    if instructed_choice not in ["L", "R"]:
        instructed_choice = ""

    try:
        movement_rt_ms = float(data.get("movement_rt_ms")) if data.get("movement_rt_ms") is not None else None
    except (TypeError, ValueError):
        movement_rt_ms = None

    try:
        top_speed_px_s = float(data.get("top_speed_px_s")) if data.get("top_speed_px_s") is not None else None
    except (TypeError, ValueError):
        top_speed_px_s = None

    valid_choice = choice in ["L", "R"]
    opponent_valid = opponent_choice in ["L", "R"]
    is_win = valid_choice and opponent_valid and choice != opponent_choice
    reward = C.REWARD_WIN if is_win else C.REWARD_LOSS
    if not valid_choice:
        outcome_reason = "no_choice"
    elif not opponent_valid:
        outcome_reason = "opponent_no_choice"
    elif choice != opponent_choice:
        outcome_reason = "mismatch"
    else:
        outcome_reason = "match"

    if calibration:
        reward = 0
        outcome_reason = "calibration"
        if movement_rt_ms is not None and top_speed_px_s is not None:
            player.calibration_rt_sum_ms += movement_rt_ms
            player.calibration_top_speed_sum_px_s += top_speed_px_s
            player.calibration_count += 1
            metrics = _calibration_metrics(player)
            metrics.append(dict(
                block_trial=block_trial,
                instructed_choice=instructed_choice,
                movement_rt_ms=movement_rt_ms,
                top_speed_px_s=top_speed_px_s,
            ))
            player.calibration_metrics_json = json.dumps(metrics)

    player.last_choice = choice
    player.last_rt_ms = rt_ms
    player.last_reward = reward
    player.total_points += reward

    pupil_sync = send_pupil_annotation(
        "single_calibration_outcome" if calibration else "single_trial_outcome",
        participant_code=player.participant.code,
        player_id=player.id_in_group,
        overall_trial=current_trial,
        block="calibration" if calibration else "single",
        block_trial=block_trial,
        instructed_choice=instructed_choice,
        player_choice=choice,
        opponent_choice=opponent_choice,
        rt_ms=rt_ms,
        movement_rt_ms=movement_rt_ms,
        top_speed_px_s=top_speed_px_s,
        reward=reward,
        outcome_reason=outcome_reason,
        timer_ms=player.trial_timer_ms,
    )

    g.append_trial(
        dict(
            overall_trial=current_trial,
            block="calibration" if calibration else "single",
            block_trial=block_trial,
            opponent_type=opponent_type,
            opponent_id=opponent_id,
            player_code=player.participant.code,
            instructed_choice=instructed_choice,
            player_choice=choice,
            valid_choice=valid_choice,
            opponent_choice=opponent_choice,
            opponent_valid_choice=opponent_valid,
            player_rt_ms=rt_ms,
            movement_rt_ms=movement_rt_ms,
            top_speed_px_s=top_speed_px_s,
            calibration_median_rt_ms=_calibration_medians(player)["rt_ms"],
            calibration_median_top_speed_px_s=_calibration_medians(player)["top_speed_px_s"],
            opponent_final_x=data.get("opponent_x"),
            opponent_final_y=data.get("opponent_y"),
            opponent_strategy=data.get("opponent_strategy"),
            follow_reaction_time_pct=g.follow_reaction_time_pct,
            follow_speed_pct=g.follow_speed_pct,
            anticipate_warmup_trials=g.anticipate_warmup_trials,
            timer_ms=player.trial_timer_ms,
            reward=reward,
            outcome_reason=outcome_reason,
            total_points_after=player.total_points,
            server_ts=time.time(),
            pupil_single_outcome_sync=pupil_sync,
        )
    )

    player.current_trial += 1

    is_last = _overall_done(player)
    next_timer_ms = 0 if is_last else _new_trial_timer_ms()
    player.trial_timer_ms = next_timer_ms
    player.instructed_choice = _new_instruction() if (not is_last and is_calibration_trial(player)) else ""
    calibration_medians = _calibration_medians(player)

    return {
        player.id_in_group: dict(
            type="feedback",
            phase="single",
            block="calibration" if calibration else "single",
            trial=block_trial,
            trial_total=block_total,
            your_choice=choice,
            other_choice=opponent_choice,
            reward=reward,
            total_points=player.total_points,
            winner=player.id_in_group if reward > 0 else 0,
            outcome_reason=outcome_reason,
            is_last=is_last,
            next_trial=0 if is_last else _single_block_trial(player),
            next_timer_ms=next_timer_ms,
            next_block="calibration" if (not is_last and is_calibration_trial(player)) else "single",
            next_instructed_choice=player.instructed_choice,
            calibration_avg_rt_ms=calibration_medians["rt_ms"],
            calibration_avg_top_speed_px_s=calibration_medians["top_speed_px_s"],
            calibration_median_rt_ms=calibration_medians["rt_ms"],
            calibration_median_top_speed_px_s=calibration_medians["top_speed_px_s"],
        )
    }


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
        if is_single_player_mode(player) or not others:
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
            g.algo_state_json = "{}"
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
                p.trial_timer_ms = _new_trial_timer_ms()
                p.instructed_choice = _new_instruction() if is_single_player_mode(p) else ""
                p.calibration_rt_sum_ms = 0
                p.calibration_top_speed_sum_px_s = 0
                p.calibration_count = 0
                p.calibration_metrics_json = "[]"

        if player.id_in_group == 1:
            g.p1_started = True
        else:
            g.p2_started = True

        phase = "single" if is_single_player_mode(player) else "multi"

        send_pupil_annotation(
            "task_start",
            participant_code=player.participant.code,
            player_id=player.id_in_group,
            phase=phase,
            trial=player.current_trial + 1,
        )

        if is_single_player_mode(player):
            return {player.id_in_group: _ready_payload(player)}

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

    if is_single_player_mode(player):
        return _single_choice(player, data, choice, rt_ms)

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

    @staticmethod
    def get_form_fields(player):
        locked_mode = player.session.config.get("game_mode")
        mode = locked_mode or selected_game_mode(player)

        if mode == "single":
            fields = [
                "single_opponent",
                "follow_reaction_time_pct",
                "follow_speed_pct",
                "anticipate_warmup_trials",
                "num_trials_single",
            ]
        else:
            fields = ["num_trials_multi"]

        if not locked_mode:
            fields.insert(0, "game_mode")
            if "single_opponent" not in fields:
                fields.extend([
                    "single_opponent",
                    "follow_reaction_time_pct",
                    "follow_speed_pct",
                    "anticipate_warmup_trials",
                    "num_trials_single",
                ])
            if "num_trials_multi" not in fields:
                fields.append("num_trials_multi")

        return fields

    @staticmethod
    def is_displayed(player):
        return player.id_in_group == 1

    @staticmethod
    def vars_for_template(player):
        locked_mode = player.session.config.get("game_mode")
        selected_mode = locked_mode or selected_game_mode(player)
        return dict(
            mode_locked=bool(locked_mode),
            selected_game_mode=selected_mode,
            selected_game_mode_label="Single player" if selected_mode == "single" else "Two players",
            show_game_mode=not locked_mode,
            show_single_settings=(locked_mode == "single") or not locked_mode,
            show_multi_settings=(locked_mode == "multi") or not locked_mode,
        )

    @staticmethod
    def before_next_page(player, timeout_happened):
        g = player.group
        g.game_mode = selected_game_mode(player)
        g.single_opponent_final = g.single_opponent


class WaitAfterSetup(WaitPage):
    pass


class Game(Page):
    live_method = live_game

    @staticmethod
    def js_vars(player: Player):
        return dict(
            num_trials_multi=num_trials_multi(player),
            num_trials_single=num_trials_single(player),
            calibration_trials=C.NUM_CALIBRATION_TRIALS,
            game_mode=selected_game_mode(player),
            single_opponent_type=player.group.single_opponent_final or player.group.single_opponent,
            follow_reaction_time_pct=player.group.follow_reaction_time_pct,
            follow_speed_pct=player.group.follow_speed_pct,
            anticipate_warmup_trials=player.group.anticipate_warmup_trials,
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
