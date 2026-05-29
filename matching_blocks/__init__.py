from otree.api import *
import random
import json
import time

class C(BaseConstants):
    NAME_IN_URL = "matching_blocks"
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1
    

    NUM_TRIALS = 20
    CORRIDOR_WIDTH = 2

    LANE_SWITCH_COOLDOWN_MS = 250
    TRIAL_DURATION_MS = 7000
    POST_TRIAL_DELAY_MS_MIN = 1000
    POST_TRIAL_DELAY_MS_MAX = 1500

    CONTROL_CUTOFF_MS_MIN = 1000
    CONTROL_CUTOFF_MS_MAX = 4000

    REWARD_WIN = 10
    REWARD_LOSS = 0

    COLLISION_X_THRESHOLD = 0.12


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    trial_index = models.IntegerField(initial=1)

    p1_final_lane = models.IntegerField(blank=True, null=True)
    p2_final_lane = models.IntegerField(blank=True, null=True)

    p1_ready = models.BooleanField(initial=False)
    p2_ready = models.BooleanField(initial=False)

    p1_start_lane = models.IntegerField(blank=True, null=True)
    p2_start_lane = models.IntegerField(blank=True, null=True)

    collision = models.BooleanField(blank=True, null=True)
    resolved = models.BooleanField(initial=False)

    control_cutoff_ms = models.IntegerField(blank=True, null=True)
    p1_control_cutoff_ms = models.IntegerField(blank=True, null=True)
    p2_control_cutoff_ms = models.IntegerField(blank=True, null=True)

    num_trials = models.IntegerField(initial=20)

    paused = models.BooleanField(initial=False)
    pause_started_ms = models.FloatField(blank=True, null=True)
    total_pause_ms = models.FloatField(initial=0)

    command_log_json = models.LongStringField(initial="[]")
    trial_log_json = models.LongStringField(initial="[]")

    shared_control_lock = models.BooleanField(initial=False)
    setup_done = models.BooleanField(initial=False)


class Player(BasePlayer):
    role_name = models.StringField()
    final_lane = models.IntegerField(blank=True, null=True)
    reward = models.IntegerField(initial=0)
    total_reward = models.IntegerField(initial=0)
    is_collision = models.BooleanField(blank=True, null=True)


def now_ms():
    return time.time() * 1000


def append_json_list(obj, field_name, row):
    rows = json.loads(getattr(obj, field_name) or "[]")
    rows.append(row)
    setattr(obj, field_name, json.dumps(rows))


def log_command(player, data):
    group = player.group
    append_json_list(group, "command_log_json", dict(
        server_ts_ms=now_ms(),
        trial_index=group.trial_index,
        player_id=player.id_in_group,
        role=player.role_name,
        msg=data,
    ))


def creating_session(subsession):
    for group in subsession.get_groups():
        players = group.get_players()
        players[0].role_name = "matcher"
        players[1].role_name = "avoider"

def make_trial_setup(group):
    group.p1_start_lane = random.randrange(C.CORRIDOR_WIDTH)
    group.p2_start_lane = random.randrange(C.CORRIDOR_WIDTH)

    if group.shared_control_lock:
        shared_cutoff = random.randint(
            C.CONTROL_CUTOFF_MS_MIN,
            C.CONTROL_CUTOFF_MS_MAX,
        )
        group.p1_control_cutoff_ms = shared_cutoff
        group.p2_control_cutoff_ms = shared_cutoff
    else:
        group.p1_control_cutoff_ms = random.randint(
            C.CONTROL_CUTOFF_MS_MIN,
            C.CONTROL_CUTOFF_MS_MAX,
        )
        group.p2_control_cutoff_ms = random.randint(
            C.CONTROL_CUTOFF_MS_MIN,
            C.CONTROL_CUTOFF_MS_MAX,
        )

def live_game(player, data):
    group = player.group
    pid = player.id_in_group
    msg_type = data.get("type")
    log_command(player, data)

    if msg_type == "set_game_settings":
        if pid != 1:
            return {}

        try:
            n = int(data.get("num_trials") or group.num_trials)
        except (TypeError, ValueError):
            n = group.num_trials
        n = max(1, min(500, n))
        group.num_trials = n

        group.shared_control_lock = bool(data.get("shared_control_lock", False))
        group.setup_done = True

        return {
            0: dict(
                type="game_settings_set",
                num_trials=group.num_trials,
                shared_control_lock=group.shared_control_lock,
            )
        }

    if msg_type == "ready":
        if not group.setup_done:
            return {
                pid: dict(
                    type="waiting_for_setup",
                )
            }

        if pid == 1:
            group.p1_ready = True
        else:
            group.p2_ready = True

        if group.p1_ready and group.p2_ready:
            make_trial_setup(group)

            return {
                1: dict(
                    type="start_trial",
                    trial_index=group.trial_index,
                    num_trials=group.num_trials,
                    p1_start_lane=group.p1_start_lane,
                    p2_start_lane=group.p2_start_lane,
                    control_cutoff_ms=group.p1_control_cutoff_ms,
                    shared_control_lock=group.shared_control_lock,
                ),
                2: dict(
                    type="start_trial",
                    trial_index=group.trial_index,
                    num_trials=group.num_trials,
                    p1_start_lane=group.p1_start_lane,
                    p2_start_lane=group.p2_start_lane,
                    control_cutoff_ms=group.p2_control_cutoff_ms,
                    shared_control_lock=group.shared_control_lock,
                ),
            }

        return {
            pid: dict(
                type="waiting_for_other_player",
                p1_ready=group.p1_ready,
                p2_ready=group.p2_ready,
                my_id=pid,
            )
        }

    if msg_type == "position":
        lane_raw = data.get("lane")
        if lane_raw is None:
            return {}

        return {
            3 - pid: dict(
                type="opponent_position",
                lane=int(lane_raw),
            )
        }

    if msg_type == "final_choice":
        if group.resolved:
            return {}

        lane_raw = data.get("lane")
        if lane_raw is None:
            return {}

        lane = int(lane_raw)
        player.final_lane = lane

        if pid == 1:
            group.p1_final_lane = lane
        else:
            group.p2_final_lane = lane

        p1_final_lane = group.field_maybe_none("p1_final_lane")
        p2_final_lane = group.field_maybe_none("p2_final_lane")

        if p1_final_lane is None or p2_final_lane is None:
            return {}

        collision = p1_final_lane == p2_final_lane
        group.collision = collision
        group.resolved = True

        rewards = {}
        total_rewards = {}

        for p in group.get_players():
            p.is_collision = collision
            won = collision if p.role_name == "matcher" else not collision

            p.reward = C.REWARD_WIN if won else C.REWARD_LOSS
            p.total_reward += p.reward
            p.payoff = p.total_reward

            rewards[p.id_in_group] = p.reward
            total_rewards[p.id_in_group] = p.total_reward

        is_last_trial = group.trial_index >= group.num_trials

        append_json_list(group, "trial_log_json", dict(
            trial_index=group.trial_index,
            p1_start_lane=group.p1_start_lane,
            p2_start_lane=group.p2_start_lane,
            p1_final_lane=p1_final_lane,
            p2_final_lane=p2_final_lane,
            collision=collision,
            p1_reward=rewards[1],
            p2_reward=rewards[2],
            p1_total_reward=total_rewards[1],
            p2_total_reward=total_rewards[2],
            shared_control_lock=group.shared_control_lock,
            p1_control_cutoff_ms=group.field_maybe_none("p1_control_cutoff_ms"),
            p2_control_cutoff_ms=group.field_maybe_none("p2_control_cutoff_ms"),
            total_pause_ms=group.total_pause_ms,
            server_ts_ms=now_ms(),
        ))

        return {
            0: dict(
                type="result",
                trial_index=group.trial_index,
                num_trials=group.num_trials,
                collision=collision,
                p1_lane=p1_final_lane,
                p2_lane=p2_final_lane,
                rewards=rewards,
                total_rewards=total_rewards,
                is_last_trial=is_last_trial,
            )
        }

    if msg_type == "next_trial":
        if player.id_in_group != 1:
            return {}
        

        if group.trial_index < group.num_trials:
            group.trial_index += 1
            group.p1_final_lane = None
            group.p2_final_lane = None
            group.collision = None
            group.resolved = False
            group.paused = False
            group.pause_started_ms = None
            group.total_pause_ms = 0

            for p in group.get_players():
                p.final_lane = None
                p.is_collision = None
                p.reward = 0

            make_trial_setup(group)

            return {
                1: dict(
                    type="start_trial",
                    trial_index=group.trial_index,
                    num_trials=group.num_trials,
                    p1_start_lane=group.p1_start_lane,
                    p2_start_lane=group.p2_start_lane,
                    control_cutoff_ms=group.p1_control_cutoff_ms,
                    shared_control_lock=group.shared_control_lock,
                ),
                2: dict(
                    type="start_trial",
                    trial_index=group.trial_index,
                    num_trials=group.num_trials,
                    p1_start_lane=group.p1_start_lane,
                    p2_start_lane=group.p2_start_lane,
                    control_cutoff_ms=group.p2_control_cutoff_ms,
                    shared_control_lock=group.shared_control_lock,
                ),
            }

        return {
            0: dict(
                type="game_over",
            )
        }

    if msg_type == "toggle_pause":
        if not group.paused:
            group.paused = True
            group.pause_started_ms = now_ms()

            return {
                0: dict(
                    type="pause_state",
                    paused=True,
                )
            }

        pause_started = group.field_maybe_none("pause_started_ms")
        if pause_started is not None:
            group.total_pause_ms += now_ms() - pause_started

        group.paused = False
        group.pause_started_ms = None

        return {
            0: dict(
                type="pause_state",
                paused=False,
                total_pause_ms=group.total_pause_ms,
            )
        }

    return {}


class WaitForOpponent(WaitPage):
    group_by_arrival_time = True


class Game(Page):
    live_method = "live_game"

    @staticmethod
    def vars_for_template(player):
        return dict(
            role_name=player.role_name,
            num_trials=C.NUM_TRIALS,
            trial_duration_ms=C.TRIAL_DURATION_MS,
            post_trial_delay_ms_min=C.POST_TRIAL_DELAY_MS_MIN,
            post_trial_delay_ms_max=C.POST_TRIAL_DELAY_MS_MAX,
            corridor_width=C.CORRIDOR_WIDTH,
            lane_switch_cooldown_ms=C.LANE_SWITCH_COOLDOWN_MS,
            reward_win=C.REWARD_WIN,
            reward_loss=C.REWARD_LOSS,
            control_cutoff_ms_min=C.CONTROL_CUTOFF_MS_MIN,
            control_cutoff_ms_max=C.CONTROL_CUTOFF_MS_MAX,
        )


class End(Page):
    @staticmethod
    def vars_for_template(player):
        return dict(
            role_name=player.role_name,
            total_payoff=player.payoff,
            total_reward=player.total_reward,
        )


page_sequence = [WaitForOpponent, Game, End]
