from otree.api import *
import json
import random
import time

from matching_live.algorithms import MatchingPennies2, BlockFlipperWithExtension


class C(BaseConstants):
    NAME_IN_URL = "matching_payoff_risk"
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    NUM_TRIALS_SINGLE = 100
    NUM_TRIALS_MULTI = 100

    SAFE_SIDE = "L"
    SAFE_PROB_PCT = 100
    SAFE_PAYOFF = 10
    RISKY_PROB1_PCT = 50
    RISKY_PAYOFF1 = 20
    RISKY_PROB2_PCT = 0
    RISKY_PAYOFF2 = 0


def get_single_algo(player):
    pv = player.participant.vars
    if "matching_payoff_risk_single_algo" not in pv:
        pv["matching_payoff_risk_single_algo"] = MatchingPennies2(N=3, alpha=0.05, invert_prediction=False)
    return pv["matching_payoff_risk_single_algo"]


def get_bandit_env(player):
    pv = player.participant.vars
    if "matching_payoff_risk_bandit_env" not in pv:
        pv["matching_payoff_risk_bandit_env"] = BlockFlipperWithExtension(
            p_high=0.6,
            p_low=0.0,
            lambda_=25.0,
            extend_block=5,
            block_extend_threshold=0.2,
        )
    return pv["matching_payoff_risk_bandit_env"]


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


def num_trials_single(player):
    g = player.group
    if getattr(g, "num_trials_single", None):
        return int(g.num_trials_single)
    return int(player.session.config.get("num_trials_single", C.NUM_TRIALS_SINGLE))


def num_trials_multi(player):
    g = player.group
    if getattr(g, "num_trials_multi", None):
        return int(g.num_trials_multi)
    return int(player.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))


def num_trials_total(player):
    return num_trials_single(player) if is_single_player_mode(player) else num_trials_multi(player)


def _overall_done(player):
    return player.current_trial >= num_trials_total(player)


def _zero_payoff(side):
    return dict(
        payoff_schedule="none",
        payoff_side=side,
        payoff_option=0,
        payoff_prob_pct=0,
        payoff_amount=0,
        payoff_lottery_won=False,
        reward=0,
    )


def _draw_safe_reward(group, side):
    won_lottery = random.random() < (group.safe_prob_pct / 100)
    return dict(
        payoff_schedule="safe",
        payoff_side=side,
        payoff_option=1,
        payoff_prob_pct=group.safe_prob_pct,
        payoff_amount=group.safe_payoff,
        payoff_lottery_won=won_lottery,
        reward=group.safe_payoff if won_lottery else 0,
    )


def _draw_risky_reward(group, side):
    prob1 = max(0, int(group.risky_prob1_pct))
    prob2 = max(0, int(group.risky_prob2_pct))
    total_prob = prob1 + prob2
    draw = random.random() * 100

    if draw < prob1:
        return dict(
            payoff_schedule="risky",
            payoff_side=side,
            payoff_option=1,
            payoff_prob_pct=prob1,
            payoff_amount=group.risky_payoff1,
            payoff_lottery_won=True,
            reward=group.risky_payoff1,
        )

    if draw < prob1 + prob2:
        return dict(
            payoff_schedule="risky",
            payoff_side=side,
            payoff_option=2,
            payoff_prob_pct=prob2,
            payoff_amount=group.risky_payoff2,
            payoff_lottery_won=True,
            reward=group.risky_payoff2,
        )

    return dict(
        payoff_schedule="risky",
        payoff_side=side,
        payoff_option=0,
        payoff_prob_pct=max(0, 100 - total_prob),
        payoff_amount=0,
        payoff_lottery_won=False,
        reward=0,
    )


def _side_params(group, side):
    safe_side = group.safe_side or C.SAFE_SIDE
    if side == safe_side:
        return "safe", group.safe_prob_pct, group.safe_payoff
    return "risky", group.risky_prob1_pct, group.risky_payoff1


def _draw_side_reward(group, side):
    if side == (group.safe_side or C.SAFE_SIDE):
        return _draw_safe_reward(group, side)
    return _draw_risky_reward(group, side)


class Subsession(BaseSubsession):
    def creating_session(self):
        default_game_mode = "single" if self.session.num_participants == 1 else "multi"
        game_mode = self.session.config.get("game_mode", default_game_mode)

        for g in self.get_groups():
            g.game_mode = game_mode
            g.num_trials_single = int(self.session.config.get("num_trials_single", C.NUM_TRIALS_SINGLE))
            g.num_trials_multi = int(self.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))
            g.safe_side = self.session.config.get("safe_side", C.SAFE_SIDE)
            g.safe_prob_pct = int(self.session.config.get("safe_prob_pct", C.SAFE_PROB_PCT))
            g.safe_payoff = int(self.session.config.get("safe_payoff", C.SAFE_PAYOFF))
            g.risky_prob1_pct = int(self.session.config.get("risky_prob1_pct", C.RISKY_PROB1_PCT))
            g.risky_payoff1 = int(self.session.config.get("risky_payoff1", C.RISKY_PAYOFF1))
            g.risky_prob2_pct = int(self.session.config.get("risky_prob2_pct", C.RISKY_PROB2_PCT))
            g.risky_payoff2 = int(self.session.config.get("risky_payoff2", C.RISKY_PAYOFF2))
            g.single_opponent = self.session.config.get("single_opponent", "random")


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

    num_trials_single = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_SINGLE)
    num_trials_multi = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_MULTI)

    single_opponent = models.StringField(
        choices=[
            ["random", "Random"],
            ["predictive", "Predictive opponent"],
            ["2ab", "2AB opponent"],
        ],
        initial="random",
        blank=False,
        widget=widgets.RadioSelect,
    )

    safe_side = models.StringField(
        choices=[["L", "Left"], ["R", "Right"]],
        initial=C.SAFE_SIDE,
        blank=False,
        widget=widgets.RadioSelect,
    )
    safe_prob_pct = models.IntegerField(min=0, max=100, initial=C.SAFE_PROB_PCT)
    safe_payoff = models.IntegerField(min=0, max=1000, initial=C.SAFE_PAYOFF)
    risky_prob1_pct = models.IntegerField(min=0, max=100, initial=C.RISKY_PROB1_PCT)
    risky_payoff1 = models.IntegerField(min=0, max=1000, initial=C.RISKY_PAYOFF1)
    risky_prob2_pct = models.IntegerField(min=0, max=100, initial=C.RISKY_PROB2_PCT)
    risky_payoff2 = models.IntegerField(min=0, max=1000, initial=C.RISKY_PAYOFF2)

    p1_choice = models.StringField(blank=True)
    p2_choice = models.StringField(blank=True)
    p1_rt_ms = models.IntegerField(initial=0)
    p2_rt_ms = models.IntegerField(initial=0)

    trial_log_json = models.LongStringField(initial="[]")

    def append_trial(self, row: dict):
        log = json.loads(self.trial_log_json or "[]")
        log.append(row)
        self.trial_log_json = json.dumps(log)


class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)
    last_rt_ms = models.IntegerField(initial=0)


def _ready_payload(player):
    phase = selected_game_mode(player)
    trial_total = num_trials_total(player)
    return dict(
        type="ready",
        phase=phase,
        trial=player.current_trial + 1,
        trial_total=trial_total,
        total_points=player.total_points,
        safe_side=player.group.safe_side,
        safe_prob_pct=player.group.safe_prob_pct,
        safe_payoff=player.group.safe_payoff,
        risky_prob1_pct=player.group.risky_prob1_pct,
        risky_payoff1=player.group.risky_payoff1,
        risky_prob2_pct=player.group.risky_prob2_pct,
        risky_payoff2=player.group.risky_payoff2,
        single_opponent=player.group.single_opponent,
    )


def _single_choice(player, choice, rt_ms):
    g = player.group
    opponent_type = g.single_opponent or "random"
    opponent_choice = None
    algo_state = None
    bandit_state = None
    bandit_reward_bin = None
    bandit_reward_prob = None
    bandit_high_side = None

    if opponent_type == "predictive":
        algo = get_single_algo(player)
        opponent_choice = algo.sample()
        is_win = choice == opponent_choice
    elif opponent_type == "2ab":
        env = get_bandit_env(player)
        bandit_reward_bin = env.trial(choice)
        bandit_state = env.to_dict()
        bandit_reward_prob = env.reward_prob(choice)
        bandit_high_side = env.high_side
        is_win = bandit_reward_bin == 1
    else:
        opponent_type = "random"
        opponent_choice = random.choice(["L", "R"])
        is_win = choice == opponent_choice

    payoff = _draw_side_reward(g, choice) if is_win else _zero_payoff(choice)
    reward = payoff["reward"]

    if opponent_type == "predictive":
        algo.update(last_choice=choice, last_reward=1 if is_win else 0)
        algo_state = algo.to_dict()

    player.last_choice = choice
    player.last_rt_ms = rt_ms
    player.last_reward = reward
    player.total_points += reward

    current_trial = player.current_trial + 1
    g.append_trial(dict(
        overall_trial=current_trial,
        block="single",
        block_trial=current_trial,
        player_code=player.participant.code,
        player_choice=choice,
        opponent_type=opponent_type,
        opponent_choice=opponent_choice,
        player_role="matcher",
        is_win=is_win,
        player_rt_ms=rt_ms,
        total_points_after=player.total_points,
        server_ts=time.time(),
        algo_state=algo_state,
        bandit_reward_bin=bandit_reward_bin,
        bandit_reward_prob=bandit_reward_prob,
        bandit_high_side=bandit_high_side,
        bandit_state=bandit_state,
        **payoff,
    ))

    player.current_trial += 1

    return {
        player.id_in_group: dict(
            type="feedback",
            phase="single",
            trial=current_trial,
            trial_total=num_trials_single(player),
            your_choice=choice,
            other_choice=opponent_choice or "NONE",
            opponent_type=opponent_type,
            winner=player.id_in_group if is_win else 0,
            is_win=is_win,
            reward=reward,
            total_points=player.total_points,
            is_last=_overall_done(player),
            next_trial=0 if _overall_done(player) else player.current_trial + 1,
        )
    }


def live_game(player: Player, data):
    print("LIVE_GAME", player.participant.code, player.id_in_group, data, flush=True)

    msg_type = data.get("type")
    g = player.group

    if msg_type == "start":
        if not g.started:
            g.started = True
            g.trial_log_json = "[]"
            g.p1_choice = ""
            g.p2_choice = ""
            g.p1_rt_ms = 0
            g.p2_rt_ms = 0

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

        if selected_game_mode(player) == "single":
            return {player.id_in_group: _ready_payload(player)}

        if not (g.p1_started and g.p2_started):
            return {
                player.id_in_group: dict(
                    type="waiting_start",
                    phase="multi",
                    message="Waiting for the other player to start...",
                    trial=player.current_trial + 1,
                    trial_total=num_trials_multi(player),
                    total_points=player.total_points,
                )
            }

        return {
            1: _ready_payload(g.get_player_by_id(1)),
            2: _ready_payload(g.get_player_by_id(2)),
        }

    if msg_type != "choice":
        return

    if _overall_done(player):
        return

    choice = data.get("choice")
    if choice not in ["L", "R"]:
        return

    rt_ms = int(data.get("rt_ms", 0) or 0)

    if selected_game_mode(player) == "single":
        return _single_choice(player, choice, rt_ms)

    player.last_choice = choice
    player.last_rt_ms = rt_ms

    if player.id_in_group == 1:
        g.p1_choice = choice
        g.p1_rt_ms = rt_ms
    else:
        g.p2_choice = choice
        g.p2_rt_ms = rt_ms

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

    p1 = g.get_player_by_id(1)
    p2 = g.get_player_by_id(2)
    c1, c2 = g.p1_choice, g.p2_choice
    rt1, rt2 = g.p1_rt_ms, g.p2_rt_ms
    current_trial = p1.current_trial + 1

    if c1 == c2:
        winner = 1
        p1_payoff = _draw_side_reward(g, c1)
        p2_payoff = _zero_payoff(c2)
    else:
        winner = 2
        p1_payoff = _zero_payoff(c1)
        p2_payoff = _draw_side_reward(g, c2)

    r1, r2 = p1_payoff["reward"], p2_payoff["reward"]
    p1.last_reward = r1
    p2.last_reward = r2
    p1.total_points += r1
    p2.total_points += r2

    g.append_trial(dict(
        overall_trial=current_trial,
        block="multi",
        block_trial=current_trial,
        opponent_type="human",
        p1_role="matcher",
        p2_role="unmatcher",
        p1_code=p1.participant.code,
        p2_code=p2.participant.code,
        p1_choice=c1,
        p2_choice=c2,
        p1_rt_ms=rt1,
        p2_rt_ms=rt2,
        winner=winner,
        p1_reward=r1,
        p2_reward=r2,
        p1_payoff_schedule=p1_payoff["payoff_schedule"],
        p1_payoff_side=p1_payoff["payoff_side"],
        p1_payoff_option=p1_payoff["payoff_option"],
        p1_payoff_prob_pct=p1_payoff["payoff_prob_pct"],
        p1_payoff_amount=p1_payoff["payoff_amount"],
        p1_payoff_lottery_won=p1_payoff["payoff_lottery_won"],
        p2_payoff_schedule=p2_payoff["payoff_schedule"],
        p2_payoff_side=p2_payoff["payoff_side"],
        p2_payoff_option=p2_payoff["payoff_option"],
        p2_payoff_prob_pct=p2_payoff["payoff_prob_pct"],
        p2_payoff_amount=p2_payoff["payoff_amount"],
        p2_payoff_lottery_won=p2_payoff["payoff_lottery_won"],
        p1_total_points_after=p1.total_points,
        p2_total_points_after=p2.total_points,
        server_ts=time.time(),
    ))

    p1.current_trial += 1
    p2.current_trial += 1
    is_last = _overall_done(p1) or _overall_done(p2)

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
            winner=winner,
            is_win=winner == 1,
            reward=r1,
            total_points=p1.total_points,
            is_last=is_last,
            next_trial=0 if is_last else p1.current_trial + 1,
        ),
        2: dict(
            type="feedback",
            phase="multi",
            trial=current_trial,
            trial_total=num_trials_multi(p2),
            your_choice=c2,
            other_choice=c1,
            winner=winner,
            is_win=winner == 2,
            reward=r2,
            total_points=p2.total_points,
            is_last=is_last,
            next_trial=0 if is_last else p2.current_trial + 1,
        ),
    }


class Setup(Page):
    form_model = "group"

    @staticmethod
    def get_form_fields(player):
        locked_mode = player.session.config.get("game_mode")
        mode = locked_mode or selected_game_mode(player)

        fields = [
            "safe_side",
            "safe_prob_pct",
            "safe_payoff",
            "risky_prob1_pct",
            "risky_payoff1",
            "risky_prob2_pct",
            "risky_payoff2",
        ]

        if mode == "single":
            fields.append("single_opponent")
            fields.append("num_trials_single")
        else:
            fields.append("num_trials_multi")

        if not locked_mode:
            fields.insert(0, "game_mode")
            if "num_trials_single" not in fields:
                fields.append("num_trials_single")
            if "num_trials_multi" not in fields:
                fields.append("num_trials_multi")

        return fields

    @staticmethod
    def is_displayed(player):
        return player.id_in_group == 1

    @staticmethod
    def error_message(player, values):
        prob1 = int(values.get("risky_prob1_pct") or 0)
        prob2 = int(values.get("risky_prob2_pct") or 0)
        if prob1 + prob2 > 100:
            return "Other side reward probability 1 plus probability 2 cannot exceed 100%."

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


class WaitAfterSetup(WaitPage):
    pass


class Game(Page):
    live_method = live_game

    @staticmethod
    def js_vars(player: Player):
        return dict(game_mode=selected_game_mode(player))


class End(Page):
    @staticmethod
    def is_displayed(player: Player):
        return _overall_done(player)

    @staticmethod
    def vars_for_template(player: Player):
        return dict(
            total_points=player.total_points,
            participant_code=player.participant.code,
        )


page_sequence = [Setup, WaitAfterSetup, Game, End]
