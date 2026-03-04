from otree.api import *
import random
import time 
import json 
from .algorithms import MatchingPennies2, BlockFlipperWithExtension



class C(BaseConstants):
    NAME_IN_URL = 'matching_live'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1  # all 10 trials happen on a single page
    NUM_TRIALS_SINGLE = 400  # fallback default
    NUM_TRIALS_MULTI = 400  # fallback default
    # If True: ensure one player gets A and the other gets B (randomly swapped)
    # If False: each player independently random (AA, AB, BA, BB all possible)
    ENFORCE_ONE_A_ONE_B_PER_GROUP = False
    REWARD_WIN = 10
    REWARD_LOSS = 0

    SINGLE_FEEDBACK_DELAY_MS_MIN = 50
    SINGLE_FEEDBACK_DELAY_MS_MAX = 250


class Subsession(BaseSubsession):

    def creating_session(self):
        # default mapping so player 2 never races ahead of Setup
        self.session.vars.setdefault("single_opponent_by_role", {1: "random", 2: "random"})


def num_trials_single(player):
    # Prefer per-group setting from Setup page
    g = player.group
    if getattr(g, "num_trials_single", None):
        return int(g.num_trials_single)
    # fallback to session config / constants
    return int(player.session.config.get("num_trials_single", C.NUM_TRIALS_SINGLE))

def num_trials_multi(player):
    g = player.group
    if getattr(g, "num_trials_multi", None):
        return int(g.num_trials_multi)
    return int(player.session.config.get("num_trials_multi", C.NUM_TRIALS_MULTI))

def num_trials_total(player):
    return num_trials_single(player) + num_trials_multi(player)


def get_single_opponent_for_player(player):
    g = player.group
    if player.id_in_group == 1:
        opp = g.single_opponent_p1_final or "algo_A"
    else:
        opp = g.single_opponent_p2_final or "algo_A"
    return opp, f"{opp}_v1"


def get_single_algo(player):
    """
    Returns a per-participant algo instance for the single-player block.
    Stored in participant.vars so it persists across live calls.
    """
    pv = player.participant.vars


    if "single_algo" not in pv:
        # choose parameters (match your MATLAB defaults)
        N = pv.get("algoA_trials_back", 3)      # or whatever you want
        alpha = pv.get("algoA_alpha", 0.05)
        # If Algorithm A is meant to *beat* the human, invert_prediction=False.
        pv["single_algo"] = MatchingPennies2(N=N, alpha=alpha, invert_prediction=False)

    return pv["single_algo"]



def get_bandit_env(player):
    pv = player.participant.vars
    if "bandit_env" not in pv:
        pv["bandit_env"] = BlockFlipperWithExtension(
            p_high=0.6,
            p_low=0.0,
            lambda_=25.0,
            extend_block=5,
            block_extend_threshold=0.2,
        )
    return pv["bandit_env"]


class Group(BaseGroup):

    started = models.BooleanField(initial=False)

    algo_split_flip = models.IntegerField(initial=-1)  # -1 unset, else 0/1


    single_opponent_p1 = models.StringField(
        choices=[["random", "Random"], ["algo_A", "Algorithm A"], ["algo_B", "Algorithm B"]],
        initial="random",
        blank=False,
        widget=widgets.RadioSelect,
    )
    single_opponent_p2 = models.StringField(
        choices=[["random", "Random"], ["algo_A", "Algorithm A"], ["algo_B", "Algorithm B"]],
        initial="random",
        blank=False,
        widget=widgets.RadioSelect,
    )

    # Trial counts set in Setup (per group)
    num_trials_single = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_SINGLE)
    num_trials_multi = models.IntegerField(min=1, max=500, initial=C.NUM_TRIALS_MULTI)
    

    # store resolved fixed choices actually used
    single_opponent_p1_final = models.StringField(blank=True)
    single_opponent_p2_final = models.StringField(blank=True)
    

    # temporary storage for multi-player phase choices
    p1_choice = models.StringField(blank=True)
    p2_choice = models.StringField(blank=True)

    # temp storage for RTs in multiplayer
    p1_rt_ms = models.IntegerField(initial=0)
    p2_rt_ms = models.IntegerField(initial=0)

    # One row per completed trial (single + multi)
    trial_log_json = models.LongStringField(initial='[]')

    # Algorithm state
    algo_state_json = models.LongStringField(initial='{}')

    def append_trial(self, row: dict):
        log = json.loads(self.trial_log_json or '[]')
        log.append(row)
        self.trial_log_json = json.dumps(log)
    



class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)
    last_rt_ms = models.IntegerField(initial=0)
    part1_points = models.IntegerField(initial=0)


def _phase_and_display_trial(player: Player):

    n_single = num_trials_single(player)
    n_multi = num_trials_multi(player)



    # This function controls whether the current trial is single-player or multiplayer
    """Returns (phase, display_trial, display_total)."""
    if player.current_trial < n_single:
        return "single", player.current_trial + 1, n_single
    else:
        # trial 11 overall becomes 1 within multiplayer block
        within = player.current_trial - n_single + 1
        return "multi", within, n_multi

def _overall_done(player: Player):
    return player.current_trial >= num_trials_total(player)


def live_game(player: Player, data):
    """
    Handles messages from the browser.

    Expected messages:
        {type: "start"}
        {type: "choice", choice: "L" or "R", rt_ms: 123}
    """
    print("LIVE_GAME", player.participant.code, player.id_in_group, data, flush=True)

    msg_type = data.get('type')

    # First click after intro: initialize the game
    if msg_type == 'start':
        g = player.group

        if not g.started:
            g.started = True
            g.trial_log_json = "[]"
            g.algo_state_json = "{}"
            g.p1_choice = g.p2_choice = ""
            g.p1_rt_ms = g.p2_rt_ms = 0

            # Reset BOTH players once
            for p in g.get_players():
                p.current_trial = 0
                p.total_points = 0
                p.last_choice = ""
                p.last_reward = 0
                p.last_rt_ms = 0

        # assign algo for THIS player (role-based or randomized)
        opp_type, opp_id = get_single_opponent_for_player(player)

        if opp_type == "algo_A":
            _ = get_single_algo(player)

        phase, disp_trial, disp_total = _phase_and_display_trial(player)

        return {
            player.id_in_group: dict(
                type="ready",
                phase=phase,
                trial=disp_trial,
                trial_total=disp_total,
                total_points=player.total_points,
            )
        }
    
    if msg_type != "choice":
        return

    choice = data.get("choice")
    if choice not in ["L", "R"]:
        return

    player.last_choice = choice

    # ---------- Phase 1: single-player ----------
    if player.current_trial < num_trials_single(player):
        g = player.group
        rt_ms = int(data.get("rt_ms", 0))
        player.last_rt_ms = rt_ms

        # Decide opponent type for this block (algo_A/algo_B)
        opponent_type, opponent_id = get_single_opponent_for_player(player)

        # Decide opponent move
        if opponent_type == "algo_A":
            # Algo A is matching pennies
            algo = get_single_algo(player)
            opponent_choice = algo.sample()
        elif opponent_type == "algo_B":
            # Algorithm B is two-armed bandit
            env = get_bandit_env(player)

            reward_bin = env.trial(choice)  # 0/1
            reward = C.REWARD_WIN if reward_bin == 1 else C.REWARD_LOSS

            # For logging/debug (optional):
            bandit_state = env.to_dict()
            high_side = env.high_side
            p_choice = env.reward_prob(choice)

            opponent_choice = None
        else:
            # keep your algo_B placeholder for now (random opponent)
            opponent_choice = random.choice(["L", "R"])

        # Resolve outcome (match your multiplayer logic: player wins if choices match)
        # Compute reward (depends on opponent_type)
        if opponent_type == "algo_B":  # two-armed bandit
            # reward already computed by bandit env above
            is_win = (reward > 0)  # just for convenience/logging
        else:
            # matching pennies style (algo_A or random opponent)
            is_win = (choice == opponent_choice)
            reward = C.REWARD_WIN if is_win else C.REWARD_LOSS
        

        player.last_reward = reward
        player.total_points += reward

        # Update algo with HUMAN (opponent-from-algo-perspective) history:
        # last_reward must be 0/1, not 0/10.
        if opponent_type == "algo_A":
            human_reward_bin = 1 if is_win else 0
            algo.update(last_choice=choice, last_reward=human_reward_bin)
        
        if opponent_type == "algo_A":
            g.algo_state_json = json.dumps(algo.to_dict())

        # Random delay
        delay_ms = random.randint(
            C.SINGLE_FEEDBACK_DELAY_MS_MIN,
            C.SINGLE_FEEDBACK_DELAY_MS_MAX
        )

        # Log BEFORE increment (trial index is stable)
        row = dict(
            overall_trial=player.current_trial + 1,   # 1-based overall
            block="single",
            block_trial=player.current_trial + 1,     # 1..NUM_TRIALS_SINGLE
            opponent_type=opponent_type,              # "algo_A" / "algo_B"
            opponent_id=opponent_id,
            player_code=player.participant.code,
            player_choice=choice,
            player_rt_ms=rt_ms,
            opponent_choice=opponent_choice,          # fill when you implement the algos
            reward=reward,
            total_points_after=player.total_points,
            server_ts=time.time(),
            feedback_delay_ms=delay_ms,
        )

        if opponent_type == "algo_B":
            row.update(dict(
                reward_bin=reward_bin,              # 0/1 (from env.trial)
                reward_prob=p_choice,               # prob used on this trial
                bandit_high_side=high_side,         # which side is high *after* env.trial() (see note below)
                bandit_block_flip=bandit_state["block_flip"],
                bandit_block_length=bandit_state["block_length"],
            ))
            g.algo_state_json = json.dumps(bandit_state)

        g.append_trial(row)

        # Now advance
        player.current_trial += 1

        # Get the total number of points for solo part (round 1)
        if player.current_trial == num_trials_single(player):
            player.part1_points = player.total_points

        is_last = _overall_done(player)
        

        return {
            player.id_in_group: dict(
                type="feedback",
                phase="single",
                trial=min(player.current_trial, num_trials_single(player)),
                trial_total=num_trials_single(player),
                reward=reward,
                total_points=player.total_points,
                is_last=is_last,
                delay_ms=delay_ms,
            )
        }

    # ---------- Phase 2: two-player matching pennies ----------
    g = player.group
    rt_ms = int(data.get("rt_ms", 0))
    player.last_rt_ms = rt_ms

    if player.id_in_group == 1:
        g.p1_choice = choice
        g.p1_rt_ms = rt_ms
    else:
        g.p2_choice = choice
        g.p2_rt_ms = rt_ms

    other = player.get_others_in_group()[0]

    # if other hasn't chosen yet, tell this player to wait
    if not g.p1_choice or not g.p2_choice:
        phase, disp_trial, disp_total = _phase_and_display_trial(player)
        return {
            player.id_in_group: dict(
                type="wait_opponent",
                phase="multi",
                trial=disp_trial,
                trial_total=disp_total,
                total_points=player.total_points,
            )
        }
    
    # both choices are in -> resolve the round for BOTH players
    p1 = g.get_player_by_id(1)
    p2 = g.get_player_by_id(2)
    c1, c2 = g.p1_choice, g.p2_choice
    rt1, rt2 = g.p1_rt_ms, g.p2_rt_ms

    if c1 == c2:
        r1, r2 = C.REWARD_WIN, C.REWARD_LOSS
        winner = 1
    else:
        r1, r2 = C.REWARD_LOSS, C.REWARD_WIN
        winner = 2

    p1.last_reward, p2.last_reward = r1, r2
    p1.total_points += r1
    p2.total_points += r2

    # current multiplayer trial number (before increment)
    current_multi_trial = (p1.current_trial - num_trials_single(player)) + 1

    # Log a single combined row for this multiplayer trial
    row = dict(
        overall_trial=p1.current_trial + 1,  # both players share the same overall trial index
        block="multi",
        block_trial=current_multi_trial,     # 1..NUM_TRIALS_MULTI
        opponent_type="human",
        p1_code=p1.participant.code,
        p2_code=p2.participant.code,
        p1_choice=c1,
        p2_choice=c2,
        p1_rt_ms=rt1,
        p2_rt_ms=rt2,
        p1_reward=r1,
        p2_reward=r2,
        winner=winner,
        p1_total_points_after=p1.total_points,
        p2_total_points_after=p2.total_points,
        server_ts=time.time(),
    )
    g.append_trial(row)

    # advance BOTH players
    p1.current_trial += 1
    p2.current_trial += 1

    is_last_1 = _overall_done(p1)
    is_last_2 = _overall_done(p2)
    is_last = is_last_1 or is_last_2   # should be the same for both if they stay in sync

    # clear group stored choices/RTs
    g.p1_choice = ""
    g.p2_choice = ""
    g.p1_rt_ms = 0
    g.p2_rt_ms = 0

    # compute next display trial values (for after feedback)
    # (but we send the *current* multiplayer trial number in the message)
    # current multiplayer trial number is: (current_trial_before_increment - 10) + 1
    current_multi_trial = (p1.current_trial - 1) - num_trials_single(player) + 1

    return {
        1: dict(
            type="feedback",
            phase="multi",
            trial=current_multi_trial,
            trial_total=num_trials_multi(player),
            your_choice=c1,
            other_choice=c2,
            reward=r1,
            total_points=p1.total_points,
            winner=winner,
            is_last=is_last,
        ),
        2: dict(
            type="feedback",
            phase="multi",
            trial=current_multi_trial,
            trial_total=num_trials_multi(player),
            your_choice=c2,
            other_choice=c1,
            reward=r2,
            total_points=p2.total_points,
            winner=winner,
            is_last=is_last,
        ),
    }

class WaitForPair(WaitPage):
    group_by_arrival_time = True


class Setup(Page):
    form_model = "group"
    form_fields = ["single_opponent_p1", "single_opponent_p2", "num_trials_single", "num_trials_multi"]

    @staticmethod
    def is_displayed(player):
        # only show to player 1 (once per group)
        return player.id_in_group == 1

    @staticmethod
    def before_next_page(player, timeout_happened):
        g = player.group

        def resolve(mode):
            return random.choice(["algo_A", "algo_B"]) if mode == "random" else mode

        g.single_opponent_p1_final = resolve(g.single_opponent_p1)
        g.single_opponent_p2_final = resolve(g.single_opponent_p2)

class WaitAfterSetup(WaitPage):
    @staticmethod
    def is_displayed(player):
        return True

class Game(Page):
    # This tells oTree to use the live_game function for WebSocket messages
    live_method = live_game

    @staticmethod
    def js_vars(player: Player):
        return dict(
            num_trials_single=num_trials_single(player),
            num_trials_multi=num_trials_multi(player),
        )

class End(Page):
    def is_displayed(player: Player):
        return _overall_done(player)

    def vars_for_template(player: Player):
        part1 = player.part1_points
        part2 = player.total_points - part1
        return dict(
            part1_points=part1,
            part2_points=part2,
            total_points=player.total_points,
            participant_code=player.participant.code,  # <-- show this on End page
            survey_url="https://forms.gle/KR7CJY4MENZ5dtX98",  # optional convenience
        )

page_sequence = [Setup, WaitAfterSetup, Game, End]
