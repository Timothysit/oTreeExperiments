from otree.api import *
import random
import time 
import json 
from .algorithms import MatchingPennies2


class C(BaseConstants):
    NAME_IN_URL = 'matching_live'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1  # all 10 trials happen on a single page
    NUM_TRIALS_SINGLE = 3
    NUM_TRIALS_MULTI = 3
    NUM_TRIALS_TOTAL = NUM_TRIALS_SINGLE + NUM_TRIALS_MULTI

    REWARD_WIN = 10
    REWARD_LOSS = 0


class Subsession(BaseSubsession):
    pass


def ensure_single_opponent_assigned(player):
    """
    Randomly assigns algo_A or algo_B ONCE per participant, stores in participant.vars.
    Returns (opponent_type, opponent_id).
    """
    pv = player.participant.vars
    if "single_opponent_type" not in pv:
        pv["single_opponent_type"] = random.choice(["algo_A", "algo_B"])
        pv["single_opponent_id"] = f'{pv["single_opponent_type"]}_v1'
    return pv["single_opponent_type"], pv["single_opponent_id"]


def get_single_algo(player):
    """
    Returns a per-participant algo instance for the single-player block.
    Stored in participant.vars so it persists across live calls.
    """
    pv = player.participant.vars

    # choose parameters (match your MATLAB defaults)
    N = pv.get("algoA_trials_back", 3)      # or whatever you want
    alpha = pv.get("algoA_alpha", 0.05)

    if "single_algo" not in pv:
        # If Algorithm A is meant to *beat* the human, invert_prediction=True.
        pv["single_algo"] = MatchingPennies2(N=N, alpha=alpha, invert_prediction=True)

    return pv["single_algo"]


class Group(BaseGroup):
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


def _phase_and_display_trial(player: Player):
    # This function controls whether the current trial is single-player or multiplayer
    """Returns (phase, display_trial, display_total)."""
    if player.current_trial < C.NUM_TRIALS_SINGLE:
        return "single", player.current_trial + 1, C.NUM_TRIALS_SINGLE
    else:
        # trial 11 overall becomes 1 within multiplayer block
        within = player.current_trial - C.NUM_TRIALS_SINGLE + 1
        return "multi", within, C.NUM_TRIALS_MULTI

def _overall_done(player: Player):
    return player.current_trial >= C.NUM_TRIALS_TOTAL


def live_game(player: Player, data):
    """
    Handles messages from the browser.

    Expected messages:
        {type: "start"}
        {type: "choice", choice: "L" or "R", rt_ms: 123}
    """
    msg_type = data.get('type')

    # First click after intro: initialize the game
    if msg_type == 'start':
        player.current_trial = 0
        player.total_points = 0
        player.last_choice = ''
        player.last_reward = 0

        # also clear any stale group fields
        g = player.group
        g.p1_choice = ""
        g.p2_choice = ""
        g.trial_log_json = "[]"  # reset the json log 

        # Assign whether solo play against computer algorithm A or B
        opp_type, opp_id = ensure_single_opponent_assigned(player)

        # If this participant is assigned algo_A, create the state now
        if opp_type == "algo_A":
            _ = get_single_algo(player)


        phase, disp_trial, disp_total = _phase_and_display_trial(player)

        # Tell the client we're ready for trial 1
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
    if player.current_trial < C.NUM_TRIALS_SINGLE:
        g = player.group
        rt_ms = int(data.get("rt_ms", 0))
        player.last_rt_ms = rt_ms

        # Decide opponent type for this block (algo_A/algo_B)
        opponent_type, opponent_id = ensure_single_opponent_assigned(player)

        # Decide opponent move
        if opponent_type == "algo_A":
            algo = get_single_algo(player)
            opponent_choice = algo.sample()
        else:
            # keep your algo_B placeholder for now (random opponent)
            opponent_choice = random.choice(["L", "R"])

        # Resolve outcome (match your multiplayer logic: player wins if choices match)
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
        )
        g.append_trial(row)

        # Now advance
        player.current_trial += 1

        is_last = _overall_done(player)

        return {
            player.id_in_group: dict(
                type="feedback",
                phase="single",
                trial=min(player.current_trial, C.NUM_TRIALS_SINGLE),
                trial_total=C.NUM_TRIALS_SINGLE,
                reward=reward,
                total_points=player.total_points,
                is_last=is_last,
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
    current_multi_trial = (p1.current_trial - C.NUM_TRIALS_SINGLE) + 1

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
    current_multi_trial = (p1.current_trial - 1) - C.NUM_TRIALS_SINGLE + 1

    return {
        1: dict(
            type="feedback",
            phase="multi",
            trial=current_multi_trial,
            trial_total=C.NUM_TRIALS_MULTI,
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
            trial_total=C.NUM_TRIALS_MULTI,
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


class Game(Page):
    # This tells oTree to use the live_game function for WebSocket messages
    live_method = live_game


page_sequence = [Game]
