from otree.api import *
import random
import time 
import json 
from .algorithms import MatchingPennies2, BlockFlipperWithExtension
from pupil_bridge.context import pupil_bridge_enabled, pupil_js_vars


class C(BaseConstants):
    NAME_IN_URL = 'matching_live'
    # None + explicit set_group_matrix() in creating_session, so the same app can
    # run paired (2/group) or solo (1/group) depending on the session config.
    PLAYERS_PER_GROUP = None
    PAIRED_GROUP_SIZE = 2
    NUM_ROUNDS = 1  # all 10 trials happen on a single page
    NUM_TRIALS_SINGLE = 400  # fallback default
    NUM_TRIALS_MULTI = 400  # fallback default
    # If True: ensure one player gets A and the other gets B (randomly swapped)
    # If False: each player independently random (AA, AB, BA, BB all possible)
    ENFORCE_ONE_A_ONE_B_PER_GROUP = False
    REWARD_WIN = 10
    REWARD_LOSS = 0

    # Original delay range (pre 2026-04-10 was 50 - 250 ms)
    SINGLE_FEEDBACK_DELAY_MS_MIN = 50
    SINGLE_FEEDBACK_DELAY_MS_MAX = 1000


def is_solo_session(session) -> bool:
    return bool(session.config.get("solo", False))


# Part 1 opponents can be set per laptop when the session is created (lab-notes passes
# opponent_laptopA / opponent_laptopB, keyed by the mp_lab room's participant labels).
# Then the Setup page is skipped. Players are numbered in the order their laptops join,
# so the opponents are matched to players by label once both have arrived.
OPPONENT_MODES = ("random", "algo_A", "algo_B")  # algo_A: matching pennies; algo_B: two-armed bandit


def opponents_preset(session) -> bool:
    return any(k.startswith("opponent_") for k in session.config)


def resolve_opponent(session, mode):
    # Solo (no-switch control) is defined as algo A across all 800 trials,
    # so Part 1 must be algo A for the Part 2 memory to be continuous.
    if is_solo_session(session):
        return "algo_A"
    return random.choice(["algo_A", "algo_B"]) if mode == "random" else mode


def preset_opponent_mode(player):
    mode = player.session.config.get(f"opponent_{player.participant.label}", "random")
    return mode if mode in OPPONENT_MODES else "random"


class Subsession(BaseSubsession):
    pass


# NOTE: this app uses oTree's "no-self" style, so oTree resolves creating_session
# on the MODULE, not on the Subsession class. Defining it as a Subsession method
# silently does nothing.
def creating_session(subsession: Subsession):
    # default mapping so player 2 never races ahead of Setup
    subsession.session.vars.setdefault("single_opponent_by_role", {1: "random", 2: "random"})

    # PLAYERS_PER_GROUP is None, so build the grouping explicitly here.
    players = subsession.get_players()
    size = 1 if is_solo_session(subsession.session) else C.PAIRED_GROUP_SIZE
    if len(players) % size != 0:
        raise ValueError(
            f"matching_live: {len(players)} participants is not divisible by "
            f"group size {size} (solo={is_solo_session(subsession.session)})"
        )
    subsession.set_group_matrix(
        [players[i:i + size] for i in range(0, len(players), size)]
    )
    # trial counts from the session config (lab-notes sets them per session); the Setup
    # page, when shown, starts from these and can still change them
    cfg = subsession.session.config
    for g in subsession.get_groups():
        g.num_trials_single = int(cfg.get("num_trials_single", C.NUM_TRIALS_SINGLE))
        g.num_trials_multi = int(cfg.get("num_trials_multi", C.NUM_TRIALS_MULTI))


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
    cfg = player.session.config

    if "single_algo" not in pv:
        # Defaults match the live Bpod config the mice faced
        # (MatchingPennies_protocol.m: trials_back = 4, alpha = 0.05).
        # Session config can override; participant.vars overrides that.
        N = pv.get("algoA_trials_back", cfg.get("algoA_trials_back", 4))
        alpha = pv.get("algoA_alpha", cfg.get("algoA_alpha", 0.05))
        # If Algorithm A is meant to *beat* the human, invert_prediction=False.
        pv["single_algo"] = MatchingPennies2(N=N, alpha=alpha, invert_prediction=False)

    return pv["single_algo"]



def get_bandit_env(player):
    pv = player.participant.vars
    if "bandit_env" not in pv:
        pv["bandit_env"] = BlockFlipperWithExtension(
            p_high=0.7,
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
        choices=[["random", "Random (A or B)"], ["algo_A", "Algorithm A (matching pennies)"], ["algo_B", "Algorithm B (two-armed bandit)"]],
        initial="random",
        blank=False,
        widget=widgets.RadioSelect,
    )
    single_opponent_p2 = models.StringField(
        choices=[["random", "Random (A or B)"], ["algo_A", "Algorithm A (matching pennies)"], ["algo_B", "Algorithm B (two-armed bandit)"]],
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

    def update_last_matching_trial(self, block, block_trial, updates: dict):
        log = json.loads(self.trial_log_json or "[]")

        for row in reversed(log):
            if row.get("block") == block and row.get("block_trial") == block_trial:
                row.update(updates)
                break

        self.trial_log_json = json.dumps(log)
    



class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)
    last_rt_ms = models.IntegerField(initial=0)
    part1_points = models.IntegerField(initial=0)


class CursorTrace(ExtraModel):
    """One row per trial: the sampled cursor trajectory during that trial's
    choice window. Stored as an ExtraModel (its own table) rather than a Group
    field so that:
      - oTree's create_all() adds the table automatically on deploy (no manual
        ALTER / resetdb, existing data untouched), and
      - each trial is a single insert (no growing-JSON re-serialization).
    `samples_json` is a JSON list of {t_ms, x, y} points (see Game.html).
    """
    group = models.Link(Group)
    player = models.Link(Player)
    participant_code = models.StringField()
    id_in_group = models.IntegerField()
    overall_trial = models.IntegerField()
    phase = models.StringField()
    block_trial = models.IntegerField()
    n_samples = models.IntegerField()
    samples_json = models.LongStringField()
    server_ts = models.FloatField()


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
    
    if msg_type == "single_feedback_shown":
        player.group.update_last_matching_trial(
            block="single",
            block_trial=data.get("block_trial"),
            updates={
                "browser_feedback_ts": data.get("browser_feedback_ts"),
            },
        )

        return {
            player.id_in_group: dict(
                type="feedback_shown_ack",
            )
        }
    
    # ---------------------------------------------------------------------
    # Cursor trajectory for one choice window (sent once per trial, batched).
    # One ExtraModel row per trial.
    # ---------------------------------------------------------------------
    if msg_type == "cursor_trace":
        g = player.group
        phase, disp_trial, _ = _phase_and_display_trial(player)
        samples = data.get("samples", []) or []

        CursorTrace.create(
            group=g,
            player=player,
            participant_code=player.participant.code,
            id_in_group=player.id_in_group,
            overall_trial=player.current_trial + 1,
            phase=phase,
            block_trial=disp_trial,
            n_samples=len(samples),
            samples_json=json.dumps(samples),
            server_ts=time.time(),
        )
        return

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
            # Pre-update state = the read the opponent acted on for THIS trial
            # (same convention as MatchingPennies_protocol.m, which logs
            # min_pvalue before mp.update).
            algo_state_pre = algo.to_dict()
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

        if opponent_type == "algo_A":
            row.update(dict(
                algo_trials_back=algo_state_pre.get("trials_back"),
                algo_best_pvalue=algo_state_pre.get("best_pvalue"),
                algo_best_p_left=algo_state_pre.get("best_p_left"),
                algo_n_trials_seen=algo_state_pre.get("n_trials_seen"),
            ))

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

    # ---------- Phase 2 (solo control): same algo A instance as Part 1 ----------
    # Memory is NOT reset at the boundary, so the opponent policy is continuous
    # across all 800 trials and the only thing that changes at trial N_single is
    # the break / Part-2 framing. Pacing (the random feedback delay) is also held
    # constant across the boundary, unlike the paired multi block which is paced
    # by the partner.
    if is_solo_session(player.session):
        g = player.group
        rt_ms = int(data.get("rt_ms", 0))
        player.last_rt_ms = rt_ms
        n_single = num_trials_single(player)
        block_trial = player.current_trial - n_single + 1

        algo = get_single_algo(player)
        opponent_choice = algo.sample()
        algo_state_pre = algo.to_dict()

        is_win = (choice == opponent_choice)
        reward = C.REWARD_WIN if is_win else C.REWARD_LOSS
        player.last_reward = reward
        player.total_points += reward

        algo.update(last_choice=choice, last_reward=1 if is_win else 0)
        g.algo_state_json = json.dumps(algo.to_dict())

        delay_ms = random.randint(
            C.SINGLE_FEEDBACK_DELAY_MS_MIN,
            C.SINGLE_FEEDBACK_DELAY_MS_MAX,
        )

        g.append_trial(dict(
            overall_trial=player.current_trial + 1,
            block="multi",
            block_trial=block_trial,
            opponent_type="algo_A",
            opponent_id="algo_A_v1",
            solo_control=True,
            player_code=player.participant.code,
            player_choice=choice,
            player_rt_ms=rt_ms,
            opponent_choice=opponent_choice,
            reward=reward,
            total_points_after=player.total_points,
            server_ts=time.time(),
            feedback_delay_ms=delay_ms,
            algo_trials_back=algo_state_pre.get("trials_back"),
            algo_best_pvalue=algo_state_pre.get("best_pvalue"),
            algo_best_p_left=algo_state_pre.get("best_p_left"),
            algo_n_trials_seen=algo_state_pre.get("n_trials_seen"),
        ))

        player.current_trial += 1

        return {
            player.id_in_group: dict(
                type="feedback",
                phase="multi",
                trial=block_trial,
                trial_total=num_trials_multi(player),
                reward=reward,
                total_points=player.total_points,
                is_last=_overall_done(player),
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


# ---------------------------------------------------------------------------
# Progress for lab-notes (read with POST /api/get_session/<code>, participant_vars=["progress"])
# and a synchronised start of Part 2. Both live in participant.vars, so no new DB columns.
# ---------------------------------------------------------------------------

def set_progress(player, stage, **fields):
    player.participant.vars["progress"] = dict(stage=stage, t=round(time.time()), **fields)


def _game_progress(player):
    n_single, n_multi = num_trials_single(player), num_trials_multi(player)
    done = player.current_trial
    pv = player.participant.vars
    if done < n_single:
        return dict(stage="part1", trial=done, of=n_single, points=player.total_points)
    if done == n_single and not pv.get("part2_started"):
        return dict(stage="break", trial=0, of=n_multi, points=player.total_points)
    if done >= n_single + n_multi:
        return dict(stage="game_done", trial=n_multi, of=n_multi, points=player.total_points)
    return dict(stage="part2", trial=done - n_single, of=n_multi, points=player.total_points)


def _at_part2_start(player):
    return player.current_trial == num_trials_single(player) and num_trials_multi(player) > 0


def _part2_ready(player):
    """This player clicked through the break (or reloaded the page there). Part 2 starts for
    everyone in the group at once, when the last of them is ready; they're only told to wait."""
    group = player.group.get_players()
    already_going = all(p.participant.vars.get("part2_ready") for p in group)
    player.participant.vars["part2_ready"] = True
    if already_going:  # a reload after Part 2 started: just this player rejoins
        targets = [player]
    elif all(p.participant.vars.get("part2_ready") for p in group):
        targets = group
    else:
        return {player.id_in_group: dict(type="part2_wait")}
    for p in targets:
        p.participant.vars["part2_started"] = True
    return {p.id_in_group: dict(type="part2_go", trial=1, trial_total=num_trials_multi(p)) for p in targets}


def live_game_with_progress(player: Player, data):
    msg_type = data.get("type")
    if msg_type == "part2_ready" and _at_part2_start(player):
        reply = _part2_ready(player)
    else:
        reply = live_game(player, data)
        # a page reload at the break comes back as "start": wait for Part 2 like a click would
        if msg_type == "start" and _at_part2_start(player) and player.group.started:
            reply = _part2_ready(player)
    for p in player.group.get_players():
        set_progress(p, **_game_progress(p))
    return reply


class EyeCheck(Page):
    """Experimenter page on the headset laptop: starts Pupil Capture and shows live
    pupil confidence per eye, to adjust the eye cameras before the game."""

    @staticmethod
    def is_displayed(player):
        return pupil_bridge_enabled(player)

    @staticmethod
    def js_vars(player):
        return dict(pupil=pupil_js_vars(player, app="matching_live"))


class Setup(Page):
    form_model = "group"
    form_fields = ["single_opponent_p1", "single_opponent_p2", "num_trials_single", "num_trials_multi"]

    @staticmethod
    def is_displayed(player):
        # only show to player 1 (once per group), and only when the session didn't set it
        return player.id_in_group == 1 and not opponents_preset(player.session)

    @staticmethod
    def before_next_page(player, timeout_happened):
        g = player.group
        g.single_opponent_p1_final = resolve_opponent(player.session, g.single_opponent_p1)
        g.single_opponent_p2_final = resolve_opponent(player.session, g.single_opponent_p2)

class WaitAfterSetup(WaitPage):
    @staticmethod
    def is_displayed(player):
        return True

    @staticmethod
    def after_all_players_arrive(group: Group):
        # everyone in the group has joined, so each player's laptop label is known
        if not opponents_preset(group.session):
            return
        for p in group.get_players():
            mode = preset_opponent_mode(p)
            setattr(group, f"single_opponent_p{p.id_in_group}", mode)
            setattr(group, f"single_opponent_p{p.id_in_group}_final", resolve_opponent(group.session, mode))

class Game(Page):
    # This tells oTree to use the live_game function for WebSocket messages
    live_method = live_game_with_progress

    @staticmethod
    def js_vars(player: Player):
        set_progress(player, **_game_progress(player))  # the page (re)loaded
        return dict(
            num_trials_single=num_trials_single(player),
            num_trials_multi=num_trials_multi(player),
            # solo_control tags every pupil annotation, so the recording's annotations.csv
            # is self-describing: block=="multi" means algo A, not a human opponent.
            pupil=pupil_js_vars(
                player,
                app="matching_live",
                solo_control=1 if is_solo_session(player.session) else 0,
            ),
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
        )

    @staticmethod
    def js_vars(player: Player):
        # End stops the pupil recording too, in case the Game page's stop didn't get through
        return dict(pupil=pupil_js_vars(player, app="matching_live"))

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        survey = "mp_survey" in player.session.config["app_sequence"] and not player.session.config.get("skip_survey")
        set_progress(player, "survey" if survey else "finished", points=player.total_points)

page_sequence = [EyeCheck, Setup, WaitAfterSetup, Game, End]


def custom_export(players):
    """Flatten cursor trajectories to one CSV row per sample.

    Downloadable from oTree's admin under Data -> the app's custom export.
    Emits every {t_ms, x, y} sample tagged with participant / trial context, so
    a single row identifies a point within a specific trial.
    """
    yield [
        "session_code",
        "participant_code",
        "id_in_group",
        "phase",
        "overall_trial",
        "block_trial",
        "n_samples",
        "server_ts",
        "sample_index",
        "t_ms",
        "x",
        "y",
    ]
    for p in players:
        for tr in CursorTrace.filter(player=p):
            try:
                samples = json.loads(tr.samples_json or "[]")
            except (TypeError, ValueError):
                samples = []
            for i, s in enumerate(samples):
                yield [
                    p.session.code,
                    tr.participant_code,
                    tr.id_in_group,
                    tr.phase,
                    tr.overall_trial,
                    tr.block_trial,
                    tr.n_samples,
                    tr.server_ts,
                    i,
                    s.get("t_ms"),
                    s.get("x"),
                    s.get("y"),
                ]
