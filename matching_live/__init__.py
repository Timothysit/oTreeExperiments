from otree.api import *
import random


class C(BaseConstants):
    NAME_IN_URL = 'matching_live'
    PLAYERS_PER_GROUP = 2
    NUM_ROUNDS = 1  # all 10 trials happen on a single page
    NUM_TRIALS_SINGLE = 10
    NUM_TRIALS_MULTI = 10
    NUM_TRIALS_TOTAL = NUM_TRIALS_SINGLE + NUM_TRIALS_MULTI

    REWARD_WIN = 10
    REWARD_LOSS = 0


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    # temporary storage for multi-player phase choices
    p1_choice = models.StringField(blank=True)
    p2_choice = models.StringField(blank=True)



class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)

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
        {type: "choice", choice: "L" or "R"}
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
        is_win = random.random() < 0.5
        reward = C.REWARD_WIN if is_win else C.REWARD_LOSS
        player.last_reward = reward
        player.total_points += reward
        player.current_trial += 1

        is_last = _overall_done(player)
        if not is_last:
            phase, disp_trial, disp_total = _phase_and_display_trial(player)
        else:
            # game ended right after increment (only possible if totals match)
            phase, disp_trial, disp_total = "end", C.NUM_TRIALS_MULTI, C.NUM_TRIALS_MULTI

        return {
            player.id_in_group: dict(
                type="feedback",
                phase="single",
                trial=min(player.current_trial, C.NUM_TRIALS_SINGLE),  # show 1..10
                trial_total=C.NUM_TRIALS_SINGLE,
                reward=reward,
                total_points=player.total_points,
                is_last=is_last,
            )
        }

    # ---------- Phase 2: two-player matching pennies ----------
    g = player.group

    # save this player's choice into the group
    if player.id_in_group == 1:
        g.p1_choice = choice
    else:
        g.p2_choice = choice

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

    # Rule (common “matching pennies” variant):
    # - if choices MATCH, Player 1 wins
    # - if choices DIFFER, Player 2 wins
    if c1 == c2:
        r1, r2 = C.REWARD_WIN, C.REWARD_LOSS
        winner = 1
    else:
        r1, r2 = C.REWARD_LOSS, C.REWARD_WIN
        winner = 2

    p1.last_reward, p2.last_reward = r1, r2
    p1.total_points += r1
    p2.total_points += r2

    # advance BOTH players to the next overall trial
    p1.current_trial += 1
    p2.current_trial += 1

    # clear group stored choices for next trial
    g.p1_choice = ""
    g.p2_choice = ""

    is_last_1 = _overall_done(p1)
    is_last_2 = _overall_done(p2)
    is_last = is_last_1 or is_last_2  # should be same if both advanced together

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
    live_method = live_game


class Game(Page):
    # This tells oTree to use the live_game function for WebSocket messages
    live_method = live_game


page_sequence = [Game]
