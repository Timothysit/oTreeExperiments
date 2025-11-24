from otree.api import *
import random


class C(BaseConstants):
    NAME_IN_URL = 'matching_live'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1  # all 10 trials happen on a single page
    NUM_TRIALS = 10
    REWARD_WIN = 10
    REWARD_LOSS = 0


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    current_trial = models.IntegerField(initial=0)
    total_points = models.IntegerField(initial=0)
    last_choice = models.StringField(blank=True)
    last_reward = models.IntegerField(initial=0)


class Game(Page):
    # This tells oTree to use the live_game function for WebSocket messages
    live_method = 'live_game'


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

        # Tell the client we're ready for trial 1
        return {
            player.id_in_group: dict(
                type='ready',
                trial=1,
                total_points=player.total_points,
            )
        }

    # Player chose left/right
    if msg_type == 'choice':
        choice = data.get('choice')
        if choice not in ['L', 'R']:
            return

        player.last_choice = choice

        # random reward: 50% chance of 10, otherwise 0
        is_win = random.random() < 0.5
        reward = C.REWARD_WIN if is_win else C.REWARD_LOSS
        player.last_reward = reward

        # update trial + total
        player.current_trial += 1
        player.total_points += reward

        is_last = player.current_trial >= C.NUM_TRIALS

        return {
            player.id_in_group: dict(
                type='feedback',
                trial=player.current_trial,
                reward=player.last_reward,
                total_points=player.total_points,
                is_last=is_last,
            )
        }


page_sequence = [Game]
