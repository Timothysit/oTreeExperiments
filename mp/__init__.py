from otree.api import *
import random

doc = """
This is a (solo) matching pennies game!
"""


class C(BaseConstants):
    NAME_IN_URL = 'mp'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 5
    STAKES = cu(100)  # I guess this is how many total points you can get?


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


class Player(BasePlayer):
    left_right_choice = models.StringField(
        choices=[['Left', 'L'], ['Right', 'R']],
        widget=widgets.RadioSelect,
        label="I choose:",
    )

    # Opponent’s choice for this round
    computer_choice = models.StringField()

    # Whether the player won this round
    is_winner = models.BooleanField(initial=False)



# Functions specific to this game 
def set_payoffs(group: Group):
    subsession = group.subsession
    session = group.session
    p1 = group.get_player_by_id(1)

    # Computer chooses randomly
    p1.computer_choice = random.choice(['Left', 'Right'])

    p1 = group.get_player_by_id(1)

    if (p1.left_right_choice == p1.computer_choice):
        p1.payoff = C.STAKES
        p1.is_winner = 1
    else:
        p1.payoff = cu(0)
        p1.is_winner = 0



# PAGES
class MyPage(Page):

    # The following method makes it so that the page is only displayed on round 1
    @staticmethod
    def is_displayed(player):
        return player.round_number == 1 

class Choice(Page):
    form_model = 'player'
    form_fields = ['left_right_choice']

    @staticmethod
    def before_next_page(player: Player, timeout_happened):
        # compute payoff after they submit their choice
        set_payoffs(player.group)


class ResultsWaitPage(WaitPage):
    after_all_players_arrive = set_payoffs


class Results(Page):
    pass


page_sequence = [MyPage, Choice, Results]
