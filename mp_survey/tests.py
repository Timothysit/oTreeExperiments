"""Bot test for mp_survey.

Run with:
    uv run --with requests otree test matching_live_solo
"""
from otree.api import Bot, expect

from . import *


class PlayerBot(Bot):
    def play_round(self):
        yield AboutTheGame, dict(
            interface_ease='Very easy',
            instructions_clear='Very clear',
            feedback_clarity=4,
            reward_rule_clear='Yes',
            earnings_display='Both points and money conversion',
            pupil_headset='Yes',
        )
        yield PartsAndOpponent, dict(
            noticed_two_parts='Yes',
            part2_opponent_belief='Another participant',
            part2_opponent_confidence=3,
            opponent_difference='Slight difference',
        )
        yield StrategyAndExperience, dict(
            performance_overall='Average',
            performance_part1='Average',
            performance_part2='Well',
            had_strategy='Yes',
            strategy_description='alternate after a loss',
            predicted_opponent='Often',
            adapted_strategy='Moderately',
            engagement=7,
            mental_demand=5,
            engagement_change='Stayed the same',
            handedness='Right-handed',
        )
        yield AQ10, {f'aq_{i:02d}': 'Slightly agree' for i in range(1, 11)}
        yield BISBrief, {f'bis_{i:02d}': 'Often' for i in range(1, 9)}
        yield GAD7, {f'gad_{i:02d}': 'Not at all' for i in range(1, 8)}
        expect(self.player.part2_opponent_belief, 'Another participant')
        expect(self.player.field_maybe_none('gad_difficulty'), None)
        expect('participant ID', 'in', self.html)


def test_payment_url_fills_in_code():
    class Obj:
        pass

    player = Obj()
    player.participant = Obj()
    player.participant.code = 'abc123'
    player.session = Obj()
    player.session.config = {
        'payment_form_url': 'https://docs.google.com/forms/d/e/X/viewform?entry.1={participant_code}'
    }
    expect(payment_form_url(player), 'https://docs.google.com/forms/d/e/X/viewform?entry.1=abc123')
    player.session.config = {'payment_form_url': ''}
    expect(payment_form_url(player), '')
