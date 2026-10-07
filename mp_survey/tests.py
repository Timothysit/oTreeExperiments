"""Bot test for mp_survey.

Run with:
    uv run --with requests otree test matching_live_solo
"""
import os
from contextlib import contextmanager

from otree.api import Bot, expect

from . import *
from .payment_crypto import (
    clean_details, decrypt_details, encrypt_details, generate_keypair, latest_per_participant,
)

GOOD_DETAILS = dict(
    full_name='Ada Lovelace', email='ada@example.com', sort_code='12-34-56', account_number='1234 5678'
)


@contextmanager
def payment_key(public_b64):
    """Set PAYMENT_PUBLIC_KEY only for the duration of a test call, so importing
    this module never changes the key a real server uses."""
    old = os.environ.get('PAYMENT_PUBLIC_KEY')
    os.environ['PAYMENT_PUBLIC_KEY'] = public_b64
    try:
        yield
    finally:
        if old is None:
            del os.environ['PAYMENT_PUBLIC_KEY']
        else:
            os.environ['PAYMENT_PUBLIC_KEY'] = old


class PlayerBot(Bot):
    def play_round(self):
        if survey_skipped(self.player):  # test run: no survey pages at all
            return
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
        check_payment_details_stored_encrypted(self.player)


def check_payment_details_stored_encrypted(player):
    private_b64, public_b64 = generate_keypair()
    with payment_key(public_b64):
        bad = save_payment_details(player, dict(GOOD_DETAILS, sort_code='12'))
        expect(bad['ok'], False)
        expect(list(bad['errors']), ['sort_code'])
        expect(len(PaymentDetails.filter(player=player)), 0)

        expect(save_payment_details(player, GOOD_DETAILS), dict(ok=True))
    [row] = PaymentDetails.filter(player=player)
    expect(row.participant_code, player.participant.code)
    for secret in ['Lovelace', 'ada@example.com', '123456', '12345678']:
        expect(secret, 'not in', row.ciphertext)
    expect(decrypt_details(row.ciphertext, private_b64), dict(
        full_name='Ada Lovelace', email='ada@example.com', sort_code='123456', account_number='12345678'
    ))
    [header, exported] = list(custom_export([player]))
    expect(exported[header.index('ciphertext')], row.ciphertext)


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


def test_clean_details_flags_each_bad_field():
    details, errors = clean_details(GOOD_DETAILS)
    expect(errors, {})
    expect(details['sort_code'], '123456')
    expect(details['account_number'], '12345678')
    _, errors = clean_details(dict(full_name=' ', email='nope', sort_code='12345a', account_number='1234567'))
    expect(sorted(errors), ['account_number', 'email', 'full_name', 'sort_code'])
    _, errors = clean_details(None)
    expect(len(errors), 4)


def test_encryption_round_trip_needs_the_private_key():
    private_b64, public_b64 = generate_keypair()
    ciphertext = encrypt_details({'a': 1}, public_b64)
    expect(decrypt_details(ciphertext, private_b64), {'a': 1})
    other_private, _ = generate_keypair()
    try:
        decrypt_details(ciphertext, other_private)
    except Exception:
        pass
    else:
        raise AssertionError('decrypted with the wrong key')


def test_latest_submission_per_participant_wins():
    rows = [
        dict(participant_code='a', submitted_at='1.0', ciphertext='old'),
        dict(participant_code='b', submitted_at='2.0', ciphertext='b'),
        dict(participant_code='a', submitted_at='3.0', ciphertext='new'),
    ]
    expect(sorted(r['ciphertext'] for r in latest_per_participant(rows)), ['b', 'new'])
