"""Bot test for mp_survey.

Run with:
    uv run --with requests otree test matching_live_solo
    uv run --with requests otree test matching_live_preset_test   # test run: all optional
"""
import csv
import os
from contextlib import contextmanager

from otree.api import Bot, SubmissionMustFail, expect

from . import *
from .payment_crypto import (
    clean_details, decrypt_details, encrypt_details, generate_keypair, latest_per_participant,
)

GOOD_DETAILS = dict(
    full_name='Ada Lovelace', email='ada@example.com', sort_code='12-34-56', account_number='1234 5678',
    address_line1='1 Example Street', address_line2='', town='London', postcode='wc1e  6bt',
    bank_name='Example Bank',
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
        if survey_optional(self.player):  # test run: every page can be submitted empty
            expect('Test run', 'in', self.html)
            for page in [AboutTheGame, PartsAndOpponent, StrategyAndExperience, AQ10, BISBrief, GAD7]:
                yield page, {}
            expect(self.player.field_maybe_none('interface_ease'), None)
            expect('Test run', 'in', self.html)
            check_payment_details_stored_encrypted(self.player, test_run=1)
            return
        yield SubmissionMustFail(AboutTheGame, {})
        yield SubmissionMustFail(AboutTheGame, dict(interface_ease='Very easy'))
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
        expect('Test run', 'not in', self.html)
        check_payment_details_stored_encrypted(self.player, test_run=0)


def check_payment_details_stored_encrypted(player, test_run):
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
        full_name='Ada Lovelace', email='ada@example.com', sort_code='123456', account_number='12345678',
        address_line1='1 Example Street', address_line2='', town='London', postcode='WC1E 6BT',
        bank_name='Example Bank',
    ))
    # lab-notes reads the latest ciphertext through the REST API (participant vars only)
    expect(player.participant.vars['payment_ciphertext'], row.ciphertext)
    [header, exported] = list(custom_export([player]))
    expect(exported[header.index('ciphertext')], row.ciphertext)
    expect(exported[header.index('test_run')], test_run)


def test_optional_fields_are_player_fields():
    for name in OPTIONAL_FIELDS:
        expect(hasattr(Player, name), True)


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
    expect(details['postcode'], 'WC1E 6BT')
    _, errors = clean_details(dict(GOOD_DETAILS, full_name=' ', email='nope', sort_code='12345a',
                                   account_number='1234567', postcode='12345', bank_name='', town='',
                                   address_line1=''))
    expect(sorted(errors), ['account_number', 'address_line1', 'bank_name', 'email', 'full_name',
                            'postcode', 'sort_code', 'town'])
    _, errors = clean_details(None)
    expect(len(errors), 8)  # all but address line 2
    for postcode in ['SW1A 1AA', 'm1 1ae', 'B33 8TH', 'CR2 6XH', 'DN55 1PT', 'EC1A1BB']:
        expect(clean_details(dict(GOOD_DETAILS, postcode=postcode))[1], {})
    # email already given on the information sheet page: blank is fine, a typo isn't
    _, errors = clean_details(dict(GOOD_DETAILS, email=''), email_on_file=True)
    expect(errors, {})
    _, errors = clean_details(dict(GOOD_DETAILS, email='nope'), email_on_file=True)
    expect(list(errors), ['email'])


def test_decrypt_fills_in_email_given_earlier(tmp_path=None):
    import tempfile
    from pathlib import Path
    from .payment_crypto import main
    private_b64, public_b64 = generate_keypair()
    d = Path(tmp_path or tempfile.mkdtemp())
    (d / 'key').write_text(private_b64)
    meta = dict(session_code='s', participant_label='', test_run='0', submitted_at='1.0')
    rows = [
        dict(meta, participant_code='a', email_ciphertext=encrypt_details(dict(email='a@x.org'), public_b64),
             ciphertext=encrypt_details(dict(GOOD_DETAILS, email=''), public_b64)),
        dict(meta, participant_code='b', email_ciphertext='',
             ciphertext=encrypt_details(GOOD_DETAILS, public_b64)),
    ]
    with open(d / 'export.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    main(['decrypt', str(d / 'export.csv'), '--key', str(d / 'key'), '--out', str(d / 'out.csv')])
    with open(d / 'out.csv', newline='') as f:
        out = {r['participant_code']: r for r in csv.DictReader(f)}
    expect(out['a']['email'], 'a@x.org')
    expect(out['b']['email'], 'ada@example.com')


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
