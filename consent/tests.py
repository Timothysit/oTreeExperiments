"""Bot test for consent.

Run with:
    uv run --with requests otree test matching_live_solo          # consent required
    uv run --with requests otree test matching_live_preset_test   # test run: consent optional
"""
import os

from otree.api import Bot, SubmissionMustFail, expect

from mp_survey.payment_crypto import decrypt_details, generate_keypair
from mp_survey.tests import payment_key

from . import *
from . import emailer

# One keypair for every bot: a pair's bots run interleaved, so a key swapped in and
# out around each bot's own submission would be changed under the other's feet.
PRIVATE_KEY, PUBLIC_KEY = generate_keypair()

SMTP = dict(host='smtp.example.com', port=587, user='study@example.com', password='x', reply_to='')


class FakeSender:
    """Stands in for emailer.send_in_background: records the message, sends nothing."""
    def __init__(self):
        self.sent = []

    def __call__(self, key, msg, settings):
        self.sent.append(msg)


class PlayerBot(Bot):
    def play_round(self):
        if not consent_pages(self.player):
            expect(self.player.field_maybe_none('consent_signed_at'), None)
            return
        expect('Participant Information Sheet', 'in', self.html)
        expect('Local Data Protection Privacy Notice', 'in', self.html)
        # demo participants have no laptop label, so count as on the headset laptop
        expect('pupil diameter', 'in', self.html)
        check_emailing(self.player)
        yield InformationSheet

        expect(digital_consent(self.player), True)
        signed = dict(
            {f: True for f in REQUIRED_STATEMENT_FIELDS},
            future_contact=C.FUTURE_CONTACT[1],
            signed_name='Ada Lovelace',
        )
        if is_test_run(self.player):
            expect('Test run', 'in', self.html)
        else:
            expect('Test run', 'not in', self.html)
            if not consent_public_key():  # first bot of the run
                # no key on the server: can't sign, whatever is ticked
                yield SubmissionMustFail(ConsentForm, signed)
        # set for the rest of the bot run (only ever runs under `otree test`)
        os.environ['PAYMENT_PUBLIC_KEY'] = PUBLIC_KEY
        # the address for a signed copy, handed over as they press Sign
        code = self.player.participant.code
        expect(copy_address(self.player, dict(copy_email='nope'))['ok'], False)
        expect(copy_address(self.player, dict(copy_email=' ada@example.com ')), dict(ok=True))
        expect(emailer.recall_address(code), 'ada@example.com')
        if not is_test_run(self.player):
            # a browser sends nothing for an unticked box
            yield SubmissionMustFail(ConsentForm, {k: v for k, v in signed.items() if k != 'consent_15'})
            yield SubmissionMustFail(ConsentForm, dict(signed, signed_name=' '))
        yield ConsentForm, signed
        expect(self.player.signed_name, '')
        expect(emailer.recall_address(code), '')  # dropped once signed (no SMTP in tests, so not sent)
        expect(self.player.consent_01, True)
        expect(self.player.consent_11, False)
        [row] = ConsentRecord.filter(player=self.player)
        expect('Lovelace', 'not in', row.ciphertext)
        record = decrypt_details(row.ciphertext, PRIVATE_KEY)
        expect(record['full_name'], 'Ada Lovelace')
        expect(record['consent_11'], False)
        expect(record['future_contact'], C.FUTURE_CONTACT[1])
        expect(record['researcher'], 'Timothy Sit')
        expect(self.player.participant.vars['consent_ciphertext'], row.ciphertext)
        [header, exported] = list(custom_export([self.player]))
        expect(exported[header.index('ciphertext')], row.ciphertext)
        check_signed_copy(self.player, record)


def check_signed_copy(player, record):
    send = FakeSender()
    expect(email_signed_copy(player, record, '', SMTP, send), False)
    expect(email_signed_copy(player, record, 'ada@example.com', SMTP, send), True)
    [msg] = send.sent
    expect(msg['To'], 'ada@example.com')
    attachments = {part.get_filename(): part.get_content() for part in msg.iter_attachments()}
    expect(sorted(attachments), ['information_sheet.pdf', 'signed_consent_form.pdf'])
    expect(attachments['signed_consent_form.pdf'][:4], b'%PDF')
    expect(decrypt_details(player.participant.vars['email_ciphertext'], PRIVATE_KEY), dict(email='ada@example.com'))


def check_emailing(player):
    send = FakeSender()
    expect(email_documents(player, dict(email='nope'), SMTP, send), dict(error='Please enter a valid email address.'))
    private_b64, public_b64 = generate_keypair()
    with payment_key(public_b64):
        expect(email_documents(player, dict(email=' ada@example.com '), SMTP, send), dict(status=emailer.SENDING))
    [msg] = send.sent
    expect(msg['To'], 'ada@example.com')
    expect(
        sorted(part.get_filename() for part in msg.iter_attachments()),
        ['consent_form.pdf', 'information_sheet.pdf'],
    )
    expect(player.info_sheet_emails, 1)
    # kept for mp_survey's payment form, encrypted only
    ciphertext = player.participant.vars['email_ciphertext']
    expect('ada@example.com', 'not in', str(player.participant.vars))
    expect(decrypt_details(ciphertext, private_b64), dict(email='ada@example.com'))
    for _ in range(C.MAX_EMAILS - 1):
        email_documents(player, dict(email='ada@example.com'), SMTP, send)
    expect('error', 'in', email_documents(player, dict(email='ada@example.com'), SMTP, send))
    expect(len(send.sent), C.MAX_EMAILS)
    # the payment form tests in mp_survey use their own key
    del player.participant.vars['email_ciphertext']


class FakePlayer:
    def __init__(self, **config):
        self.session = type('Session', (), dict(config=config))()


def test_consent_errors_require_all_but_statement_11():
    player = FakePlayer()
    full = dict({f: True for f in REQUIRED_STATEMENT_FIELDS}, future_contact=C.FUTURE_CONTACT[0], signed_name='A B')
    expect(consent_errors(player, full), {})
    expect('consent_11', 'not in', REQUIRED_STATEMENT_FIELDS)
    expect(sorted(consent_errors(player, dict(full, consent_15=False, signed_name=' '))), ['consent_15', 'signed_name'])
    expect(len(consent_errors(player, {})), len(REQUIRED_STATEMENT_FIELDS) + 2)
    expect(consent_errors(FakePlayer(skip_survey=True), {}), {})


def test_statement_fields_match_player():
    expect(len(STATEMENT_FIELDS), 16)
    for f in STATEMENT_FIELDS:
        expect(hasattr(Player, f), True)


def test_smtp_settings_need_user_and_password():
    expect(emailer.smtp_settings({}), None)
    expect(emailer.smtp_settings(dict(SMTP_USER='a@b.co')), None)
    s = emailer.smtp_settings(dict(SMTP_USER='a@b.co', SMTP_PASSWORD='pw'))
    expect((s['host'], s['port']), ('smtp.gmail.com', 587))


def test_background_send_reports_status():
    import threading
    called = threading.Event()
    emailer.send_in_background('k', None, SMTP, send=lambda key, msg, settings: called.set())
    expect(called.wait(5), True)
    expect(emailer.status('k'), emailer.SENDING)  # the fake never reports back


def test_remembered_address_expires():
    emailer.remember_address('p1', 'a@b.co', now=1000)
    expect(emailer.recall_address('p1', now=1000 + emailer.ADDRESS_TTL), 'a@b.co')
    expect(emailer.recall_address('p1', now=1001 + emailer.ADDRESS_TTL), '')
    emailer.forget_address('p1')
    expect(emailer.recall_address('p1', now=1000), '')


def test_documents_exist():
    for name in ['information_sheet.pdf', 'consent_form.pdf']:
        expect((C.DOCS_DIR / name).read_bytes()[:4], b'%PDF')


def make_record(name='Zoë Łukasiewicz-Ng', signed_at='2026-10-08 16:38:12 UTC'):
    return dict(full_name=name, signed_at=signed_at, future_contact=C.FUTURE_CONTACT[0],
                researcher='Timothy Sit', **{f: f != 'consent_11' for f in STATEMENT_FIELDS})


def test_signed_form_renders_any_european_name():
    from .signed_form import render_signed_form, signed_date
    expect(render_signed_form(make_record(), 'abc')[:4], b'%PDF')
    expect(signed_date(make_record()), '8 October 2026')


def test_archive_writes_one_pdf_per_real_participant(tmp_path=None):
    import csv
    import tempfile
    from pathlib import Path
    from mp_survey.payment_crypto import encrypt_details
    from .signed_form import archive
    d = Path(tmp_path or tempfile.mkdtemp())
    rows = [  # p1 signed twice (latest wins), p2 is a test run
        dict(participant_code='p1', test_run='0', submitted_at='1.0',
             ciphertext=encrypt_details(make_record('Old'), PUBLIC_KEY)),
        dict(participant_code='p1', test_run='0', submitted_at='2.0',
             ciphertext=encrypt_details(make_record('Ada'), PUBLIC_KEY)),
        dict(participant_code='p2', test_run='1', submitted_at='3.0',
             ciphertext=encrypt_details(make_record('Test'), PUBLIC_KEY)),
    ]
    with open(d / 'export.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    written, existing = archive(d / 'export.csv', PRIVATE_KEY, d / 'forms')
    expect([p.name for p in written], ['2026-10-08_p1_consent.pdf'])
    expect(existing, [])
    written, existing = archive(d / 'export.csv', PRIVATE_KEY, d / 'forms')  # re-run: nothing new
    expect((len(written), len(existing)), (0, 1))
