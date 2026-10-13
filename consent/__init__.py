import os
import time
from pathlib import Path

from otree.api import *

from mp_survey.payment_crypto import encrypt_details
from pupil_bridge.context import pupil_bridge_enabled

from . import emailer, form_text
from .signed_form import render_signed_form


doc = """
Before the game: the Participant Information Sheet (InfoSheetText.html), with
the option to have it emailed (see emailer.py; the address is then kept,
encrypted, for mp_survey's payment form, so they aren't asked twice), then the
Consent Form ticked and signed on screen, with a PDF of the signed form emailed
to them (signed_form.py, which also archives the forms from the export).
Session config consent_pages=False
skips the app (paper copies); digital_consent=False skips just the form.

Signing needs PAYMENT_PUBLIC_KEY on the server: the typed name
is encrypted with the same public key as the payment details
(PAYMENT_PUBLIC_KEY, see mp_survey/payment_crypto.py) and stored only in the
ConsentRecord table, which is in this app's custom export, never the research
export. The ticks themselves are Player fields, so e.g. consent to data sharing
(statement 11) joins to the game data.

The PDFs in _static/consent/ (emailed to participants) are copies of the
approved versions in JuliaAndTimHumanMP/ethics/. When the documents change,
replace them and update InfoSheetText.html and form_text.py to match.
"""


class C(BaseConstants):
    NAME_IN_URL = 'consent'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    STUDY_TITLE = form_text.STUDY_TITLE
    ETHICS_ID = form_text.ETHICS_ID
    DOCS_DIR = Path(__file__).resolve().parent.parent / '_static' / 'consent'
    INFO_SHEET_PDF = 'consent/information_sheet.pdf'
    CONSENT_FORM_PDF = 'consent/consent_form.pdf'
    MAX_EMAILS = 3

    STATEMENTS = form_text.STATEMENTS
    OPTIONAL_STATEMENTS = form_text.OPTIONAL_STATEMENTS
    FUTURE_CONTACT = form_text.FUTURE_CONTACT


STATEMENT_FIELDS = form_text.STATEMENT_FIELDS
REQUIRED_STATEMENT_FIELDS = [
    f for i, f in enumerate(STATEMENT_FIELDS, start=1) if i not in C.OPTIONAL_STATEMENTS
]


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


def _tick():
    return models.BooleanField(widget=widgets.CheckboxInput, blank=True)


class Player(BasePlayer):
    # how many times the documents were emailed (the address itself is never stored)
    info_sheet_emails = models.IntegerField(initial=0)

    consent_01 = _tick()
    consent_02 = _tick()
    consent_03 = _tick()
    consent_04 = _tick()
    consent_05 = _tick()
    consent_06 = _tick()
    consent_07 = _tick()
    consent_08 = _tick()
    consent_09 = _tick()
    consent_10 = _tick()
    consent_11 = _tick()
    consent_12 = _tick()
    consent_13 = _tick()
    consent_14 = _tick()
    consent_15 = _tick()
    consent_16 = _tick()
    future_contact = models.StringField(choices=C.FUTURE_CONTACT, widget=widgets.RadioSelect, blank=True)
    # The typed name only passes through: before_next_page encrypts it into a
    # ConsentRecord and blanks this field, so it is never stored in plain text.
    signed_name = models.StringField(blank=True)
    consent_signed_at = models.FloatField(blank=True)


class ConsentRecord(ExtraModel):
    """One row per signed consent form: `ciphertext` is the encrypted JSON of the
    name, time and answers. Its own table, so it stays out of oTree's standard exports."""
    player = models.Link(Player)
    participant_code = models.StringField()
    ciphertext = models.LongStringField()
    submitted_at = models.FloatField()


def is_test_run(player):
    """lab-notes test sessions (session config skip_survey): consent optional, as in mp_survey."""
    return bool(player.session.config.get('skip_survey', False))


def consent_pages(player):
    """Session config `consent_pages`: show this app at all (off: paper copies)."""
    return bool(player.session.config.get('consent_pages', True))


def digital_consent(player):
    return consent_pages(player) and bool(player.session.config.get('digital_consent', True))


def researcher_name(player):
    """Countersigns the consent form: session config `researcher_name` (lab-notes can
    set whoever runs the session)."""
    return player.session.config.get('researcher_name', '') or 'Timothy Sit'


def consent_public_key():
    return os.environ.get('PAYMENT_PUBLIC_KEY', '').strip()


def set_progress(player, stage):
    # progress for lab-notes (same record matching_live keeps up to date)
    player.participant.vars['progress'] = dict(stage=stage, t=round(time.time()))


# --- Emailing the documents -------------------------------------------------

EMAIL_SUBJECT = f'Your copy: Participant Information Sheet, "{C.STUDY_TITLE}"'
EMAIL_BODY = f"""Hello,

Thank you for your interest in the UCL study "{C.STUDY_TITLE}"
(UCL Research Ethics Committee approval {C.ETHICS_ID}).

Attached are the Participant Information Sheet and the Consent Form for you
to keep. The research team's contact details are in the Information Sheet.

UCL Psychology and Language Sciences
"""


def email_documents(player, data, settings=None, send=emailer.send_in_background):
    """Live method of InformationSheet. {'email': ...} starts sending;
    {'check': True} asks how it went (the page polls)."""
    key = player.participant.code
    data = data if isinstance(data, dict) else {}
    if data.get('check'):
        return dict(status=emailer.status(key))
    settings = settings or emailer.smtp_settings()
    if settings is None:
        return dict(error='Email is not set up; please ask the experimenter for a printed copy.')
    address, error = emailer.clean_address(data.get('email'))
    if error:
        return dict(error=error)
    if player.info_sheet_emails >= C.MAX_EMAILS:
        return dict(error='The documents have already been sent several times; '
                          'please ask the experimenter if they have not arrived.')
    msg = emailer.build_message(
        address, settings, EMAIL_SUBJECT, EMAIL_BODY,
        [C.DOCS_DIR / 'information_sheet.pdf', C.DOCS_DIR / 'consent_form.pdf'],
    )
    send(key, msg, settings)
    player.info_sheet_emails += 1
    emailer.remember_address(key, address)  # offered again for the signed consent form
    keep_email_for_payment(player, address)
    return dict(status=emailer.SENDING)


def keep_email_for_payment(player, address):
    """So mp_survey's payment form needn't ask again: encrypted like the payment
    details, never kept in plain text."""
    if consent_public_key():
        player.participant.vars['email_ciphertext'] = encrypt_details(dict(email=address), consent_public_key())


# --- Signing the consent form -------------------------------------------------

def consent_errors(player, values):
    """Field -> message for what's missing; empty when the form can be accepted."""
    if is_test_run(player):
        return {}
    errors = {f: 'Please tick this box to take part.' for f in REQUIRED_STATEMENT_FIELDS if not values.get(f)}
    if not values.get('future_contact'):
        errors['future_contact'] = 'Please choose one.'
    if not str(values.get('signed_name') or '').strip():
        errors['signed_name'] = 'Please type your full name.'
    elif len(values['signed_name']) > 100:
        errors['signed_name'] = 'Name is too long.'
    return errors


def save_consent(player):
    """Encrypt the signed form into a ConsentRecord, then blank the name on the Player.
    Returns the record (None if no name was typed: only possible in a test run)."""
    name = (player.field_maybe_none('signed_name') or '').strip()
    player.signed_name = ''
    player.consent_signed_at = time.time()
    if not name:
        return None
    record = dict(
        full_name=name,
        signed_at=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(player.consent_signed_at)),
        future_contact=player.field_maybe_none('future_contact') or '',
        researcher=researcher_name(player),
        **{f: bool(player.field_maybe_none(f)) for f in STATEMENT_FIELDS},
    )
    key = consent_public_key()
    if key:  # always, except in a test run (see ConsentForm.error_message)
        ciphertext = encrypt_details(record, key)
        ConsentRecord.create(
            player=player,
            participant_code=player.participant.code,
            ciphertext=ciphertext,
            submitted_at=player.consent_signed_at,
        )
        # for lab-notes, which archives the signed forms (REST get_session returns
        # participant vars only); still ciphertext, like payment_ciphertext
        player.participant.vars['consent_ciphertext'] = ciphertext
    return record


SIGNED_SUBJECT = f'Your signed consent form, "{C.STUDY_TITLE}"'
SIGNED_BODY = f"""Hello,

Thank you for taking part in the UCL study "{C.STUDY_TITLE}"
(UCL Research Ethics Committee approval {C.ETHICS_ID}).

Attached is a copy of the consent form you signed, and the Participant
Information Sheet, for you to keep. The research team's contact details are
in the Information Sheet.

UCL Psychology and Language Sciences
"""


def copy_address(player, data):
    """Live method of ConsentForm, sent as they press Sign: {'copy_email': ...}
    ('' for no copy). The address is held in memory only, until before_next_page."""
    key = player.participant.code
    raw = str((data if isinstance(data, dict) else {}).get('copy_email') or '').strip()
    if not raw:
        emailer.forget_address(key)
        return dict(ok=True)
    address, error = emailer.clean_address(raw)
    if error:
        return dict(ok=False, error=error)
    emailer.remember_address(key, address)
    return dict(ok=True)


def email_signed_copy(player, record, address, settings=None, send=emailer.send_in_background):
    """Email the signed form (as a PDF) and the information sheet. Returns whether it was sent."""
    settings = settings or emailer.smtp_settings()
    if not (record and address and settings):
        return False
    code = player.participant.code
    msg = emailer.build_message(address, settings, SIGNED_SUBJECT, SIGNED_BODY, [
        ('signed_consent_form.pdf', render_signed_form(record, code)),
        C.DOCS_DIR / 'information_sheet.pdf',
    ])
    send(f'{code}:signed', msg, settings)
    keep_email_for_payment(player, address)
    return True


# PAGES
class InformationSheet(Page):
    @staticmethod
    def is_displayed(player):
        return consent_pages(player)

    @staticmethod
    def vars_for_template(player):
        set_progress(player, 'info_sheet')
        return dict(
            info_sheet_pdf=C.INFO_SHEET_PDF,
            pupillometry=pupil_bridge_enabled(player),
            consent_form_pdf=C.CONSENT_FORM_PDF,
            can_email=emailer.smtp_settings() is not None,
            keep_email=bool(consent_public_key()),
            digital_consent=digital_consent(player),
        )

    @staticmethod
    def live_method(player, data):
        return {player.id_in_group: email_documents(player, data)}


class ConsentForm(Page):
    form_model = 'player'
    form_fields = STATEMENT_FIELDS + ['future_contact', 'signed_name']

    @staticmethod
    def is_displayed(player):
        return digital_consent(player)

    @staticmethod
    def vars_for_template(player):
        set_progress(player, 'consent')
        statements = [
            dict(
                number=i,
                field=f,
                optional=i in C.OPTIONAL_STATEMENTS,
                paragraphs=[p for p in paras if not p.startswith('- ')],
                bullets=[p[2:] for p in paras if p.startswith('- ')],
            )
            for i, (f, paras) in enumerate(zip(STATEMENT_FIELDS, C.STATEMENTS), start=1)
        ]
        return dict(
            statements=statements,
            today=time.strftime('%d %B %Y'),
            researcher=researcher_name(player),
            can_email=emailer.smtp_settings() is not None,
            copy_email=emailer.recall_address(player.participant.code),
            not_set_up=not consent_public_key(),
            test_run=is_test_run(player),
            consent_form_pdf=C.CONSENT_FORM_PDF,
        )

    @staticmethod
    def error_message(player, values):
        if not consent_public_key() and not is_test_run(player):
            return 'Digital consent is not set up on this server. Please ask the experimenter for a paper form.'
        return consent_errors(player, values) or None

    @staticmethod
    def live_method(player, data):
        return {player.id_in_group: copy_address(player, data)}

    @staticmethod
    def before_next_page(player, timeout_happened):
        record = save_consent(player)
        code = player.participant.code
        email_signed_copy(player, record, emailer.recall_address(code))
        emailer.forget_address(code)


page_sequence = [InformationSheet, ConsentForm]


def custom_export(players):
    """Encrypted consent records; decrypt with
    `python mp_survey/payment_crypto.py decrypt <this export> --key ... --out consent.csv`."""
    yield ['session_code', 'participant_code', 'participant_label', 'test_run', 'submitted_at', 'ciphertext']
    for p in players:
        for row in ConsentRecord.filter(player=p):
            yield [
                p.session.code, row.participant_code, p.participant.label or '',
                int(is_test_run(p)), row.submitted_at, row.ciphertext,
            ]
