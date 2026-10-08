import os
import time
from pathlib import Path

from otree.api import *

from mp_survey.payment_crypto import encrypt_details
from pupil_bridge.context import pupil_bridge_enabled

from . import emailer


doc = """
Before the game: the Participant Information Sheet (InfoSheetText.html), with
the option to have it emailed (see emailer.py; the address is then kept,
encrypted, for mp_survey's payment form, so they aren't asked twice), then the
Consent Form ticked and signed on screen. Session config consent_pages turns
the app on (off by default for now: paper copies); digital_consent=False then
skips just the form.

Signing needs PAYMENT_PUBLIC_KEY on the server: the typed name
is encrypted with the same public key as the payment details
(PAYMENT_PUBLIC_KEY, see mp_survey/payment_crypto.py) and stored only in the
ConsentRecord table, which is in this app's custom export, never the research
export. The ticks themselves are Player fields, so e.g. consent to data sharing
(statement 11) joins to the game data.

The PDFs in _static/consent/ (emailed to participants) are copies of the
approved versions in JuliaAndTimHumanMP/ethics/. When the documents change,
replace them and update InfoSheetText.html and C.STATEMENTS to match.
"""


class C(BaseConstants):
    NAME_IN_URL = 'consent'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    STUDY_TITLE = 'Decision-Making in a Dynamic Social Context'
    ETHICS_ID = 'CEHP/2024/596'
    DOCS_DIR = Path(__file__).resolve().parent.parent / '_static' / 'consent'
    INFO_SHEET_PDF = 'consent/information_sheet.pdf'
    CONSENT_FORM_PDF = 'consent/consent_form.pdf'
    MAX_EMAILS = 3

    # Consent form statements, worded as in consent_form_Tim.pdf. Each is a list
    # of paragraphs; the lines starting with '- ' render as a list.
    STATEMENTS = [
        ['I confirm that I have read and understood the Information Sheet for the above study. I have had '
         'an opportunity to consider the information and what will be expected of me. I have also had the '
         'opportunity to ask questions which have been answered to my satisfaction.'],
        ['I understand that my participation is voluntary and that I am free to withdraw at any time without '
         'giving a reason, without the care I receive or my legal rights being affected.'],
        ['I understand that I will be able to withdraw my anonymous data at any point in time up until the '
         'publication of this data, and withdraw my personal information at any point without requiring a '
         'reason.',
         'I understand that if I decide to withdraw:',
         '- Any personal data I have provided up to that point will be deleted unless I agree otherwise.',
         '- Any published or pre-print (anonymous) data will remain available and open-access.'],
        ['I consent to participate in the study. I understand that my personal information (name, email '
         'address, gender, and date of birth) will be used for the purposes explained to me. I understand '
         'that according to data protection legislation, public task will be the lawful basis for processing.'],
        ['Use of the information',
         'I understand that all personal information will remain confidential and that all efforts will be '
         'made to ensure I cannot be identified (personal information that can be used to identify my data '
         'will be protected and only accessible to researchers on the study, and stored for up to 10 years '
         'after the completion of the project).'],
        ['I understand that the data gathered in this study will be stored pseudonymously and securely. It '
         'will be assigned a coded designation that will deprive the collected data of any connection to my '
         'identity. It will not be possible to identify me in any publications or scientific communication, '
         'where this data will be presented anonymously.'],
        ['I understand the potential risks of participating and the support that will be available to me '
         'should I become distressed during the course of the research.'],
        ['I understand that the data will not be made available to any commercial organisations but is '
         'solely the responsibility of the researcher(s) undertaking this study.'],
        ['I understand that I will not benefit financially from this study or from any possible outcome it '
         'may result in in the future.'],
        ['I understand that I will be compensated for the portion of time spent in the study, with additional '
         'compensation based on my performance, and will still be fully compensated if I later choose to '
         'withdraw.'],
        ['I agree that my pseudonymised research data may be used by others for future research. Only '
         'researchers undertaking the current study will be able to identify me from this data, and no one '
         'will be able to identify me from any published data. *Note, not agreeing to this will not preclude '
         'you from taking part in this study'],
        ['I hereby confirm that I understand the inclusion criteria as detailed in the Information Sheet and '
         'explained to me by the researcher.'],
        ['I hereby confirm that:',
         '- (a) I understand the exclusion criteria as detailed in the Information Sheet and explained to me '
         'by the researcher; and',
         '- (b) I do not fall under the exclusion criteria.'],
        ['I am aware of who I should contact if I wish to lodge a complaint.'],
        ['I voluntarily agree to take part in this study.'],
        ['Use of information for this project and beyond:',
         'I agree that my personal information (name, date of birth, gender, and email) will be stored '
         'securely in on UCL’s Data Safe Haven, for up to 10 years after the completion of this study, and '
         'that only study researchers will be able to associate my person information with my data. I am '
         'aware that no personal information will be included in any publication resulting from this study '
         'or otherwise.',
         'I would be happy for the data I provide to be securely archived at UCL until project completion.',
         'I understand that other authenticated researchers working on this study at UCL will have access '
         'to my pseudonymised data.'],
    ]
    OPTIONAL_STATEMENTS = [11]  # data sharing: "not agreeing to this will not preclude you"
    FUTURE_CONTACT = [
        'Yes, I would be happy to be contacted in this way',
        'No, I would not like to be contacted',
    ]


STATEMENT_FIELDS = [f'consent_{i:02d}' for i in range(1, len(C.STATEMENTS) + 1)]
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
    return bool(player.session.config.get('consent_pages', False))


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
    # so mp_survey's payment form needn't ask again; encrypted like the payment
    # details, never kept in plain text
    if consent_public_key():
        player.participant.vars['email_ciphertext'] = encrypt_details(dict(email=address), consent_public_key())
    return dict(status=emailer.SENDING)


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
    """Encrypt the signed form into a ConsentRecord, then blank the name on the Player."""
    name = (player.field_maybe_none('signed_name') or '').strip()
    player.signed_name = ''
    player.consent_signed_at = time.time()
    key = consent_public_key()
    if not (name and key):  # only possible in a test run (see ConsentForm.error_message)
        return
    record = dict(
        full_name=name,
        signed_at=time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime(player.consent_signed_at)),
        future_contact=player.field_maybe_none('future_contact') or '',
        researcher=researcher_name(player),
        **{f: bool(player.field_maybe_none(f)) for f in STATEMENT_FIELDS},
    )
    ConsentRecord.create(
        player=player,
        participant_code=player.participant.code,
        ciphertext=encrypt_details(record, key),
        submitted_at=player.consent_signed_at,
    )


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
    def before_next_page(player, timeout_happened):
        save_consent(player)


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
