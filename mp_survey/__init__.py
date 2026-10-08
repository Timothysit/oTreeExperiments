import os
import time

from otree.api import *

from .payment_crypto import clean_details, encrypt_details


doc = """
Post-task questionnaire for matching_live, replacing the separate Google Form
("Course Evaluation (Responses)") used up to 2026-10.

Answers are stored on the Player, so they join to the game data by
participant.code automatically. Deliberately collects NO personal details on
the Player. Payment details (name, email, UK bank details) are entered on the
last page, encrypted on the server with the public key in env var
PAYMENT_PUBLIC_KEY, and stored as ciphertext only in the PaymentDetails table,
which never appears in the research export (see payment_crypto.py for the
payer's decrypt step). Without that key, the page falls back to linking an
external payment form (session config `payment_form_url`). A participant who
had the information sheet emailed to them (consent app) isn't asked for their
email again: it was encrypted then, see email_on_file().

Answer text matches the old Google Form, so old and new responses can be
pooled. Exceptions: the interface-ease question is now single-choice (the Form's
grid let people tick several), and three typos were fixed ("Mich more engaging",
"stop of control", "looking art").
"""


class C(BaseConstants):
    NAME_IN_URL = 'mp_survey'
    PLAYERS_PER_GROUP = None
    NUM_ROUNDS = 1

    EASE = ['Very easy', 'Somewhat easy', 'Neutral', 'Somewhat difficult', 'Very difficult']
    CLARITY = ['Very clear', 'Somewhat clear', 'Neutral', 'Somewhat unclear', 'Very unclear']
    YES_NO_UNSURE = ['Yes', 'No', 'Unsure']
    PERFORMANCE = ['Very poorly', 'Poorly', 'Average', 'Well', 'Very well']
    AQ = ['Definitely agree', 'Slightly agree', 'Slightly disagree', 'Definitely disagree']
    BIS = ['Rarely/never', 'Occasionally', 'Often', 'Almost always/always']
    GAD = ['Not at all', 'Several days', 'More than half the days', 'Nearly every day']


class Subsession(BaseSubsession):
    pass


class Group(BaseGroup):
    pass


# Every field is blank=True so test runs can leave anything empty; for real
# participants, SurveyPage.error_message requires all but OPTIONAL_FIELDS.
def _radio(label, choices):
    return models.StringField(label=label, choices=choices, widget=widgets.RadioSelect, blank=True)


def _scale(label, n):
    return models.IntegerField(
        label=label, choices=list(range(1, n + 1)), widget=widgets.RadioSelectHorizontal, blank=True
    )


def _text(label):
    return models.LongStringField(label=label, blank=True)


def _aq(label):
    return _radio(label, C.AQ)


def _bis(label):
    return _radio(label, C.BIS)


def _gad(label):
    return _radio(label, C.GAD)


class Player(BasePlayer):
    # --- The game -----------------------------------------------------------
    interface_ease = _radio('How easy was it to understand the game interface?', C.EASE)
    interface_clear = _text('What aspects of the interface were clear and easy to use?')
    interface_confusing = _text('What aspects of the interface were confusing or unclear?')
    instructions_clear = _radio('Were the game instructions clear and sufficient?', C.CLARITY)
    instructions_missing = _text(
        'What additional instructions or information would have helped you understand the game better?'
    )
    feedback_clarity = _scale('How clear was the feedback you received after each round?', 5)
    reward_rule_clear = _radio(
        'Was the underlying rule to receive reward on each round clear to you?',
        ['Yes', 'Somewhat', 'No'],
    )
    earnings_display = _radio(
        'Would you prefer to see your earnings displayed as:',
        [
            'Points only (e.g., "you won 10 points!")',
            'Money only (e.g., "you won £0.50!")',
            'Both points and money conversion',
            "No preference/wouldn't make a difference",
        ],
    )
    pupil_headset = _radio('Did you wear the pupillometry headset?', ['Yes', 'No'])

    # --- Parts 1 and 2, opponent ------------------------------------------
    noticed_two_parts = _radio('Did you notice that the game had two distinct parts?', C.YES_NO_UNSURE)
    two_parts_cue = _text('If yes, what made it clear that Parts 1 and 2 were different?')
    part2_opponent_belief = _radio(
        'In Part 2, who did you think you were playing against?',
        ['A computer algorithm', 'Another participant', 'Not sure'],
    )
    part2_opponent_confidence = _scale(
        'How confident are you in your answer to the previous question?', 5
    )
    opponent_difference = _radio(
        "Did your opponent's behaviour feel different in Part 1 compared to Part 2?",
        ['No difference', 'Slight difference', 'Moderate difference', 'Strong difference'],
    )

    # --- Performance and strategy -----------------------------------------
    performance_overall = _radio('How well do you think you performed in the game overall?', C.PERFORMANCE)
    performance_part1 = _radio('How well do you think you performed in Part 1 of the game?', C.PERFORMANCE)
    performance_part2 = _radio('How well do you think you performed in Part 2 of the game?', C.PERFORMANCE)
    had_strategy = _radio('Did you develop a strategy whilst playing?', C.YES_NO_UNSURE)
    strategy_description = _text('If yes, can you briefly describe your strategy?')
    predicted_opponent = _radio(
        'Did you try to predict what your opponent would do?',
        ['Never', 'Rarely', 'Sometimes', 'Often', 'Always'],
    )
    adapted_strategy = _radio(
        'Do you feel like you adapted your strategy during the game?',
        ['Not at all', 'Slightly', 'Moderately', 'Quite a bit', 'Extensively'],
    )

    # --- Experience --------------------------------------------------------
    engagement = _scale('How engaging did you find the game?', 10)
    mental_demand = _scale('How mentally demanding was the task?', 10)
    engagement_change = _radio(
        'Did the game become more or less engaging as the game progressed?',
        [
            'Much less engaging',
            'Slightly less engaging',
            'Stayed the same',
            'Slightly more engaging',
            'Much more engaging',
        ],
    )
    felt_lost = _text('Were there any points where you felt lost or confused during the game?')
    technical_problems = _text('Did you experience any technical problems during the game?')
    suggestions = _text(
        'What did you like the most/least about the game? Do you have any suggestions for improving this set-up?'
    )
    handedness = _radio('Are you left-handed or right-handed?', ['Left-handed', 'Right-handed'])

    # --- AQ-10 -------------------------------------------------------------
    aq_01 = _aq('I often notice small sounds when others do not')
    aq_02 = _aq('I usually concentrate more on the whole picture, rather than the small details')
    aq_03 = _aq('I find it easy to do more than one thing at once')
    aq_04 = _aq('If there is an interruption, I can switch back to what I was doing very quickly')
    aq_05 = _aq("I find it easy to 'read between the lines' when someone is talking to me")
    aq_06 = _aq(
        'I like to collect information about categories of things '
        '(e.g. types of car, types of bird, types of train, types of plant, etc.)'
    )
    aq_07 = _aq('I know how to tell if someone listening to me is getting bored')
    aq_08 = _aq("When I'm reading a story I find it difficult to work out the characters' intentions")
    aq_09 = _aq('I find it easy to work out what someone is thinking or feeling just by looking at their face')
    aq_10 = _aq("I find it difficult to work out people's intentions")

    # --- BIS-Brief -----------------------------------------------------------
    bis_01 = _bis('I plan tasks carefully')
    bis_02 = _bis('I concentrate easily')
    bis_03 = _bis("I don't pay attention")
    bis_04 = _bis('I say things without thinking')
    bis_05 = _bis('I am a careful thinker')
    bis_06 = _bis('I act on the spur of the moment')
    bis_07 = _bis('I am self-controlled')
    bis_08 = _bis('I do things without thinking')

    # --- GAD-7 ---------------------------------------------------------------
    gad_01 = _gad('Feeling nervous, anxious or on edge')
    gad_02 = _gad('Not being able to stop or control worrying')
    gad_03 = _gad('Worrying too much about different things')
    gad_04 = _gad('Trouble relaxing')
    gad_05 = _gad('Being so restless that it is hard to sit still')
    gad_06 = _gad('Becoming easily annoyed or irritable')
    gad_07 = _gad('Feeling afraid, as if something awful might happen')
    gad_difficulty = models.StringField(
        label=(
            'If you checked any problems, how difficult have they made it for you to do your work, '
            'take care of things at home, or get along with other people?'
        ),
        choices=['Not difficult at all', 'Somewhat difficult', 'Very difficult', 'Extremely difficult'],
        widget=widgets.RadioSelect,
        blank=True,
    )


class PaymentDetails(ExtraModel):
    """One row per submission of the payment form. `ciphertext` is the encrypted
    JSON of payment_crypto.FIELDS; the plain text is never stored. Its own table, so
    it stays out of oTree's standard exports and is created automatically on deploy."""
    player = models.Link(Player)
    participant_code = models.StringField()
    ciphertext = models.LongStringField()
    submitted_at = models.FloatField()


# free-text answers, and gad_difficulty (only asked if a GAD-7 problem was ticked)
OPTIONAL_FIELDS = {
    'interface_clear', 'interface_confusing', 'instructions_missing', 'two_parts_cue',
    'strategy_description', 'felt_lost', 'technical_problems', 'suggestions', 'gad_difficulty',
}


def survey_optional(player):
    """Test runs show the questionnaire and payment form but let you skip every
    question: session config skip_survey (lab-notes sets it; the name predates
    the pages being shown)."""
    return bool(player.session.config.get('skip_survey', False))


def page_vars(player, heading, intro=''):
    return dict(heading=heading, intro=intro, survey_optional=survey_optional(player))


# PAGES
class SurveyPage(Page):
    form_model = 'player'
    template_name = 'mp_survey/SurveyPage.html'

    @staticmethod
    def error_message(player, values):
        if survey_optional(player):
            return None
        missing = {
            f: 'Please answer this question.'
            for f, v in values.items()
            if f not in OPTIONAL_FIELDS and v in (None, '')
        }
        return missing or None


class AboutTheGame(SurveyPage):
    form_fields = [
        'interface_ease', 'interface_clear', 'interface_confusing',
        'instructions_clear', 'instructions_missing',
        'feedback_clarity', 'reward_rule_clear', 'earnings_display', 'pupil_headset',
    ]

    @staticmethod
    def vars_for_template(player):
        return page_vars(player, 'About the game')


class PartsAndOpponent(SurveyPage):
    form_fields = [
        'noticed_two_parts', 'two_parts_cue',
        'part2_opponent_belief', 'part2_opponent_confidence', 'opponent_difference',
    ]

    @staticmethod
    def vars_for_template(player):
        return page_vars(player, 'The two parts of the game')


class StrategyAndExperience(SurveyPage):
    form_fields = [
        'performance_overall', 'performance_part1', 'performance_part2',
        'had_strategy', 'strategy_description', 'predicted_opponent', 'adapted_strategy',
        'engagement', 'mental_demand', 'engagement_change',
        'felt_lost', 'technical_problems', 'suggestions', 'handedness',
    ]

    @staticmethod
    def vars_for_template(player):
        return page_vars(player, 'Your strategy and experience')


class AQ10(SurveyPage):
    form_fields = [f'aq_{i:02d}' for i in range(1, 11)]

    @staticmethod
    def vars_for_template(player):
        return page_vars(
            player,
            heading='About you (1 of 3)',
            intro='Please indicate how much you agree with each statement.',
        )


class BISBrief(SurveyPage):
    form_fields = [f'bis_{i:02d}' for i in range(1, 9)]

    @staticmethod
    def vars_for_template(player):
        return page_vars(
            player,
            heading='About you (2 of 3)',
            intro='People differ in the ways they act and think in different situations. '
                  'Please indicate how often each statement applies to you.',
        )


class GAD7(SurveyPage):
    form_fields = [f'gad_{i:02d}' for i in range(1, 8)] + ['gad_difficulty']

    @staticmethod
    def vars_for_template(player):
        return page_vars(
            player,
            heading='About you (3 of 3)',
            intro='Over the last 2 weeks, how often have you been bothered by the following problems?',
        )


def payment_form_url(player):
    """Payment form link with this participant's code filled in.

    The configured URL may contain the placeholder {participant_code} (e.g. a
    Google Forms pre-filled link), so the participant never has to type it.
    """
    url = player.session.config.get('payment_form_url', '')
    return url.replace('{participant_code}', player.participant.code)


def payment_public_key():
    return os.environ.get('PAYMENT_PUBLIC_KEY', '').strip()


def email_on_file(player):
    """The participant's email, encrypted with PAYMENT_PUBLIC_KEY when the consent
    app emailed them the information sheet; '' if they weren't asked there."""
    return player.participant.vars.get('email_ciphertext', '')


def save_payment_details(player, data):
    """Live method of the Payment page: validate, encrypt, store. Replies with
    the field errors, or ok. Plain-text details are never stored or logged."""
    details, errors = clean_details(data, email_on_file=bool(email_on_file(player)))
    if errors:
        return dict(ok=False, errors=errors)
    ciphertext = encrypt_details(details, payment_public_key())
    PaymentDetails.create(
        player=player,
        participant_code=player.participant.code,
        ciphertext=ciphertext,
        submitted_at=time.time(),
    )
    player.participant.vars['payment_details_submitted'] = True
    # the latest submission, for lab-notes (REST get_session returns participant vars only);
    # still ciphertext, like email_ciphertext
    player.participant.vars['payment_ciphertext'] = ciphertext
    return dict(ok=True)


class Payment(Page):
    @staticmethod
    def vars_for_template(player):
        # progress for lab-notes (same record matching_live keeps up to date)
        player.participant.vars["progress"] = dict(
            player.participant.vars.get("progress") or {}, stage="finished", t=round(time.time())
        )
        return dict(
            participant_code=player.participant.code,
            collect_payment_details=bool(payment_public_key()),
            email_on_file=bool(email_on_file(player)),
            survey_optional=survey_optional(player),
            payment_form_url=payment_form_url(player),
        )

    @staticmethod
    def live_method(player, data):
        if not payment_public_key():
            return {player.id_in_group: dict(ok=False, errors=dict(form='Payment form is not set up.'))}
        return {player.id_in_group: save_payment_details(player, data)}


page_sequence = [AboutTheGame, PartsAndOpponent, StrategyAndExperience, AQ10, BISBrief, GAD7, Payment]


def custom_export(players):
    """Encrypted payment submissions, for payment_crypto.py's decrypt step.
    Ciphertext only: unreadable without the private key."""
    yield [
        'session_code', 'participant_code', 'participant_label', 'test_run', 'submitted_at',
        'ciphertext', 'email_ciphertext',
    ]
    for p in players:
        for row in PaymentDetails.filter(player=p):
            yield [
                p.session.code, row.participant_code, p.participant.label or '',
                int(survey_optional(p)), row.submitted_at, row.ciphertext, email_on_file(p),
            ]
