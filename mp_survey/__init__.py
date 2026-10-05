import time

from otree.api import *


doc = """
Post-task questionnaire for matching_live, replacing the separate Google Form
("Course Evaluation (Responses)") used up to 2026-10.

Answers are stored on the Player, so they join to the game data by
participant.code automatically. Deliberately collects NO personal details:
payment details go in a separate payment-only form (session config
`payment_form_url`), keeping identities out of the research export.

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


def _radio(label, choices):
    return models.StringField(label=label, choices=choices, widget=widgets.RadioSelect)


def _scale(label, n):
    return models.IntegerField(
        label=label, choices=list(range(1, n + 1)), widget=widgets.RadioSelectHorizontal
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


def survey_skipped(player):
    """Test runs can skip the questionnaire: session config skip_survey (lab-notes sets it)."""
    return bool(player.session.config.get('skip_survey', False))


# PAGES
class SurveyPage(Page):
    form_model = 'player'
    template_name = 'mp_survey/SurveyPage.html'

    @staticmethod
    def is_displayed(player):
        return not survey_skipped(player)


class AboutTheGame(SurveyPage):
    form_fields = [
        'interface_ease', 'interface_clear', 'interface_confusing',
        'instructions_clear', 'instructions_missing',
        'feedback_clarity', 'reward_rule_clear', 'earnings_display', 'pupil_headset',
    ]

    @staticmethod
    def vars_for_template(player):
        return dict(heading='About the game', intro='')


class PartsAndOpponent(SurveyPage):
    form_fields = [
        'noticed_two_parts', 'two_parts_cue',
        'part2_opponent_belief', 'part2_opponent_confidence', 'opponent_difference',
    ]

    @staticmethod
    def vars_for_template(player):
        return dict(heading='The two parts of the game', intro='')


class StrategyAndExperience(SurveyPage):
    form_fields = [
        'performance_overall', 'performance_part1', 'performance_part2',
        'had_strategy', 'strategy_description', 'predicted_opponent', 'adapted_strategy',
        'engagement', 'mental_demand', 'engagement_change',
        'felt_lost', 'technical_problems', 'suggestions', 'handedness',
    ]

    @staticmethod
    def vars_for_template(player):
        return dict(heading='Your strategy and experience', intro='')


class AQ10(SurveyPage):
    form_fields = [f'aq_{i:02d}' for i in range(1, 11)]

    @staticmethod
    def vars_for_template(player):
        return dict(
            heading='About you (1 of 3)',
            intro='Please indicate how much you agree with each statement.',
        )


class BISBrief(SurveyPage):
    form_fields = [f'bis_{i:02d}' for i in range(1, 9)]

    @staticmethod
    def vars_for_template(player):
        return dict(
            heading='About you (2 of 3)',
            intro='People differ in the ways they act and think in different situations. '
                  'Please indicate how often each statement applies to you.',
        )


class GAD7(SurveyPage):
    form_fields = [f'gad_{i:02d}' for i in range(1, 8)] + ['gad_difficulty']

    @staticmethod
    def vars_for_template(player):
        return dict(
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


class Payment(Page):
    @staticmethod
    def is_displayed(player):
        return not survey_skipped(player)

    @staticmethod
    def vars_for_template(player):
        # progress for lab-notes (same record matching_live keeps up to date)
        player.participant.vars["progress"] = dict(
            player.participant.vars.get("progress") or {}, stage="finished", t=round(time.time())
        )
        return dict(
            participant_code=player.participant.code,
            payment_form_url=payment_form_url(player),
        )


page_sequence = [AboutTheGame, PartsAndOpponent, StrategyAndExperience, AQ10, BISBrief, GAD7, Payment]
