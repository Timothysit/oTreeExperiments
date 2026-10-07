from os import environ


SESSION_CONFIGS = [
    dict(
        name='matching_live',
        display_name='Matching Pennies (live)',
        app_sequence=['matching_live', 'mp_survey'],
        num_demo_participants=2,
        num_trials_single=400,
        num_trials_multi=400,
    ),
    dict(
        name='matching_live_solo',
        display_name='Matching Pennies (solo — no-switch control)',
        app_sequence=['matching_live', 'mp_survey'],
        num_demo_participants=1,
        num_trials_single=400,
        num_trials_multi=400,
        # One participant per group; Part 2 is played against the SAME algo A
        # instance as Part 1 (memory carries across the block boundary), so the
        # only thing that changes at the boundary is the break/Part-2 framing.
        solo=True,
        algoA_trials_back=4,
    ),
    dict(
        # bot test only (otree test matching_live_preset_test): opponents per laptop label set
        # when the session is created, as lab-notes does, so the Setup page is skipped; and a
        # test run's skip_survey
        name='matching_live_preset_test',
        display_name='(test) Matching Pennies, opponents set at creation, survey optional',
        app_sequence=['matching_live', 'mp_survey'],
        num_demo_participants=2,
        num_trials_single=30,
        num_trials_multi=30,
        opponent_laptopA='algo_A',
        opponent_laptopB='algo_B',
        skip_survey=True,
    ),
    dict(
        name='guess_two_thirds',
        display_name="Guess 2/3 of the Average",
        app_sequence=['guess_two_thirds', 'payment_info'],
        num_demo_participants=3,
    ),
    dict(
        name='survey', app_sequence=['survey', 'payment_info'], num_demo_participants=1
    ),
    dict(
        name='matching_pennies', app_sequence=['mp'], num_demo_participants=1
    ),
    dict(
        name="pupil_sync_test",
        display_name="Pupil Sync Test",
        app_sequence=["pupil_sync_test"],
        num_demo_participants=1,
    ),
    dict(
        name="matching_retreat_1",
        display_name="MP Real time (two players)",
        app_sequence=["matching_retreat_1"],
        num_demo_participants=2,
        game_mode="multi",
        num_trials_multi=400,
    ),
    dict(
        name="matching_retreat_1_single",
        display_name="MP Real time (single player)",
        app_sequence=["matching_retreat_1"],
        num_demo_participants=1,
        game_mode="single",
        single_opponent="follow",
        follow_reaction_time_pct=100,
        follow_speed_pct=80,
        anticipate_warmup_trials=5,
        num_trials_single=400,
    ),
    dict(
        name="matching_blocks",
        display_name="Matching Blocks",
        app_sequence=["matching_blocks"],
        num_demo_participants=2,
    ),
    dict(
        name="matching_payoff_risk",
        display_name="Risky choice matching (two players)",
        app_sequence=["matching_payoff_risk"],
        num_demo_participants=2,
        game_mode="multi",
        num_trials_multi=100,
        safe_side="L",
        safe_prob_pct=100,
        safe_payoff=10,
        risky_prob1_pct=50,
        risky_payoff1=20,
        risky_prob2_pct=0,
        risky_payoff2=0,
    ),
    dict(
        name="matching_payoff_risk_single",
        display_name="Risky choice matching (single player)",
        app_sequence=["matching_payoff_risk"],
        num_demo_participants=1,
        game_mode="single",
        single_opponent="random",
        num_trials_single=100,
        safe_side="L",
        safe_prob_pct=100,
        safe_payoff=10,
        risky_prob1_pct=50,
        risky_payoff1=20,
        risky_prob2_pct=0,
        risky_payoff2=0,
    ),
]

# if you set a property in SESSION_CONFIG_DEFAULTS, it will be inherited by all configs
# in SESSION_CONFIGS, except those that explicitly override it.
# the session config can be accessed from methods in your apps as self.session.config,
# e.g. self.session.config['participation_fee']

# Set once here; every session uses it (can still be overridden per session).
PAYMENT_FORM_URL = ""

SESSION_CONFIG_DEFAULTS = dict(
    real_world_currency_per_point=1.00, participation_fee=0.00, doc="",
    # mp_survey collects payment details itself, encrypted, when the env var
    # PAYMENT_PUBLIC_KEY is set (see mp_survey/payment_crypto.py). Without it, its
    # last page links this external payment-only form instead.
    # {participant_code} is replaced with the participant's oTree code, so use a
    # Google Forms pre-filled link with that as the participant-ID answer.
    payment_form_url=PAYMENT_FORM_URL,
    # mp_lab laptops with a Pupil headset (comma-separated labels); only these send
    # annotations to the pupil bridge, see pupil_bridge/README.md
    pupil_bridge_labels="laptopA",
    # free space the eye tracker check page asks for on the recordings drive; a full
    # 400+400-trial session recorded 21-28 GB (May-July 2026), plus a margin
    pupil_min_free_gb=35,
)

SESSION_CONFIG_CHOICES = dict(
    game_mode=["single", "multi"],
    single_opponent=["follow", "adaptive_follower", "anticipate", "random", "predictive", "2ab"],
    single_opponent_p1=["random", "algo_A", "algo_B"],
    single_opponent_p2=["random", "algo_A", "algo_B"],
    safe_side=["L", "R"],
)

PARTICIPANT_FIELDS = []
SESSION_FIELDS = []

# ISO-639 code
# for example: de, fr, ja, ko, zh-hans
LANGUAGE_CODE = 'en'

# e.g. EUR, GBP, CNY, JPY
REAL_WORLD_CURRENCY_CODE = 'USD'
USE_POINTS = True

ROOMS = [
    dict(
        name='econ101',
        display_name='Econ 101 class',
        participant_label_file='_rooms/econ101.txt',
    ),
    dict(name='live_demo', display_name='Room for live demo (no participant labels)'),
    # The two testing laptops. Each keeps its own link open between participants:
    #   <server>/room/mp_lab/?participant_label=laptopA   (laptopB on the other)
    # A session created in this room (admin page, or lab-notes via POST /api/sessions with
    # room_name='mp_lab') starts on both at once, and the label records which laptop was used.
    dict(
        name='mp_lab',
        display_name='MP testing room (laptops A and B)',
        participant_label_file='_rooms/mp_lab.txt',
    ),
]

ADMIN_USERNAME = 'admin'
# for security, best to set admin password in an environment variable
ADMIN_PASSWORD = environ.get('OTREE_ADMIN_PASSWORD')

DEMO_PAGE_INTRO_HTML = """
Here are some oTree games.
"""


SECRET_KEY = '5432611698031'

INSTALLED_APPS = ['otree']

# Whether to show the debug info
DEBUG = 0
