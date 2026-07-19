"""Smoke tests for matching_live.

Run with:
    uv run otree test matching_live_solo
    uv run otree test matching_live

Exercises the full live_game loop (both blocks, the block boundary, and the
End page) for the solo no-switch control and the paired 2-player config.
"""
from os import environ

from otree.api import Bot, Submission, expect

from . import *

# kept small so the test runs fast; the real sessions use 400/400.
# Set MATCHING_LIVE_TEST_TRIALS=400 to rehearse a full-length session.
_n = int(environ.get("MATCHING_LIVE_TEST_TRIALS", 30))
TEST_TRIALS_SINGLE = _n
TEST_TRIALS_MULTI = _n


class PlayerBot(Bot):
    def play_round(self):
        if self.player.id_in_group == 1:
            yield Setup, dict(
                single_opponent_p1="algo_A",
                single_opponent_p2="algo_A",
                num_trials_single=TEST_TRIALS_SINGLE,
                num_trials_multi=TEST_TRIALS_MULTI,
            )
        # Game advances via JS/live messages, so there is no submit button
        yield Submission(Game, {}, check_html=False)

        total = TEST_TRIALS_SINGLE + TEST_TRIALS_MULTI
        expect(self.player.current_trial, total)


def _choice_for(pid, trial):
    # deterministic, mildly exploitable pattern so algo A actually gets reads
    return "L" if (trial + pid) % 3 else "R"


def call_live_method(method, group, page_class, **kwargs):
    if page_class.__name__ != "Game":
        return

    players = group.get_players()
    solo = len(players) == 1
    n_single = group.num_trials_single
    n_multi = group.num_trials_multi

    for p in players:
        method(p.id_in_group, {"type": "start"})

    for trial in range(n_single + n_multi):
        for p in players:
            method(
                p.id_in_group,
                {"type": "choice", "choice": _choice_for(p.id_in_group, trial), "rt_ms": 400},
            )

    log = json.loads(group.trial_log_json)
    expect(len(log), (n_single + n_multi) if solo else (n_single * len(players) + n_multi))

    single_rows = [r for r in log if r["block"] == "single"]
    multi_rows = [r for r in log if r["block"] == "multi"]
    expect(len(multi_rows), n_multi)

    if solo:
        # Part 2 must be algo A, and its memory must CARRY OVER from Part 1
        # rather than resetting at the boundary.
        expect(all(r["opponent_type"] == "algo_A" for r in multi_rows), True)
        expect(all(r.get("solo_control") is True for r in multi_rows), True)

        seen = [r["algo_n_trials_seen"] for r in log]
        expect(seen, list(range(len(log))))  # strictly continuous 0..N-1

        expect(all(r["algo_trials_back"] == 4 for r in log), True)
        expect(all(r["feedback_delay_ms"] is not None for r in multi_rows), True)
    else:
        expect(all(r["opponent_type"] == "human" for r in multi_rows), True)
        expect(len(single_rows), n_single * len(players))
