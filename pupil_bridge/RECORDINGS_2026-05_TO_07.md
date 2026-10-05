# Known issues in the May–July 2026 Pupil recordings

Applies to all recordings made with the old bridge and page code (before branch
`pupil-bridge`), i.e. everything in `C:\Users\tomha\recordings\` on the headset
laptop from 2026-05-12 to 2026-07-23. Checked on 2026-10-05 by reading every
recording's `annotation.pldata`.

The oTree data (trial log, cursor traces) is **not** affected: these issues are
only in the Pupil annotations. Matching an annotation to an oTree trial is where
they matter.

Summary, in order of how likely they are to affect an analysis:

1. [`trial_start.displayed_trial` is one trial behind](#1-trial_startdisplayed_trial-is-one-trial-behind)
2. [Some annotation timestamps are up to 2.6 s late](#2-some-annotation-timestamps-are-up-to-26-s-late)
3. [A few events are missing](#3-a-few-events-are-missing)
4. [Smaller labelling issues](#4-smaller-labelling-issues)

## Annotations recorded

Main task (`matching_live`, 400 + 400 trials), three annotations per trial, in
this order:

| label | fields | sent when |
|---|---|---|
| `trial_start` | `displayed_trial`, `phase` | choice window opens (2 s after the previous feedback, or after the break) |
| `choice_submitted` | `choice`, `rt_ms`, `cursor_x`, `cursor_y` | cursor reaches a target |
| `feedback_shown` | `block`, `block_trial`, `overall_trial`, `choice`, `reward`, `total_points` | reward shown |

All three also have `browser_ts` (the page's `performance.now()` in ms when the
event happened). The 23 July recording also has `solo_control` (0 throughout).
No annotation carries a participant or session code.

`2026_05_12/008` and `2026_05_15/002` are 10-trial pilots that used the label
`single_feedback_shown` instead of `feedback_shown`. `2026_05_15/003` is the
pupil sync test.

## 1. `trial_start.displayed_trial` is one trial behind

`displayed_trial` is the trial counter shown on screen. The counter is only
updated when feedback arrives, so while trial N is being played it still shows
the number of trial N−1:

| trial | `displayed_trial` |
|---|---|
| Part 1 trial 1 | 1 |
| Part 1 trial N (N ≥ 2) | N−1 |
| Part 2 trial 1 | **400** (last Part 1 trial) |
| Part 2 trial N (N ≥ 2) | N−1 |

So `displayed_trial` cannot tell Part 1 trial 1 from trial 2 (both show `1`),
and Part 2 trial 1 shows `400`. Participants also saw the lagging counter on
screen.

`choice_submitted` has no trial number at all.

**Check in your analysis:** if `trial_start` or `choice_submitted` were matched to
trials using `displayed_trial`, or by counting `trial_start` events, the match is
off by one trial (from trial 2 on), and the first Part 2 trial was probably
assigned to Part 1.

**Correct trial number:** an event belongs to the trial of the next
`feedback_shown` (its `block` and `block_trial`). Sort by corrected time first
(section 2), and allow for missing events (section 3).

## 2. Some annotation timestamps are up to 2.6 s late

The old bridge stamped each annotation with Pupil's clock when it *received* it,
not when it happened in the page. Usually that's about 1 ms later, but some
annotations arrived much later. A few even arrived after the next event, so in
time order `feedback_shown` comes before its `choice_submitted`.

Delay = Pupil `timestamp` minus the event's own time (`browser_ts`), relative to
the fastest deliveries in that recording:

| recording | median | 99th pct | max | > 200 ms | > 1 s |
|---|---|---|---|---|---|
| 2026_05_19/000 | 14 ms | 59 ms | 68 ms | 0 | 0 |
| 2026_05_19/001 | 4 ms | 57 ms | 900 ms | 4 | 0 |
| 2026_05_19/002 | 1 ms | 7 ms | 47 ms | 0 | 0 |
| 2026_05_19/003 | 1 ms | 8 ms | 654 ms | 2 | 0 |
| 2026_05_26/000 | 1 ms | 8 ms | 1661 ms | 9 | 3 |
| 2026_05_26/001 | 1 ms | 8 ms | 617 ms | 3 | 0 |
| 2026_05_28/000 | 1 ms | 7 ms | 1336 ms | 5 | 1 |
| 2026_05_28/001 | 1 ms | 7 ms | 14 ms | 0 | 0 |
| 2026_05_28/002 | 1 ms | 7 ms | 10 ms | 0 | 0 |
| 2026_05_29/000 | 1 ms | 8 ms | 26 ms | 0 | 0 |
| 2026_05_29/001 | 1 ms | 7 ms | 12 ms | 0 | 0 |
| 2026_06_04/000 | 1 ms | 8 ms | 1953 ms | 3 | 3 |
| 2026_06_04/001 | 1 ms | 7 ms | 10 ms | 0 | 0 |
| 2026_06_04/002 | 1 ms | 7 ms | 9 ms | 0 | 0 |
| 2026_07_23/000 | 1 ms | 10 ms | 2557 ms | 21 | 14 |

(Each recording has about 2,400 annotations.) The late annotations come in
clusters around single trials. A likely cause is the old bridge sharing one
Pupil connection between simultaneous requests.

**Check in your analysis:** epochs time-locked to an annotation's `timestamp`
(e.g. pupil size around feedback onset) are misaligned on those trials. That
mainly matters for 2026_05_26/000, 2026_05_28/000, 2026_06_04/000 and
2026_07_23/000.

**Correction:** `browser_ts` is accurate. The page's clock and Pupil's clock
differ by a constant within one page load: the drift is under 0.5 ms per hour.
So:

```python
import pandas as pd

# Pupil Player export (annotations.csv keeps the custom fields as columns),
# rows in the order they were recorded
a = pd.read_csv("exports/000/annotations.csv")

# browser_ts restarts from 0 when the page is reloaded (once in 2026_05_19/001
# and 2026_05_12/008), so estimate the clock offset per page load
page_load = (a["browser_ts"].diff() < -1000).cumsum()
offset = a["timestamp"] - a["browser_ts"] / 1000
# delays are never negative, so the fastest deliveries give the true offset
a["timestamp_corrected"] = a["browser_ts"] / 1000 + offset.groupby(page_load).transform(
    lambda o: o.quantile(0.01)
)
a["delay_ms"] = (a["timestamp"] - a["timestamp_corrected"]) * 1000
a = a.sort_values(["timestamp_corrected"])  # restores the true event order
```

Without a Pupil Player export, the same fields can be read from
`annotation.pldata` with `msgpack`.

## 3. A few events are missing

After restoring the true order (section 2), every trial has exactly
`trial_start`, `choice_submitted` and `feedback_shown`, except:

| recording | trial | missing |
|---|---|---|
| 2026_05_19/001 | Part 1 trial 106 | page reloaded during the trial: `choice_submitted` sent, then a new page load; check this trial in the oTree trial log |
| 2026_05_19/003 | Part 2 trial 185 | `trial_start` |
| 2026_05_26/000 | Part 1 trial 166 | `trial_start` |
| 2026_05_26/000 | Part 1 trial 167 | `feedback_shown` |
| 2026_05_26/000 | Part 1 trial 190 | `trial_start` |
| 2026_05_26/000 | Part 1 trial 400 | `feedback_shown` |
| 2026_05_28/000 | Part 2 trial 82 | `choice_submitted` |
| 2026_07_23/000 | Part 2 trial 330 | `feedback_shown` |

These are the same recordings that have the late timestamps, which points to
the same cause.

**Check in your analysis:** code that assumes three events per trial, or pairs
the k-th `trial_start` with the k-th `feedback_shown`, is shifted by one trial
after each gap.

## 4. Smaller labelling issues

- **`feedback_shown.overall_trial` in Part 2** is the trial number within Part 2
  (1–400), the same as `block_trial`, not 401–800. Use `block` + `block_trial`.
- **19 May recordings:** the first Part 2 `trial_start` has `phase = "single"`.
  From 26 May on it has `phase = "multi"`.
- **No participant or session code** in any annotation. Recordings can only be
  matched to oTree participants by date/time and lab notes.

## Fixed in the new code (branch `pupil-bridge`)

Every annotation now carries `participant_code`, `session_code`,
`overall_trial` and `block_trial` (taken from the server's messages, not the
on-screen counter). It is stamped with the time it happened in the page, and
the bridge handles simultaneous requests safely. See [README.md](README.md).
The on-screen counter still lags by one trial; that is a separate decision.
