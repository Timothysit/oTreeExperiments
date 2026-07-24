# Cursor trajectory recording (`matching_live`)

Added 2026-07-22. Records the participant's virtual-cursor path during each
choice window and stores it on the server. Ported from `matching_retreat_1`,
trimmed down to keep the data small (see "Data volume" below).

## What is recorded

During each trial's choice window (from the moment the cursor task becomes
active until a choice is submitted), the browser samples the virtual cursor at
a fixed wall-clock rate and buffers the samples locally. When the choice is
submitted, the whole buffer is shipped to the server in **one** `liveSend`
message (one message per trial, not one per sample).

Each sample is:

| field  | meaning                                                        |
|--------|----------------------------------------------------------------|
| `t_ms` | ms since the choice window opened (0 = window start)           |
| `x`    | cursor x, normalized to viewport width (0 = left, 1 = right)   |
| `y`    | cursor y, normalized to viewport height (0 = top, 1 = bottom)  |

`x`/`y` are normalized (not pixels) so traces are comparable across screen
sizes, and rounded to 4 decimals to save space. Velocity/speed are **not**
stored — they are trivially derivable from consecutive `(t_ms, x, y)` samples
in analysis.

## Sampling rate

Set by `CURSOR_SAMPLE_INTERVAL_MS` in `Game.html` (default **50 ms = 20 Hz**).
This is a single named constant near the top of the recording block — change it
there if you need finer/coarser sampling.

For reference, `matching_retreat_1` samples at 30 Hz and stores 9 fields per
sample; this app samples at 20 Hz and stores 3, so it produces roughly 5–6×
less data per trial. That matters here because a session can run up to
`NUM_TRIALS_SINGLE + NUM_TRIALS_MULTI` = 400 + 400 = 800 trials.

## Where the data lives

Stored on the **Group** model in `matching_live/__init__.py`:

```python
cursor_log_json = models.LongStringField(initial='[]')
```

It is reset to `"[]"` when the game starts. It holds a JSON list with one entry
per trial:

```json
{
  "player_id": 1,
  "participant_code": "abc123",
  "overall_trial": 12,
  "phase": "single",
  "block_trial": 12,
  "samples": [{"t_ms": 0, "x": 0.5, "y": 0.5}, ...],
  "server_ts": 1753200000.0
}
```

`cursor_log_json` is a standard group field, so it appears in oTree's normal
data export and can also be read directly from the DB / admin data view. (There
is no `custom_export` yet — if you want the trajectory flattened into tidy rows,
that can be added.)

## Code changes

**`__init__.py`**
- `Group.cursor_log_json` field added.
- Reset to `"[]"` in the `start` handler of `live_game`.
- New `msg_type == "cursor_trace"` branch in `live_game` that appends one entry
  (with player/trial context) to `cursor_log_json`.

**`Game.html`**
- `CURSOR_SAMPLE_INTERVAL_MS` constant + `startCursorRecording()` /
  `stopCursorRecordingAndSend()` helpers.
- `startCursorRecording()` called when the choice window opens
  (`waitingForChoice = true` in `startChoicePhase`).
- `stopCursorRecordingAndSend()` called inside `submitChoice()` before the
  `choice` message is sent, so the trace and choice stay paired. The F/J
  keyboard fallback also routes through `submitChoice`, so it is covered.
- Interval cleared defensively on `finished` and at the start of each recording
  so it can never double-sample.

## Implementation note / caveat

Like `matching_retreat_1`, each incoming trace is appended by parsing and
re-serializing the entire growing `cursor_log_json` string. At 20 Hz / 3 fields
over ~800 trials the total stays in the low hundreds of KB, which is fine. If
the rate is cranked up substantially or many fields are added back, that
per-trial re-serialization cost is the thing to watch.
