# Cursor trajectory recording (`matching_live`)

Added 2026-07-24. Records the participant's virtual-cursor path during each
choice window and stores it on the server, one row per trial.

## What is recorded

During each trial's choice window (from when the cursor task becomes active
until a choice is submitted), the browser samples the virtual cursor at a fixed
wall-clock rate, buffers the samples, and ships the whole buffer to the server
in **one** `liveSend` per trial (one message per trial, not one per sample).

Each sample is `{ t_ms, x, y }`:

| field  | meaning                                                       |
|--------|---------------------------------------------------------------|
| `t_ms` | ms since the choice window opened (0 = window start)          |
| `x`    | cursor x, normalized to viewport width (0 = left, 1 = right)  |
| `y`    | cursor y, normalized to viewport height (0 = top, 1 = bottom) |

`x`/`y` are normalized (not pixels) so traces compare across screen sizes, and
rounded to 4 decimals. Velocity/speed are derivable from consecutive samples, so
they are not stored.

## Sampling rate

`CURSOR_SAMPLE_INTERVAL_MS` in `Game.html`, default **50 ms = 20 Hz**. Single
named constant near the top of the recording block.

## Storage: an ExtraModel (NOT a Group field)

Traces live in a dedicated `ExtraModel` table, **not** a new column on `Group`:

```python
class CursorTrace(ExtraModel):
    group = models.Link(Group)
    player = models.Link(Player)
    participant_code = models.StringField()
    id_in_group = models.IntegerField()
    overall_trial = models.IntegerField()
    phase = models.StringField()
    block_trial = models.IntegerField()
    n_samples = models.IntegerField()
    samples_json = models.LongStringField()   # JSON list of {t_ms, x, y}
    server_ts = models.FloatField()
```

The `cursor_trace` live message creates one `CursorTrace` row per trial.

### Why ExtraModel and not a Group column — this is the important part

oTree's startup (`init_orm` -> SQLAlchemy `metadata.create_all`) **creates
missing tables but never adds missing columns** to existing tables, and there is
no auto-migration on `prodserver`.

- A **new column** on `Group` requires the physical column to already exist.
  On the Heroku **Postgres** DB (prodserver, no in-memory copy) that column is
  missing after a deploy, so `INSERT` fails at session creation until you run a
  manual `ALTER TABLE`. Getting that wrong tempts a `resetdb`, which **wipes all
  data**.
- A **new table** (this ExtraModel) is created automatically by `create_all` on
  the next server boot / deploy. No `ALTER`, no `resetdb`, existing rows
  untouched. Verified locally: booting after adding `CursorTrace` created
  `matching_live_cursortrace` with all columns and left the existing 14 groups /
  27 players intact.

Extra benefit: one insert per trial instead of re-serializing a growing JSON
blob every trial (no O(n^2) write cost).

## Getting the data out

`custom_export(players)` (bottom of `__init__.py`) flattens the traces to **one
CSV row per sample** (`session_code, participant_code, id_in_group, phase,
overall_trial, block_trial, n_samples, server_ts, sample_index, t_ms, x, y`).
Download it from the oTree admin under the app's custom/per-app export, or query
the `matching_live_cursortrace` table directly.

## Deploying to Heroku (safe procedure, no data loss)

Because this is a new table, the deploy itself is enough:

1. `git push heroku main` (or your normal deploy).
2. Restart is automatic; on boot `create_all` creates `matching_live_cursortrace`.
3. **Do NOT run `otree resetdb`** (that would drop all tables/data).

Optional but recommended before any prod change: `heroku pg:backups:capture`.

No `ALTER TABLE` is required for this feature. (Contrast: the earlier
Group-column version *did* require a manual Postgres `ALTER` — that is why it was
changed to an ExtraModel.)

## Code changes

**`__init__.py`**
- `CursorTrace(ExtraModel)` class added.
- `msg_type == "cursor_trace"` branch in `live_game` does `CursorTrace.create(...)`.
- `custom_export(players)` added.

**`Game.html`**
- `CURSOR_SAMPLE_INTERVAL_MS` constant + `startCursorRecording()` /
  `stopCursorRecordingAndSend()` helpers.
- `startCursorRecording()` when the choice window opens;
  `stopCursorRecordingAndSend()` inside `submitChoice()` before the `choice`
  message (so trace and choice stay paired; the F/J keyboard fallback routes
  through `submitChoice` too).
- Interval cleared defensively on `finished` and at the start of each recording.

## Local devserver note

`otree devserver` uses an in-memory SQLite DB and only writes to `db.sqlite3`
on a **graceful** shutdown (Ctrl-C). A hard kill or crash discards that
session's data. Stop the dev server cleanly if you want locally-collected trials
persisted. (Heroku/prodserver uses Postgres directly and is not affected by
this.)
