# Pupil bridge

Sends Pupil Capture annotations from the oTree pages to the headset laptop.
The oTree pages are served from Heroku, so the oTree server can't reach Pupil
Capture; the browser on the headset laptop can, through this small local server.

```
Game.html + _static/pupil_bridge.js  --HTTP 127.0.0.1:8765-->  pupil_bridge.py
pupil_bridge.py                      --ZMQ  127.0.0.1:50020->  Pupil Capture
```

## Running a session (headset laptop)

1. Start Pupil Capture (Pupil Remote enabled, port 50020 is the default).
2. Start the bridge from the repository root, in a terminal with your Python
   (e.g. Anaconda Prompt):

   ```
   python pupil_bridge/pupil_bridge.py
   ```

   It prints `Connected to Pupil Capture ...`. Order doesn't matter: if Pupil
   Capture isn't up yet, or is restarted, the bridge reconnects on the next request.
3. Start recording in Pupil Capture.
4. Open the room link, e.g. `.../room/mp_lab/?participant_label=laptopA`.

The bridge prints one line per annotation with its latency. To check it is
working, open <http://127.0.0.1:8765/status> in the browser.

The first time a page from the Heroku site contacts the bridge, Chrome may ask
whether the site may access devices on your local network. Allow it once
on the headset laptop.

Setup once per laptop: `pip install -r pupil_bridge/requirements.txt`

## Which laptops send annotations

The session config `pupil_bridge_labels` (default `"laptopA"`, see `settings.py`)
lists the room labels of laptops with a headset; other laptops never contact
the bridge. Participants without a label (demo links) always do. It can be
changed per session on the oTree "create session" page.

## What gets recorded

Every annotation carries `participant_code`, `participant_label`,
`session_code`, `player_id` and `app`, plus the page's own fields, so a
recording can be matched to the oTree data (`overall_trial` matches the trial
log). The bridge adds:

| field | meaning |
|---|---|
| `timestamp` | Pupil time of the event (see below) |
| `timestamp_source` | `page`: the time the event happened in the page; `receive`: the time the bridge received it (fallback when the page has no clock sync yet) |
| `receive_ts` | Pupil time the bridge received the annotation |
| `latency_ms` | `receive_ts - timestamp`, normally a few ms |
| `browser_ts` | the event's `performance.now()` in the page |
| `clock_sync_rtt_ms` | round trip of the clock sync the page used |

Timing: the page measures the offset between its `performance.now()` clock and
Pupil's clock through `GET /clock` (best of 10 round trips, repeated every 30 s)
and sends each event's time converted to Pupil time. The bridge only uses it
if it lies within 5 s before receipt, so a stale offset (e.g. after Pupil
Capture restarts) falls back to the receive time.

A field sent by the page that clashes with one the bridge sets (`topic`,
`timestamp`, ...) is kept as `page_<name>`.

Each annotation is also appended to `pupil_bridge/logs/annotations_<start time>.jsonl`
(not in git) as a backup.

## Adding annotations to a page

```html
<script src="{{ static 'pupil_bridge.js' }}"></script>
<script>
  PupilBridge.init(js_vars.pupil);
  PupilBridge.annotate("trial_start", {overall_trial: 12});
  PupilBridge.annotate("stim_onset", {}, onsetMs);  // event time as performance.now()
</script>
```

with `pupil=pupil_js_vars(player, app="...")` (from `pupil_bridge.context`)
in the page's `js_vars`. Used in `matching_live`, `matching_retreat_1` and
`pupil_sync_test`.
