# Pupil bridge

Sends Pupil Capture annotations from the oTree pages to the headset laptop.
The oTree pages are served from Heroku, so the oTree server can't reach Pupil
Capture; the browser on the headset laptop can, through this small local server.

> Recordings made before this branch (May–July 2026) have known timing and
> trial-numbering issues, see [RECORDINGS_2026-05_TO_07.md](RECORDINGS_2026-05_TO_07.md).

```
Game.html + _static/pupil_bridge.js  --HTTP 127.0.0.1:8765-->  pupil_bridge.py
pupil_bridge.py                      --ZMQ  127.0.0.1:50020->  Pupil Capture
```

## Running a session (headset laptop)

The bridge starts by itself at Windows login (see Setup) and runs in the
background. For a `matching_live` session:

1. Plug in the headset and open the room link,
   e.g. `.../room/mp_lab/?participant_label=laptopA`.
2. **Eye tracker check** (first page, headset laptop only, for the experimenter):
   it starts Pupil Capture if it isn't running and shows live confidence, pupil
   size and samples/s for each eye. With the headset on the participant, switch
   to Pupil Capture (Alt+Tab) and adjust each eye camera until the pupil is
   tracked (confidence above 0.8, green), then click **Eye cameras OK, continue**.
3. The participant's **Click anywhere to begin** starts the recording. The bridge
   checks that both eye cameras deliver pupil data and starts a recording named
   `<date>_<room label>_<participant code>`,
   e.g. `recordings\2026-10-05_laptopA_abc123de\000`. The game starts once Pupil
   Capture confirms it is recording. If Pupil Capture had to be started first,
   the page asks for one more click.
4. After the last trial's feedback the recording stops, before the survey.
   (The End page stops it too, in case that didn't get through.)

If the recording can't be started, the page says **The eye tracker is not
recording. Please call the experimenter**, with the reason, and the game does
not start; clicking again retries. Common reasons: headset unplugged, an eye
window closed, Pupil Capture already recording (stop it by hand), or the
bridge not running (start it with the "Pupil bridge" shortcut in the Startup
folder, or `python pupil_bridge\pupil_bridge.py`).

Recording stays on during the Part 1/2 break. A page reload doesn't start a
second recording.

To check the bridge: <http://127.0.0.1:8765/status> shows whether it is
connected and recording. Its output goes to `pupil_bridge/logs/bridge_*.log`
when it runs in the background, or the console when started by hand.

The first time a page from the Heroku site contacts the bridge, Chrome may ask
whether the site may access devices on your local network. Allow it once
on the headset laptop.

## Setup (once per laptop)

The bridge only needs `pyzmq` and `msgpack`
(`pip install -r pupil_bridge/requirements.txt`), which oTree's environment
already has, so the same environment can run both (see below for creating it).
Then, with that environment's Python, add the bridge to Windows startup:

```
python pupil_bridge\install_startup.py
```

This puts a "Pupil bridge" shortcut in your Startup folder that runs this
repository's bridge with `pythonw` (no window). Double-click it to start the
bridge now; `--remove` takes it out of Startup again. Only one bridge can run
at a time: a second one exits with "Port 8765 is in use".

The bridge must be run from this repository: an older copy elsewhere
(e.g. `Documents\pupil_bridge`) is the previous version without these fixes.

## Testing locally with the pupil sync test

To try changes before they are on Heroku, run oTree on the headset laptop.
In an Anaconda Prompt, once:

```
conda create -n otree python=3.12 -y
conda activate otree
pip install otree==5.11.4 numpy scipy -r pupil_bridge/requirements.txt
```

Then, from the repository root in an Anaconda Prompt after
`conda activate otree` (the bridge is already running from Startup):

```
otree devserver
```

For the sync test, start recording in Pupil Capture by hand (only
`matching_live` controls recording), open <http://localhost:8000/demo/pupil_sync_test>,
click the session-wide link, go full screen (F11) and click **Start sync test**
(about 42 s of black/white screens).

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

## Validation (2026-10-05)

Pupil sync test run locally on the headset laptop with the new bridge, with
the headset turned to face the screen so the world camera filmed the
black/white changes (recording `recordings\2026_10_05\000`).

- All 8 annotations arrived, all with `timestamp_source = page`, delivered
  2–8 ms after the event, with the participant/session fields.
- World video (31 fps, so a resolution of about 32 ms): stimuli 1–2 are
  unusable (headset still being moved). For stimuli 3–6 the last unchanged
  frame is 43–54 ms and the first changed frame 8–18 ms *before* the
  annotation timestamp, the same frame slot every time. Two of those first
  frames are only partly changed, i.e. caught mid-change.

| stimulus | last unchanged frame | first changed frame |
|---|---|---|
| 3 (black) | −45 ms | −13 ms (partly changed) |
| 4 (white) | −43 ms | −11 ms |
| 5 (black) | −54 ms | −18 ms (partly changed) |
| 6 (white) | −43 ms | −8 ms |

Conclusions:

- **Jitter** between annotations and the world video is below one camera
  frame (32 ms), the resolution of this test.
- **Offset:** onsets are stamped at the browser frame that draws the change,
  so the screen cannot change before the annotation time. The world camera
  showing it 10–45 ms earlier therefore suggests the world video's timestamps
  run late by at least that much (e.g. stamped on frame arrival rather than
  exposure). Not confirmed; a photodiode on the screen would measure absolute
  timing. For pupil-size analysis this is negligible next to the pupil light
  response latency (> 200 ms); it matters when aligning events to world video.
- The eye cameras faced the screen, so this run has no pupil data.
- In `pupil_sync_test`, each stimulus lasts 6–21 ms longer than its
  `duration_ms`, because the page waits for the annotation request before
  starting the timer. Onset timestamps are unaffected.
