# Pupil bridge

Small local HTTP server that forwards annotations from the oTree pages in the
browser to Pupil Capture. It runs on the **participant laptop** (the machine with
the Pupil headset and Pupil Capture), not on the oTree server.

```
browser (Game.html)  --POST http://127.0.0.1:8765/annotation-->  pupil_bridge.py
pupil_bridge.py      --ZMQ tcp://127.0.0.1:50020-->              Pupil Capture
```

Each annotation is timestamped with Pupil Capture's clock when the bridge
receives it. All JSON fields sent by the page are stored on the annotation and
end up in the recording's `annotation.pldata` (and `annotations.csv` after
export in Pupil Player).

## Setup (once per laptop)

From a terminal where your Python is available (e.g. Anaconda Prompt):

```
pip install -r pupil_bridge/requirements.txt
```

## Running a session

1. Start Pupil Capture (Pupil Remote must be enabled, port 50020 is the default).
2. Start the bridge from the repository root:

   ```
   python pupil_bridge/pupil_bridge.py
   ```

   It should print `Connected to Pupil Capture PUB port ...`.
3. Start recording in Pupil Capture.
4. Open the oTree participant link in a browser **on the same laptop**.

Annotations appear in the Pupil Capture log as e.g.
`annotations: trial_start annotation`.

## Pages that send annotations

- `matching_live/Game.html`
- `matching_retreat_1/Game.html`
- `pupil_sync_test/Stimulus.html`
