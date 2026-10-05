"""Local bridge from the oTree pages in the browser to Pupil Capture.

Runs on the laptop with the headset. The page (via _static/pupil_bridge.js)
POSTs annotations here; they are forwarded to Pupil Capture over its ZMQ API
and also appended to a local JSONL log as a backup.

    GET  /clock       Pupil time now, used by the page to sync its clock
    POST /annotation  send one annotation
    GET  /status      connection state and timing stats
"""
import argparse
import json
import logging
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import msgpack
import zmq
from flask import Flask, jsonify, request

LOG_DIR = Path(__file__).resolve().parent / "logs"

# An event time from the page is only trusted if it falls in this window
# before the bridge received it; otherwise the receive time is used.
MAX_EVENT_AGE_S = 5.0
MAX_EVENT_AHEAD_S = 0.05

# Set by the bridge; a page field with one of these names is kept as page_<name>.
RESERVED_FIELDS = {"topic", "timestamp", "receive_ts", "latency_ms", "timestamp_source"}


class PupilConnection:
    """Pupil Remote (REQ) + IPC backbone (PUB) sockets, shared by the request
    threads. Connects lazily and reconnects after any failure, so the bridge
    can be started before Pupil Capture and survives a Pupil Capture restart."""

    def __init__(self, host, port, timeout_ms=1000):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.ctx = zmq.Context.instance()
        self.lock = threading.Lock()
        self.remote = None
        self.pub = None
        self.pub_port = None
        self.last_error = None

    def _close(self):
        for sock in (self.remote, self.pub):
            if sock is not None:
                sock.close(linger=0)
        self.remote = self.pub = self.pub_port = None

    def _request(self, *frames):
        for frame in frames[:-1]:
            self.remote.send(frame, flags=zmq.SNDMORE)
        self.remote.send(frames[-1])
        return self.remote.recv_string()

    def _connect(self):
        self.remote = self.ctx.socket(zmq.REQ)
        self.remote.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.remote.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.remote.setsockopt(zmq.LINGER, 0)
        self.remote.connect(f"tcp://{self.host}:{self.port}")

        self.pub_port = self._request(b"PUB_PORT")
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.connect(f"tcp://{self.host}:{self.pub_port}")

        notification = {"subject": "start_plugin", "name": "Annotation_Capture", "args": {}}
        self._request(b"notify.start_plugin", msgpack.dumps(notification, use_bin_type=True))

        # PUB connections drop messages sent before they are fully set up
        time.sleep(0.5)
        print(f"Connected to Pupil Capture (PUB port {self.pub_port})", flush=True)

    def _run(self, fn):
        with self.lock:
            try:
                if self.remote is None:
                    self._connect()
                result = fn()
                self.last_error = None
                return result
            except zmq.ZMQError as e:
                # a timed-out REQ socket can't be reused, so start over next time
                self._close()
                self.last_error = f"{type(e).__name__}: {e}"
                raise PupilUnavailable(self.last_error) from e

    def time(self):
        return self._run(lambda: float(self._request(b"t")))

    def time_and_publish(self, build_annotation):
        """Read Pupil time, build the annotation from it and publish it, all
        under one lock so no other request runs in between."""
        def run():
            annotation = build_annotation(float(self._request(b"t")))
            self.pub.send_string("annotation", flags=zmq.SNDMORE)
            self.pub.send(msgpack.dumps(annotation, use_bin_type=True))
            return annotation
        return self._run(run)

    @property
    def connected(self):
        return self.remote is not None


class PupilUnavailable(Exception):
    pass


def build_annotation(data, receive_ts):
    label = str(data.pop("label", "event"))
    event_ts = data.pop("event_pupil_ts", None)
    duration = data.pop("duration", 0.0)
    if not isinstance(duration, (int, float)):
        data["page_duration"] = duration
        duration = 0.0

    annotation = {"topic": "annotation", "label": label, "duration": float(duration)}
    for key, value in data.items():
        annotation[f"page_{key}" if key in RESERVED_FIELDS else key] = value

    # Prefer the time the page says the event happened (converted to Pupil time
    # with the clock offset it measured through /clock) over the time it got here.
    if (isinstance(event_ts, (int, float))
            and receive_ts - MAX_EVENT_AGE_S <= event_ts <= receive_ts + MAX_EVENT_AHEAD_S):
        annotation["timestamp"] = float(event_ts)
        annotation["timestamp_source"] = "page"
    else:
        annotation["timestamp"] = receive_ts
        annotation["timestamp_source"] = "receive"
    annotation["receive_ts"] = receive_ts
    annotation["latency_ms"] = round((receive_ts - annotation["timestamp"]) * 1000, 3)
    return annotation


def create_app(pupil):
    app = Flask(__name__)
    stats = {"sent": 0, "receive_fallbacks": 0, "latencies_ms": deque(maxlen=500)}
    LOG_DIR.mkdir(exist_ok=True)
    log_path = LOG_DIR / f"annotations_{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
    log_lock = threading.Lock()

    @app.after_request
    def add_cors_headers(response):
        # the page is served from the oTree server (e.g. Heroku), so every
        # response needs CORS headers, and Chrome asks before a public site
        # may reach a local address
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        if request.headers.get("Access-Control-Request-Private-Network"):
            response.headers["Access-Control-Allow-Private-Network"] = "true"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/clock")
    def clock():
        try:
            return jsonify(pupil_time=pupil.time())
        except PupilUnavailable as e:
            return jsonify(error=str(e)), 503

    @app.post("/annotation")
    def annotation():
        data = request.get_json(force=True, silent=True)
        if not isinstance(data, dict):
            return jsonify(error="expected a JSON object"), 400
        try:
            sent = pupil.time_and_publish(lambda t: build_annotation(dict(data), t))
        except PupilUnavailable as e:
            print(f"NOT SENT ({e}): {data.get('label')}", flush=True)
            return jsonify(error=str(e)), 503

        stats["sent"] += 1
        if sent["timestamp_source"] == "receive":
            stats["receive_fallbacks"] += 1
        else:
            stats["latencies_ms"].append(sent["latency_ms"])
        with log_lock, log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(sent, default=str) + "\n")
        print(f"{sent['label']:<28} {sent['timestamp_source']:<7} "
              f"latency {sent['latency_ms']:7.1f} ms", flush=True)
        return jsonify(status="ok", label=sent["label"], timestamp=sent["timestamp"])

    @app.get("/status")
    def status():
        try:
            pupil_time = pupil.time()
        except PupilUnavailable:
            pupil_time = None
        latencies = list(stats["latencies_ms"])
        return jsonify(
            pupil_connected=pupil.connected,
            pupil_time=pupil_time,
            pub_port=pupil.pub_port,
            last_error=pupil.last_error,
            annotations_sent=stats["sent"],
            receive_time_fallbacks=stats["receive_fallbacks"],
            latency_ms_median=statistics.median(latencies) if latencies else None,
            latency_ms_max=max(latencies) if latencies else None,
            log_file=str(log_path),
        )

    return app


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the page (default 8765)")
    parser.add_argument("--pupil-host", default="127.0.0.1")
    parser.add_argument("--pupil-port", type=int, default=50020, help="Pupil Remote port (default 50020)")
    args = parser.parse_args(argv)

    pupil = PupilConnection(args.pupil_host, args.pupil_port)
    try:
        pupil.time()
    except PupilUnavailable as e:
        print(f"Pupil Capture not reachable yet ({e}); will retry on each request.", flush=True)

    app = create_app(pupil)
    logging.getLogger("werkzeug").setLevel(logging.WARNING)  # one line per annotation is enough
    print(f"Pupil bridge on http://127.0.0.1:{args.port}  (status: /status)", flush=True)
    app.run(host="127.0.0.1", port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    sys.exit(main())
