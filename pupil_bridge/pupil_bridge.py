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
import statistics
import sys
import threading
import time
from collections import deque
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import msgpack
import zmq

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
        self.warned = False  # warned that Pupil Capture is unreachable, since the last success

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
                self.warned = False
                return result
            except zmq.ZMQError as e:
                # a timed-out REQ socket can't be reused, so start over next time
                self._close()
                self.last_error = f"Pupil Capture not reachable at {self.host}:{self.port} ({e})"
                if not self.warned:
                    self.warned = True
                    print(f"WARNING: Pupil Capture is not running, or Pupil Remote is not on "
                          f"port {self.port}. Annotations are not being recorded. The bridge "
                          f"keeps trying and connects as soon as Pupil Capture is up.", flush=True)
                raise PupilUnavailable("Pupil Capture not reachable") from e

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


class Bridge:
    """HTTP side of the bridge: request handling, stats and the backup log."""

    def __init__(self, pupil):
        self.pupil = pupil
        self.sent = 0
        self.receive_fallbacks = 0
        self.latencies_ms = deque(maxlen=500)
        LOG_DIR.mkdir(exist_ok=True)
        self.log_path = LOG_DIR / f"annotations_{datetime.now():%Y-%m-%d_%H%M%S}.jsonl"
        self.lock = threading.Lock()

    def clock(self):
        try:
            return 200, {"pupil_time": self.pupil.time()}
        except PupilUnavailable as e:
            return 503, {"error": str(e)}

    def annotation(self, body):
        try:
            data = json.loads(body)
        except ValueError:
            data = None
        if not isinstance(data, dict):
            return 400, {"error": "expected a JSON object"}
        try:
            sent = self.pupil.time_and_publish(lambda t: build_annotation(dict(data), t))
        except PupilUnavailable as e:
            print(f"NOT SENT ({e}): {data.get('label')}", flush=True)
            return 503, {"error": str(e)}

        with self.lock:
            self.sent += 1
            if sent["timestamp_source"] == "receive":
                self.receive_fallbacks += 1
            else:
                self.latencies_ms.append(sent["latency_ms"])
            with self.log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(sent, default=str) + "\n")
        print(f"{sent['label']:<28} {sent['timestamp_source']:<7} "
              f"latency {sent['latency_ms']:7.1f} ms", flush=True)
        return 200, {"status": "ok", "label": sent["label"], "timestamp": sent["timestamp"]}

    def status(self):
        try:
            pupil_time = self.pupil.time()
        except PupilUnavailable:
            pupil_time = None
        latencies = list(self.latencies_ms)
        return 200, {
            "pupil_connected": self.pupil.connected,
            "pupil_time": pupil_time,
            "pub_port": self.pupil.pub_port,
            "last_error": self.pupil.last_error,
            "annotations_sent": self.sent,
            "receive_time_fallbacks": self.receive_fallbacks,
            "latency_ms_median": statistics.median(latencies) if latencies else None,
            "latency_ms_max": max(latencies) if latencies else None,
            "log_file": str(self.log_path),
        }


def make_handler(bridge):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, status, payload=None):
            body = b"" if payload is None else json.dumps(payload).encode()
            self.send_response(status)
            # the page is served from the oTree server (e.g. Heroku), so every
            # response needs CORS headers, and Chrome asks before a public site
            # may reach a local address
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            if self.headers.get("Access-Control-Request-Private-Network"):
                self.send_header("Access-Control-Allow-Private-Network", "true")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_OPTIONS(self):
            self._reply(204)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/clock":
                self._reply(*bridge.clock())
            elif path == "/status":
                self._reply(*bridge.status())
            else:
                self._reply(404, {"error": "not found"})

        def do_POST(self):
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            if self.path.split("?")[0] == "/annotation":
                self._reply(*bridge.annotation(body))
            else:
                self._reply(404, {"error": "not found"})

        def log_message(self, format, *args):
            pass  # one line per annotation is enough

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # default 5 refuses bursts of simultaneous requests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the page (default 8765)")
    parser.add_argument("--pupil-host", default="127.0.0.1")
    parser.add_argument("--pupil-port", type=int, default=50020, help="Pupil Remote port (default 50020)")
    args = parser.parse_args(argv)

    pupil = PupilConnection(args.pupil_host, args.pupil_port)
    try:
        pupil.time()
    except PupilUnavailable:
        pass  # already warned; it connects on a later request

    server = Server(("127.0.0.1", args.port), make_handler(Bridge(pupil)))
    print(f"Pupil bridge on http://127.0.0.1:{args.port}  (status: /status, stop: Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
