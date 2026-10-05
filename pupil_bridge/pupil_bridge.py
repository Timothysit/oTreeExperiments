"""Local bridge from the oTree pages in the browser to Pupil Capture.

Runs on the laptop with the headset. The page (via _static/pupil_bridge.js)
POSTs annotations here; they are forwarded to Pupil Capture over its ZMQ API
and also appended to a local JSONL log as a backup.

    GET  /clock             Pupil time now, used by the page to sync its clock
    POST /annotation        send one annotation
    POST /recording/start   start Pupil Capture if needed, check the eye cameras,
                            start a recording named after the participant
    POST /recording/stop    stop the recording
    GET  /status            connection state and timing stats

Under pythonw (no console, e.g. started at login), output goes to
logs/bridge_<date>.log instead.
"""
import argparse
import json
import re
import statistics
import subprocess
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
PUPIL_CAPTURE_GLOB = "Pupil-Labs/*/Pupil Capture*/pupil_capture.exe"
PROGRAM_DIRS = [Path(r"C:\Program Files (x86)"), Path(r"C:\Program Files")]

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
        self.sub = None  # recording.* notifications
        self.pub_port = None
        self.sub_port = None
        self.recording = None  # {"session_name", "rec_path"} while Pupil Capture records
        self.instance_pub_port = None
        self.last_error = None
        self.warned = False  # warned that Pupil Capture is unreachable, since the last success

    def _close(self):
        for sock in (self.remote, self.pub, self.sub):
            if sock is not None:
                sock.close(linger=0)
        self.remote = self.pub = self.sub = self.pub_port = self.sub_port = None

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
        # Pupil Capture picks new ports each launch: a different port means it
        # restarted, which ends any recording; the same port is a mere reconnect
        if self.pub_port != self.instance_pub_port:
            self.recording = None
            self.instance_pub_port = self.pub_port
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.setsockopt(zmq.LINGER, 0)
        self.pub.connect(f"tcp://{self.host}:{self.pub_port}")

        self.sub_port = self._request(b"SUB_PORT")
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.setsockopt(zmq.LINGER, 0)
        self.sub.connect(f"tcp://{self.host}:{self.sub_port}")
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "notify.recording.")

        notification = {"subject": "start_plugin", "name": "Annotation_Capture", "args": {}}
        self._request(b"notify.start_plugin", msgpack.dumps(notification, use_bin_type=True))

        # PUB/SUB connections drop messages sent before they are fully set up
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

    def _drain_notifications(self):
        while self.sub.poll(0):
            frames = self.sub.recv_multipart()
            note = msgpack.unpackb(frames[1], raw=False)
            if note.get("subject") == "recording.started":
                self.recording = {"session_name": note.get("session_name"),
                                  "rec_path": note.get("rec_path")}
            elif note.get("subject") == "recording.stopped":
                self.recording = None

    def _wait_for(self, done, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            self._run(self._drain_notifications)
            if done():
                return True
            time.sleep(0.1)
        return False

    def start_recording(self, session_name, timeout_s=5.0):
        """Returns {"session_name", "rec_path"} once Pupil Capture reports it started."""
        def send():
            self._drain_notifications()
            self._request(f"R {session_name}".encode())
        self._run(send)
        if not self._wait_for(lambda: self.recording is not None, timeout_s):
            raise RecordingError("Pupil Capture did not start recording. If it is "
                                 "already recording, stop that recording and retry.")
        return self.recording

    def stop_recording(self, timeout_s=15.0):
        """Returns the rec_path of the stopped recording (None if none was running)."""
        self._run(self._drain_notifications)
        if self.recording is None:
            return None
        rec_path = self.recording["rec_path"]
        self._run(lambda: self._request(b"r"))
        if not self._wait_for(lambda: self.recording is None, timeout_s):
            raise RecordingError("Pupil Capture did not confirm that the recording stopped.")
        return rec_path

    def eyes_streaming(self, timeout_s=3.0):
        """Ids of the eye cameras that delivered pupil data within timeout_s."""
        sub_port = self._run(lambda: self.sub_port)
        sock = self.ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.LINGER, 0)
        sock.connect(f"tcp://{self.host}:{sub_port}")
        sock.setsockopt_string(zmq.SUBSCRIBE, "pupil.")
        seen = set()
        deadline = time.monotonic() + timeout_s
        try:
            while time.monotonic() < deadline and len(seen) < 2:
                if sock.poll(100):
                    topic = sock.recv_multipart()[0].decode()  # e.g. pupil.0.2d
                    seen.add(int(topic.split(".")[1]))
        finally:
            sock.close()
        return seen

    @property
    def connected(self):
        return self.remote is not None


class PupilUnavailable(Exception):
    pass


class RecordingError(Exception):
    pass


def find_pupil_capture():
    found = sorted(p for d in PROGRAM_DIRS for p in d.glob(PUPIL_CAPTURE_GLOB))
    return found[-1] if found else None


def pupil_capture_running():
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq pupil_capture.exe", "/NH"],
                         capture_output=True, text=True).stdout
    return "pupil_capture.exe" in out


def recording_name(data):
    """Recording folder name: date, room label and participant code."""
    parts = [datetime.now().strftime("%Y-%m-%d"),
             data.get("participant_label") or "nolabel",
             data.get("participant_code") or "nocode"]
    return re.sub(r"[^A-Za-z0-9_-]", "-", "_".join(str(p) for p in parts))


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

    def __init__(self, pupil, pupil_capture_exe=None, eyes=(0, 1)):
        self.pupil = pupil
        self.pupil_capture_exe = pupil_capture_exe
        self.eyes = set(eyes)
        self.recording_lock = threading.Lock()
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

    def ensure_pupil_capture(self, timeout_s=90):
        """Start Pupil Capture if it isn't running and wait until it answers."""
        try:
            self.pupil.time()
            return
        except PupilUnavailable:
            pass
        if not pupil_capture_running():
            if not self.pupil_capture_exe:
                raise RecordingError("Pupil Capture is not running and was not found in "
                                     "Program Files; start it by hand.")
            print(f"Starting Pupil Capture: {self.pupil_capture_exe}", flush=True)
            subprocess.Popen([str(self.pupil_capture_exe)], cwd=self.pupil_capture_exe.parent,
                             creationflags=subprocess.DETACHED_PROCESS
                             | subprocess.CREATE_NEW_PROCESS_GROUP)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                self.pupil.time()
                return
            except PupilUnavailable:
                time.sleep(1)
        raise RecordingError(f"Pupil Capture did not respond within {timeout_s} s of starting.")

    def recording_start(self, body):
        data = json.loads(body or b"{}")
        if not isinstance(data, dict):
            raise ValueError("expected a JSON object")
        name = recording_name(data)
        with self.recording_lock:
            try:
                self.pupil._run(self.pupil._drain_notifications)
            except PupilUnavailable:
                pass
            current = self.pupil.recording
            if current and current["session_name"] == name:
                return 200, {"status": "already recording", **current}
            try:
                if current:  # left running for a previous participant
                    print(f"Stopping leftover recording {current['rec_path']}", flush=True)
                    self.pupil.stop_recording()
                self.ensure_pupil_capture()
                # the eye windows can take a few seconds more than Pupil Capture itself
                deadline = time.monotonic() + 20
                missing = self.eyes - self.pupil.eyes_streaming()
                while missing and time.monotonic() < deadline:
                    missing = self.eyes - self.pupil.eyes_streaming()
                if missing:
                    raise RecordingError(
                        "No pupil data from eye camera " + " and ".join(map(str, sorted(missing)))
                        + ". Check that the headset is plugged in and the eye windows are open.")
                started = self.pupil.start_recording(name)
            except (PupilUnavailable, RecordingError) as e:
                print(f"RECORDING NOT STARTED: {e}", flush=True)
                return 503, {"error": str(e)}
        print(f"Recording started: {started['rec_path']}", flush=True)
        return 200, {"status": "started", **started}

    def recording_stop(self, body):
        with self.recording_lock:
            try:
                rec_path = self.pupil.stop_recording()
            except (PupilUnavailable, RecordingError) as e:
                print(f"RECORDING NOT STOPPED: {e}", flush=True)
                return 503, {"error": str(e)}
        if rec_path:
            print(f"Recording stopped: {rec_path}", flush=True)
        return 200, {"status": "stopped" if rec_path else "not recording", "rec_path": rec_path}

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
            "recording": self.pupil.recording,
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
            routes = {
                "/annotation": bridge.annotation,
                "/recording/start": bridge.recording_start,
                "/recording/stop": bridge.recording_stop,
            }
            route = routes.get(self.path.split("?")[0])
            if route is None:
                self._reply(404, {"error": "not found"})
                return
            try:
                self._reply(*route(body))
            except ValueError:
                self._reply(400, {"error": "expected a JSON object"})

        def log_message(self, format, *args):
            pass  # one line per annotation is enough

    return Handler


class Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128  # default 5 refuses bursts of simultaneous requests
    # on Windows, SO_REUSEADDR would let a second bridge bind the same port
    allow_reuse_address = False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=8765, help="HTTP port for the page (default 8765)")
    parser.add_argument("--pupil-host", default="127.0.0.1")
    parser.add_argument("--pupil-port", type=int, default=50020, help="Pupil Remote port (default 50020)")
    parser.add_argument("--pupil-capture-exe", type=Path, default=None,
                        help="pupil_capture.exe to start when needed (default: newest in Program Files)")
    parser.add_argument("--eyes", default="0,1",
                        help="eye cameras that must deliver data before recording (default 0,1)")
    args = parser.parse_args(argv)

    if sys.stdout is None:  # pythonw: no console
        LOG_DIR.mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(LOG_DIR / f"bridge_{datetime.now():%Y-%m-%d_%H%M%S}.log",
                                       "a", encoding="utf-8", buffering=1)

    pupil = PupilConnection(args.pupil_host, args.pupil_port)
    exe = args.pupil_capture_exe or find_pupil_capture()
    eyes = [int(e) for e in args.eyes.split(",") if e.strip()]
    try:
        server = Server(("127.0.0.1", args.port), make_handler(Bridge(pupil, exe, eyes)))
    except OSError:
        print(f"Port {args.port} is in use: the bridge is probably already running "
              f"(see http://127.0.0.1:{args.port}/status).", flush=True)
        return 1

    try:
        pupil.time()
    except PupilUnavailable:
        pass  # already warned; it connects on a later request
    print(f"Pupil bridge on http://127.0.0.1:{args.port}  (status: /status, stop: Ctrl+C)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
