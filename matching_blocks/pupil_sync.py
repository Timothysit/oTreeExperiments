import time
import msgpack
import zmq


class PupilSync:
    def __init__(self, ip="127.0.0.1", port=50020):
        self.ip = ip
        self.port = port
        self.ctx = zmq.Context.instance()

        self.remote = self.ctx.socket(zmq.REQ)
        
        self.remote.setsockopt(zmq.RCVTIMEO, 1000)
        self.remote.setsockopt(zmq.SNDTIMEO, 1000)

        self.remote.connect(f"tcp://{ip}:{port}")

        # Get Pupil Capture PUB port
        self.remote.send_string("PUB_PORT")
        pub_port = self.remote.recv_string()

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(f"tcp://{ip}:{pub_port}")

        # Estimate offset between local Python clock and Pupil clock
        self.clock_offset = self.measure_clock_offset_stable()

        # Start Annotation Capture plugin
        self.notify({
            "subject": "start_plugin",
            "name": "Annotation_Capture",
            "args": {}
        })

        time.sleep(0.5)

    def request_pupil_time(self):
        self.remote.send_string("t")
        return float(self.remote.recv_string())

    def measure_clock_offset(self):
        local_before = time.time()
        pupil_time = self.request_pupil_time()
        local_after = time.time()

        local_midpoint = (local_before + local_after) / 2
        return pupil_time - local_midpoint

    def measure_clock_offset_stable(self, n_samples=10):
        offsets = [self.measure_clock_offset() for _ in range(n_samples)]
        return sum(offsets) / len(offsets)

    def notify(self, notification):
        topic = "notify." + notification["subject"]
        payload = msgpack.dumps(notification, use_bin_type=True)

        self.remote.send_string(topic, flags=zmq.SNDMORE)
        self.remote.send(payload)

        return self.remote.recv_string()

    def send_annotation(self, label, duration=0.0, **custom_fields):
        local_ts = time.time()
        pupil_ts = local_ts + self.clock_offset

        annotation = {
            "topic": "annotation",
            "label": label,
            "timestamp": pupil_ts,
            "duration": duration,
            **custom_fields,
        }

        payload = msgpack.dumps(annotation, use_bin_type=True)

        self.pub.send_string("annotation", flags=zmq.SNDMORE)
        self.pub.send(payload)

        return {
            "label": label,
            "local_ts": local_ts,
            "pupil_ts": pupil_ts,
            **custom_fields,
        }


_pupil_sync = None
_pupil_available = True


def get_pupil_sync():
    global _pupil_sync
    global _pupil_available

    if not _pupil_available:
        return None

    if _pupil_sync is None:
        try:
            _pupil_sync = PupilSync()
            print("[PupilSync] Connected to Pupil Capture")
        except Exception as e:
            print(f"[PupilSync WARNING] Could not connect: {e}")
            _pupil_available = False
            return None

    return _pupil_sync


def send_pupil_annotation(label, **custom_fields):
    sync = get_pupil_sync()

    if sync is None:
        return None

    try:
        return sync.send_annotation(label, **custom_fields)

    except Exception as e:
        print(f"[PupilSync WARNING] Could not send annotation: {e}")
        return None
