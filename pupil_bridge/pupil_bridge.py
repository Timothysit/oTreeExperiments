from flask import Flask, request, jsonify
from flask_cors import CORS
import zmq
import msgpack
import time

app = Flask(__name__)
CORS(app)

ctx = zmq.Context.instance()

remote = ctx.socket(zmq.REQ)
remote.setsockopt(zmq.RCVTIMEO, 1000)
remote.setsockopt(zmq.SNDTIMEO, 1000)
remote.connect("tcp://127.0.0.1:50020")


def remote_notify(notification):
    topic = "notify." + notification["subject"]
    payload = msgpack.dumps(notification, use_bin_type=True)

    remote.send_string(topic, flags=zmq.SNDMORE)
    remote.send(payload)

    return remote.recv_string()


# Get Pupil PUB port
remote.send_string("PUB_PORT")
pub_port = remote.recv_string()

pub = ctx.socket(zmq.PUB)
pub.connect(f"tcp://127.0.0.1:{pub_port}")

# Start Annotation Capture plugin
remote_notify({
    "subject": "start_plugin",
    "name": "Annotation_Capture",
    "args": {}
})

time.sleep(0.5)

print(f"Connected to Pupil Capture PUB port {pub_port}")


@app.route("/annotation", methods=["POST", "OPTIONS"])
def annotation():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"})

    data = request.get_json(force=True)

    # Ask Pupil for its current clock time
    remote.send_string("t")
    pupil_ts = float(remote.recv_string())

    annotation = {
        "topic": "annotation",
        "label": data.get("label", "event"),
        "timestamp": pupil_ts,
        "duration": 0.0,
        **data,
    }

    payload = msgpack.dumps(annotation, use_bin_type=True)

    pub.send_string("annotation", flags=zmq.SNDMORE)
    pub.send(payload)

    print("Sent annotation:", annotation["label"], data)

    return jsonify({"status": "ok", "label": annotation["label"]})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)