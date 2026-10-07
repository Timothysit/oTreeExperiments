// Sends Pupil Capture annotations from an experiment page to the pupil bridge
// running on the same laptop (pupil_bridge/pupil_bridge.py).
//
//   PupilBridge.init(js_vars.pupil);       // once, js_vars.pupil from pupil_bridge.context
//   PupilBridge.annotate("trial_start", {block_trial: 3});
//   PupilBridge.annotate("stim_onset", {}, onsetMs);  // event time as performance.now()
//   await PupilBridge.startRecording();  /  await PupilBridge.stopRecording();
//
// The page measures the offset between its performance.now() clock and Pupil's
// clock through the bridge, so each annotation is stamped with the time the
// event happened rather than the time the request arrived.
(function () {
  const SYNC_SAMPLES = 10;
  const RESYNC_MS = 30000;
  const RETRY_MS = 5000;

  let enabled = false;
  let baseUrl = "http://127.0.0.1:8765";
  let context = {};
  let clockOffsetS = null;  // Pupil time (s) minus performance.now() (s)
  let clockRttMs = null;
  let syncTimer = null;

  async function measureClockOffset() {
    let best = null;
    for (let i = 0; i < SYNC_SAMPLES; i++) {
      const t0 = performance.now();
      const response = await fetch(baseUrl + "/clock", { cache: "no-store" });
      const t1 = performance.now();
      if (!response.ok) throw new Error("pupil bridge /clock returned " + response.status);
      const { pupil_time } = await response.json();
      // the sample with the shortest round trip pins Pupil time down most tightly
      if (best === null || t1 - t0 < best.rttMs) {
        best = { rttMs: t1 - t0, offsetS: pupil_time - (t0 + t1) / 2000 };
      }
    }
    return best;
  }

  async function syncClock() {
    clearTimeout(syncTimer);
    try {
      const best = await measureClockOffset();
      clockOffsetS = best.offsetS;
      clockRttMs = best.rttMs;
      syncTimer = setTimeout(syncClock, RESYNC_MS);
    } catch (err) {
      clockOffsetS = null;
      console.log("Pupil bridge unavailable", err);
      syncTimer = setTimeout(syncClock, RETRY_MS);
    }
  }

  // options: js_vars.pupil; pass syncClock: false on pages that only control recording
  function init(options = {}) {
    const { enabled: isEnabled = true, url, syncClock: shouldSync = true, ...fields } = options;
    enabled = isEnabled;
    if (url) baseUrl = url.replace(/\/$/, "");
    context = fields;
    if (enabled && shouldSync) syncClock();
  }

  async function post(path, body, timeoutMs) {
    try {
      const response = await fetch(baseUrl + path, {
        method: "POST",
        headers: { "Content-Type": "text/plain" },
        body: JSON.stringify(body),
        signal: AbortSignal.timeout(timeoutMs),
      });
      const data = await response.json();
      return response.ok
        ? { ok: true, ...data }
        : { ok: false, error: data.error || "pupil bridge returned " + response.status };
    } catch (err) {
      return bridgeDown(err);
    }
  }

  function bridgeDown(err) {
    return {
      ok: false,
      bridgeDown: true,
      error: "The pupil bridge is not running on this laptop (" + err.message + ").",
    };
  }

  // For the eye check page: {ok: true, recordings_dir, free_gb, total_gb}.
  async function diskSpace() {
    try {
      const response = await fetch(baseUrl + "/disk", { signal: AbortSignal.timeout(5000) });
      const data = await response.json();
      return response.ok ? { ok: true, ...data } : { ok: false, error: data.error };
    } catch (err) {
      return bridgeDown(err);
    }
  }

  // Starts the bridge through the pupilbridge: link type (pupil_bridge/install_windows.py).
  // Call it from a click: Chrome only opens such links on a user action, and asks
  // once whether to allow it. Resolves to {ok: true} once the bridge answers.
  async function launchBridge(timeoutMs = 30000) {
    const link = document.createElement("a");
    link.href = "pupilbridge:start";
    link.click();
    const deadline = performance.now() + timeoutMs;
    while (performance.now() < deadline) {
      try {
        const response = await fetch(baseUrl + "/status", { signal: AbortSignal.timeout(2000) });
        if (response.ok) return { ok: true };
      } catch (err) {
        // not up yet
      }
      await new Promise((resolve) => setTimeout(resolve, 500));
    }
    return {
      ok: false,
      bridgeDown: true,
      error: "The pupil bridge did not start. Allow Chrome to open it, or run "
        + "pupil_bridge/install_windows.py once on this laptop.",
    };
  }

  // Starts a Pupil Capture recording named after the participant; the bridge starts
  // Pupil Capture first if needed, so this can take a minute. Resolves to
  // {ok: true, rec_path, ...} or {ok: false, error}. Repeating it (page reload) is harmless.
  function startRecording() {
    if (!enabled) return Promise.resolve({ ok: true, skipped: true });
    return post("/recording/start", context, 150000);
  }

  function stopRecording() {
    if (!enabled) return Promise.resolve({ ok: true, skipped: true });
    return post("/recording/stop", context, 20000);
  }

  // For the eye check page: start Pupil Capture if it isn't running (can take a minute).
  function startPupilCapture() {
    return post("/pupil/start", {}, 120000);
  }

  // For the eye check page: {ok: true, eyes: {"0": {fps, confidence, diameter_px,
  // diameter_mm}, ...}, required: [0, 1]} from about one second of pupil data.
  async function eyeStats() {
    try {
      const response = await fetch(baseUrl + "/eyes", { signal: AbortSignal.timeout(10000) });
      const data = await response.json();
      return response.ok ? { ok: true, ...data } : { ok: false, error: data.error };
    } catch (err) {
      return bridgeDown(err);
    }
  }

  // Resolves to true if Pupil Capture received the annotation.
  async function annotate(label, fields = {}, eventTimeMs = performance.now()) {
    if (!enabled) return false;
    const body = {
      ...fields,
      ...context,
      label: label,
      browser_ts: eventTimeMs,
    };
    if (clockOffsetS !== null) {
      body.event_pupil_ts = eventTimeMs / 1000 + clockOffsetS;
      body.clock_sync_rtt_ms = clockRttMs;
    }
    try {
      const response = await fetch(baseUrl + "/annotation", {
        method: "POST",
        // text/plain keeps this a "simple" request (no CORS preflight round trip);
        // the bridge parses the body as JSON regardless
        headers: { "Content-Type": "text/plain" },
        body: JSON.stringify(body),
        // lets the request finish if the page navigates away right after
        keepalive: true,
      });
      return response.ok;
    } catch (err) {
      console.log("Pupil bridge unavailable", err);
      return false;
    }
  }

  window.PupilBridge = {
    init,
    annotate,
    startRecording,
    stopRecording,
    startPupilCapture,
    eyeStats,
    diskSpace,
    launchBridge,
    isEnabled: () => enabled,
  };
})();
