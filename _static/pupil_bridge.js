// Sends Pupil Capture annotations from an experiment page to the pupil bridge
// running on the same laptop (pupil_bridge/pupil_bridge.py).
//
//   PupilBridge.init(js_vars.pupil);       // once, js_vars.pupil from pupil_bridge.context
//   PupilBridge.annotate("trial_start", {block_trial: 3});
//   PupilBridge.annotate("stim_onset", {}, onsetMs);  // event time as performance.now()
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

  function init(options = {}) {
    const { enabled: isEnabled = true, url, ...fields } = options;
    enabled = isEnabled;
    if (url) baseUrl = url.replace(/\/$/, "");
    context = fields;
    if (enabled) syncClock();
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

  window.PupilBridge = { init, annotate };
})();
