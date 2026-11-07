#!/usr/bin/env python3
"""
presence-brightness — webcam presence sensor that controls display brightness.

If no one is in front of the webcam, brightness goes to 0.
If a person is present, brightness goes to max during the day and half otherwise.

Designed to be tiny: one 320x240 frame every 3s, YuNet face detector (a small
ONNX model, CPU-only; Haar cascades as fallback). Zero GPU usage. Runs as a
plain background process (or systemd user service). Measured: ~230 MB RAM,
≤ ~0.5% of one CPU core, 0 VRAM.

Usage:
  python3 presence_brightness.py            # run
  python3 presence_brightness.py --dry-run  # log decisions, don't touch hardware
  python3 presence_brightness.py --once     # single detect-and-set pass, exit
  python3 presence_brightness.py --preview  # also serve a live camera stream
                                            # at http://localhost:8123
  ./setup.sh                                # venv + systemd user service + tray

Tray icon (tray.py, started automatically after ./setup.sh):
  — status line, Auto (webcam control), manual cap 10..100%,
    pause/resume webcam control. The tray only quits itself; the daemon
    (systemd service) keeps running.
  Control: unix socket XDG_RUNTIME_DIR/screen-sentry.sock (fallback
  ~/.cache/screen-sentry.sock). HTTP-style JSON, e.g.:
    curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock http://s/status
    curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{"value":30}' http://s/override
    curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/auto
    curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/pause
    curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/enable
"""

import json
import os
import re
import socketserver
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

# --- config ---------------------------------------------------------------
CAMERA_INDEX = 0
SAMPLE_INTERVAL_ABSENT = 3  # seconds between checks (always, constant)
SAMPLE_INTERVAL_PRESENT = 3 # same at both ends of the state: constant checking

# Hysteresis so the screen doesn't flicker:
OFF_AFTER_ABSENT = 30       # absent for this long -> screen off (slow, deliberate)
ON_AFTER_PRESENT = 0        # present even once -> screen on instantly

# brightness levels (percent 0-100)
BRIGHTNESS_FULL = 100
BRIGHTNESS_HALF = 50
# "day time" window; inside it -> full, outside -> half
DAY_START = 7 * 60          # 07:00 — from here full brightness
DAY_END = 18 * 60 + 30      # 18:30 — after this half brightness

# --- detection --------------------------------------------------------------
GRAB_WIDTH = 320            # camera frame; YuNet gets this size directly
GRAB_HEIGHT = 240
DET_SCORE = 0.7             # YuNet minimum face confidence (0..1); lower =
                            # catches more tilted/away poses, keep >= 0.4

# YuNet (ONNX, ~230 KB CPU model, shipped in models/) — much better than Haar
# for off-angle faces: head tilted, looking down/sideways, leaning, glasses.
MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "face_detection_yunet_2023mar.onnx"
)

# Haar fallback (used only if the YuNet model file is missing)
CASCADES = (
    "haarcascade_frontalface_default.xml",
    "haarcascade_profileface.xml",
)
PROFILE_EVERY = 3
DET_SCALE_FACTOR = 1.2
DET_MIN_NEIGHBORS = 6
DET_MIN_FACE_PX = 20
HAAR_GRAY_SIZE = (160, 120)

BACKOFF_ON_ERROR = 10       # seconds to wait if camera/errors
BLIP_TOLERANCE = 1          # missed frames right after a confirmed person that
                            # don't restart the 30s off-countdown (flicker guard)

VERIFY_ATTEMPTS = 4         # set -> read back VCP 10 per monitor -> retry if off
VERIFY_DELAY = 1.5          # settle seconds between attempts
VERIFY_TOLERANCE = 1        # acceptable deviation in percent

PREVIEW = "--preview" in sys.argv
PREVIEW_PORT = 8123         # http://localhost:8123 (only bound to 127.0.0.1)
PREVIEW_FPS = 4             # stream rate between detection samples
CONTROL_SOCK = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache"),
    "screen-sentry.sock",      # local control socket used by tray.py
)

# --- manual control (set by the tray via the control API) --------------------
_paused = False     # tray: stop touching brightness, daemon keeps running
_override = None    # tray: cap applied while someone is present (None = auto)
STATUS = {          # snapshot for GET /status, updated by the main loop
    "state": "starting", "present": False, "paused": False,
    "override": None, "last_set": None,
}
# -------------------------------------------------------------------------


def detect_brightness_backend():
    """Return 'brightness', 'sysfs', or None."""
    if subprocess.run(["sh", "-c", "command -v brightness"], capture_output=True).returncode == 0:
        return "brightness"
    # fallback: internal panel backlight
    if os.path.isdir("/sys/class/backlight") and os.listdir("/sys/class/backlight"):
        return "sysfs"
    return None


_ddc_buses = None


def ddc_buses():
    """I2C bus numbers of all detected monitors (cached)."""
    global _ddc_buses
    if _ddc_buses is None:
        out = subprocess.run(
            ["ddcutil", "detect"], capture_output=True, text=True, timeout=90
        )
        _ddc_buses = [
            int(m.group(1))
            for line in out.stdout.splitlines()
            for m in [re.search(r"/dev/i2c-(\d+)", line)]
            if m
        ]
    return _ddc_buses


def read_brightness_bus(bus):
    """Read VCP 10 off one monitor; returns percent or None."""
    r = subprocess.run(
        ["ddcutil", "--brief", "getvcp", "10", "--bus", str(bus)],
        capture_output=True,
        text=True,
        timeout=60,
    )
    for line in r.stdout.splitlines():
        f = line.split()
        # brief format: "VCP 10 C <current> <max>"
        if len(f) >= 5 and f[0] == "VCP" and f[1] == "10":
            try:
                cur, mx = int(f[3]), int(f[4])
            except ValueError:
                return None
            return round(cur * 100 / mx) if mx else cur
    return None


def set_brightness(backend, value):
    """Set brightness, then VERIFY it actually took effect on every monitor;
    re-set stragglers until they confirm (or attempts run out).
    Returns (ok, detail)."""
    if backend == "brightness":
        try:
            subprocess.run(["brightness", str(value)], capture_output=True, timeout=90)
        except subprocess.SubprocessError as e:
            return False, f"brightness command failed: {e}"
        buses = ddc_buses()
        if not buses:
            return True, "no ddcutil monitors found to verify"
        mismatches = []
        for attempt in range(1, VERIFY_ATTEMPTS + 1):
            time.sleep(VERIFY_DELAY if attempt > 1 else 0.8)
            mismatches = [
                (b, c)
                for b in buses
                for c in [read_brightness_bus(b)]
                if c is None or abs(c - value) > VERIFY_TOLERANCE
            ]
            if not mismatches:
                return True, f"verified {value}% on {len(buses)} monitor(s) {buses}"
            for b, _c in mismatches:
                cur = read_brightness_bus(b)
                print(
                    f"{stamp()} verify: monitor bus {b} reports {cur}% — wanted {value}%, re-setting"
                )
                try:
                    subprocess.run(
                        ["ddcutil", "setvcp", "10", str(value), "--bus", str(b)],
                        capture_output=True,
                        timeout=60,
                    )
                except subprocess.SubprocessError as e:
                    print(f"{stamp()} re-set bus {b} failed: {e}")
        detail = ", ".join(f"bus {b} stuck at {c}%" for b, c in mismatches)
        return False, f"gave up after {VERIFY_ATTEMPTS} attempts ({detail})"
    # sysfs: internal panel, verify by reading back
    ok = False
    for name in os.listdir("/sys/class/backlight"):
        bl = f"/sys/class/backlight/{name}"
        try:
            cur_max = 255
            try:
                with open(f"{bl}/device/max_brightness") as f:
                    cur_max = int(f.read())
            except OSError:
                pass
            level = round(value * cur_max / 100)
            for attempt in range(1, VERIFY_ATTEMPTS + 1):
                with open(f"{bl}/brightness", "w") as fh:
                    fh.write(str(level))
                time.sleep(0.2 if attempt > 1 else 0)
                with open(f"{bl}/actual_brightness") as fh:
                    actual = round(int(fh.read()) * 100 / cur_max)
                if abs(actual - value) <= VERIFY_TOLERANCE + 2:
                    ok = True
                    break
        except (OSError, PermissionError):
            continue
    return ok, "panel verified" if ok else "panel did not confirm"


# --- live camera preview (optional, `--preview`) --------------------------
_preview_jpeg = None        # latest encoded frame, shared with the server thread


class _PreviewHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request noise
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/snapshot" or self.path.endswith(".jpg"):
            self._snapshot()
        else:
            self._stream()

    def _snapshot(self):
        if _preview_jpeg is None:
            self.send_response(503)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(_preview_jpeg)))
        self.end_headers()
        self.wfile.write(_preview_jpeg)

    def _stream(self):
        boundary = b"--frame"
        self.send_response(200)
        self.send_header(
            "Content-Type", "multipart/x-mixed-replace; boundary=frame"
        )
        self.end_headers()
        while True:
            if _preview_jpeg is None:
                time.sleep(0.1)
                continue
            try:
                self.wfile.write(boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n")
                self.wfile.write(_preview_jpeg)
                self.wfile.write(b"\r\n")
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            time.sleep(1.0 / PREVIEW_FPS)


def start_preview_server():
    """Serve the camera stream on 127.0.0.1 only. Returns the bound port."""
    for port in range(PREVIEW_PORT, PREVIEW_PORT + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), _PreviewHandler)
        except OSError:
            continue  # port taken, try the next one
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return port
    print("warning: preview server could not bind any port, continuing without preview")
    return None


def _grab_preview_frame(sensor, status_text):
    """Grab one frame, annotate it, publish it for the preview stream."""
    global _preview_jpeg
    ok, frame = sensor.cap.read()
    if not ok or frame is None:
        return
    annotated = frame.copy()
    cv2.putText(
        annotated,
        status_text,
        (8, 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        _preview_jpeg = buf.tobytes()


def preview_status_line(present_since, absent_since, countdown_on, last_state, now):
    """One short text line shown on the preview overlay."""
    if present_since is not None:
        level = target_brightness(True)
        return f"person seen - brightness {level}% ({'day full' if current_day_time() else 'half'})"
    if absent_since is not None and countdown_on:
        remain = max(0, int(OFF_AFTER_ABSENT - (now - absent_since)))
        return f"nobody - screen off in {remain}s (30s rule)"
    return "starting..."


# --- local control API (used by tray.py) ------------------------------------
class _ControlHandler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silence per-request noise
        pass

    def _send(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/status":
            self._send(200, dict(STATUS))
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _paused, _override
        if self.path == "/pause":
            if not _paused:
                _paused = True
                print(f"{stamp()} control: paused — webcam control off, brightness untouched")
        elif self.path == "/enable":
            if _paused:
                _paused = False
                print(f"{stamp()} control: resumed — webcam control back on")
        elif self.path == "/auto":
            if _override is not None:
                _override = None
                print(f"{stamp()} control: override cleared — back to automatic")
        elif self.path == "/override":
            try:
                length = int(self.headers.get("Content-Length", 0))
                value = int(json.loads(self.rfile.read(length) or b"{}").get("value"))
            except (ValueError, json.JSONDecodeError):
                return self._send(400, {"error": "bad request"})
            if not 10 <= value <= 100:
                return self._send(400, {"error": "value out of range 10..100"})
            if value != _override:
                _override = value
                print(f"{stamp()} control: manual cap {value}%")
        else:
            return self._send(404, {"error": "not found"})
        STATUS["paused"] = _paused
        STATUS["override"] = _override
        self._send(200, dict(STATUS))


class _ThreadingUnixHTTPServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True


def start_control_server():
    """Serve the tray control API as a unix socket (no TCP ports)."""
    if not os.environ.get("XDG_RUNTIME_DIR"):
        os.makedirs(os.path.dirname(CONTROL_SOCK), exist_ok=True)
        os.chmod(os.path.dirname(CONTROL_SOCK), 0o700)
    try:
        os.unlink(CONTROL_SOCK)
    except FileNotFoundError:
        pass
    try:
        server = _ThreadingUnixHTTPServer(CONTROL_SOCK, _ControlHandler)
    except OSError as e:
        print(f"warning: control socket busy ({e}) — tray control disabled")
        return
    os.chmod(CONTROL_SOCK, 0o600)
    threading.Thread(target=server.serve_forever, daemon=True).start()


# --- cascades -------------------------------------------------------------------------
_LOCAL_CASCADES = os.path.join(os.path.dirname(__file__), "cascades")
_HAS_LOCAL_CASCADES = os.path.isfile(
    os.path.join(_LOCAL_CASCADES, "haarcascade_frontalface_default.xml")
)
CASCADE_PATHS = [
    os.path.join(_LOCAL_CASCADES if _HAS_LOCAL_CASCADES else cv2.data.haarcascades, f)
    for f in CASCADES
]


class PresenceSensor:
    cascades = [cv2.CascadeClassifier(p) for p in CASCADE_PATHS]

    def __init__(self):
        cv2.setNumThreads(2)     # keep OpenCV's thread pool from spinning CPU
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_V4L2)
        if self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, GRAB_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, GRAB_HEIGHT)
        self.n_surveys = 0
        self.detector = None     # YuNet, lazily built on first frame
        self.last_score = None   # confidence of the last positive detection

    def close(self):
        if self.cap.isOpened():
            self.cap.release()

    def _ensure_detector(self, frame):
        """YuNet must match the actual frame size the camera delivers."""
        h, w = frame.shape[:2]
        self.detector = cv2.FaceDetectorYN.create(
            MODEL_PATH, "", (w, h), score_threshold=DET_SCORE
        )
        self.detector.setInputSize((w, h))

    def present(self) -> bool:
        if not self.cap.isOpened():
            raise RuntimeError("camera not open")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("frame grab failed")
        self.last_score = None

        if os.path.isfile(MODEL_PATH):
            if self.detector is None:
                self._ensure_detector(frame)
            _, faces = self.detector.detect(frame)
            if faces is not None and len(faces) > 0:
                self.last_score = float(faces[:, 14].max())
                return True
            return False

        # Haar fallback (less robust to tilted/turned heads)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, HAAR_GRAY_SIZE)
        gray = cv2.equalizeHist(gray)
        self.n_surveys += 1
        for i, cas in enumerate(self.cascades):
            if i > 0 and self.n_surveys % PROFILE_EVERY != 0:
                continue
            faces = cas.detectMultiScale(
                gray,
                scaleFactor=DET_SCALE_FACTOR,
                minNeighbors=DET_MIN_NEIGHBORS,
                minSize=(DET_MIN_FACE_PX, DET_MIN_FACE_PX),
            )
            if len(faces) > 0:
                return True
        return False


def current_day_time():
    lt = time.localtime()
    minutes = lt.tm_hour * 60 + lt.tm_min
    return DAY_START <= minutes < DAY_END


def stamp():
    return time.strftime("%H:%M:%S")


def target_brightness(present):
    return (BRIGHTNESS_FULL if current_day_time() else BRIGHTNESS_HALF) if present else 0


def main():
    dry_run = "--dry-run" in sys.argv
    once = "--once" in sys.argv

    backend = detect_brightness_backend()
    if backend is None and not dry_run:
        sys.exit("no brightness backend found (no `brightness` command, no /sys/class/backlight)")
    print(f"brightness backend: {'dry-run' if dry_run else backend}")

    start_control_server()

    sensor = PresenceSensor()
    if not sensor.cap.isOpened():
        sys.exit(f"cannot open camera {CAMERA_INDEX}")
    print(f"camera {CAMERA_INDEX} open — checking every {SAMPLE_INTERVAL_ABSENT}s")

    preview_port = None
    if PREVIEW:
        preview_port = start_preview_server()
        if preview_port:
            print(
                f"preview: camera view on http://localhost:{preview_port} "
                f"(still image: /snapshot)"
            )

    def refresh_status():
        """Publish loop state for the control API right away (not only at the
        end of an iteration, which can be delayed by slow DDC verification)."""
        STATUS.update(
            state=last_state
            or ("countdown" if countdown_running else "starting"),
            present=present_since is not None,
            paused=_paused,
            override=_override,
            last_set=None if last_set < 0 else last_set,
        )

    present_since = None     # when presence was first seen
    absent_since = None      # when absence was first seen
    last_state = None        # last resolved state ("present"/"absent")
    countdown_running = False
    miss_streak = 0          # consecutive person-free samples
    first_miss_time = None   # when the current absence streak began
    last_set = -1            # last brightness value we actually applied

    try:
        while True:
            # only sleep the remainder of the interval so we keep the average rate
            interval = (
                SAMPLE_INTERVAL_PRESENT if last_state == "present"
                else SAMPLE_INTERVAL_ABSENT
            )
            t0 = time.monotonic()
            try:
                detected = sensor.present()
            except RuntimeError as e:
                print(f"camera error: {e}; retrying in {BACKOFF_ON_ERROR}s")
                time.sleep(BACKOFF_ON_ERROR)
                sensor.close()
                sensor = PresenceSensor()
                if not sensor.cap.isOpened():
                    print("camera re-open failed, will keep retrying")
                continue

            now = time.monotonic()
            if detected:
                miss_streak = 0
                if present_since is None:
                    present_since = now
                    absent_since = None
                    if countdown_running:
                        countdown_running = False
                        score = f" ({sensor.last_score:.2f})" if sensor.last_score else ""
                        print(f"{stamp()} person seen{score} — countdown canceled, screen on")
                    else:
                        score = f" ({sensor.last_score:.2f})" if sensor.last_score else ""
                        print(f"{stamp()} person seen{score}")
            else:
                if miss_streak == 0:
                    first_miss_time = now
                miss_streak += 1
                if absent_since is None:
                    if last_state == "present" and miss_streak <= BLIP_TOLERANCE:
                        pass  # detection blip right after confirmed presence: ignore
                    else:
                        # count the 30s from the first person-free check
                        absent_since = first_miss_time if first_miss_time else now
                        present_since = None
                        remaining = max(
                            0.0, OFF_AFTER_ABSENT - (now - absent_since)
                        )
                        off_at = time.strftime(
                            "%H:%M:%S",
                            time.localtime(time.time() + remaining),
                        )
                        if countdown_running:
                            print(
                                f"{stamp()} nobody again — countdown restarted, off at {off_at}"
                            )
                        else:
                            countdown_running = True
                            print(
                                f"{stamp()} nobody seen — screen off at {off_at} unless someone appears"
                            )

            decided = None
            if present_since is not None and now - present_since >= ON_AFTER_PRESENT:
                decided = "present"
            elif absent_since is not None and now - absent_since >= OFF_AFTER_ABSENT:
                decided = "absent"

            if decided:
                if decided == "present":
                    value = _override if _override is not None else target_brightness(True)
                else:
                    value = 0
                if not _paused:
                    # log only on state change or level change (was spamming
                    # every sample while a person is present)
                    if last_state != decided or value != last_set:
                        if decided == "present":
                            why = (
                                f"manual cap {_override}%"
                                if _override is not None
                                else (
                                    "day time, full"
                                    if current_day_time()
                                    else "half brightness"
                                )
                            )
                            print(
                                f"{stamp()} countdown ended with person seen — screen on ({why})"
                            )
                        else:
                            print(
                                f"{stamp()} countdown expired, nobody for {OFF_AFTER_ABSENT}s — screen off"
                            )
                    if value != last_set:
                        refresh_status()  # latest published before the slow set+verify
                        if dry_run:
                            ok, detail = True, "dry-run, hardware untouched"
                        else:
                            try:
                                ok, detail = set_brightness(backend, value)
                            except Exception as e:
                                ok, detail = False, f"set failed: {e}"
                        print(
                            f"{stamp()} brightness -> {value}% "
                            f"{'set' if ok else 'set FAILED'} ({detail})"
                        )
                        if ok:
                            last_set = value
                if decided == "present":
                    countdown_running = False
                last_state = decided
                present_since = now if decided == "present" else None
                absent_since = now if decided == "absent" else None

            refresh_status()

            if once:
                break

            wait = max(0.0, interval - (time.monotonic() - t0))
            if preview_port:
                # stream live frames between detection samples
                deadline = time.monotonic() + wait
                while time.monotonic() < deadline:
                    _grab_preview_frame(
                        sensor,
                        preview_status_line(
                            present_since, absent_since, countdown_running, last_state,
                            time.monotonic(),
                        ),
                    )
                    time.sleep(1.0 / PREVIEW_FPS)
            else:
                time.sleep(wait)
    finally:
        sensor.close()


if __name__ == "__main__":
    main()
