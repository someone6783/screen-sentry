#!/usr/bin/env python3
"""
screen-sentry tray — appindicator icon for presence-brightness (Cinnamon/GNOME/etc.)

Menu:
  <status line>
  ---
  Auto (webcam control)   — daemon decides: present = day full / else half, away = 0
  10% ... 100%            — manual cap used instead of auto while someone is present
  ---
  Pause webcam control    — daemon keeps running/sampling, stops touching brightness
  ---
  Quit (tray only — the daemon keeps running as a systemd service)

Talks to the daemon over its unix control socket (XDG_RUNTIME_DIR/screen-sentry.sock,
fallback ~/.cache/screen-sentry.sock — same file the daemon creates). Starts
automatically on login via ~/.config/autostart (installed by setup.sh).
"""

import http.client
import json
import os
import socket
import sys
from urllib.error import URLError
from urllib.error import HTTPError

import gi

gi.require_version("AyatanaAppIndicator3", "0.1")
from gi.repository import AyatanaAppIndicator3, GLib, Gtk  # noqa: E402

ICON_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
SOCK_PATH = os.path.join(
    os.environ.get("XDG_RUNTIME_DIR") or os.path.expanduser("~/.cache"),
    "screen-sentry.sock",
)
POLL_SECONDS = 3
LEVELS = range(10, 101, 10)


# --- daemon client (HTTP semantics over a unix socket) ----------------------
class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("screen-sentry")
        self._path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout if isinstance(self.timeout, (int, float)) else 5)
        self.sock.connect(self._path)


def call(path, value=None, method=None):
    if method is None:
        method = "POST" if path != "/status" else "GET"
    body = None if method == "GET" else json.dumps(
        {} if value is None else {"value": value}
    )
    conn = _UnixConnection(SOCK_PATH)
    try:
        conn.request(method, path, body=body,
                     headers={"Content-Type": "application/json"})
        resp = conn.getresponse()
        payload = resp.read()
        if resp.status != 200:
            raise URLError(f"daemon returned {resp.status}: {payload[:120]}")
        return json.loads(payload or b"{}")
    finally:
        conn.close()


# --- tray -------------------------------------------------------------------
class Tray:
    def __init__(self):
        self.indicator = AyatanaAppIndicator3.Indicator.new(
            "screen-sentry", "screen-sentry",
            AyatanaAppIndicator3.IndicatorCategory.APPLICATION_STATUS,
        )
        self.indicator.set_status(AyatanaAppIndicator3.IndicatorStatus.ACTIVE)
        self._icon("screen-sentry")
        self._menu_signature = None
        self._build_menu(None)
        self.refresh()
        GLib.timeout_add_seconds(POLL_SECONDS, self.refresh)

    def _icon(self, name):
        try:  # theme lookup with our assets dir searched first
            self.indicator.set_icon_theme_search_path([ICON_DIR])
        except Exception:  # older bindings may not have it
            pass
        if hasattr(self.indicator, "set_icon_full"):
            self.indicator.set_icon_full(name, "screen-sentry")
        else:
            self.indicator.set_icon(name)

    def refresh(self, *_):
        try:
            st = call("/status")
        except (URLError, HTTPError, OSError, ValueError, TypeError):
            self._icon("screen-sentry-paused")
            self.indicator.set_title("daemon not reachable")
            if self._menu_signature != "error":
                self._menu_signature = "error"
                self._build_menu(None)
            return GLib.SOURCE_CONTINUE

        icon = (
            "screen-sentry-paused" if st["paused"]
            else "screen-sentry-manual" if st["override"] is not None
            else "screen-sentry"
        )
        self._icon(icon)
        title_text = (
            f"{'paused' if st['paused'] else 'webcam control: on'}"
            f"{' — cap ' + str(st['override']) + '%' if st['override'] is not None else ''}"
        )
        try:  # tooltips render only in some environments; never fatal
            self.indicator.set_title(title_text)
        except Exception:
            pass

        signature = (
            st["state"], st["present"], st["paused"], st["override"], st["last_set"],
            st.get("holder"),
        )
        if signature != self._menu_signature:
            self._menu_signature = signature
            self._build_menu(st)
        return GLib.SOURCE_CONTINUE

    def _apply(self, *path_and_value):
        """Menu callback wrapper; errors surface as a stderr traceback-free noop."""
        try:
            call(*path_and_value)
        except (URLError, HTTPError, OSError, ValueError) as e:
            print(f"screen-sentry-tray: {path_and_value[0]} failed: {e}", file=sys.stderr)
        self.refresh()
        return False

    def _toggle_pause(self, item):
        self._apply("/enable" if not item.get_active() else "/pause")
        return False

    def _sep(self, menu):
        menu.append(Gtk.SeparatorMenuItem())

    def _build_menu(self, st):
        menu = Gtk.Menu()
        dead = st is None

        status = Gtk.MenuItem(label=(
            f"daemon not reachable ({SOCK_PATH})" if dead
            else self._status_text(st)
        ))
        status.set_sensitive(False)
        menu.append(status)

        self._sep(menu)
        auto = Gtk.CheckMenuItem(label="Auto (webcam control)")
        active = not dead and st["override"] is None and not st["paused"]
        auto.set_active(active)
        if dead:
            auto.set_sensitive(False)
        else:
            auto.connect("activate", lambda *_: self._apply("/auto"))
        menu.append(auto)

        for level in LEVELS:
            item = Gtk.CheckMenuItem(label=f"{level}%")
            item.set_active(not dead and st["override"] == level)
            if dead:
                item.set_sensitive(False)
            else:
                item.connect("activate",
                             lambda _, v=level: self._apply("/override", v))
            menu.append(item)

        self._sep(menu)
        is_paused = not dead and st["paused"]
        pause = Gtk.CheckMenuItem(
            label="Resume webcam control" if is_paused else "Pause webcam control"
        )
        pause.set_active(is_paused)
        if dead:
            pause.set_sensitive(False)
        else:
            pause.connect("toggled", self._toggle_pause)
        menu.append(pause)

        self._sep(menu)
        quit_item = Gtk.MenuItem(label="Quit (daemon keeps running)")
        quit_item.connect("activate", lambda *_: Gtk.main_quit())
        menu.append(quit_item)

        menu.show_all()
        self.indicator.set_menu(menu)
        self._current_menu = menu  # keep a reference so it isn't garbage collected

    @staticmethod
    def _status_text(st):
        if st["paused"]:
            return "paused — daemon running, brightness untouched"
        if st.get("holder"):
            return f"camera shared with {st['holder']} — screen stays on"
        level = st["last_set"]
        mode = f"manual cap {st['override']}%" if st["override"] is not None else (
            "auto (day full)" if st["state"] == "present" else "auto"
        )
        if st["state"] == "countdown":
            return f"away — screen off in ≤30s, now {level}%"
        if st["state"] == "absent":
            return f"away — screen off ({level}%)"
        if st["present"] or st["state"] == "present":
            return f"present — {level}% ({mode})"
        return "starting…"


def main():
    if not os.path.exists(SOCK_PATH):
        print(
            f"warning: no daemon socket at {SOCK_PATH} — is presence-brightness running?",
            file=sys.stderr,
        )
    tray = Tray()
    Gtk.main()


if __name__ == "__main__":
    main()
