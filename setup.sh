#!/usr/bin/env bash
# setup.sh — create venv, install deps, enable + start the systemd user service
set -euo pipefail
cd "$(dirname "$0")"

# exists and actually works (guards against venvs created/copied elsewhere)
if [ ! -x .venv/bin/python3 ] || ! .venv/bin/python3 -m pip --version >/dev/null 2>&1; then
  echo "creating venv…"
  rm -rf .venv
  python3 -m venv .venv
fi

echo "installing dependencies…"
.venv/bin/python3 -m pip install -q -r requirements.txt

echo "installing service…"
mkdir -p ~/.config/systemd/user
cp presence-brightness.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now presence-brightness.service

echo "installing tray…"
# theme icons (used by the tray by name) + start-on-login entry
mkdir -p ~/.local/share/icons/hicolor/64x64/apps ~/.config/autostart
cp assets/screen-sentry*.png ~/.local/share/icons/hicolor/64x64/apps/
if command -v gtk-update-icon-cache >/dev/null; then
  gtk-update-icon-cache -f -t ~/.local/share/icons/hicolor 2>/dev/null || true
fi
cat > ~/.config/autostart/screen-sentry-tray.desktop <<EOF
[Desktop Entry]
Type=Application
Name=screen-sentry tray
Comment=Tray control for presence-brightness
Exec=/usr/bin/python3 $HOME/Documents/projects/screen-sentry/tray.py
Icon=screen-sentry
Terminal=false
Categories=Utility;
EOF

echo "running. logs: journalctl --user -u presence-brightness -f"
systemctl --user status presence-brightness.service --no-pager || true
