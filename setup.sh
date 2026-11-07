#!/usr/bin/env bash
# setup.sh — create venv, install deps, enable + start the systemd user service
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
  echo "creating venv…"
  python3 -m venv .venv
fi

echo "installing dependencies…"
.venv/bin/pip install -q -r requirements.txt

echo "installing service…"
mkdir -p ~/.config/systemd/user
cp presence-brightness.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now presence-brightness.service

echo "running. logs: journalctl --user -u presence-brightness -f"
systemctl --user status presence-brightness.service --no-pager || true
