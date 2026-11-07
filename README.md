**Screen-sentry**  
Watches the webcam for a person. No one there → display brightness goes to **0**.  
   
 Someone there → **max** brightness during the day, **half** otherwise.  
Runs as a plain background process (systemd user service included), with a
tray icon for manual control.  
**Resource usage (measured on this machine)**  
  
|  |  |  |  
| - | - | - |  
| **Resource** |  **Budget** |  **Measured** |  
| RAM |  < 500 MB |  **\~62 MB** |  
| CPU (avg) |  < 1% of one core |  **≤ \~0.5%** |  
| GPU / VRAM |  < 50 MB |  **0** (pure CPU, no GPU pipeline) |  
  
   
**How it stays that small**  
- Grabs one **320×240** frame every **3 seconds**, constantly (in both states).  
- Runs the **YuNet face detector** (a small CPU-only ONNX model bundled in  
 `models/`, 230 KB) — robust to heads tilted/turned away, looking down,  
   
 glasses, etc., unlike the old Haar cascades. If the model file is missing it  
   
 automatically falls back to Haar cascades (weaker but works).  
- Tolerates **1 detection blip** (`BLIP_TOLERANCE`) right after a confirmed  
   
 person, so brief detection flickers don't trigger the 30s off-countdown or  
   
 restart it while you're still there. The countdown runs from the first truly  
   
 person-free check.  
- Between samples the process is asleep — CPU is only used in short \~30 ms  
   
 bursts once every few seconds.  
- No deep learning, no GPU: VRAM usage is zero.  
**Setup**  
```  
./setup.sh          # creates .venv, installs deps, installs + starts the service  
 # or manually:  
 python3 -m venv .venv  
 .venv/bin/pip install -r requirements.txt  
   
 systemctl --user enable --now presence-brightness.service  
 journalctl --user -u presence-brightness -f  
   
```  
Manual testing:  
```  
.venv/bin/python presence_brightness.py --dry-run                       # log decisions, don't touch hardware  
 .venv/bin/python presence_brightness.py --once                          # single check, sets brightness, exits  
 .venv/bin/python presence_brightness.py --preview                       # live camera view at http://localhost:8123  
 .venv/bin/python presence_brightness.py --preview --dry-run             # do both without touching brightness  
   
```  
`--preview` serves an MJPEG stream on `127.0.0.1:8123` — open `http://localhost:8123`  
   
 in any browser to watch the camera live (with the current state drawn in the  
   
 corner), or grab a still PNG/JPEG from `http://localhost:8123/snapshot`. It is  
   
 only ever bound to localhost, nothing leaves the machine.  
Dry-run log example:  
```  
10:13:16 nobody seen — screen off at 10:13:46 unless someone appears  
 10:13:48 countdown expired, nobody for 30s — screen off  
 10:13:48 brightness -> 0% set (dry-run, hardware untouched)  
   
```  
The countdown states you'll see: `nobody seen — screen off at …` (timer starts),  
   
 `person seen — countdown canceled, screen on` (instant on), `nobody again — countdown restarted` (a missed detection resets the 30s), `countdown expired … screen off`.  
**Tray / manual override**
`./setup.sh` also installs a tray icon (appindicator, starts on login via
`~/.config/autostart`):
- status line — present/away, countdown and the currently applied level
- **Auto (webcam control)** — the normal automatic behavior
- **10% … 100%** — manual cap: while someone is present the daemon applies
  this level instead of the day/night automatic value (away still goes to 0)
- **Pause webcam control** — the daemon keeps running and watching, but stops
  touching brightness until you resume
- **Quit** — only quits the tray; the daemon service keeps running

Control channel: a unix socket, `XDG_RUNTIME_DIR/screen-sentry.sock` (fallback
`~/.cache/screen-sentry.sock`) — no TCP ports. Same thing by hand:
```
curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock http://s/status
curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{"value":30}' http://s/override
curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/auto
curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/pause
curl --unix-socket $XDG_RUNTIME_DIR/screen-sentry.sock -d '{}' http://s/enable
```
Start the tray by hand: `python3 tray.py`.

**Config**  
All knobs are constants at the top of `presence_brightness.py`:  
  
|  |  |  |  
| - | - | - |  
| **Constant** |  **Meaning** |  **Default** |  
| `SAMPLE_INTERVAL_ABSENT` / `_PRESENT` |  seconds between checks (constant, same in both states) |  3 |  
| `DET_SCORE` |  YuNet minimum face confidence (0–1); raise to reject more false faces, lower to catch harder poses |  0.5 |  
| `BLIP_TOLERANCE` |  missed checks right after a confirmed person that don't start the off-timer |  1 |  
| `PREVIEW_PORT` |  localhost port for the `--preview` camera stream |  8123 |  
| `OFF_AFTER_ABSENT` |  screen-off delay (30 s before it turns dark) |  30 s |  
| `ON_AFTER_PRESENT` |  screen-on delay (instant, first detection) |  0 |  
| `BRIGHTNESS_FULL` / `BRIGHTNESS_HALF` |  levels when present |  100 / 50 |  
| `DAY_START` / `DAY_END` |  minutes-from-midnight window that counts as day time (full when person present, half otherwise) |  07:00 → full, 18:30 → half |  
  
   
Brightness is set with the `brightness <0-100>` command (which drives all  
   
 detected ddcutil monitors); if that command is missing it falls back to writing  
   
 `/sys/class/backlight/*/brightness` directly.  
Camera feed is sampled, never saved or displayed.  
