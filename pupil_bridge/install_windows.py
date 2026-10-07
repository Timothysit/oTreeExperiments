"""Set up the pupil bridge on the headset laptop (Windows, current user only).

Run with the Python environment the bridge should use, from anywhere:

    python pupil_bridge/install_windows.py           # install
    python pupil_bridge/install_windows.py --remove  # undo

It installs two ways of starting this repository's pupil_bridge.py with
pythonw.exe (no window; output goes to pupil_bridge/logs/):

- a "Pupil bridge" shortcut in your Startup folder, so it starts at login;
- the link type pupilbridge: (registry key HKEY_CURRENT_USER\\Software\\Classes\\pupilbridge),
  so the experiment pages can offer a "Start pupil bridge" button. Chrome asks
  once before opening it; tick "Always allow" for the experiment site.

Neither needs admin rights. Starting the bridge while it is already running is
harmless: the second copy exits.
"""
import argparse
import os
import subprocess
import sys
import winreg
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent / "pupil_bridge.py"
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
SHORTCUT = STARTUP / "Pupil bridge.lnk"
PROTOCOL_KEY = r"Software\Classes\pupilbridge"


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def install_shortcut(pythonw):
    script = "; ".join([
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + ps_quote(SHORTCUT) + ")",
        "$s.TargetPath = " + ps_quote(pythonw),
        "$s.Arguments = " + ps_quote(f'"{BRIDGE}"'),
        "$s.WorkingDirectory = " + ps_quote(BRIDGE.parent.parent),
        "$s.Description = 'Pupil bridge for the oTree experiments'",
        "$s.Save()",
    ])
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    print(f"Startup shortcut: {SHORTCUT}")


def install_protocol(pythonw):
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, PROTOCOL_KEY) as key:
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, "URL:Pupil bridge")
        winreg.SetValueEx(key, "URL Protocol", 0, winreg.REG_SZ, "")
    with winreg.CreateKey(winreg.HKEY_CURRENT_USER, PROTOCOL_KEY + r"\shell\open\command") as key:
        # the link itself (pupilbridge:start) is not passed on; it only starts the bridge
        winreg.SetValueEx(key, None, 0, winreg.REG_SZ, f'"{pythonw}" "{BRIDGE}"')
    print(rf"Link type pupilbridge: HKEY_CURRENT_USER\{PROTOCOL_KEY}")


def remove():
    SHORTCUT.unlink(missing_ok=True)
    print(f"Removed {SHORTCUT}")
    for sub in (r"\shell\open\command", r"\shell\open", r"\shell", ""):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, PROTOCOL_KEY + sub)
        except FileNotFoundError:
            pass
    print(rf"Removed HKEY_CURRENT_USER\{PROTOCOL_KEY}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remove", action="store_true", help="remove the shortcut and link type")
    args = parser.parse_args()

    if args.remove:
        remove()
        return

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        sys.exit(f"pythonw.exe not found next to {sys.executable}")
    install_shortcut(pythonw)
    install_protocol(pythonw)
    print(f"Both run: {pythonw} \"{BRIDGE}\"\n"
          "The bridge starts at the next login; to start it now, double-click the shortcut.")


if __name__ == "__main__":
    main()
