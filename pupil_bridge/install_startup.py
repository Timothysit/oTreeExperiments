"""Start the pupil bridge automatically at Windows login (no console window).

Run with the Python environment the bridge should use, from anywhere:

    python pupil_bridge/install_startup.py           # add to Startup
    python pupil_bridge/install_startup.py --remove  # remove from Startup

It puts a shortcut "Pupil bridge" in your Startup folder that runs this
repository's pupil_bridge.py with pythonw.exe. Output goes to pupil_bridge/logs/.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

BRIDGE = Path(__file__).resolve().parent / "pupil_bridge.py"
STARTUP = Path(os.environ["APPDATA"]) / "Microsoft/Windows/Start Menu/Programs/Startup"
SHORTCUT = STARTUP / "Pupil bridge.lnk"


def ps_quote(value):
    return "'" + str(value).replace("'", "''") + "'"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--remove", action="store_true", help="remove the Startup shortcut")
    args = parser.parse_args()

    if args.remove:
        SHORTCUT.unlink(missing_ok=True)
        print(f"Removed {SHORTCUT}")
        return

    pythonw = Path(sys.executable).with_name("pythonw.exe")
    if not pythonw.exists():
        sys.exit(f"pythonw.exe not found next to {sys.executable}")
    script = "; ".join([
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(" + ps_quote(SHORTCUT) + ")",
        "$s.TargetPath = " + ps_quote(pythonw),
        "$s.Arguments = " + ps_quote(f'"{BRIDGE}"'),
        "$s.WorkingDirectory = " + ps_quote(BRIDGE.parent.parent),
        "$s.Description = 'Pupil bridge for the oTree experiments'",
        "$s.Save()",
    ])
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    print(f"Added {SHORTCUT}\n  runs: {pythonw} \"{BRIDGE}\"\n"
          "It starts at the next login; to start it now, double-click the shortcut.")


if __name__ == "__main__":
    main()
