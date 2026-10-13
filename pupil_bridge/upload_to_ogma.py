"""Upload finished Pupil recordings and the bridge's logs to ogma, then free the disk.

Runs nightly on the headset laptop (Windows scheduled task, see --install).
Each recording the bridge named (<date>_<room label>_<participant code> in
Pupil Capture's recordings folder) and each closed log in pupil_bridge/logs
(bridge_*.log, annotations_*.jsonl) is

1. copied to ogma over SSH (host "swc" in ~/.ssh/config), resuming files
   that were only partly copied last time;
2. checked: the SHA-256 of every file on ogma must match the local one;
3. deleted locally, only if all its files matched (not with --keep).

Skipped and retried next time: everything while Pupil Capture records (the
run stops), recordings changed in the last hour or without info.player.json,
and the logs the running bridge still writes to. Older recordings named
<YYYY_MM_DD> (Pupil Capture's default) are left alone.

    python pupil_bridge/upload_to_ogma.py --dry-run   # show what would be copied
    python pupil_bridge/upload_to_ogma.py             # copy, check, delete
    python pupil_bridge/upload_to_ogma.py --install   # every night at 02:00
    python pupil_bridge/upload_to_ogma.py --remove    # remove the nightly task

Under pythonw (the scheduled task), output goes to logs/ogma_upload_<date>.log.
"""
import argparse
import hashlib
import json
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path

from install_windows import ps_quote
from pupil_bridge import LOG_DIR, recordings_dir

SSH_HOST = "swc"
REMOTE_ROOT = "/ceph/ogma/delab/lab-members/Tim/human_mp/experiment_data"
REMOTE_LOGS = REMOTE_ROOT + "/pupil_bridge_logs"
STATUS_URL = "http://127.0.0.1:8765/status"

RECORDING_NAME = re.compile(r"^\d{4}-\d{2}-\d{2}_(.+)_[^_]+$")  # see pupil_bridge.recording_name
# Only sessions from the lab room are uploaded; demo links ("nolabel") and tests stay local.
ROOM_LABELS = Path(__file__).resolve().parent.parent / "_rooms" / "mp_lab.txt"
LOG_PATTERNS = ("bridge_*.log", "annotations_*.jsonl")
MIN_AGE_S = 3600  # a recording untouched this long is finished

TASK_NAME = "Pupil upload to ogma"
TASK_TIME = "02:00"
SCRIPT = Path(__file__).resolve()

# Windows' own OpenSSH, also when the scheduled task's PATH lacks it
OPENSSH_DIR = Path(r"C:\Windows\System32\OpenSSH")
SSH = str(OPENSSH_DIR / "ssh.exe") if OPENSSH_DIR.exists() else "ssh"
SFTP = str(OPENSSH_DIR / "sftp.exe") if OPENSSH_DIR.exists() else "sftp"
SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=30"]


class PupilRecording(Exception):
    """Pupil Capture started recording: stop uploading until next time."""


class Remote:
    def ssh(self, command, stdin=None):
        result = subprocess.run([SSH, *SSH_OPTIONS, SSH_HOST, command], input=stdin,
                                capture_output=True, text=True, timeout=3 * 3600)
        if result.returncode != 0:
            raise RuntimeError(f"ssh {command!r} failed: {result.stderr.strip()}")
        return result.stdout

    def sizes(self, remote_dir):
        """{relative path: size} of the files under remote_dir ({} if it doesn't exist)."""
        out = self.ssh(f"cd {shlex.quote(remote_dir)} 2>/dev/null && find . -type f -printf '%P\\t%s\\n' || true")
        return {rel: int(size) for rel, size in (line.rsplit("\t", 1) for line in out.splitlines())}

    def sha256(self, remote_dir, rels):
        out = self.ssh(f"cd {shlex.quote(remote_dir)} && xargs -0 -r sha256sum --",
                       stdin="\0".join(rels))
        return {line[66:]: line[:64] for line in out.splitlines()}

    def mkdirs(self, dirs):
        self.ssh("mkdir -p " + " ".join(shlex.quote(d) for d in sorted(dirs)))

    def put(self, local, remote_path, resume):
        command = "reput" if resume else "put"
        result = subprocess.run([SFTP, *SSH_OPTIONS, "-b", "-", SSH_HOST],
                                input=f'{command} "{local.as_posix()}" "{remote_path}"\n',
                                capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"sftp {command} {local} failed: {result.stderr.strip()}")

    def remove(self, remote_dir, rels):
        self.ssh(f"cd {shlex.quote(remote_dir)} && xargs -0 -r rm -f --", stdin="\0".join(rels))


def bridge_status():
    """The running bridge's /status, or {} when it isn't running."""
    try:
        with urllib.request.urlopen(STATUS_URL, timeout=5) as response:
            return json.load(response)
    except (OSError, ValueError):
        return {}


def check_not_recording():
    current = bridge_status().get("recording")
    if current:
        raise PupilRecording(f"Pupil Capture is recording {current.get('rec_path')}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def in_use(path, bridge_log):
    """True for the running bridge's annotation log (opened for each line) and
    for files another process keeps open (its output log): Windows then
    refuses to rename them."""
    if bridge_log and path == Path(bridge_log):
        return True
    probe = path.with_name(path.name + ".probe")
    try:
        path.rename(probe)
    except PermissionError:
        return True
    probe.rename(path)
    return False


def gb(n_bytes):
    return f"{n_bytes / 1e9:.1f} GB"


def upload(remote, files, remote_dir, dry_run):
    """Copy files ({relative path: local Path}) to remote_dir. Returns the
    relative paths whose copy on ogma has the same SHA-256 (none on a dry run)."""
    have = remote.sizes(remote_dir)
    # a bigger file on ogma isn't ours to overwrite (e.g. a log the bridge restarted)
    bigger = {rel for rel, path in files.items() if have.get(rel, -1) > path.stat().st_size}
    for rel in sorted(bigger):
        print(f"  {rel}: ogma's copy is bigger than the local one, left as is", flush=True)
    files = {rel: path for rel, path in files.items() if rel not in bigger}
    todo = {rel: path for rel, path in files.items() if have.get(rel) != path.stat().st_size}
    print(f"  {len(files)} files, {gb(sum(p.stat().st_size for p in files.values()))}; "
          f"to copy: {len(todo)} files, {gb(sum(p.stat().st_size for p in todo.values()))}", flush=True)
    if dry_run:
        return set()

    remote.mkdirs({posixpath.dirname(f"{remote_dir}/{rel}") for rel in files})
    for rel, path in todo.items():
        check_not_recording()
        size = path.stat().st_size
        resume = 0 < have.get(rel, 0) < size
        started = time.monotonic()
        remote.put(path, f"{remote_dir}/{rel}", resume)
        seconds = time.monotonic() - started
        print(f"  {'resumed' if resume else 'copied'} {rel} ({gb(size)}, "
              f"{size / 1e6 / max(seconds, 1e-3):.0f} MB/s)", flush=True)

    theirs = remote.sha256(remote_dir, list(files))
    ok = {rel for rel, path in files.items() if theirs.get(rel) == sha256(path)}
    bad = sorted(set(files) - ok)
    if bad:
        # removed so the next run copies them again from the start
        print(f"  CHECKSUM MISMATCH, removed from ogma: {', '.join(bad)}", flush=True)
        remote.remove(remote_dir, bad)
    return ok


def finished_recordings(root):
    labels = set(ROOM_LABELS.read_text(encoding="utf-8").split())
    now = time.time()
    for folder in sorted(root.iterdir()):
        name = RECORDING_NAME.match(folder.name)
        if not (folder.is_dir() and name):
            continue
        if name.group(1) not in labels:
            print(f"{folder.name}: not a lab room label, left local", flush=True)
            continue
        files = [p for p in folder.rglob("*") if p.is_file()]
        sessions = [d for d in folder.iterdir() if d.is_dir()]
        if not files or not all((d / "info.player.json").exists() for d in sessions):
            print(f"{folder.name}: not finished (no info.player.json), skipped", flush=True)
        elif now - max(p.stat().st_mtime for p in files) < MIN_AGE_S:
            print(f"{folder.name}: changed in the last hour, skipped", flush=True)
        else:
            yield folder


def upload_recordings(remote, dry_run, keep):
    root = recordings_dir()
    print(f"Recordings in {root}", flush=True)
    for folder in finished_recordings(root):
        check_not_recording()
        print(f"{folder.name}:", flush=True)
        files = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*")) if p.is_file()}
        ok = upload(remote, files, f"{REMOTE_ROOT}/{folder.name}", dry_run)
        if dry_run:
            continue
        if ok != set(files):
            print(f"  not deleted: {len(files) - len(ok)} files differ, retried next time", flush=True)
        elif keep:
            print("  all files match on ogma (kept, --keep)", flush=True)
        else:
            shutil.rmtree(folder)
            print("  all files match on ogma, deleted locally", flush=True)


def upload_logs(remote, dry_run, keep):
    logs = sorted(p for pattern in LOG_PATTERNS for p in LOG_DIR.glob(pattern))
    bridge_log = bridge_status().get("log_file")
    active = [p for p in logs if in_use(p, bridge_log)]
    files = {p.name: p for p in logs if p not in active}
    print(f"Bridge logs in {LOG_DIR} ({len(active)} still open, skipped):", flush=True)
    if not files:
        return
    ok = upload(remote, files, REMOTE_LOGS, dry_run)
    if dry_run:
        return
    if not keep:
        for name in ok:
            files[name].unlink()
    print(f"  {len(ok)} of {len(files)} match on ogma{' (kept, --keep)' if keep else ', deleted locally'}",
          flush=True)


def install(pythonw):
    script = "; ".join([
        "$a = New-ScheduledTaskAction -Execute " + ps_quote(pythonw)
        + " -Argument " + ps_quote(f'"{SCRIPT}"') + " -WorkingDirectory " + ps_quote(SCRIPT.parent.parent),
        "$t = New-ScheduledTaskTrigger -Daily -At " + ps_quote(TASK_TIME),
        # runs only on mains power (the default), and wakes the laptop for it
        "$s = New-ScheduledTaskSettingsSet -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 12)",
        "Register-ScheduledTask -TaskName " + ps_quote(TASK_NAME) + " -Action $a -Trigger $t -Settings $s"
        " -Description 'Uploads Pupil recordings and bridge logs to ogma' -Force | Out-Null",
    ])
    subprocess.run(["powershell", "-NoProfile", "-Command", script], check=True)
    print(f'Scheduled task "{TASK_NAME}": daily at {TASK_TIME}, runs {pythonw} "{SCRIPT}"')


def remove():
    subprocess.run(["powershell", "-NoProfile", "-Command",
                    "Unregister-ScheduledTask -TaskName " + ps_quote(TASK_NAME) + " -Confirm:$false"],
                   check=True)
    print(f'Removed scheduled task "{TASK_NAME}"')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="only show what would be copied")
    parser.add_argument("--keep", action="store_true", help="don't delete local copies after checking")
    parser.add_argument("--install", action="store_true", help=f"run this nightly at {TASK_TIME}")
    parser.add_argument("--remove", action="store_true", help="remove the nightly task")
    args = parser.parse_args(argv)

    if args.install:
        pythonw = Path(sys.executable).with_name("pythonw.exe")
        if not pythonw.exists():
            sys.exit(f"pythonw.exe not found next to {sys.executable}")
        install(pythonw)
        return
    if args.remove:
        remove()
        return

    if sys.stdout is None:  # pythonw: no console
        LOG_DIR.mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(LOG_DIR / f"ogma_upload_{datetime.now():%Y-%m-%d_%H%M%S}.log",
                                       "a", encoding="utf-8", buffering=1)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} upload to {SSH_HOST}:{REMOTE_ROOT}"
          f"{' (dry run)' if args.dry_run else ''}", flush=True)
    remote = Remote()
    try:
        check_not_recording()
        upload_recordings(remote, args.dry_run, args.keep)
        upload_logs(remote, args.dry_run, args.keep)
    except PupilRecording as e:
        print(f"Stopped: {e}. The rest is uploaded next time.", flush=True)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        print(f"FAILED: {e}", flush=True)
        sys.exit(1)
    print(f"{datetime.now():%Y-%m-%d %H:%M:%S} done", flush=True)


if __name__ == "__main__":
    main()
