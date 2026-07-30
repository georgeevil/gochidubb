#!/usr/bin/env python3
"""TachiDUBB Server Manager — start, stop, restart, and monitor the server.

Usage:
    python tools/tachidubb_serverctl.py start     # Start server in background
    python tools/tachidubb_serverctl.py stop      # Graceful stop
    python tools/tachidubb_serverctl.py restart   # Restart (stop + start)
    python tools/tachidubb_serverctl.py status    # Check if running
    python tools/tachidubb_serverctl.py logs      # Tail server logs
    python tools/tachidubb_serverctl.py foreground # Run in foreground (for dev)

The server writes a PID file to {project_root}/.tachidubb.pid on startup
and removes it on clean shutdown. The manager uses this to track the process.

On macOS, a launchd plist is also provided for auto-start on login.
"""
from __future__ import annotations

import argparse
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
SERVER_SCRIPT = PROJECT_ROOT / "server.py"
PID_FILE = PROJECT_ROOT / ".tachidubb.pid"
LOG_FILE = PROJECT_ROOT / "tachidubb.log"
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
# Fallback: use system python if venv not found
FALLBACK_PYTHON = sys.executable

log = logging.getLogger("tachidubb.serverctl")


def _python() -> str:
    """Return the venv python if it exists, else system python."""
    if VENV_PYTHON.exists():
        return str(VENV_PYTHON)
    return FALLBACK_PYTHON


def _read_pid() -> int | None:
    """Read PID from the PID file. Returns None if file doesn't exist or is invalid."""
    if not PID_FILE.exists():
        return None
    try:
        raw = PID_FILE.read_text().strip()
        return int(raw) if raw else None
    except (ValueError, OSError):
        return None


def _write_pid(pid: int) -> None:
    """Write PID to the PID file atomically."""
    PID_FILE.write_text(str(pid))


def _remove_pid() -> None:
    """Remove the PID file if it exists."""
    try:
        PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


def _is_running(pid: int | None) -> bool:
    """Check if a process with the given PID is alive and is a TachiDUBB server."""
    if pid is None:
        return False
    try:
        # Sending signal 0 checks if the process exists without killing it
        os.kill(pid, 0)
        return True
    except (OSError, PermissionError):
        return False


def _find_server_processes() -> list[int]:
    """Find all TachiDUBB server processes (not just the one in the PID file).

    Uses `ps` to find python processes running server.py. This catches
    orphaned processes whose PID file was deleted or stale.
    """
    pids = []
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            if "server.py" in line and "python" in line:
                parts = line.split()
                if parts:
                    try:
                        pids.append(int(parts[1]))
                    except (ValueError, IndexError):
                        pass
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return pids


# ── Commands ───────────────────────────────────────────────────────────

def cmd_status() -> dict:
    """Check server status. Returns a dict with status info."""
    pid = _read_pid()
    running = _is_running(pid)

    # Also scan for orphaned server processes
    all_server_pids = _find_server_processes()
    orphaned = [p for p in all_server_pids if p != pid]

    info = {
        "running": running,
        "pid": pid,
        "pid_file": str(PID_FILE),
        "orphaned_pids": orphaned,
        "log_file": str(LOG_FILE),
    }

    if running:
        print(f"✅ TachiDUBB server is RUNNING (PID {pid})")
    else:
        print("❌ TachiDUBB server is NOT running")
        if pid is not None:
            print(f"   Stale PID file found: {pid} (process no longer exists)")
            _remove_pid()

    if orphaned:
        print(f"⚠️  Found {len(orphaned)} orphaned server process(es): {orphaned}")
        print("   Run 'tachidubb-serverctl stop' to clean them up")

    return info


def cmd_start(daemon: bool = True, port: int = 8910, reload: bool = False) -> None:
    """Start the TachiDUBB server.

    Args:
        daemon: If True, run in background (detached). If False, run in foreground.
        port: Server port (default 8910).
        reload: Enable uvicorn auto-reload (for development).
    """
    # Check if already running
    pid = _read_pid()
    if _is_running(pid):
        print(f"⚠️  Server is already running (PID {pid})")
        return

    # Clean up stale PID file
    _remove_pid()

    python = _python()
    cmd = [python, str(SERVER_SCRIPT)]

    if reload:
        cmd.append("--reload")

    print(f"🚀 Starting TachiDUBB server on port {port}...")

    if daemon:
        # Start in background: redirect stdout/stderr to log file
        log_fh = open(LOG_FILE, "a")
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            cwd=str(PROJECT_ROOT),
            # Create a new process group so we can kill the whole tree
            start_new_session=True,
        )
        _write_pid(proc.pid)
        log_fh.close()

        # Wait a moment and check if it's still alive
        time.sleep(2)
        if _is_running(proc.pid):
            print(f"✅ Server started (PID {proc.pid})")
            print(f"   Logs: {LOG_FILE}")
            print(f"   URL:  http://localhost:{port}")
        else:
            print("❌ Server failed to start. Check logs:")
            _tail_logs(10)
    else:
        # Foreground: replace current process
        os.execvp(python, cmd)


def cmd_stop(force: bool = False) -> None:
    """Stop the TachiDUBB server gracefully.

    Args:
        force: If True, use SIGKILL instead of SIGTERM.
    """
    pid = _read_pid()
    all_pids = _find_server_processes()

    if not all_pids:
        print("ℹ️  No server processes found")
        _remove_pid()
        return

    signal_name = "SIGKILL" if force else "SIGTERM"
    sig = signal.SIGKILL if force else signal.SIGTERM

    # Kill the tracked PID first
    if pid and pid in all_pids:
        print(f"🛑 Stopping server (PID {pid}) with {signal_name}...")
        try:
            os.kill(pid, sig)
        except OSError as e:
            print(f"   Warning: {e}")

    # Kill any orphaned server processes
    for p in all_pids:
        if p != pid:
            print(f"🛑 Stopping orphaned server process (PID {p}) with {signal_name}...")
            try:
                os.kill(p, sig)
            except OSError as e:
                print(f"   Warning: {e}")

    # Wait for graceful shutdown (only for SIGTERM)
    if not force:
        deadline = time.time() + 10  # 10 second grace period
        while time.time() < deadline:
            still_alive = [p for p in all_pids if _is_running(p)]
            if not still_alive:
                break
            time.sleep(0.5)

        # Force kill any remaining
        for p in still_alive:
            try:
                os.kill(p, signal.SIGKILL)
            except OSError:
                pass

    _remove_pid()
    print("✅ Server stopped")


def cmd_restart(port: int = 8910, reload: bool = False) -> None:
    """Restart the server (stop + start)."""
    print("🔄 Restarting TachiDUBB server...")
    cmd_stop()
    # Brief pause to ensure port is released
    time.sleep(1)
    cmd_start(daemon=True, port=port, reload=reload)


def cmd_foreground(port: int = 8910, reload: bool = False) -> None:
    """Run the server in the foreground (for development)."""
    cmd_start(daemon=False, port=port, reload=reload)


def _tail_logs(lines: int = 30) -> None:
    """Print the last N lines of the log file."""
    if not LOG_FILE.exists():
        print("   (No log file yet)")
        return
    try:
        with open(LOG_FILE) as f:
            content = f.read()
        all_lines = content.splitlines()
        for line in all_lines[-lines:]:
            print(f"   {line}")
    except OSError as e:
        print(f"   Error reading log: {e}")


def cmd_logs(follow: bool = False, lines: int = 30) -> None:
    """Show server logs.

    Args:
        follow: If True, tail the log file continuously (like `tail -f`).
        lines: Number of recent lines to show.
    """
    if not LOG_FILE.exists():
        print("ℹ️  No log file found. Start the server first.")
        return

    if follow:
        # Use `tail -f` for live following
        try:
            subprocess.run(
                ["tail", "-f", "-n", str(lines), str(LOG_FILE)],
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: simple Python follow
            print(f"📋 Following {LOG_FILE} (Ctrl+C to stop)...")
            try:
                with open(LOG_FILE) as f:
                    # Seek to end
                    f.seek(0, 2)
                    while True:
                        line = f.readline()
                        if line:
                            print(line.rstrip())
                        else:
                            time.sleep(0.1)
            except KeyboardInterrupt:
                pass
    else:
        print(f"📋 Last {lines} lines of {LOG_FILE}:")
        _tail_logs(lines)


def cmd_install_launchd() -> None:
    """Install a macOS launchd plist for auto-start on login.

    This creates ~/Library/LaunchAgents/com.tachidubb.server.plist
    which starts the server automatically when you log in.
    """
    if sys.platform != "darwin":
        print("❌ launchd is only available on macOS")
        return

    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / "com.tachidubb.server.plist"

    python = _python()
    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.tachidubb.server</string>
    <key>ProgramArguments</key>
    <array>
        <string>{python}</string>
        <string>{SERVER_SCRIPT}</string>
    </array>
    <key>WorkingDirectory</key>
    <string>{PROJECT_ROOT}</string>
    <key>StandardOutPath</key>
    <string>{LOG_FILE}</string>
    <key>StandardErrorPath</key>
    <string>{LOG_FILE}</string>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>ThrottleInterval</key>
    <integer>5</integer>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>{os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin")}</string>
    </dict>
</dict>
</plist>
"""
    plist_path.write_text(plist_content)
    print(f"✅ Created {plist_path}")
    print("   To load it now: launchctl load ~/Library/LaunchAgents/com.tachidubb.server.plist")
    print("   To unload:      launchctl unload ~/Library/LaunchAgents/com.tachidubb.server.plist")


def cmd_uninstall_launchd() -> None:
    """Remove the macOS launchd plist."""
    plist_path = Path.home() / "Library" / "LaunchAgents" / "com.tachidubb.server.plist"
    if plist_path.exists():
        # Try to unload first
        subprocess.run(
            ["launchctl", "unload", str(plist_path)],
            capture_output=True,
        )
        plist_path.unlink()
        print(f"✅ Removed {plist_path}")
    else:
        print("ℹ️  No launchd plist found")


# ── CLI ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TachiDUBB Server Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/tachidubb_serverctl.py start
  python tools/tachidubb_serverctl.py stop
  python tools/tachidubb_serverctl.py restart
  python tools/tachidubb_serverctl.py status
  python tools/tachidubb_serverctl.py logs --follow
  python tools/tachidubb_serverctl.py foreground --reload
  python tools/tachidubb_serverctl.py install-launchd
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # start
    start_parser = subparsers.add_parser("start", help="Start the server in background")
    start_parser.add_argument("--port", type=int, default=8910, help="Server port (default: 8910)")
    start_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    # stop
    stop_parser = subparsers.add_parser("stop", help="Stop the server gracefully")
    stop_parser.add_argument("--force", "-f", action="store_true", help="Force kill (SIGKILL)")

    # restart
    restart_parser = subparsers.add_parser("restart", help="Restart the server")
    restart_parser.add_argument("--port", type=int, default=8910, help="Server port (default: 8910)")
    restart_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    # status
    subparsers.add_parser("status", help="Check server status")

    # logs
    logs_parser = subparsers.add_parser("logs", help="Show server logs")
    logs_parser.add_argument("--follow", "-f", action="store_true", help="Follow log output")
    logs_parser.add_argument("--lines", "-n", type=int, default=30, help="Number of lines (default: 30)")

    # foreground
    fg_parser = subparsers.add_parser("foreground", help="Run server in foreground (dev)")
    fg_parser.add_argument("--port", type=int, default=8910, help="Server port (default: 8910)")
    fg_parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")

    # launchd
    subparsers.add_parser("install-launchd", help="Install macOS launchd auto-start plist")
    subparsers.add_parser("uninstall-launchd", help="Remove macOS launchd plist")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "start":
        cmd_start(daemon=True, port=args.port, reload=args.reload)
    elif args.command == "stop":
        cmd_stop(force=args.force)
    elif args.command == "restart":
        cmd_restart(port=args.port, reload=args.reload)
    elif args.command == "status":
        cmd_status()
    elif args.command == "logs":
        cmd_logs(follow=args.follow, lines=args.lines)
    elif args.command == "foreground":
        cmd_foreground(port=args.port, reload=args.reload)
    elif args.command == "install-launchd":
        cmd_install_launchd()
    elif args.command == "uninstall-launchd":
        cmd_uninstall_launchd()


if __name__ == "__main__":
    main()