"""Launch the Web UI through a Cloudflare quick tunnel and hand back its link.

``python main.py webui --public`` should be the whole ceremony: this module owns
the cloudflared child process, the public hostname it is assigned, and tearing
both down again when the server stops. Two details matter for correctness.

The hostname is *discovered*, not configured. A quick tunnel asks Cloudflare for
a random name on every start, so the address of a guest's page cannot be known
before cloudflared reports it. The web UI's Host fence is therefore updated with
that exact name the moment it appears.

The discovery thread is untrusted input. Everything cloudflared prints is
matched against a strict hostname pattern and rejected when it does not fit, so
a garbled log line can never widen the fence or end up inside a link that is
printed for a guest to open.
"""
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import threading
import time
from urllib.parse import urlsplit

DEFAULT_TIMEOUT = 40.0
WINDOWS_PATHS = (
    Path(r"C:\Program Files (x86)\cloudflared\cloudflared.exe"),
    Path(r"C:\Program Files\cloudflared\cloudflared.exe"),
    Path(os.environ.get("LOCALAPPDATA", "")) / "cloudflared" / "cloudflared.exe",
)
# RFC 1123 hostname: labels of letters, digits and inner hyphens. Brackets,
# colons, spaces and '#' can therefore never reach a fence entry or a URL.
HOSTNAME = re.compile(r"^[a-z0-9]([a-z0-9-]*[a-z0-9])?(\.[a-z0-9]([a-z0-9-]*[a-z0-9])?)+$")
# Only a quick-tunnel address is ever taken from the log. cloudflared's startup
# banner also links to its own terms and documentation, and the first URL in
# that text is not the tunnel: accepting "any https link" would hand the
# operator cloudflare.com as their invite address.
URL_IN_LOG = re.compile(r"https://([a-z0-9-]+(?:\.[a-z0-9-]+)*\.trycloudflare\.com)\b")
# Cloudflare prints the hostname before the tunnel can carry traffic. The
# connection registration is what the edge needs before a guest could get in.
READY_MARKERS = ("Registered tunnel connection", "Connection registered")


def find_executable(value=None):
    """Locate cloudflared: an explicit path, the PATH, then the usual installs."""
    if value:
        candidate = Path(value)
        if candidate.is_file():
            return str(candidate)
        raise FileNotFoundError(f"找不到 cloudflared：{value}")
    found = shutil.which("cloudflared")
    if found:
        return found
    for candidate in WINDOWS_PATHS:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError(
        "找不到 cloudflared。请先安装（winget install --id Cloudflare.cloudflared），"
        "或用 --cloudflared <路径> 指定。")


def hostname_ok(value):
    """Whether ``value`` is a hostname that may be trusted as a tunnel address."""
    if not value or len(value) > 253 or ":" in value:
        return False
    return bool(HOSTNAME.match(value))


def link_hostname(link):
    """The bare hostname of a discovered ``https://`` link, or None if unusable."""
    if not link:
        return None
    try:
        parts = urlsplit(link)
    except ValueError:
        return None
    if parts.scheme != "https" or parts.path not in ("", "/"):
        return None
    hostname = (parts.hostname or "").lower()
    return hostname if hostname_ok(hostname) else None


def parse_log_line(line):
    """The tunnel hostname announced on one log line, or None.

    Only a quick-tunnel address is ever taken from the log. cloudflared's
    startup banner also links to its own terms and documentation, so accepting
    "the first https link" would hand the operator ``cloudflare.com`` as their
    invite address; the pattern pins the ``.trycloudflare.com`` suffix instead.
    """
    for match in URL_IN_LOG.finditer(line):
        found = link_hostname(f"https://{match.group(1)}")
        if found:
            return found
    return None


class TunnelError(RuntimeError):
    """cloudflared could not be started or never produced a public hostname."""


class Tunnel:
    """A running cloudflared quick tunnel and the hostname it was given.

    ``start`` returns once the edge has registered the connection, so a printed
    link is one that already works. ``stop`` is idempotent and safe to call from
    a ``finally`` block, which is what keeps a Ctrl+C from stranding a public
    route to the operator's machine.
    """

    def __init__(self, port, executable=None, timeout=DEFAULT_TIMEOUT):
        self.port = int(port)
        self.executable = executable
        self.timeout = float(timeout)
        self.link = None
        self.hostname = None
        self._process = None
        self._lines = None
        self._reader = None

    def start(self):
        """Spawn cloudflared and block until the public hostname is known."""
        if self._process is not None:
            raise TunnelError("隧道已经启动。")
        self.executable = find_executable(self.executable)
        # cloudflared logs to stderr; the origin is always loopback because the
        # web UI is expected to bind every interface on the same machine.
        command = [self.executable, "tunnel", "--url", f"http://127.0.0.1:{self.port}",
                   "--no-autoupdate"]
        try:
            self._process = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except OSError as exc:
            raise TunnelError(f"无法启动 cloudflared：{exc}") from exc
        self._lines = queue.Queue()
        self._reader = threading.Thread(target=self._pump, args=(self._process, self._lines),
                                        daemon=True)
        self._reader.start()
        self._discover()
        return self.link

    def _pump(self, process, lines):
        """Forward cloudflared's log lines to a queue, marking the last one.

        Reading runs on its own thread so a silent cloudflared can never block
        the wait below, and the sentinel keeps the reader from outliving the
        process it reads.
        """
        try:
            for line in process.stderr:
                lines.put(line)
        except (OSError, ValueError):
            pass
        finally:
            lines.put(None)

    def _discover(self):
        """Wait for the link and the connection registration, in either order."""
        deadline = time.monotonic() + self.timeout
        registered = False
        while time.monotonic() < deadline:
            try:
                line = self._lines.get(timeout=max(0.1, deadline - time.monotonic()))
            except queue.Empty:
                break
            if line is None:
                self.stop()
                raise TunnelError(self._reason())
            if not registered and any(marker in line for marker in READY_MARKERS):
                registered = True
            if self.hostname is None:
                found = parse_log_line(line)
                if found:
                    self.hostname = found
                    self.link = f"https://{found}"
            if registered and self.hostname is not None:
                return
        self.stop()
        raise TunnelError(
            f"cloudflared 在 {self.timeout:.0f} 秒内没有给出可用的公网地址。"
            "请确认本机能访问外网（防火墙 / 代理），或稍后重试。")

    def _reason(self):
        """cloudflared's own last words, for when it quit instead of connecting."""
        tail = []
        if self._lines is not None:
            while True:
                try:
                    line = self._lines.get_nowait()
                except queue.Empty:
                    break
                if line is None:
                    break
                tail.append(line.rstrip())
        code = self._process.returncode if self._process else None
        detail = " / ".join(tail[-3:]) if tail else "没有输出"
        return f"cloudflared 启动失败（退出码 {code}）：{detail}"

    def wait(self):
        """Block until cloudflared exits, for a foreground ``--public`` run."""
        if self._process is not None:
            self._process.wait()

    def alive(self):
        return self._process is not None and self._process.poll() is None

    def stop(self):
        """Kill cloudflared and wait for it, so no public route is left behind."""
        process, self._process = self._process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                process.terminate()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    process.kill()
                    process.wait(timeout=5)
                except (OSError, subprocess.TimeoutExpired):
                    pass
        self.link = None
        self.hostname = None
