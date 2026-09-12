"""CPU inference UI for playing the model; training files are always read-only.

Loopback stays the default and an unshared server behaves exactly as before: one
table, no credentials, no session. Sharing is opt-in and adds the two things a
guest needs: a bind host others can reach, and a per-browser session so each
player owns a private table instead of fighting over one shared board.

``--public`` adds the third: a Cloudflare quick tunnel started alongside the
server, whose randomly assigned hostname is admitted into the Host fence and
printed as one ready-to-open link.

Every share also prints a read-only ``/watch`` link. It mirrors every game in
progress for spectators, who hold no table and no CSRF token, so watching can
never move a stone or consume one of the seats.

A shared table whose browser has gone quiet for ``--table-ttl`` minutes is
released on its own, so a closed tab cannot hold a seat until restart.
"""
from concurrent.futures import ThreadPoolExecutor
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hmac
import json
import secrets
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.parse import unquote_plus, urlparse

import numpy as np
import torch

from .candidates import immediate_wins
from .game import Game
from .network import Network, Evaluator, architecture_of
from .search import MCTS
from .config import mix_vector
from .search import search_options
from .storage import load_model_state
from .tunnel import Tunnel, TunnelError

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("web")
RULES = ("freestyle", "renju")
RECENT_LIMIT = 8
# Root value (the AI's own perspective) above which it considers the game clearly
# its own and starts to talk. The board-threat categories below are exact; this
# one is only the search's self-assessment, so it stays deliberately high.
TAUNT_VALUE = 0.9
CHECKPOINT_HELP = "请让训练保存检查点后再试（每轮结束时写入）。"
# Concurrent guest tables. Each one holds its own network and MCTS tree, so the
# cap bounds the service's memory and CPU rather than being a nicety: measured
# at roughly 16-30 MB per table with a game in progress under the legacy 64x6
# network (weights ~2 MB; the search tree dominated). The hybrid 128x10 network
# carries about 2.8 M parameters (~11 MB), which shifts that figure but leaves
# the search tree and CPU contention as the binding constraints.
LAN_SESSION_LIMIT = 10
# A table whose browser has gone quiet this long is released: an open page
# polls every second, so only an abandoned tab ever goes quiet.
DEFAULT_IDLE_TTL = 5 * 60.0
ALL_INTERFACES = "0.0.0.0"
SESSION_COOKIE = "renju_session"
WATCH_COOKIE = "renju_watch"
TOKEN_QUERY = "k"
WATCH_QUERY = "w"
CONFIG_TTL_SECONDS = 2.0
PLAYER_ASSETS = {"/": ("index.html", "text/html"),
                 "/app.js": ("app.js", "text/javascript"),
                 "/board.js": ("board.js", "text/javascript"),
                 "/style.css": ("style.css", "text/css")}
WATCH_ASSETS = {"/watch": ("watch.html", "text/html"),
                "/watch.js": ("watch.js", "text/javascript")}
# The watch page shares the stylesheet and the board code with the player
# page: a player's browser never holds a watch cookie and a spectator's never
# holds a session, so exactly these two files accept either credential.
SHARED_ASSETS = ("/style.css", "/board.js")
WATCH_SERVED = dict(WATCH_ASSETS, **{name: PLAYER_ASSETS[name] for name in SHARED_ASSETS})
WATCH_ROUTES = ("/watch", "/watch.js", "/api/watch", *SHARED_ASSETS)


def routed_ipv4():
    """The address this machine would use to reach the outside world.

    A UDP connect sends nothing; it only asks the routing table which local
    address a real flow would leave from. That is the address a guest on the
    same network can reach, unlike a VMware or Hyper-V adapter that may answer
    name resolution first.
    """
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        probe.connect(("8.8.8.8", 53))
        return probe.getsockname()[0]
    except OSError:
        return None
    finally:
        probe.close()


def get_local_ipv4():
    """Non-loopback IPv4 address to hand to guests, best effort.

    Prefers the routed address and falls back to whatever the hostname resolves
    to, which is enough on a single-adapter machine.
    """
    routed = routed_ipv4()
    if routed and not routed.startswith("127."):
        return routed
    found = []
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in found:
                found.append(address)
    except OSError:
        pass
    for address in found:
        if not address.startswith("127."):
            return address
    return found[0] if found else None


def bind_host(value):
    """Validate a ``--host`` value that is a literal address or a local name."""
    if value in (ALL_INTERFACES, "::"):
        return ALL_INTERFACES
    if value == "localhost" or value == get_local_ipv4():
        return value
    try:
        socket.getaddrinfo(value, None, socket.AF_INET)
    except OSError as exc:
        raise ValueError(f"无法解析监听地址 {value}：{exc}") from exc
    return value


def authorities(host, port, trusted=(), aliases=()):
    """Every ``Host`` value the fence accepts: the bind target plus loopback.

    A guest reaches the page by IP, so each literal address this machine owns is
    accepted on this port only, and DNS names only when the operator named them
    on the command line. Rebinding attacks need an attacker-controlled name,
    which is exactly what stays out of this set. ``aliases`` are the addresses
    other than the bind target that the operator meant to serve: the addresses
    of this machine for an all-interfaces bind, and the ones a shared server
    prints for its guests.

    A trusted name from ``--trusted-host`` is accepted both bare and with this
    port, because a reverse proxy or tunnel decides the port itself: cloudflared
    forwards the host without one, while a proxy on a non-default port may keep
    it. Names that carry an explicit port are taken literally. A name that
    starts with a dot trusts a whole subdomain tree, which is how a rotating
    quick-tunnel hostname stays reachable without editing the command.
    """
    accepted = {f"127.0.0.1:{port}", f"localhost:{port}"}
    for name in (host, *aliases):
        if not name:
            continue
        accepted.add(name if ":" in name else f"{name}:{port}")
    for name in trusted:
        if not name:
            continue
        accepted.add(name)
        if ":" not in name:
            accepted.add(f"{name}:{port}")
    return accepted


def trusted_host_match(authority, trusted):
    """Whether an authority matches a subdomain pattern in ``--trusted-host``.

    ``--trusted-host .example.com`` matches ``play.example.com`` and
    ``play.example.com:8443``, but never a bare ``example.com`` and never a
    lookalike such as ``notexample.com``. Aliases without the leading dot are
    exact entries and are handled by the plain authority set.
    """
    host = authority.rsplit(":", 1)[0] if not authority.startswith("[") else authority
    for name in trusted:
        if name.startswith(".") and (host == name[1:] or host.endswith(name)):
            return True
    return False


def proxy_patterns(trusted, tunnel):
    """The wildcard patterns this request may be matched against.

    A subdomain pattern is how a rotating tunnel hostname stays reachable. A
    running quick tunnel contributes its own ``.trycloudflare.com`` tree, so
    ``--public`` needs no ``--trusted-host`` typed by hand; an operator's
    patterns are honoured on top, because naming a tree is their explicit call.
    """
    patterns = list(trusted)
    if tunnel is not None and tunnel.hostname:
        overlay = "." + tunnel.hostname.split(".", 1)[-1]
        if overlay not in patterns:
            patterns.append(overlay)
    return [name for name in patterns if name]


def start_tunnel(server, port, executable=None, timeout=40.0):
    """Start a quick tunnel and admit its hostname in the server's Host fence.

    The hostname is known only after cloudflared reports it, so the tunnel state
    is published while the server is already serving. That is safe: the name was
    validated by :mod:`vk.tunnel`, and a request arriving before it exists is
    simply the 403 it would have received anyway.

    The name is deliberately *not* added to the fence. The fence holds the names
    that mean "this machine, reached directly", and a tunnel guest must keep
    being answered with the tunnel's own https origin rather than the operator's
    LAN link; admission comes from the tunnel pattern instead.

    Returns the public ``https://`` link, or raises :class:`TunnelError`.
    """
    tunnel = Tunnel(port, executable=executable, timeout=timeout)
    link = tunnel.start()
    with server.tunnel_lock:
        server.tunnel = tunnel
        server.tunnel_host = tunnel.hostname
    return link


def invite_url(host, port, key):
    """The link a guest opens once; the token is exchanged for a session cookie."""
    return f"http://{host}:{port}/?{TOKEN_QUERY}={key}"


def watch_url(host, port, key):
    """The read-only link: mirrors the games in progress, moves nothing."""
    return f"http://{host}:{port}/watch?{WATCH_QUERY}={key}"


def taunt_category(game, human, value=None):
    """Classify the position for a taunt bubble, or None when silence fits.

    Derived from the position alone every time the snapshot is taken, so the
    field needs no bookkeeping: a blocked open four degrades from "crushing"
    to "threat" and vanishes once the human answers it, and an undo simply
    re-derives the earlier position. A threat is only spoken while the
    threatened side must actually move -- the model brags when the human has
    to parry, pleads when it has to parry itself, and stays quiet the moment
    either side can just complete a five on its own turn.
    """
    if game.winner is not None:
        return "win" if game.winner == -human else None
    theirs = immediate_wins(game, human)
    if theirs.any() and game.player == human:
        return None
    mine = immediate_wins(game, -human)
    if mine.any():
        return ("crushing" if int(mine.sum()) >= 2 else "threat") if game.player == human else None
    if theirs.any():
        return "panic"
    if game.player == human and value is not None and value >= TAUNT_VALUE:
        return "dominant"
    return None


class Table:
    def __init__(self, root=ROOT / "runs", catalog=None, sessions=0):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.sessions = sessions
        self.game = None
        self.history = []
        self.busy = False
        self.error = None
        self.id = None
        self.info = {}
        self.legal = []
        self.seconds = None
        self.last_value = None
        self.touched = time.monotonic()
        shared = catalog if catalog is not None else Catalog(self.root)
        self.catalog = shared.catalog
        self.entries = shared.entries

    def touch(self):
        """Mark this table as just used, for the activity view and idle release."""
        self.touched = time.monotonic()

    def catalog(self, rule):
        """Validated checkpoints across legacy, pretrain, and hybrid run folders.

        The scan itself belongs to the server-wide :class:`Catalog`, so several
        guest tables never multiply the disk work that lists checkpoints.
        """
        return [dict(item) for item in self.entries(rule)]

    def models(self):
        """Checkpoint identifiers per rule; aliases follow newest valid files."""
        return {rule: {key: self.resolve(rule, key) for key in ("latest", "best")} for rule in RULES}

    def resolve(self, rule, name):
        """Map a UI selection to the checkpoint file a new game would load.

        'latest'/'best' are the two aliases the UI offers; any other string is
        taken as a checkpoint file name and reported unchanged, so the client
        can trust a listed file. Whether the file really loads is decided when
        the model is read, which happens off the request thread.
        """
        catalog = self.catalog(rule)
        if name == "latest":
            candidates = [item for item in catalog if item["kind"] == "checkpoint"] or catalog
            return candidates[0]["name"] if candidates else None
        if name == "best":
            candidates = [item for item in catalog if item["kind"] == "best"]
            return candidates[0]["name"] if candidates else None
        if any(item["name"] == name for item in catalog):
            return name
        # A simple checkpoint filename may have been pruned between the list
        # refresh and the click; let the async loader report that stale file.
        if "/" not in name and name.startswith("checkpoint-") and name.endswith(".pt"):
            return name
        return None

    def recent(self, rule, limit=RECENT_LIMIT):
        """Recently written checkpoints, newest first, for manual selection."""
        return [{key: item[key] for key in ("name", "mtime", "mb", "run", "kind")}
                for item in self.catalog(rule)[:limit]]

    def snapshot(self):
        with self.lock:
            return dict(id=self.id, busy=self.busy, error=self.error, model=dict(self.info),
                        board=self.game.board.tolist() if self.game else [0]*225,
                        player=self.game.player if self.game else 1,
                        winner=self.game.winner if self.game else None,
                        human=getattr(self, "human", 1), legal=self.legal,
                        history=list(self.history), seconds=self.seconds,
                        taunt=taunt_category(self.game, getattr(self, "human", 1),
                                             self.last_value) if self.game else None,
                        sessions=self.sessions)

    def refresh(self):
        self.game.adjudicate()
        self.legal = np.flatnonzero(self.game.legal()).tolist()

    def new(self, data):
        rule, color, checkpoint = data.get("rule"), data.get("color"), data.get("checkpoint")
        simulations = data.get("simulations", 64)
        if rule not in RULES or color not in ("black", "white") or not isinstance(checkpoint, str):
            raise ValueError("请选择有效的规则、执色和模型。")
        if type(simulations) is not int or simulations not in (32, 64, 200):
            raise ValueError("搜索次数必须为 32、64 或 200。")
        if "\\" in checkpoint or ".." in checkpoint or checkpoint.startswith("/"):
            raise ValueError("检查点名称无效。")
        with self.lock:
            self.touch()
            if self.busy:
                raise ValueError("模型正在思考，请等本手完成后再开始新局。")
            requested = checkpoint
            checkpoint = self.resolve(rule, requested)
            if "/" in requested and not checkpoint:
                raise ValueError("检查点名称无效。")
            if not checkpoint:
                raise ValueError(f"{'连珠' if rule == 'renju' else '自由五子棋'}还没有可用的模型。{CHECKPOINT_HELP}")
            self.id = secrets.token_hex(16)
            self.game = Game(rule)
            self.human = 1 if color == "black" else -1
            self.history, self.error, self.seconds = [], None, None
            self.last_value = None
            self.info = dict(rule=rule, checkpoint=requested,
                             latest=self.resolve(rule, "latest"), simulations=simulations, device="CPU")
            self.busy = True
            self.legal = []
            self.pool.submit(self.initialize, rule, checkpoint, simulations)
            return self.snapshot()

    def initialize(self, rule, checkpoint, simulations):
        try:
            path = ((self.root / rule / checkpoint) if "/" not in checkpoint
                    else (self.root / Path(checkpoint))).resolve()
            root = self.root.resolve()
            if not path.is_relative_to(root) or checkpoint not in {item["name"] for item in self.catalog(rule)}:
                raise FileNotFoundError(checkpoint)
            state = load_model_state(path, rule)
            cfg = state["config"]
            model = Network(architecture_of(cfg))
            model.load_state_dict(state["model"])
            self.tree = MCTS(Evaluator(model, "cpu", mix=mix_vector(cfg)), simulations,
                         cfg["cpuct"], **search_options(cfg))
            with self.lock:
                self.info.update(file=checkpoint, step=state.get("step"), round=state.get("round"))
            del state
            self.reply()
        except Exception as exc:
            self.failed(exc)

    def failed(self, exc):
        with self.lock:
            self.busy = False
            self.error = f"模型运行失败：{exc}。请重新开始一局。"

    def reply(self):
        start = time.monotonic()
        try:
            # Mutations are disallowed while busy. Search a copy to keep GET nonblocking.
            with self.lock:
                game = self.game.copy()
            game.adjudicate()
            action = None
            if game.winner is None and game.player != self.human:
                action = int(np.argmax(self.tree.policy(game)))
                game.move(action)
                self.tree.advance(action, game)
            with self.lock:
                self.game = game
                if action is not None:
                    self.history.append(action)
                    self.seconds = round(time.monotonic() - start, 2)
                    # The search just ran for the position the model moved from;
                    # its root value is the model's own assessment of the game.
                    self.last_value = float(self.tree.last_result.value)
                self.refresh()
                self.busy = False
        except Exception as exc:
            self.failed(exc)

    def act(self, data, undo=False):
        with self.lock:
            self.touch()
            if not self.game or data.get("id") != self.id:
                raise ValueError("对局已更新，请刷新页面。")
            if self.busy or self.error:
                raise ValueError("当前无法操作，请等待思考结束或重新开始。")
            if undo:
                # Restore the position just before the most recent human move.
                indices = [i for i in range(len(self.history)) if (1 if i % 2 == 0 else -1) == self.human]
                if not indices:
                    raise ValueError("还没有可以撤回的落子。")
                self.history = self.history[:indices[-1]]
                self.game = Game(self.info["rule"])
                for action in self.history:
                    self.game.move(action)
                self.tree = MCTS(self.tree.evaluate, self.tree.simulations, self.tree.cpuct)
                self.seconds = None
                self.last_value = None
                self.refresh()
            else:
                action = data.get("action")
                if type(action) is not int or action not in self.legal or self.game.player != self.human:
                    raise ValueError("这里不能落子：已占用、禁手或尚未轮到你。")
                self.game.move(action)
                self.history.append(action)
                self.tree.advance(action, self.game)
                self.refresh()  # sets self.legal, and adjudicates a finished game
                if self.game.winner is not None or self.game.player == self.human:
                    # The human's move ended the game: no reply is pending, so the
                    # result must be visible immediately instead of "thinking".
                    self.busy = False
                    return self.snapshot()
                self.busy = True
                self.legal = []
                self.pool.submit(self.reply)
            return self.snapshot()


class Catalog:
    """Checkpoint lists that every table shares, so guests never reload them.

    Only the file listing and its small caches are shared; the network weights
    and the search tree stay private to a table.
    """

    def __init__(self, root=ROOT / "runs"):
        self.root = Path(root)
        self.lock = threading.Lock()
        self.checkpoint_cache = {}
        self.run_cache = {}
        self.catalog_cache = {}

    def entries(self, rule):
        """Paths and names for one rule, newest first."""
        with self.lock:
            cached = self.catalog_cache.get(rule)
            now = time.monotonic()
            if cached is None or now - cached[0] >= CONFIG_TTL_SECONDS:
                cached = (now, self._scan(rule))
                self.catalog_cache[rule] = cached
            return list(cached[1])

    def catalog(self, rule):
        return [dict(item) for item in self.entries(rule)]

    def _scan(self, rule):
        items = []
        if not self.root.exists():
            return items
        for folder in self.root.iterdir():
            if not folder.is_dir():
                continue
            checkpoints = [path for path in folder.glob("*.pt")
                           if path.name == "best.pt" or path.name.startswith("checkpoint-")]
            if self._run_rule(folder, checkpoints) != rule:
                continue
            for path in checkpoints:
                stat = path.stat()
                items.append({"path": path.resolve(), "name": self._identifier(path, rule),
                              "kind": "best" if path.name == "best.pt" else "checkpoint",
                              "mtime_ns": stat.st_mtime_ns, "mb": round(stat.st_size / 1048576, 1),
                              "mtime": time.strftime("%m-%d %H:%M", time.localtime(stat.st_mtime)),
                              "run": path.parent.name})
        return sorted(items, key=lambda item: (item["mtime_ns"], item["name"]), reverse=True)

    def _checkpoint_rule(self, path):
        try:
            stat = path.stat()
            key = (str(path.resolve()), stat.st_mtime_ns, stat.st_size)
            if key not in self.checkpoint_cache:
                state = torch.load(path, map_location="cpu", weights_only=False)
                # Every format this project writes carries its rule in the config;
                # the format check that used to be here predates format 2 and hid
                # a distilled run from the model list.
                found = state.get("config", {}).get("rule")
                self.checkpoint_cache[key] = found if found in RULES else None
            return self.checkpoint_cache[key]
        except Exception:
            return None

    def _run_rule(self, folder, checkpoints):
        config = folder / "config.json"
        signature = tuple((item.name, item.stat().st_mtime_ns, item.stat().st_size)
                          for item in checkpoints)
        if config.exists():
            stat = config.stat()
            signature += ((config.name, stat.st_mtime_ns, stat.st_size),)
        key = (str(folder.resolve()), signature)
        if key not in self.run_cache:
            found = None
            if config.exists():
                try:
                    found = json.loads(config.read_text(encoding="utf-8")).get("rule")
                except (OSError, ValueError):
                    pass
            if found not in RULES and checkpoints:
                probe = min(checkpoints, key=lambda item: item.stat().st_size)
                found = self._checkpoint_rule(probe)
            self.run_cache[key] = found if found in RULES else None
        return self.run_cache[key]

    def _identifier(self, path, rule):
        legacy = (self.root / rule).resolve()
        return path.name if path.parent.resolve() == legacy else path.resolve().relative_to(self.root.resolve()).as_posix()


class Sessions:
    """One private table per browser, with a capacity that is never stolen.

    Capacity is a resource limit, not a queue: every table holds its own network
    and search tree, so admitting a fourth player would quietly multiply the
    service's memory. A full share therefore turns the next guest away with an
    explanation instead of ending somebody's game.

    What it does reclaim is seats nobody is using. A table whose browser has
    not been heard from for the idle limit is released — by the background
    reaper, or just before a full share refuses the next guest. An open page
    polls every second, so only an abandoned tab ever goes quiet.
    """

    def __init__(self, maximum=LAN_SESSION_LIMIT, catalog=None, root=ROOT / "runs",
                 idle_ttl=DEFAULT_IDLE_TTL):
        self.maximum = max(1, int(maximum))
        self.idle_ttl = None if not idle_ttl else float(idle_ttl)
        self.catalog = catalog if catalog is not None else Catalog(root)
        self.root = self.catalog.root
        # Reentrant so open() can call sweep() while holding the lock.
        self.lock = threading.RLock()
        self.tables = {}
        self._stop = threading.Event()
        if self.idle_ttl:
            threading.Thread(target=self._reap, daemon=True).start()

    def _reap(self):
        interval = min(60.0, max(0.5, self.idle_ttl / 2))
        while not self._stop.wait(interval):
            self.sweep()

    def fetch(self, token):
        """The table that ``token`` owns, or None when it is unknown or gone."""
        if not token:
            return None
        with self.lock:
            table = self.tables.get(token)
            if table is not None:
                table.touch()
        return table

    def open(self):
        """Create a table, or None while the share is at capacity."""
        with self.lock:
            if len(self.tables) >= self.maximum:
                # A stale tab holds its seat only until this sweep; after it,
                # the late guest is admitted instead of refused.
                self.sweep()
            if len(self.tables) >= self.maximum:
                return None
            token = secrets.token_urlsafe(24)
            self.tables[token] = Table(self.root, catalog=self.catalog)
            self._renumber()
            return token, self.tables[token]

    def sweep(self):
        """Release every table idle past the limit; returns the tokens released."""
        if not self.idle_ttl:
            return []
        with self.lock:
            now = time.monotonic()
            expired = [token for token, table in self.tables.items()
                       if now - table.touched > self.idle_ttl]
            dropped = [self.tables.pop(token) for token in expired]
            self._renumber()
        for table in dropped:
            # Decided under the lock; the release happens outside of it.
            table.pool.shutdown(wait=False)
        return expired

    def close(self, token):
        """Drop one table (a guest leaving) without touching the others."""
        with self.lock:
            table = self.tables.pop(token, None)
            self._renumber()
        if table is not None:
            # Nothing waits on an in-flight move: this guest's next request is
            # simply unknown, which the page already reports as "对局已更新".
            table.pool.shutdown(wait=False)
        return table is not None

    def _renumber(self):
        for index, table in enumerate(self.tables.values(), start=1):
            table.sessions = index

    def count(self):
        with self.lock:
            return len(self.tables)

    def shutdown(self):
        self._stop.set()
        with self.lock:
            tables, self.tables = list(self.tables.values()), {}
        for table in tables:
            table.pool.shutdown(wait=True)


def make_server(port=8765, root=ROOT / "runs", host="127.0.0.1", share=False,
                max_sessions=LAN_SESSION_LIMIT, password=None, trusted_hosts=(),
                idle_ttl=DEFAULT_IDLE_TTL):
    catalog = Catalog(root)
    sessions = Sessions(max_sessions, catalog, idle_ttl=idle_ttl) if share else None
    table = Table(root, catalog=catalog, sessions=1)
    request_token = secrets.token_hex(24)
    join_key = secrets.token_urlsafe(18)
    # Spectators get their own ticket: the watch link is read-only, so it
    # neither consumes a table slot nor ever satisfies a play route.
    watch_key = secrets.token_urlsafe(18)
    # Filled in after the bind, because port 0 only becomes a real port then.
    fence = set()
    configs = {}
    configs_lock = threading.Lock()
    # Set by start_tunnel() once cloudflared has reported its hostname; the
    # pattern guard and the guest-facing link both depend on it.
    tunnel_lock = threading.Lock()

    def config_payload(key):
        """Model lists per rule, cached briefly: guests poll this every 10 s."""
        now = time.monotonic()
        with configs_lock:
            cached = configs.get(key)
            if cached is None or now - cached[0] >= CONFIG_TTL_SECONDS:
                cached = (now, {"token": request_token, "models": table.models(),
                                "recent": {rule: table.recent(rule) for rule in RULES}})
                configs[key] = cached
            return cached[1]

    def guest_address():
        """The address to put in the invite link, or None when it is unreachable.

        A loopback bind cannot be reached from another machine, and an
        all-interfaces bind has no name of its own, so both are replaced by the
        machine's LAN address when one exists.
        """
        if host == ALL_INTERFACES:
            return get_local_ipv4()
        if host == "127.0.0.1":
            return None
        return host

    # Host values other than the bind target that this server should answer to.
    # An all-interfaces bind is reachable under any address of this machine, so
    # each of them has to pass the fence; a specific bind has exactly one name.
    aliases = []
    if host == ALL_INTERFACES:
        for source in (routed_ipv4(), get_local_ipv4()):
            if source and source not in aliases:
                aliases.append(source)
        try:
            aliases.extend(info[4][0] for info in
                           socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
                           if info[4][0] not in aliases)
        except OSError:
            pass

    def invite(port_number=None):
        """Public join link, or None when this bind is not reachable by others."""
        if not share:
            return None
        address = guest_address()
        if address is None:
            return None
        return invite_url(address, port_number or port, join_key)

    def watch(port_number=None):
        """Read-only spectate link, or None when this bind is not reachable."""
        if not share:
            return None
        address = guest_address()
        if address is None:
            return None
        return watch_url(address, port_number or port, watch_key)

    def watch_state():
        """Every game in progress, in table order, for the watch page feed.

        A spectator owns no table, so this is read-only by construction: the
        payload carries positions and history, while every mutating route
        still demands a session cookie the watch page never had.
        """
        with sessions.lock:
            tables = sorted(sessions.tables.values(), key=lambda item: item.sessions)
        playing = [item.snapshot() for item in tables]
        return {"tables": [snap for snap in playing if snap["id"] is not None]}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "ViskWebUI"

        # --- plumbing -----------------------------------------------------
        def send(self, status, body, kind="application/json; charset=utf-8", headers=()):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(body)

        def redirect(self, location, headers=()):
            self.send_response(HTTPStatus.SEE_OTHER)
            self.send_header("Location", location)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            for name, value in headers:
                self.send_header(name, value)
            self.end_headers()

        def host_ok(self):
            authority = self.headers.get("Host", "")
            return authority in fence or \
                trusted_host_match(authority, proxy_patterns(trusted_hosts, server.tunnel))

        def proxied_authority(self):
            """This request's own name, when it arrived through a proxy.

            A guest behind a tunnel must be handed a link it can actually open;
            the LAN address in the printed link would not resolve for it. Direct
            LAN visitors keep the address the operator printed at startup.

            The comparison is against the whole authority, port included,
            because that is what the fence stores; cloudflared forwards the bare
            hostname, which no fence entry ever matches.
            """
            authority = self.headers.get("Host", "")
            if not authority or authority in fence:
                return None
            if server.tunnel_host and authority.split(":")[0] == server.tunnel_host:
                return authority
            return authority if trusted_host_match(authority, trusted_hosts) else None

        def public_invite(self):
            """An invite link on this request's own origin, when a proxy served it.

            A guest behind a tunnel must be handed a link it can actually open;
            the LAN address in the printed link would not resolve for it. Direct
            LAN visitors keep the address the operator printed at startup.
            """
            origin = self.proxied_origin()
            return None if origin is None else f"{origin}/?{TOKEN_QUERY}={join_key}"

        def public_watch(self):
            """The spectate link on this request's own origin, same as invites."""
            origin = self.proxied_origin()
            return None if origin is None else f"{origin}/watch?{WATCH_QUERY}={watch_key}"

        def proxied_origin(self):
            """``scheme://authority`` of this request when a proxy served it."""
            authority = self.proxied_authority()
            if authority is None:
                return None
            forwarded = self.headers.get("X-Forwarded-Proto", "").lower()
            scheme = "https" if forwarded == "https" or self.server.server_port in (80, 443) else "http"
            return f"{scheme}://{authority}"

        def cookie(self, name=SESSION_COOKIE):
            for chunk in self.headers.get("Cookie", "").split(";"):
                key, _, value = chunk.strip().partition("=")
                if key == name:
                    return value
            return None

        def watch_header(self):
            return ("Set-Cookie",
                    f"{WATCH_COOKIE}={watch_key}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800")

        def watch_route(self, path):
            """Serve the read-only spectate page: HTML, script, and live feed.

            A spectator owns no table and never sees the CSRF token, so every
            play route keeps refusing them; the watch credential only ever
            reads. The ticket in the link is exchanged for a cookie exactly
            like an invite key, so it leaves the address bar after one visit.
            """
            key = self.query().get(WATCH_QUERY, "")
            granted = bool(key and hmac.compare_digest(key, watch_key)) or \
                bool(hmac.compare_digest(self.cookie(WATCH_COOKIE) or "", watch_key))
            if granted:
                if key and path == "/watch":
                    return self.redirect("/watch", [self.watch_header()])
            elif not (path in SHARED_ASSETS and sessions.fetch(self.cookie()) is not None):
                if path == "/api/watch":
                    return self.send(HTTPStatus.FORBIDDEN, {"error": "请使用观战链接进入。"})
                return self.notice(HTTPStatus.FORBIDDEN, "棋间 · 观战需要链接",
                                   "请使用服务提供者给出的观战链接进入。观战是只读的，也不会占用棋桌。")
            if path == "/api/watch":
                return self.send(HTTPStatus.OK, watch_state())
            if path not in WATCH_SERVED:
                return self.send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            name, kind = WATCH_SERVED[path]
            self.send(HTTPStatus.OK, (STATIC / name).read_bytes(), kind + "; charset=utf-8")

        def cookie_header(self, token):
            return ("Set-Cookie",
                    f"{SESSION_COOKIE}={token}; Path=/; HttpOnly; SameSite=Strict; Max-Age=604800")

        def query(self):
            """Single-valued query parameters of this request, last one wins."""
            found = {}
            for chunk in urlparse(self.path).query.split("&"):
                name, _, value = chunk.partition("=")
                if name:
                    found[name] = value
            return found

        def page(self, title, body):
            self.send(HTTPStatus.OK, LOGIN_PAGE.replace("{{TITLE}}", title).replace("{{BODY}}", body).encode("utf-8"),
                      "text/html; charset=utf-8", [("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")])

        def notice(self, status, title, message):
            self.send(status, LOGIN_PAGE.replace("{{TITLE}}", title)
                      .replace("{{BODY}}", f'<p class="notice">{message}</p>').encode("utf-8"),
                      "text/html; charset=utf-8", [("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")])

        # --- access control ----------------------------------------------
        def submitted(self):
            """Read this request's body once and parse it as a URL-encoded form.

            The one form in this UI is the password prompt, and the value rides
            the body rather than the URL, so any character is legal in a
            password and nothing lands in browser history. The raw body is kept
            because :meth:`do_POST` may read the same bytes as JSON.
            """
            try:
                size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                size = 0
            self.body = self.rfile.read(size) if 0 < size <= 4096 else b""
            fields = {}
            for chunk in self.body.decode("utf-8", "replace").split("&"):
                name, _, value = chunk.partition("=")
                if name:
                    fields[unquote_plus(name)] = unquote_plus(value)
            return fields

        def admit(self):
            """Resolve this request to a table, or answer it and set ``answered``.

            Without sharing the server keeps its original single-table
            behaviour. With sharing, the first visit exchanges the invite key
            (or the password) for a session cookie, so the key never has to
            stay in the address bar. The body is read exactly once here, so a
            JSON route never sees a drained stream and a form never sees a
            half-read one.
            """
            self.body, self.secret, self.answered = None, "", False
            try:
                self.size = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.size = 0
            if sessions is None:
                return table
            path = urlparse(self.path).path
            if self.command == "POST" and path == "/":
                self.secret = self.submitted().get("p", "")
            elif self.command == "GET":
                self.secret = self.query().get("p", "")
            key = self.query().get(TOKEN_QUERY, "")
            granted = bool(key and hmac.compare_digest(key, join_key)) or \
                bool(password and self.secret and hmac.compare_digest(self.secret, password))
            if granted:
                opened = sessions.open()
                if opened is None:
                    self.answered = True
                    later = ("闲置的棋桌会自动让出，请稍后再试"
                             if sessions.idle_ttl else "请稍后再试")
                    self.notice(HTTPStatus.TOO_MANY_REQUESTS, "棋间 · 名额已满",
                                f"分享最多同时开 {sessions.maximum} 桌，现在都有人在用。"
                                f"{later}，或请服务提供者用 Ctrl+C 停止服务后重新开启。")
                    return None
                token, mine = opened
                self.answered = True
                self.redirect("/", [self.cookie_header(token)])
                return None
            mine = sessions.fetch(self.cookie())
            if mine is not None:
                return mine
            self.answered = True
            if path != "/":
                self.send(HTTPStatus.FORBIDDEN, {"error": "请从邀请链接进入。"})
                return None
            if password:
                # A submitted password that did not match is an error, not a
                # fresh page: say so instead of silently re-rendering the form.
                wrong = bool(self.secret)
                self.notice(HTTPStatus.BAD_REQUEST if wrong else HTTPStatus.OK, "棋间 · 需要口令",
                            ('<form method="POST" action="/"><input type="password" name="p" placeholder="访问口令" '
                             'autofocus><button type="submit">进入棋桌</button></form>'
                             + ('<p class="notice">口令不正确，请重新输入。</p>' if wrong else '')
                             + f'<p class="notice">口令由服务提供者告知；进入后本浏览器独占一张棋桌'
                               f'（最多 {sessions.maximum} 桌）。</p>'))
            else:
                self.notice(HTTPStatus.FORBIDDEN, "棋间 · 需要邀请链接",
                            "请使用服务提供者给出的邀请链接进入。")
            return None

        # --- routes -------------------------------------------------------
        def do_GET(self):
            if not self.host_ok():
                return self.send(HTTPStatus.FORBIDDEN, {"error": "Local access only"})
            path = urlparse(self.path).path
            if sessions is not None and path in WATCH_ROUTES:
                return self.watch_route(path)
            mine = self.admit()
            if self.answered:  # the password form or the invite exchange replied
                return
            if mine is None:
                return
            if path == "/api/state":
                return self.send(HTTPStatus.OK, mine.snapshot())
            if path == "/api/config":
                payload = dict(config_payload(self.cookie() or "-"))
                if sessions is not None:
                    payload.update(share=True, sessions=sessions.count(),
                                   max_sessions=sessions.maximum, password=bool(password),
                                   invite=self.public_invite() or invite(self.server.server_port),
                                   watch=self.public_watch() or watch(self.server.server_port),
                                   table_ttl=idle_ttl,
                                   invite_reachable=guest_address() is not None)
                return self.send(HTTPStatus.OK, payload)
            if path not in PLAYER_ASSETS:
                return self.send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
            name, kind = PLAYER_ASSETS[path]
            self.send(HTTPStatus.OK, (STATIC / name).read_bytes(), kind + "; charset=utf-8")

        def do_POST(self):
            if not self.host_ok():
                return self.send(HTTPStatus.FORBIDDEN, {"error": "请从本地页面操作。"})
            mine = self.admit()
            if self.answered:  # the password form or the invite exchange replied
                return
            if mine is None:
                return
            if self.headers.get("X-Renju-Token") != request_token:
                return self.send(HTTPStatus.FORBIDDEN, {"error": "请从本地页面操作。"})
            try:
                if not 0 < self.size <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.body if self.body is not None else self.rfile.read(self.size))
                if not isinstance(data, dict):
                    raise ValueError("Invalid request")
                path = urlparse(self.path).path
                if path == "/api/new":
                    result = mine.new(data)
                elif path == "/api/leave":
                    if sessions is None:
                        raise ValueError("当前服务没有开启分享。")
                    sessions.close(self.cookie())
                    return self.send(HTTPStatus.OK, {"left": True, "sessions": sessions.count()})
                elif path in ("/api/move", "/api/undo"):
                    result = mine.act(data, undo=path.endswith("undo"))
                else:
                    return self.send(HTTPStatus.NOT_FOUND, {"error": "Not found"})
                if sessions is not None:
                    result = dict(result, sessions=sessions.count())
                self.send(HTTPStatus.OK, result)
            except (ValueError, TypeError) as exc:
                self.send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer((host, port), Handler)
    server.daemon_threads = True
    # The fence is per bound port, so it is computed once the OS has picked one.
    fence.update(authorities(host, server.server_port, trusted_hosts, aliases))
    server.table = table
    server.sessions = sessions
    server.fence = fence
    server.trusted_hosts = list(trusted_hosts)
    server.tunnel_lock = tunnel_lock
    server.tunnel = None
    server.tunnel_host = None
    server.invite_key = join_key
    server.invite = invite
    server.watch_key = watch_key
    server.watch = watch
    return server


def serve(port=8765, root=ROOT / "runs", open_browser=True, host="127.0.0.1", share=False,
          max_sessions=LAN_SESSION_LIMIT, password=None, trusted_hosts=(),
          public=False, cloudflared=None, tunnel_timeout=40.0, idle_ttl=DEFAULT_IDLE_TTL):
    # Explicit CPU mode, with one inference thread per table to limit competition
    # with self-play. `set_num_interop_threads` refuses a second call in one
    # process, which is what a repeated serve() would do.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if public:
        # The tunnel carries the page to strangers, so a shared instance without
        # a password would be an open board. One is minted here and printed with
        # the link, which is both safer and easier than making the operator
        # invent one before they can start.
        password = password or secrets.token_urlsafe(9)
    server = make_server(port, root, host=host, share=share, max_sessions=max_sessions,
                         password=password, trusted_hosts=trusted_hosts, idle_ttl=idle_ttl)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Visk Web UI: {url} (CPU inference; training continues)", flush=True)
    if share:
        link = server.invite(server.server_port)
        if link is None:
            print(f"分享已开启，但服务只监听 {host}，其他机器无法连接。"
                  f"请改用 --host {get_local_ipv4() or '<局域网IP>'} 后重试。", flush=True, file=sys.stderr)
        else:
            note = "，进入时需输入口令" if password else ""
            release = f"，闲置 {idle_ttl / 60:.0f} 分钟自动释放" if idle_ttl else ""
            print(f"分享链接（最多 {server.sessions.maximum} 桌{release}{note}）: {link}", flush=True)
            print(f"观战链接（只读，不占棋桌）: {server.watch(server.server_port)}", flush=True)
    if public:
        print(f"正在为公网访问启动 Cloudflare 隧道（cloudflared 需能连上外网）……", flush=True)
        try:
            tunnel_link = start_tunnel(server, server.server_port, executable=cloudflared,
                                       timeout=tunnel_timeout)
        except TunnelError as exc:
            server.server_close()
            if server.sessions is not None:
                server.sessions.shutdown()
            else:
                server.table.pool.shutdown(wait=True)
            raise SystemExit(f"公网暴露失败：{exc}")
        print("", flush=True)
        print(f"公网访问口令: {password}", flush=True)
        print(f"公网邀请链接: {tunnel_link}/?{TOKEN_QUERY}={server.invite_key}", flush=True)
        print(f"公网观战链接: {tunnel_link}/watch?{WATCH_QUERY}={server.watch_key}", flush=True)
        print("把邀请链接和口令发给对弈的人，把观战链接发给只看不下的人（观战不占棋桌，也无法落子）；"
              "隧道关闭或 Ctrl+C 后链接立即失效。公网访客会各自拿到一张独立棋桌。", flush=True)
        print("", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        if server.tunnel is not None:
            print("正在关闭公网隧道……", flush=True)
            server.tunnel.stop()
        if server.sessions is not None:
            server.sessions.shutdown()
        else:
            server.table.pool.shutdown(wait=True)


LOGIN_PAGE = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{TITLE}}</title><style>
:root{color-scheme:light dark}
body{margin:0;min-height:100vh;display:grid;place-items:center;background:#12100e;color:#efe7db;
font:16px/1.6 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}
main{width:min(90vw,26rem);background:#1c1917;border:1px solid #322c26;border-radius:14px;padding:2rem}
h1{font-size:1.1rem;margin:0 0 .5rem}
p{margin:.5rem 0 0;color:#b7ab9c;font-size:.9rem}
form{display:flex;gap:.5rem;margin-top:1rem}
input{flex:1;padding:.6rem .7rem;border-radius:8px;border:1px solid #443c33;background:#12100e;color:inherit}
button{padding:.6rem 1rem;border-radius:8px;border:0;background:#c9a227;color:#1c1917;font-weight:600;cursor:pointer}
.notice{color:#c9a227}
</style></head><body><main><h1>{{TITLE}}</h1>{{BODY}}</main></body></html>
"""
