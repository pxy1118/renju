"""Loopback-only, CPU inference UI; training files are always read-only."""
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import threading
import time
from urllib.parse import urlparse

import numpy as np
import torch

from .game import Game
from .network import Network, Evaluator
from .search import MCTS
from .training import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).with_name("web")
RULES = ("freestyle", "renju")
RECENT_LIMIT = 8
CHECKPOINT_HELP = "请让训练保存检查点后再试（每轮结束时写入）。"


class Table:
    def __init__(self, root=ROOT / "runs"):
        self.root = Path(root)
        self.lock = threading.RLock()
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.game = None
        self.history = []
        self.busy = False
        self.error = None
        self.id = None
        self.info = {}
        self.legal = []
        self.seconds = None

    def models(self):
        """Checkpoint names per rule; 'latest'/'best' resolve to a filename."""
        return {rule: {key: self.resolve(rule, key) for key in ("latest", "best")} for rule in RULES}

    def resolve(self, rule, name):
        """Map a UI selection to the checkpoint file a new game would load.

        'latest'/'best' are the two aliases the UI offers; any other string is
        taken as a checkpoint file name and reported unchanged, so the client
        can trust a listed file. Whether the file really loads is decided when
        the model is read, which happens off the request thread.
        """
        folder = self.root / rule
        if name == "latest":
            files = sorted(folder.glob("checkpoint-*.pt"))
            return files[-1].name if files else None
        if name == "best":
            return "best.pt" if (folder / "best.pt").is_file() else None
        return name or None

    def recent(self, rule, limit=RECENT_LIMIT):
        """Recently written checkpoints, newest first, for manual selection."""
        items = []
        for path in sorted((self.root / rule).glob("checkpoint-*.pt"),
                           key=lambda p: p.stat().st_mtime, reverse=True)[:limit]:
            stat = path.stat()
            # Local clock is UTC+8; the field is for human orientation only.
            stamp = time.strftime("%m-%d %H:%M", time.localtime(stat.st_mtime + 8 * 3600))
            items.append(dict(name=path.name, mtime=stamp, mb=round(stat.st_size / 1048576, 1)))
        return items

    def snapshot(self):
        with self.lock:
            return dict(id=self.id, busy=self.busy, error=self.error, model=dict(self.info),
                        board=self.game.board.tolist() if self.game else [0]*225,
                        player=self.game.player if self.game else 1,
                        winner=self.game.winner if self.game else None,
                        human=getattr(self, "human", 1), legal=self.legal,
                        history=list(self.history), seconds=self.seconds)

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
        if "/" in checkpoint or "\\" in checkpoint or ".." in checkpoint:
            raise ValueError("检查点名称无效。")
        with self.lock:
            if self.busy:
                raise ValueError("模型正在思考，请等本手完成后再开始新局。")
            checkpoint = self.resolve(rule, checkpoint)
            if not checkpoint:
                raise ValueError(f"{'连珠' if rule == 'renju' else '自由五子棋'}还没有可用的模型。{CHECKPOINT_HELP}")
            self.id = secrets.token_hex(16)
            self.game = Game(rule)
            self.human = 1 if color == "black" else -1
            self.history, self.error, self.seconds = [], None, None
            self.info = dict(rule=rule, checkpoint="best" if checkpoint == "best.pt" else checkpoint,
                             latest=self.resolve(rule, "latest"), simulations=simulations, device="CPU")
            self.busy = True
            self.legal = []
            self.pool.submit(self.initialize, rule, checkpoint, simulations)
            return self.snapshot()

    def initialize(self, rule, checkpoint, simulations):
        try:
            # The name comes from the client, so confirm it stays inside this rule.
            path = (self.root / rule / checkpoint).resolve()
            if path.parent != (self.root / rule).resolve():
                raise FileNotFoundError(checkpoint)
            state = load_checkpoint(path, rule)
            cfg = state["config"]
            model = Network(cfg["channels"], cfg["blocks"])
            model.load_state_dict(state["model"])
            self.tree = MCTS(Evaluator(model, "cpu"), simulations, cfg["cpuct"])
            with self.lock:
                self.info.update(file=path.name, step=state.get("step"), round=state.get("round"))
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
                self.refresh()
                self.busy = False
        except Exception as exc:
            self.failed(exc)

    def act(self, data, undo=False):
        with self.lock:
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


def make_server(port=8765, root=ROOT / "runs"):
    table = Table(root)
    token = secrets.token_hex(24)

    class Handler(BaseHTTPRequestHandler):
        def send(self, status, body, kind="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def valid_host(self):
            return self.headers.get("Host") in (f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}")

        def do_GET(self):
            if not self.valid_host():
                return self.send(403, {"error": "Local access only"})
            path = urlparse(self.path).path
            if path == "/api/state":
                return self.send(200, table.snapshot())
            if path == "/api/config":
                return self.send(200, {"token": token, "models": table.models(),
                                       "recent": {rule: table.recent(rule) for rule in RULES}})
            assets = {"/": ("index.html", "text/html"), "/app.js": ("app.js", "text/javascript"), "/style.css": ("style.css", "text/css")}
            if path not in assets:
                return self.send(404, {"error": "Not found"})
            name, kind = assets[path]
            self.send(200, (STATIC / name).read_bytes(), kind + "; charset=utf-8")

        def do_POST(self):
            if not self.valid_host() or self.headers.get("X-Renju-Token") != token:
                return self.send(403, {"error": "请从本地页面操作。"})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 4096:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(size))
                if not isinstance(data, dict):
                    raise ValueError("Invalid request")
                if self.path == "/api/new":
                    result = table.new(data)
                elif self.path in ("/api/move", "/api/undo"):
                    result = table.act(data, undo=self.path.endswith("undo"))
                else:
                    return self.send(404, {"error": "Not found"})
                self.send(200, result)
            except (ValueError, TypeError) as exc:
                self.send(400, {"error": str(exc)})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    server.table = table
    return server


def serve(port=8765, root=ROOT / "runs", open_browser=True):
    # Explicit CPU mode, with one inference thread to limit competition with self-play.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    server = make_server(port, root)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Renju Web UI: {url} (CPU inference; training continues)", flush=True)
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        server.table.pool.shutdown(wait=True)
