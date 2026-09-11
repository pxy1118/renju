"""Process-protocol adapter for a user-installed Rapfi engine.

Rapfi is deliberately an external runtime dependency.  No engine code, binary,
configuration, or weight is copied into this project.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import queue
import re
import subprocess
import sys
import threading
import time


class RapfiError(RuntimeError):
    pass


@dataclass(frozen=True)
class RapfiMove:
    action: int
    winrate: float
    nodes: int
    depth: int
    pv: tuple[int, ...]


@dataclass(frozen=True)
class RapfiAnalysis:
    moves: tuple[RapfiMove, ...]
    version: str


_COORD = re.compile(r"^(\d+),(\d+)$")


def _action(text):
    match = _COORD.match(text.strip())
    if not match:
        raise RapfiError(f"Invalid Rapfi coordinate: {text!r}")
    x, y = map(int, match.groups())
    if not (0 <= x < 15 and 0 <= y < 15):
        raise RapfiError(f"Out-of-range Rapfi coordinate: {text!r}")
    return y * 15 + x


def board_command(game):
    """Encode YXBOARD roles relative to the side that is about to move."""
    occupied = set(map(int, np.flatnonzero(game.board)))
    history = list(getattr(game, "history", ()))
    if len(history) != len(occupied) or set(history) != occupied:
        black = list(map(int, np.flatnonzero(game.board == 1)))
        white = list(map(int, np.flatnonzero(game.board == -1)))
        if len(black) not in (len(white), len(white) + 1):
            raise RapfiError("Board cannot be represented as an alternating move sequence")
        history = []
        for index in range(len(black)):
            history.append(black[index])
            if index < len(white):
                history.append(white[index])
    stones = []
    for action in history:
        y, x = divmod(action, 15)
        role = 1 if game.board[action] == game.player else 2
        stones.append(f"{x},{y},{role}")
    return "YXBOARD " + " ".join(stones + ["DONE"])


class RapfiClient:
    def __init__(self, engine, engine_dir, threads=4, hash_mb=256,
                 max_nodes=200_000, timeout=5.0, retries=2):
        self.engine = Path(engine).resolve()
        self.engine_dir = Path(engine_dir).resolve()
        if not self.engine.is_file() or not self.engine_dir.is_dir():
            raise FileNotFoundError("Rapfi executable or engine directory does not exist")
        self.threads, self.hash_mb = int(threads), int(hash_mb)
        self.max_nodes, self.timeout, self.retries = int(max_nodes), float(timeout), int(retries)
        self.process = None
        self.lines = None
        self.version = "unknown"
        self.start()

    def _reader(self, stream):
        for line in iter(stream.readline, ""):
            self.lines.put(line.rstrip("\r\n"))
        self.lines.put(None)

    def _send(self, command):
        if self.process is None or self.process.poll() is not None:
            raise RapfiError("Rapfi process is not running")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read(self, deadline=None):
        wait = self.timeout if deadline is None else min(self.timeout, max(0.0, deadline - time.monotonic()))
        try:
            line = self.lines.get(timeout=wait)
        except queue.Empty as exc:
            raise TimeoutError("Rapfi response timed out") from exc
        if line is None:
            raise RapfiError("Rapfi process exited")
        if line.startswith("MESSAGE Rapfi "):
            self.version = line.removeprefix("MESSAGE Rapfi ").strip()
        if line.startswith("ERROR") or line.startswith("UNKNOWN"):
            raise RapfiError(line)
        return line

    def _wait_for(self, predicate):
        while True:
            line = self._read()
            if predicate(line):
                return line

    def start(self):
        self.close()
        self.lines = queue.Queue()
        command = [sys.executable, str(self.engine)] if self.engine.suffix.lower() == ".py" else [str(self.engine)]
        self.process = subprocess.Popen(
            command, cwd=str(self.engine_dir), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._reader, args=(self.process.stdout,), daemon=True).start()
        self._send("START 15")
        self._wait_for(lambda line: line == "OK")
        for command in (
            "INFO RULE 0", f"INFO THREAD_NUM {self.threads}",
            f"INFO HASH_SIZE {self.hash_mb}", f"INFO MAX_NODE {self.max_nodes}",
            "INFO SHOW_DETAIL 3"):
            self._send(command)
        self._send("YXSHOWINFO")

    def close(self):
        process = getattr(self, "process", None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write("END\n")
                process.stdin.flush()
                process.wait(timeout=1)
            except Exception:
                process.terminate()
                try:
                    process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    process.kill()
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _analyze_once(self, game, multipv):
        if game.rule != "freestyle":
            raise ValueError("The first teacher phase supports freestyle only")
        self._send(board_command(game))
        self._send(f"YXNBEST {int(multipv)}")
        deadline = time.monotonic() + self.timeout

        current = None
        completed = []
        fallback_action = None
        while True:
            line = self._read(deadline).strip()
            if _COORD.match(line):
                fallback_action = _action(line)
                break
            if line.startswith("INFO PV "):
                tail = line[8:].strip()
                if tail == "DONE":
                    if current is not None and current.get("action") is not None:
                        completed.append(current)
                    current = None
                elif tail.isdigit():
                    current = {"index": int(tail), "depth": 0, "winrate": None,
                               "nodes": 0, "action": None, "pv": (), "numpv": multipv}
                continue
            if current is None or not line.startswith("INFO "):
                continue
            key, _, value = line[5:].partition(" ")
            try:
                if key == "DEPTH":
                    current["depth"] = int(value.split()[0])
                elif key == "NUMPV":
                    current["numpv"] = int(value.split()[0])
                elif key == "WINRATE":
                    number = float(value.split()[0])
                    current["winrate"] = number / 100.0 if number > 1 else number
                elif key == "NODES":
                    current["nodes"] = int(value.split()[0])
                elif key == "BESTLINE":
                    coords = tuple(_action(token) for token in value.split() if _COORD.match(token))
                    if coords:
                        current["action"], current["pv"] = coords[0], coords
            except (ValueError, RapfiError):
                raise RapfiError(f"Malformed Rapfi detail line: {line!r}")

        # Only one fully completed depth is safe: take the deepest complete group.
        valid = [record for record in completed if record["winrate"] is not None]
        if not valid:
            raise RapfiError("Rapfi returned no complete MultiPV records")
        groups = {}
        for record in valid:
            groups.setdefault(record["depth"], {})[record["index"]] = record
        complete_depths = [depth for depth, group in groups.items()
                           if len(group) >= min(multipv, next(iter(group.values()))["numpv"])]
        if not complete_depths:
            raise RapfiError("Rapfi returned no complete MultiPV depth")
        depth = max(complete_depths)
        by_index = groups[depth]
        records = [by_index[index] for index in sorted(by_index)[:multipv]]
        legal = game.legal()
        moves = []
        seen = set()
        for record in records:
            action = record["action"]
            if action in seen or not legal[action]:
                raise RapfiError(f"Rapfi returned duplicate or illegal move {action}")
            seen.add(action)
            moves.append(RapfiMove(action, min(1.0, max(0.0, record["winrate"])),
                                   record["nodes"], record["depth"], record["pv"]))
        if not legal[fallback_action]:
            raise RapfiError("Rapfi final move is illegal")
        return RapfiAnalysis(tuple(moves), self.version)

    def analyze(self, game, multipv=5):
        last = None
        for attempt in range(self.retries + 1):
            try:
                return self._analyze_once(game, multipv)
            except (RapfiError, TimeoutError, BrokenPipeError, OSError) as exc:
                last = exc
                if attempt < self.retries:
                    self.start()
        raise RapfiError(f"Rapfi analysis failed after {self.retries + 1} attempts: {last}") from last
