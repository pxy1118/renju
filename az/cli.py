import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import time
import numpy as np
import torch
from .game import Game
from .network import Network, Evaluator, device_check
from .search import MCTS
from .training import DEFAULTS, train, checkpoint_path, load_checkpoint

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def gpu_lock(device):
    if device != "cuda":
        yield
        return
    folder = ROOT / "artifacts"
    folder.mkdir(exist_ok=True)
    f = (folder / "gpu.lock").open("a+b")
    if os.fstat(f.fileno()).st_size == 0:
        f.write(b"0")
        f.flush()
    f.seek(0)
    try:
        if os.name == "nt":
            import msvcrt
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        f.close()
        raise RuntimeError("Another project process is using the GPU. Run modes sequentially.")
    try:
        yield
    finally:
        f.close()


def read_config(path, rule):
    cfg = dict(DEFAULTS)
    if path:
        values = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if set(values)-set(cfg):
            raise ValueError(f"Unknown configuration keys: {set(values)-set(cfg)}")
        cfg.update(values)
    cfg["rule"] = rule
    for k, v in cfg.items():
        if k != "rule" and (not isinstance(v,(float,int)) or isinstance(v,bool) or v < 0):
            raise ValueError(f"Invalid configuration: {k}")
    for k in ("channels", "blocks", "simulations", "workers", "games_per_round", "train_steps", "replay_capacity", "batch_size", "eval_every", "eval_pairs"):
        if not isinstance(cfg[k],int) or cfg[k] < 1:
            raise ValueError(f"{k} must be a positive integer")
    return cfg


def display(g):
    print("    " + " ".join(f"{i:2}" for i in range(1,16)))
    for r in range(15):
        print(f"{r+1:2}  " + "  ".join({0:".",1:"X",-1:"O"}[int(v)] for v in g.board.reshape(15,15)[r]))


def main():
    p = argparse.ArgumentParser(description="Dual-rule AlphaZero")
    p.add_argument("command", choices=["doctor", "train", "benchmark", "evaluate", "play", "webui"])
    p.add_argument("--port", type=int, default=8765, help="local Web UI port")
    p.add_argument("--no-browser", action="store_true", help="do not open the Web UI automatically")
    p.add_argument("--rule", choices=["freestyle", "renju"], default="freestyle")
    p.add_argument("--config")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--hours", type=float, default=2)
    p.add_argument("--minutes", type=float, default=3)
    p.add_argument("--output")
    p.add_argument("--resume")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--pairs", type=int, default=10)
    p.add_argument("--workers", type=int, help="parallel self-play processes; may be changed when resuming")
    p.add_argument("--human-color", choices=["black","white"], default="black")
    args = p.parse_args()
    if args.command == "webui":
        from .webui import serve
        if not 1 <= args.port <= 65535:
            p.error("--port must be between 1 and 65535")
        serve(args.port, Path(args.output) if args.output else ROOT / "runs", not args.no_browser)
        return
    if args.hours <= 0 or args.minutes <= 0 or args.pairs <= 0 or (args.max_rounds is not None and args.max_rounds < 1):
        p.error("Budgets, pairs and max-rounds must be positive")
    torch.set_num_threads(4)
    cfg = read_config(args.config, args.rule)
    root = Path(args.output) if args.output else ROOT / "runs" / args.rule
    stopping = [False]
    def stop_handler(signum, frame):
        if stopping[0]:
            raise KeyboardInterrupt()
        stopping[0] = True
        print("Stopping at a safe boundary; incomplete games are discarded. Saving checkpoint...", flush=True)
    if args.command in ("train","benchmark","evaluate"):
        signal.signal(signal.SIGINT, stop_handler)
    with gpu_lock(args.device):
        print(json.dumps(device_check(args.device)), flush=True)
        if args.command == "doctor":
            model = Network().to(args.device)
            x = torch.randn(4,3,15,15,device=args.device)
            logits, value = model(x)
            (logits.square().mean()+value.square().mean()).backward()
            print("Network CUDA forward/backward OK", flush=True)
            return
        if args.command == "train":
            if args.resume and not args.config:
                state = load_checkpoint(checkpoint_path(root,args.resume),args.rule)
                cfg = state["config"]
                del state
            if args.workers is not None:
                if args.workers < 1:
                    p.error("--workers must be positive")
                cfg["workers"] = args.workers
            print(json.dumps(train(cfg, root, args.device, args.hours*3600, args.resume,
                                   lambda: stopping[0], args.max_rounds)), flush=True)
        elif args.command == "benchmark":
            from .selfplay import collect
            if args.workers is not None:
                if args.workers < 1:
                    p.error("--workers must be positive")
                cfg["workers"] = args.workers
            torch.manual_seed(cfg["seed"])
            model = Network(cfg["channels"],cfg["blocks"]).to(args.device)
            model.eval()
            start = time.monotonic()
            data, games, perf = collect(cfg, Evaluator(model,args.device), range(10000),
                                        start+args.minutes*60, lambda: stopping[0])
            # Separate directory, never writes to runs/.
            dest = Path(args.output) if args.output else ROOT / "artifacts" / f"benchmark-{args.rule}-{time.time_ns()}"
            dest.mkdir(parents=True,exist_ok=True)
            rate = len(games)/perf["seconds"]*3600
            report = {"rule": args.rule, "config": cfg, "games":len(games), "positions":len(data),
                      "games_per_hour":rate, "inference_positions_per_second":perf["inference_positions"]/perf["seconds"],
                      "completed_game_simulations_per_second":sum(g["simulations"] for g in games)/perf["seconds"],
                      "estimated_32_game_selfplay_minutes":32/rate*60 if rate else None,
                      "note":"Random untrained weights; inference includes discarded partial games; training/evaluation time excluded.", **perf}
            (dest/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
            print(json.dumps(report),flush=True)
        else:
            state = load_checkpoint(checkpoint_path(root,args.checkpoint),args.rule)
            cfg = state["config"]
            model = Network(cfg["channels"],cfg["blocks"]).to(args.device)
            model.load_state_dict(state["model"])
            del state
            if args.command == "evaluate":
                from .evaluation import evaluate_suite
                report = evaluate_suite(model,cfg,args.device,root,args.pairs,time.monotonic()+args.minutes*60,lambda: stopping[0])
                destination = root / f"evaluation-{time.time_ns()}.json"
                destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps(report),flush=True)
            else:
                g = Game(args.rule)
                human = 1 if args.human_color == "black" else -1
                tree = MCTS(Evaluator(model,args.device),cfg["simulations"],cfg["cpuct"])
                while g.adjudicate() is None:
                    display(g)
                    if g.player == human:
                        try:
                            raw = input("row col (1-15), q to quit: ").strip()
                            if raw.lower() == "q":
                                return
                            row,col = map(int,raw.split())
                            if not (1 <= row <= 15 and 1 <= col <= 15):
                                raise ValueError()
                            action = (row-1)*15+col-1
                            g.move(action)
                        except ValueError:
                            print("Invalid or forbidden move.")
                            continue
                    else:
                        action = int(np.argmax(tree.policy(g)))
                        print(f"AI: {action//15+1} {action%15+1}")
                        g.move(action,validate=False)
                    tree.advance(action,g)
                display(g)
                print({0:"Draw",1:"Black wins",-1:"White wins"}[g.winner])


if __name__ == "__main__":
    main()
