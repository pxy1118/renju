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
from .network import (ARCHITECTURES, Network, Evaluator, architecture,
                      architecture_of, device_check, parameter_count)
from .search import MCTS
from .config import from_file, mix_vector  # noqa: F401  (mix_vector used by play)
from .storage import checkpoint_path, load_checkpoint
from .training import train

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
    """Configuration parsing lives in vk/config; this is the CLI entry point."""
    return from_file(path, rule)


def display(g):
    print("    " + " ".join(f"{i:2}" for i in range(1,16)))
    for r in range(15):
        print(f"{r+1:2}  " + "  ".join({0:".",1:"X",-1:"O"}[int(v)] for v in g.board.reshape(15,15)[r]))


def main():
    p = argparse.ArgumentParser(description="Visk — dual-rule Gomoku/Renju self-play training")
    p.add_argument("command", choices=["doctor", "teacher-generate", "pretrain", "train", "benchmark", "evaluate", "play", "webui"])
    p.add_argument("--port", type=int, default=8765, help="local Web UI port")
    p.add_argument("--no-browser", action="store_true", help="do not open the Web UI automatically")
    p.add_argument("--host", default="127.0.0.1",
                   help="Web UI bind address; 127.0.0.1 keeps it local, use your LAN IP to share (0.0.0.0 = every interface)")
    p.add_argument("--share", action="store_true",
                   help="give each visiting browser its own table; prints an invite link guests can open")
    p.add_argument("--public", action="store_true",
                   help="expose the Web UI on the internet with a Cloudflare tunnel and print a shareable link "
                        "(implies --share and --host 0.0.0.0)")
    p.add_argument("--cloudflared", metavar="PATH",
                   help="cloudflared executable to use with --public (default: found on PATH)")
    p.add_argument("--tunnel-timeout", type=float, default=40.0,
                   help="seconds to wait for the public link with --public (default 40)")
    p.add_argument("--max-sessions", type=int, default=10,
                   help="concurrent shared tables (default 10; each costs tens of MB, "
                        "the CPU is what runs out first)")
    p.add_argument("--table-ttl", type=float, default=5,
                   help="minutes an unused shared table waits before it is released (default 5; "
                        "0 keeps a table until the service stops)")
    p.add_argument("--password", help="optional access password for shared tables")
    p.add_argument("--trusted-host", action="append", default=[], metavar="AUTHORITY",
                   help="extra Host authority the UI accepts, for a reverse proxy (repeatable)")
    p.add_argument("--rule", choices=["freestyle", "renju"], default="freestyle")
    p.add_argument("--config")
    p.add_argument("--device", choices=["cuda","cpu"], default="cuda")
    p.add_argument("--hours", type=float, default=2)
    p.add_argument("--minutes", type=float, default=3)
    p.add_argument("--output")
    p.add_argument("--resume")
    p.add_argument("--init-checkpoint")
    p.add_argument("--checkpoint", default="latest")
    p.add_argument("--max-rounds", type=int)
    p.add_argument("--pairs", type=int, default=10,
                   help="opening pairs; a pair plays both colours, so games = 2 x pairs")
    p.add_argument("--workers", type=int, help="parallel self-play processes; may be changed when resuming")
    p.add_argument("--human-color", choices=["black","white"], default="black")
    p.add_argument("--dataset")
    p.add_argument("--mix-dataset", help="second teacher dataset mixed into pretraining")
    p.add_argument("--mix-share", type=float, default=0.5,
                   help="share of pretraining batches drawn from --mix-dataset (default 0.5)")
    p.add_argument("--policy-weight", type=float, default=1.0,
                   help="weight of the policy cross-entropy in pretraining")
    p.add_argument("--value-weight", type=float, default=1.0,
                   help="weight of the value MSE in pretraining; 0 trains policy only")
    p.add_argument("--critical-weighting", action="store_true",
                   help="draw pretraining batches with a weight set by the teacher's "
                        "top-1/top-2 winrate gap, so decision-critical positions get "
                        "more of the budget (needs format-3 top-k data)")
    p.add_argument("--hard-rules", choices=["forced","none"],
                   help="deterministic tactics that may exclude legal points: forced keeps "
                        "complete-a-five and block-a-five, none keeps every legal point")
    p.add_argument("--search-bias", choices=["none","tactical"],
                   help="soft prior bias for forcing fours and the local neighbourhood; "
                        "never removes a legal point")
    p.add_argument("--search", choices=["mcts","policy"],
                   help="how the model picks a move: MCTS+value, or the raw policy argmax")
    p.add_argument("--opening-mode", choices=["sampled","teacher","book","none"],
                   help="opening procedure: sampled random stones, Rapfi-following, "
                        "a verified balanced book, or the empty board")
    p.add_argument("--opening-book", metavar="PATH",
                   help="balanced opening book JSON (required by --opening-mode book)")
    p.add_argument("--opening-seed", type=int,
                   help="first opening seed; the same seed reproduces the same pair schedule")
    p.add_argument("--positions", type=int, default=50000)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--engine")
    p.add_argument("--engine-dir")
    p.add_argument("--opponent", choices=["suite", "rapfi", "checkpoint", "self-policy"],
                   default="suite",
                   help="who the model plays: the fixed suite, Rapfi, a checkpoint, or "
                        "its own raw policy (measuring what search adds)")
    p.add_argument("--opponent-checkpoint")
    p.add_argument("--engine-workers", type=int, default=4)
    p.add_argument("--engine-threads", type=int, default=4)
    p.add_argument("--engine-hash-mb", type=int, default=256)
    p.add_argument("--max-nodes", type=int, default=200000)
    p.add_argument("--engine-timeout", type=float, default=5.0)
    args = p.parse_args()
    if args.command == "webui":
        from .webui import bind_host, serve
        if not 1 <= args.port <= 65535:
            p.error("--port must be between 1 and 65535")
        # The upper bound is a guard against a typo turning into a memory
        # exhaustion, not a statement about what this machine can hold: every
        # table carries its own network and search tree.
        if not 1 <= args.max_sessions <= 16:
            p.error("--max-sessions must be between 1 and 16")
        if args.table_ttl < 0:
            p.error("--table-ttl must not be negative")
        if args.public:
            # A tunnel cannot reach a loopback-only server, and a public table is
            # never a single shared board: both are implied rather than asked for.
            if not args.share:
                print("--public 已自动开启分享（每位访客一张独立棋桌）。", flush=True)
            args.share = True
            if args.host == "127.0.0.1":
                args.host = "0.0.0.0"
            if args.host not in ("0.0.0.0", "::"):
                p.error("--public 需要 cloudflared 能连上的监听地址，请用 --host 0.0.0.0 "
                        "（或不写 --host，它会被自动设为 0.0.0.0）")
            if args.tunnel_timeout <= 0:
                p.error("--tunnel-timeout must be positive")
        try:
            host = bind_host(args.host)
        except ValueError as exc:
            p.error(str(exc))
        serve(args.port, Path(args.output) if args.output else ROOT / "runs", not args.no_browser,
              host=host, share=args.share, max_sessions=args.max_sessions,
              password=args.password,
              trusted_hosts=[item for item in args.trusted_host if item],
              public=args.public, cloudflared=args.cloudflared,
              tunnel_timeout=args.tunnel_timeout,
              idle_ttl=args.table_ttl * 60 if args.table_ttl > 0 else None)
        return
    if args.resume and args.init_checkpoint:
        p.error("--resume and --init-checkpoint are mutually exclusive")
    if args.command == "teacher-generate":
        if not args.engine or not args.engine_dir or not args.output:
            p.error("teacher-generate requires --engine, --engine-dir and --output")
        if min(args.positions, args.engine_workers, args.engine_threads, args.engine_hash_mb,
               args.max_nodes, args.engine_timeout) <= 0:
            p.error("Teacher generation numeric arguments must be positive")
        from .teacher import generate_teacher_dataset
        print(json.dumps(generate_teacher_dataset(
            args.engine, args.engine_dir, args.output, args.positions, args.engine_workers,
            args.engine_threads, args.engine_hash_mb, args.max_nodes, args.engine_timeout,
            rule=args.rule)), flush=True)
        return
    if args.hours <= 0 or args.minutes <= 0 or args.pairs <= 0 or (args.max_rounds is not None and args.max_rounds < 1):
        p.error("Budgets, pairs and max-rounds must be positive")
    torch.set_num_threads(4)
    cfg = read_config(args.config, args.rule)
    # Explicit flags win over both the configuration file and the checkpoint's
    # stored config, so one run directory can be evaluated under several
    # decoupled switches without editing JSON or writing a new checkpoint.
    overrides = {key: value for key, value in (("hard_rules", args.hard_rules),
                                               ("search_bias", args.search_bias),
                                               ("search", args.search),
                                               ("opening_mode", args.opening_mode),
                                               ("opening_book", args.opening_book))
                 if value is not None}
    cfg.update(overrides)
    # Refuse to start a run that cannot load its book, rather than failing on
    # the first game.
    from .evaluation import load_book
    loaded_book = load_book(cfg)
    if loaded_book is not None:
        print(json.dumps({"opening_book": cfg["opening_book"],
                          "openings": loaded_book["count"],
                          "acceptance": loaded_book.get("acceptance")}), flush=True)
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
        if args.command == "pretrain":
            if not args.dataset or not args.output:
                p.error("pretrain requires --dataset and --output")
            if args.steps <= 0:
                p.error("--steps must be positive")
            from .pretraining import pretrain
            print(json.dumps(pretrain(args.dataset, args.output, args.rule, args.steps,
                                      cfg["batch_size"], cfg["arch"],
                                      args.device, cfg["seed"],
                                      policy_weight=args.policy_weight,
                                      value_weight=args.value_weight,
                                      mix_dataset=args.mix_dataset,
                                      mix_share=args.mix_share,
                                      critical_weighting=args.critical_weighting)), flush=True)
            return
        if args.command == "doctor":
            model = Network().to(args.device)
            x = torch.randn(4,3,15,15,device=args.device)
            logits, value = model(x)
            (logits.square().mean()+value.square().mean()).backward()
            print(json.dumps({"arch": model.arch, "pattern": model.pattern,
                              "width": model.width, "heads": model.heads,
                              "value_heads": list(model.value_heads),
                              "blocks": model.blocks, "parameters": parameter_count(model)}),
                  flush=True)
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
                                   lambda: stopping[0], args.max_rounds, args.init_checkpoint)), flush=True)
        elif args.command == "benchmark":
            from .selfplay import collect
            if args.workers is not None:
                if args.workers < 1:
                    p.error("--workers must be positive")
                cfg["workers"] = args.workers
            torch.manual_seed(cfg["seed"])
            model = Network(cfg["arch"]).to(args.device)
            model.eval()
            start = time.monotonic()
            data, games, perf = collect(cfg, Evaluator(model, args.device, mix=mix_vector(cfg)),
                                        range(10000),
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
            cfg.update(overrides)
            model = Network(architecture_of(cfg)).to(args.device)
            model.load_state_dict(state["model"])
            del state
            if args.command == "evaluate":
                from .evaluation import evaluate_suite, evaluate_rapfi
                if args.opponent == "rapfi":
                    if not args.engine or not args.engine_dir:
                        p.error("Rapfi evaluation requires --engine and --engine-dir")
                    report = {"rapfi": evaluate_rapfi(model, cfg, args.device, args.engine,
                              args.engine_dir, args.pairs, time.monotonic()+args.minutes*60,
                              lambda: stopping[0], args.engine_threads, args.engine_hash_mb,
                              args.max_nodes, args.engine_timeout,
                              args.opening_seed if args.opening_seed is not None else 91823)}
                elif args.opponent == "self-policy":
                    # The same weights, once with search and once without: the
                    # one measurement that says whether search adds anything.
                    from .evaluation import evaluate_search_gap
                    report = {"search_gap": evaluate_search_gap(
                        model, cfg, args.device, args.pairs,
                        time.monotonic() + args.minutes * 60, lambda: stopping[0],
                        args.opening_seed if args.opening_seed is not None else 91823)}
                elif args.opponent == "checkpoint":
                    if not args.opponent_checkpoint:
                        p.error("Checkpoint evaluation requires --opponent-checkpoint")
                    from .evaluation import match
                    opponent_state = load_checkpoint(Path(args.opponent_checkpoint), args.rule)
                    opponent_model = Network(architecture_of(opponent_state["config"])).to(args.device)
                    opponent_model.load_state_dict(opponent_state["model"])
                    report = {"checkpoint": match(model, cfg, args.device, opponent_model,
                                                   args.pairs, time.monotonic()+args.minutes*60,
                                                   lambda: stopping[0])}
                else:
                    report = evaluate_suite(model,cfg,args.device,root,args.pairs,time.monotonic()+args.minutes*60,lambda: stopping[0])
                destination = root / f"evaluation-{time.time_ns()}.json"
                destination.write_text(json.dumps(report,indent=2),encoding="utf-8")
                print(json.dumps(report),flush=True)
            else:
                g = Game(args.rule)
                human = 1 if args.human_color == "black" else -1
                from .evaluation import search_options
                tree = MCTS(Evaluator(model, args.device, mix=mix_vector(cfg)),
                            cfg["simulations"], cfg["cpuct"], **search_options(cfg))
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
