"""The self-play training loop: collect a round, update, evaluate, checkpoint.

The loop owns three things and delegates the rest: it decides when a round runs
and when parameters move, it keeps the promoted champion, and it writes one
metrics record per round. Search lives in vk/search, targets in vk/targets, the
loss in vk/objective, storage in vk/storage and the round diagnostics in
vk/diagnostics.
"""
import json
from pathlib import Path
import random
import time
import copy
import numpy as np
import torch

from .config import mix_vector, resume_incompatible
from .diagnostics import record_round, target_report
from .network import Evaluator, Network, architecture_of
from .objective import compute_loss, require_multi_head, tensor_batch
from .records import augment_batch, blank
from .replay import ReplayBuffer
from .selfplay import collect
from .storage import (append_json, atomic_save, checkpoint_path, load_checkpoint,
                      load_model_state, reconcile_jsonl, save, save_best)


def update(model, optimizer, replay, cfg, device, rng):
    """One optimisation step over a surprise-weighted replay batch."""
    batch = replay.sample(cfg["batch_size"], rng, cfg)
    states, policies = augment_batch(batch["state"], batch["policy"], rng)
    tensors = tensor_batch(batch, device, states, policies)
    model.train()
    logits, values = model(tensors["state"])
    report = compute_loss(logits, values, tensors, cfg)
    optimizer.zero_grad(set_to_none=True)
    report["loss"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()
    metrics = {key: value for key, value in report.items()
               if isinstance(value, (int, float, type(None)))}
    metrics["loss"] = float(report["loss"].detach())
    metrics["policy_loss"] = float(report["policy_loss"].detach())
    metrics["value_loss"] = float(report["value_loss"].detach())
    weights = batch["weight"].astype(np.float64) if len(batch) else np.zeros(1)
    metrics["batch_weight_mean"] = float(weights.mean())
    metrics["batch_weight_max"] = float(weights.max())
    metrics["batch_full_search_share"] = float((batch["full_search"] != 0).mean()) if len(batch) else 0.0
    return metrics


def head_line(targets, network):
    """The handful of numbers worth reading at a glance in metrics.jsonl."""
    final = targets.get("target_saturation_final", {})
    return {
        "value_target_abs_mean_final": final.get("mean_abs"),
        "value_target_abs_gt_0.9_share_final": final.get("abs_gt_0.9_share"),
        "value_target_abs_lt_0.5_share_final": final.get("abs_lt_0.5_share"),
        "policy_entropy_target": targets.get("policy_target_entropy"),
        "policy_entropy_network": network.get("policy_entropy_network"),
        "kl_target_prior": targets.get("policy_surprise_mean"),
        "kl_target_network": network.get("kl_target_network"),
        "q_spread_mean": targets.get("q_spread_mean"),
        "policy_valid_share": targets.get("policy_valid_share"),
        "full_search_share": targets.get("full_search_share"),
        "sample_weight_mean": targets.get("sample_weight_mean"),
        "sample_weight_max": targets.get("sample_weight_max"),
    }


def train(cfg, root, device, seconds, resume=None, stop=lambda: False, max_rounds=None,
          init_checkpoint=None):
    root = Path(root)
    if resume and init_checkpoint:
        raise ValueError("--resume and --init-checkpoint are mutually exclusive")
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError("Existing output found: use --resume latest or a different --output.")
    root.mkdir(parents=True, exist_ok=True)
    state = load_checkpoint(checkpoint_path(root, resume), cfg["rule"]) if resume else None
    if state:
        if "optimizer" not in state:
            raise ValueError("Inference-only best checkpoint cannot resume training; use latest.")
        incompatible = resume_incompatible(cfg, state["config"])
        if incompatible:
            raise ValueError(f"Resume configuration differs in incompatible fields: "
                             f"{sorted(incompatible)}")
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = require_multi_head(Network(cfg["arch"])).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"],
                                 weight_decay=cfg["weight_decay"])
    replay = ReplayBuffer(cfg["replay_capacity"])
    round_id = step = total_games = pending_steps = 0
    teacher = None
    if init_checkpoint:
        initial = load_model_state(checkpoint_path(root, init_checkpoint), cfg["rule"])
        if architecture_of(initial["config"]) != cfg["arch"]:
            raise ValueError("Initial checkpoint architecture mismatch")
        if initial.get("pretrain_report") and not initial["pretrain_report"].get("accepted"):
            raise ValueError("Initial pretraining checkpoint did not pass the formal acceptance gates")
        model.load_state_dict(initial["model"])
        teacher = initial.get("teacher")
    if state:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        replay = ReplayBuffer.from_state(cfg["replay_capacity"], state["replay"])
        round_id, step, total_games = state["round"], state["step"], state["total_games"]
        pending_steps = state.get("pending_steps", 0)
        rng.bit_generator.state = state["rng"]
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if state["cuda_rng"] is not None and device == "cuda":
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        champion_state = copy.deepcopy(state.get("champion_model") or state["model"])
        champion_optimizer = copy.deepcopy(state.get("champion_optimizer") or optimizer.state_dict())
        champion_step = state.get("champion_step", step)
        teacher = state.get("teacher")
        removed = {name: reconcile_jsonl(root / name, round_id)
                   for name in ("games.jsonl", "metrics.jsonl", "evaluations.jsonl")}
        if any(removed.values()):
            print(json.dumps({"resume_reconciled": removed,
                              "checkpoint_round": round_id}), flush=True)
        del state
    else:
        champion_state = copy.deepcopy(model.state_dict())
        champion_optimizer = copy.deepcopy(optimizer.state_dict())
        champion_step = step
    champion = Network(cfg["arch"]).to(device)
    champion.load_state_dict(champion_state)
    (root / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    start = time.monotonic()
    deadline = start + seconds
    evaluator = Evaluator(champion, device, mix=mix_vector(cfg))
    minimum = cfg.get("min_replay_size", 1)
    if not (root / "best.pt").exists():
        save_best(root, cfg, champion_state, champion_step, teacher)
    rounds_this_run = 0
    last_path = None
    while time.monotonic() < deadline and not stop():
        if max_rounds is not None and rounds_this_run >= max_rounds:
            break
        rounds_this_run += 1
        games, records = [], blank(0)
        perf = {"inference_positions": 0, "batches": 0, "average_inference_batch_size": 0,
                "largest_inference_batch_size": 0, "seconds": 0}
        if pending_steps == 0:
            round_id += 1
            seeds = rng.integers(0, 2 ** 32, cfg["games_per_round"])
            records, games, perf = collect(cfg, evaluator, seeds, deadline, stop)
            replay.extend(records)
            total_games += len(games)
            for game in games:
                append_json(root / "games.jsonl", {"round": round_id, **game})
            pending_steps = cfg["train_steps"] if games and len(replay) >= minimum else 0
        trainable = len(replay) >= minimum
        metrics = {}
        updates = 0
        if trainable:
            for _ in range(pending_steps):
                if stop() or time.monotonic() >= deadline:
                    break
                metrics = update(model, optimizer, replay, cfg, device, rng)
                step += 1
                updates += 1
                pending_steps -= 1
        champion_result = None
        if (updates and pending_steps == 0 and
                round_id % cfg.get("promotion_every", cfg.get("eval_every", 10)) == 0 and
                not stop() and time.monotonic() < deadline):
            from .evaluation import match, tactical_gate
            tactics = tactical_gate(model, cfg, device)
            arena = match(model, cfg, device, champion, cfg.get("promotion_pairs", 100),
                          deadline, stop, sequential=True) if tactics["passed"] else {}
            promoted = tactics["passed"] and arena.get("wilson_lower", 0) > 0.5
            champion_result = {"promoted": promoted, "tactical": tactics, "arena": arena}
            if promoted:
                champion_state = copy.deepcopy(model.state_dict())
                champion_optimizer = copy.deepcopy(optimizer.state_dict())
                champion_step = step
                champion.load_state_dict(champion_state)
                save_best(root, cfg, champion_state, champion_step, teacher)
            else:
                model.load_state_dict(champion_state)
                optimizer.load_state_dict(champion_optimizer)
            append_json(root / "evaluations.jsonl", {"round": round_id, **champion_result})
        last_path = save(root, cfg, model, optimizer, replay.to_state(), rng, round_id, step,
                         total_games, pending_steps, champion_state, champion_optimizer,
                         champion_step, teacher)

        def average(name):
            values = [game.get(name) for game in games if game.get(name) is not None]
            return float(np.mean(values)) if values else None

        targets = target_report(records) if len(records) else {}
        network = record_round(model, records, cfg, device, rng=rng) if len(records) else {}
        openings = [len(game["opening"]) for game in games if game.get("opening") is not None]
        record = {"round": round_id, "step": step, "updates": updates, "games": len(games),
                  "total_games": total_games, "replay_size": len(replay),
                  "pending_steps": pending_steps,
                  "trainable": trainable, "min_replay_size": minimum,
                  "opening_distinct": len({tuple(g["opening"]) for g in games if g.get("opening")}),
                  "opening_sample_size": float(np.mean(openings)) if openings else None,
                  "elapsed_seconds": time.monotonic() - start,
                  "remaining_seconds": max(0, deadline - time.monotonic()),
                  "black_wins": sum(g["winner"] == 1 for g in games),
                  "white_wins": sum(g["winner"] == -1 for g in games),
                  "draws": sum(g["winner"] == 0 for g in games),
                  "black_win_rate": sum(g["winner"] == 1 for g in games) / len(games) if games else None,
                  "white_win_rate": sum(g["winner"] == -1 for g in games) / len(games) if games else None,
                  "draw_rate": sum(g["winner"] == 0 for g in games) / len(games) if games else None,
                  "candidate_count_mean": average("candidate_count_mean"),
                  "forced_win_count": sum(g.get("forced_win_count", 0) for g in games),
                  "forced_defense_count": sum(g.get("forced_defense_count", 0) for g in games),
                  "bias_moves_mean": average("bias_moves_mean"),
                  "search_max_depth": max((g.get("search_max_depth", 0) for g in games), default=0),
                  "cheap_search_share": average("cheap_search_share"),
                  "horizon_bootstrap_share": average("horizon_bootstrap_share"),
                  "champion": champion_result,
                  "completed_game_simulations_per_second":
                      sum(g["simulations"] for g in games) / max(perf["seconds"], 1e-6),
                  "diagnostics": {"targets": targets, "network": network,
                                  "replay": replay.stats(cfg)},
                  **head_line(targets, network), **perf, **metrics}
        append_json(root / "metrics.jsonl", record)
        print(json.dumps(record), flush=True)
    if last_path is None:
        last_path = save(root, cfg, model, optimizer, replay.to_state(), rng, round_id, step,
                         total_games, pending_steps, champion_state, champion_optimizer,
                         champion_step, teacher)
    return {"checkpoint": str(last_path), "step": step, "total_games": total_games,
            "elapsed_seconds": time.monotonic() - start}
