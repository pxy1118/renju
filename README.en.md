# Dual-Rule Gomoku / Renju AlphaZero

English | [简体中文](README.md)

Starting from random weights, this project learns to play through **MCTS self-play → replay pool → policy-value network updates**. Both rule sets share one implementation, while weights and data are stored separately. The project ships a training system; it does not yet include a trained, strong playing model.

Local test results, GPU short runs, and throughput limits are in the [acceptance record](ACCEPTANCE.md).

## Quick Start

### Playing in the browser (can run alongside training)

Double-click `start-webui.bat` in the project. It starts the service in the current window and opens the browser automatically at
**http://127.0.0.1:8765**. From then on you play entirely on the web page — no coordinates typed into the terminal.
**The service runs in that window; press Ctrl+C to stop it** (alternatively, run `python main.py webui` directly in the foreground).
Double-clicking again detects that the port is already in use and simply opens the running interface instead of starting a second service. You can also run:

```powershell
.\.venv\Scripts\python.exe main.py webui
# custom port; here --output means the model root directory containing the freestyle/renju subdirectories
.\.venv\Scripts\python.exe main.py webui --port 8766 --output runs
```

The interface supports choosing the rule and your color, manually selecting the latest / historical best / several most recent checkpoints, 32/64/200 MCTS searches, a move counter, undoing the last round, restarting, and an end-of-game notice; at the end of a game the winning five stones are highlighted and the dot on the left of the status line becomes the winning side's piece (hidden on a draw), so "who won" is stated consistently in three places: text, piece, and line. The model drop-down lists the most recently written checkpoints for that rule (name, time, size), letting you revisit early training weights; if the selected file is removed by checkpoint rotation, it automatically falls back to "latest". Renju legality reuses the training rules; when no checkpoint exists, random weights are not passed off as a trained model — "Start Game" becomes "No model available" together with the reason.

The Web UI always uses **CPU with a single inference thread**: it neither requests CUDA nor the training GPU lock, and it does not write to the training directory. It still consumes CPU and memory — a single service process has been measured at roughly a 550 MB resident working set and about 1.5 GB of private memory (including the CUDA-enabled PyTorch runtime), so running it alongside training causes memory contention. A new game loads the model saved at that moment, and the weights stay fixed for the whole game. Clicking "Restart" picks up an updated checkpoint. The service listens on localhost only, all browser tabs share a single board, and refreshing restores the current game; the game is not preserved after the service exits. While the model is thinking, wait for the move to finish before undoing or restarting. The client polls status every 250 ms while the model is thinking and every 1 second when idle, so even thinking times of a few hundred milliseconds show as "model thinking". When you run `python main.py webui` directly, the log is printed to the terminal. A model being trained does not mean mature playing strength already exists.

### Stopping the service

Press **Ctrl+C** in the window running the service. The service shuts down immediately, the current game is not saved, and the next game re-reads the latest checkpoint; the training process and `runs/` are unaffected. When started via `start-webui.bat`, after Ctrl+C cmd may ask once more "Terminate batch job (Y/N)?" — press Y or simply close the window, either works.

If the window is gone but the port is still in use, first confirm who is listening before deciding whether to end it:

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen | Select-Object OwningProcess
Get-CimInstance Win32_Process -Filter "ProcessId=<PID from the previous step>" | Select-Object CommandLine
# only if the command line is main.py webui is it the interface service; main.py train is training, leave it alone
Stop-Process -Id <PID>
```

Install dependencies in the project directory and check CUDA:

```powershell
Set-Location D:\Workplace\Renju
uv sync --locked
python main.py doctor

# optional: measure throughput on your own machine; does not write to the real training directory
python main.py benchmark --rule freestyle --minutes 3
python main.py benchmark --rule renju --minutes 3

# first training run, specify the duration of this run each time; run them sequentially
python main.py train --rule freestyle --hours 2
python main.py train --rule renju --hours 2

# subsequent training: run for another two hours this time
python main.py train --rule freestyle --hours 2 --resume latest --workers 16
python main.py train --rule renju --hours 2 --resume latest --workers 16

# play against the model (enter row, column, range 1–15; q to quit)
python main.py play --rule freestyle --checkpoint latest --human-color black
python main.py play --rule renju --checkpoint latest --human-color white

# 10 pairs of color-swapped games per opponent; prints a partial report when the 60-minute budget expires
python main.py evaluate --rule freestyle --checkpoint latest --pairs 10 --minutes 60
```

If the project environment is not activated, you can replace `python` with `.\.venv\Scripts\python.exe`. `uv sync --locked` installs only the locked dependencies and does not start training. The locked versions are PyTorch 2.14.0+cu130, torchvision 0.29.0+cu130, torchaudio 2.11.0+cu130, NumPy 2.5.3, and Pillow 12.3.0. When CUDA is unavailable the program reports an error and does not silently fall back to CPU. A project process lock is used between GPU runs, so the two rules cannot be started simultaneously and contend for the graphics card. That lock does not manage GPU usage by other software.

## The Two Rules

| Option | Opening | Win conditions and restrictions |
|---|---|---|
| `freestyle` | Black moves first, any empty point | Either side wins with five or more consecutive stones; no forbidden moves |
| `renju` | Black's first stone is fixed at the center, then free placement alternating | Black wins with exactly five; White wins with five or more; Black is forbidden from double-three, double-four, and overline |

Renju here is the **standard opening with forbidden moves** mode; it does not include swap/choice openings, passing, a game clock, or human appeals. Adjudication is automatic: forbidden points are masked out of the action space; if the board is not full and the side to move has no legal point, that side loses, and a full board with no winner is a draw. These environment conventions are not the complete RIF competition protocol.

The forbidden-move rules follow [RIF international rules, articles 9.1–9.3](https://www.renju.net/rifrules/): forming exactly five at the same time takes priority as a win; the two endpoints of a single open four are not counted twice as two fours; an open three must be legally extendable into a straight four, and the three-three status at the extension point is checked recursively — simple string matching cannot replace this. The fixed positions in the tests are constructed according to these definitions, including a double-four in the same direction and a false open three whose extension point is itself a three-three.

## Default Training Configuration

`configs/default.json` lists all default values. You can copy that file and pass it in via `-Config`. A new configuration should use a new `-Output` directory; by default, continuing training reads the configuration stored in the checkpoint and rejects rule or configuration mismatches, preventing accidental mixing of training runs.

```powershell
python main.py train --rule renju --hours 4 --config configs/default.json --output runs/renju-experiment
python main.py train --rule renju --hours 2 --output runs/renju-experiment --resume latest
```

- 15×15; three input planes: the current side's stones, the opponent's stones, and whether the current side is Black.
- 6 residual blocks and 64 channels with a shared trunk, a 225-action policy head, and a value head from the perspective of the side to move.
- PUCT coefficient 2, 200 new searches per move; subtrees and existing visit counts are reused, and the legal candidate region is not pruned.
- Self-play root nodes use 25% Dirichlet noise with alpha=0.3; the first 20 moves are sampled from the root visit-count distribution, after which the maximum visit count is taken.
- The model is frozen each round, and by default 16 Windows spawn CPU game processes generate 32 games; the main process merges requests into GPU batches, waiting up to about 3 ms per batch. On an RTX 5070 Ti, measurements at 4 workers left both CPU and GPU underutilized, hence the higher concurrency; adjust `--workers` according to your machine's load.
- After sampling finishes, train for 200 steps with batch size 256 and a replay pool of the most recent 100,000 positions; Adam with learning rate 0.001, L2 coefficient 0.0001, and gradient norm clipped at 5.
- After each game ends, assign win +1 / loss −1 / draw 0, and always label from the perspective of the side to move in that position; no intermediate shape rewards, human game records, or tactical teacher are used.
- The optimization objective is policy cross-entropy plus value mean squared error, with L2 regularization applied by the optimizer; rotations/mirrors are applied to both the board and the policy, while the "Black to move" plane is unchanged.

The hyperparameters are a workable starting point, not the result of completed tuning. Self-play carries a first-move advantage, and a standard opening in particular is not a fair competitive opening; therefore Black and White results are reported separately, and playing strength cannot be judged from the self-play win rate or the training loss alone.

## Duration, Stopping, and Resuming

`-Hours` is the total budget for this run's sampling, optimization, and automatic evaluation; saving files and shutting down the process may take a few extra seconds. The first Ctrl+C issues a safe-stop request: completed games are kept, unfinished games are discarded, and the process saves and exits. A second Ctrl+C is a forced interrupt and may preserve only the previous checkpoint.

If fewer than 32 games are finished when the time budget is reached, the completed positions are still saved. If the current round has an unfinished parameter update, the checkpoint records `pending_steps`; the next resumed run completes those updates first and then begins the next sampling round. Short segmented training therefore does not keep sampling without ever updating. If not even one game was completed, no fabricated draw labels are written.

Each round atomically saves one complete checkpoint and keeps the three most recent. A checkpoint contains the network, optimizer, configuration, replay pool, total game count, round count, training step count, pending updates, and random number state. Resuming allows adjusting the self-play concurrency via `--workers`; all other configuration must still match the checkpoint. Process scheduling and GPU operations affect numerics and the order in which samples arrive, so resuming guarantees a continuous training state, not bit-for-bit reproducibility.

Only `.pt` files produced by this project and trusted should be loaded: full recovery uses Python pickle. `best.pt` holds inference weights only, without the optimizer or replay pool, and cannot be used to resume; use `latest` or a complete `checkpoint-*.pt` to resume. An existing training directory requires an explicit `-Resume` and will not be overwritten as a new training run.

## Files and Evaluation

```text
runs/freestyle/              freestyle model, replay, and logs
runs/renju/                  renju model, replay, and logs
  config.json                current configuration
  checkpoint-*.pt            the three most recent complete checkpoints
  best.pt                    first updated model, later replaced by head-to-head results
  metrics.jsonl              time, throughput, loss, policy entropy, per-color win rates
  games.jsonl                moves and results of completed games
  evaluations.jsonl          periodic evaluations during training
  evaluation-*.json          manual evaluation reports
artifacts/                   benchmarks and short-run tests, not real training output
```

Every 10 rounds the model is evaluated in turn against the historical `best`, a random baseline, and a fixed tactical baseline; both sides use the same search budget and no exploration noise is added. Fixed seeds generate legal four-move openings, and the model's color is swapped for each opening. The report gives total wins and losses, per-color results, the score, and a conservative 95% Hoeffding interval based on opening pairing. When the time budget is reached it is marked `complete: false`, and half-finished games are not counted as draws.

`best` is replaced when the comparison against the historical model completes and the score exceeds 55%. The initial `best` is merely the first updated model, and 55% is an engineering promotion threshold, not proof of statistical significance or expert-level strength. By default the confidence interval for 20 games per opponent is wide; `-Pairs` increases the number of independent opening groups. The random and tactical baselines are used for evaluation only and produce no rewards or labels for training.

## Testing and Development

```powershell
python -m pytest -q
python main.py doctor

# short run with a small model / low search budget, for functional acceptance only
python main.py train --rule freestyle --config configs/smoke.json --output artifacts/my-smoke-freestyle --hours 0.04 --max-rounds 1
python main.py train --rule renju --config configs/smoke.json --output artifacts/my-smoke-renju --hours 0.04 --max-rounds 1
```

Tests cover wins in all four directions, overlines, real and false double-threes, same-direction and crossing double-fours, edges and a full board, legal actions, the sign of search backup, immediate wins and necessary defenses, augmentation alignment, full-game labels, actual parameter updates, checkpoint resumption, stopping, rule isolation, and worker exception propagation.

The implementation is layered: `main.py` is the unified command entry point; `az/game.py` holds the rules; `az/search.py` is MCTS; `az/selfplay.py` is multi-process sampling; `az/network.py` is the network and inference; `az/training.py` covers optimization and resumption; `az/evaluation.py` is head-to-head evaluation; `az/cli.py` handles command parsing.

Method references: [AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/), [DeepMind OpenSpiel AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md), [PyTorch CUDA installation](https://pytorch.org/get-started/previous-versions/). This project is a simplified AlphaZero system for Gomoku, not a reproduction of the original paper's compute scale or a complete implementation.
