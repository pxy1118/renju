# Visk — Dual-Rule Gomoku / Renju

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

The Web UI always uses **CPU with a single inference thread**: it neither requests CUDA nor the training GPU lock, and it does not write to the training directory. It still consumes CPU and memory — a single service process has been measured at roughly a 550 MB resident working set and about 1.5 GB of private memory (including the CUDA-enabled PyTorch runtime), so running it alongside training causes memory contention. A new game loads the model saved at that moment, and the weights stay fixed for the whole game. Clicking "Restart" picks up an updated checkpoint. The service listens on localhost only by default, all browser tabs share a single board, and refreshing restores the current game; the game is not preserved after the service exits. While the model is thinking, wait for the move to finish before undoing or restarting. The client polls status every 250 ms while the model is thinking and every 1 second when idle, so even thinking times of a few hundred milliseconds show as "model thinking". When you run `python main.py webui` directly, the log is printed to the terminal. A model being trained does not mean mature playing strength already exists.

### Sharing it with friends for a while (10 tables by default)

Sharing is off by default. Add `--share` and `--host <LAN-IP>` and the service prints two links: an **invite link** that lets a friend on the same network sit down and play, and a **watch link** for read-only spectating. For a quick share, double-click `share-webui.bat` (the same as `start-webui.ps1 -Share`, which picks the default-route LAN address itself), or run it by hand:

```powershell
# find this machine's LAN address, then pass it to --host
Get-NetIPAddress -AddressFamily IPv4 | Where-Object AddressState -eq Preferred
.\.venv\Scripts\python.exe main.py webui --host 192.168.1.3 --share
# optional: require a password in addition to the link
.\.venv\Scripts\python.exe main.py webui --host 192.168.1.3 --share --password <password>
# or bind every interface; the invite link then picks the default-route address
.\.venv\Scripts\python.exe main.py webui --host 0.0.0.0 --share --max-sessions 10
```

Startup prints something like:

```text
Visk Web UI: http://127.0.0.1:8765 (CPU inference; training continues)
分享链接（最多 10 桌）: http://192.168.1.3:8765/?k=<random string>
观战链接（只读，不占棋桌）: http://192.168.1.3:8765/watch?w=<random string>
```

What sharing does and does not do:

- **One private table per player.** A browser that opens the invite link receives a session cookie and its own game; the page grows an "Invite & spectate" panel with the current table count and copyable links.
- **The watch link is read-only.** Opening `/watch?w=…` leads to a spectate page that mirrors every game in progress on the service (table picker, board, and move list, refreshed every second), but a spectator occupies no table seat and never sees the CSRF token, so every play route refuses them — watching cannot move a stone by construction. The watch string and the invite string are two independent keys that do not interchange.
- **The link stays in the address bar only once.** The server exchanges `?k=` for a cookie and redirects to a clean `/`, so refreshes and back-navigation never carry the credential; the cookie is `HttpOnly` and `SameSite=Strict`.
- **Ten tables at most by default** (`--max-sessions`, 1–16). Capacity is a resource limit, not a queue: every table holds its own network and search tree, so an eleventh visitor gets a 429 "share is full" page instead of displacing somebody's game. "End my table" in the page releases the slot. Memory is not the bottleneck — measured at roughly 16–30 MB per table (the network weights are only about 2 MB; the search tree dominates), so ten tables cost a few hundred MB. **CPU is what runs out first**: each table searches independently, so the more people think at once, the longer each of them waits.
- **Temporary by construction.** The invite string and every table live in memory only and die with the service; after Ctrl+C the guests' next poll simply fails.
- **Trusted networks only.** The link is a ticket: whoever holds it can spend your CPU and memory (each table adds a network and a search tree, so memory grows with the table count). LAN sharing is fine; to expose this to the internet use `--public` below, which brings its own password.
- With `--share` but the default host, the service still listens on 127.0.0.1 only; startup says so and suggests the `--host` value to use. With `--host` set to a LAN address but no `--share`, the service is a single table with **no credentials**, which is only appropriate on a network you trust.

### Exposing it to the internet in one command (`--public`)

Adding `--public` is the whole ceremony: the service turns sharing on, binds every interface, starts a Cloudflare quick tunnel (cloudflared must be installed), waits until the tunnel is actually registered, and then prints the public link and password together.

```powershell
.\.venv\Scripts\python.exe main.py webui --public
```

The output looks like:

```text
Visk Web UI: http://127.0.0.1:8765 (CPU inference; training continues)
分享链接（最多 10 桌，进入时需输入口令）: http://192.168.1.3:8765/?k=<random>
观战链接（只读，不占棋桌）: http://192.168.1.3:8765/watch?w=<random>
正在为公网访问启动 Cloudflare 隧道（cloudflared 需能连上外网）……

公网访问口令: <random password>
公网邀请链接: https://<random-name>.trycloudflare.com/?k=<random>
公网观战链接: https://<random-name>.trycloudflare.com/watch?w=<random>
```

Send players **the invite link and the password**; send spectators **the watch link** (read-only, no seat consumed). Details:

- **The password is generated for you.** Going public hands the board to strangers, so `--public` never opens a passwordless share; pass `--password <password>` to choose your own. The password only gates entry (typed once, then exchanged for a cookie) while the invite string is itself a ticket — either one gets in, so give both only to people you mean to.
- **One service, two kinds of guest.** A tunnel guest is handed `https://<tunnel host>/?k=…` (built from the request's `Host` and `X-Forwarded-Proto`; the watch link becomes `/watch?w=…` the same way), while a LAN guest still sees the `http://192.168.1.3:8765/?k=…` printed at startup. No `--trusted-host` is needed: `--public` admits only the hostname this run's tunnel was actually given.
- **If the tunnel cannot start, the whole command exits** rather than leaving a service you believe is public when it is not. The wait defaults to 40 seconds; use `--tunnel-timeout` to change it.
- **The hostname changes on every restart**, and quick tunnels carry no uptime guarantee. Cloudflare itself warns that a new hostname "may take some time to be reachable" (about 20 seconds of DNS propagation in testing here). If cloudflared is not on PATH, pass `--cloudflared <path>` or run `winget install --id Cloudflare.cloudflared`.
- **Ctrl+C takes the tunnel down with it**, so the public link stops working immediately. The tunnel process is owned by the service; no second window is needed.
- `--public` cannot be combined with a specific interface address (cloudflared always dials `127.0.0.1`) and says so instead of failing obscurely; for another bind address use the manual route below.
- **`--max-sessions` sets how many tables run at once** (10 by default, 16 maximum). Each holds its own network and search tree at roughly 16–30 MB, so memory is not the issue; CPU is, because ten concurrent searches slow every one of them down. Lower it when the machine struggles.

### Sharing through a Cloudflare tunnel (manual)

When you need your own domain, a reverse proxy, or a named tunnel, the pieces still combine by hand. The Host fence accepts loopback, this machine's addresses, and exactly the names you pass, so DNS-rebinding style attacks stay outside.

```powershell
# 1) the service must be reachable by the tunnel: cloudflared dials 127.0.0.1:8765, so bind every interface
.\.venv\Scripts\python.exe main.py webui --host 0.0.0.0 --share --password <password> `
    --trusted-host .trycloudflare.com
# 2) start the tunnel in another window; the script prints this run's public URL
.\scripts\tunnel-webui.ps1
```

Details:

- **`--trusted-host` takes two forms**: a full name (`board.example.com`) admits just that host, both bare and with this port because the proxy decides the port; a leading dot (`.trycloudflare.com`) admits the whole subdomain tree, which suits a quick tunnel whose hostname changes on every restart.
- **Tunnel guests get public links**: the "Invite & spectate" panel shows the invite link as `https://<host you are visiting>/?k=…` and the watch link as `/watch?w=…`, derived from the request's `Host` and `X-Forwarded-Proto`, while LAN visitors still see the address printed at startup.
- **A quick tunnel's hostname is random** and changes when cloudflared restarts; `scripts/tunnel-webui.ps1` prints the new one. The tunnel itself has **no authentication at all**, so always pair it with `--password`.
- Before starting the tunnel, make sure port 8765 serves the **shared** instance: if an older loopback-only service without `--share` holds the port, the tunnel reaches that one and guests see "Local access only" or a single unauthenticated table.


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
- The architecture is chosen by `arch`, and `ARCHITECTURES` in `vk/network.py` is the single source of truth (a width plus an `R`/`T` block-pattern string); the block count and the transformer count are derived from that string, so a name and the layers actually built cannot disagree. The default `hybrid-128-10` is 128 channels, 10 blocks, pattern `RRTRRTRRTR` — 7 residual blocks and 3 transformer blocks, about **2.80 M** parameters. A residual block is `Conv3×3 → BN → ReLU → Conv3×3 → BN` with a residual add and ReLU; a transformer block is pre-LN, 8-head self-attention and a `128→512→128` GELU MLP, with a **2D relative position bias** (one 29×29 table per head, indexed by `(dr, dc)`) and **no CLS token** — the 225 board points are the tokens. `legacy-64-6` is the pre-hybrid 64-channel, 6-residual-block network, kept parameter-name-for-parameter-name so old checkpoints still load (see the resume section).
- Policy and value share the whole trunk; the policy head emits 225 logits (MCTS applies the softmax) and the value heads are `tanh` into `[-1, 1]`. A head pools the trunk globally and then applies `128→64→1`: the original flattened `1×15×15` into 225 entries, which collapses to one dimension at batch size 1, and MCTS expands one leaf at a time. There are now **three heads** (`final`/`mid`/`short`): the search leaf value is `search_value_mix` (default final 0.5 + mid 0.5), so Q is no longer decided by one saturated scalar.
- PUCT coefficient 2 and `simulations` new searches per move; subtrees and existing visit counts are reused. **Hard rules cover only the deterministic cases**: complete a five if one is available, otherwise block the opponent's five; every other legal point stays reachable. Forcing fours and the neighbourhood of the stones became a **soft additive logit bias** (`search_bias_four=2.0`, `search_bias_neighbour=0.5`) instead of a hard filter that permanently hid moves a heuristic disliked. An empty board only tilts the prior towards the centre.
- **Playout cap randomization**: each move draws its budget from `cheap_search_prob=0.75` — the full search uses `simulations` (400 by default), the cheap one `cheap_search_simulations` (64, never more than the full budget). Only full moves supervise the policy head (`cheap_search_target_weight=0.0`); cheap moves contribute value data only. A move whose hard rules leave one candidate runs a single simulation and still produces value.
- **The policy target is noise-corrected before it is pruned**: `policy_noise_correction` subtracts the expected visit mass the Dirichlet exploration contributed, then `policy_target_prune_prop=0.02` / `policy_target_prune_min_count=2` drop children the search never really expanded, and the rest is renormalised. The loss adds a soft target at `policy_soft_temperature=2.0` (`policy_soft_weight=0.25`) so the head cannot collapse onto one move per position.
- **Replay is sampled by surprise**: `policy_surprise`/`value_surprise` are written into each record at generation time and the sampling weight is `w = (1-0.5) + 0.5·clip(s/1.0, 0, 5)` (`surprise_uniform_share`/`surprise_ref`/`surprise_cap`), so there is both a floor and a ceiling and one anomalous position cannot dominate a batch. Weights are reported every round.
- Games no longer start from an empty board. `balanced_opening` samples an 8-move balanced opening (the first 4 moves anywhere legal, the rest only near stones already on the board). Self-play, the champion promotion match, and Rapfi evaluation share the same sampler and the same move count, so evaluation and training face the same distribution. Setting `opening_plies: 0` restores empty-board openings for comparison.
- Self-play root nodes use 25% Dirichlet noise with total concentration 10.83 (alpha = 10.83/candidates); the noise is applied to **full moves only**. The first 20 moves are sampled from the root visit-count distribution, after which the maximum visit count is taken.
- The model is frozen each round, and by default 16 Windows spawn CPU game processes generate 32 games; the main process merges requests into GPU batches, waiting up to about 3 ms per batch. On an RTX 5070 Ti, measurements at 4 workers left both CPU and GPU underutilized, hence the higher concurrency; adjust `--workers` according to your machine's load. A single forward pass costs about 2.2 ms under `hybrid-128-10` (about 1.0 ms under the legacy 64×6), which tightens the batching window — read `average_inference_batch_size` before changing `--workers` or `simulations`.
- After sampling finishes, train for 200 steps with batch size 256 and a replay pool of the most recent 100,000 positions; Adam with learning rate 0.001, L2 coefficient 0.0001, and gradient norm clipped at 5. No parameter is updated until the replay pool reaches `min_replay_size`, so that threshold must stay **well below** one round's output — including a round cut short by the time budget — or training never begins. The per-round `trainable` and `min_replay_size` fields exist to make that visible.
- The loss is assembled in `vk/objective.py`: cross-entropy against the hard target at `policy_weight=1.0` plus a soft-target term at 0.25, and a **masked mean squared error per value head** (`value_weight_final=1.0`, `value_weight_mid=0.5`, `value_weight_short=0.25`). A head with no valid row in a batch contributes no gradient at all rather than being pulled towards zero. Training and pretraining call the same function, and every weight lives in `vk/config.py`; the optimizer applies L2, and rotations/mirrors are applied to both the board and the policy while the "Black to move" plane is unchanged.
- Every round writes diagnostics into `metrics.jsonl`: target and prediction saturation per head (`mean|v|`, `|v|>0.9`, `|v|<0.5`), policy target and network entropy, `kl_target_prior` (what the search added over the prior) and `kl_target_network` (how stale the target is), `q_spread_mean` (sibling Q spread), `policy_valid_share`/`full_search_share` (whether PCR is doing what it says), `horizon_bootstrap_share`, and the surprise and sampling-weight distributions. `vk/diagnostics.py` defines the field list.

The hyperparameters are a workable starting point, not the result of completed tuning. A standard freestyle opening gives Black an overwhelming first-move advantage, so empty-board self-play degenerates into one colour winning everything; training therefore starts from balanced openings. Even so, Black and White results must be reported separately, and playing strength cannot be judged from the self-play win rate or the training loss alone.

## Duration, Stopping, and Resuming

`-Hours` is the total budget for this run's sampling, optimization, and automatic evaluation; saving files and shutting down the process may take a few extra seconds. The first Ctrl+C issues a safe-stop request: completed games are kept, unfinished games are discarded, and the process saves and exits. A second Ctrl+C is a forced interrupt and may preserve only the previous checkpoint.

If fewer than 32 games are finished when the time budget is reached, the completed positions are still saved. If the current round has an unfinished parameter update, the checkpoint records `pending_steps`; the next resumed run completes those updates first and then begins the next sampling round. Short segmented training therefore does not keep sampling without ever updating. If not even one game was completed, no fabricated draw labels are written.

Each round atomically saves one complete checkpoint (format 2) and keeps the three most recent. A checkpoint contains the network, optimizer, configuration, the **structured replay array**, total game count, round count, training step count, pending updates, and random number state. Resuming allows adjusting the self-play concurrency via `--workers`; all other configuration must still match the checkpoint — `vk/config.RESUME_FREE` is the complete set of fields that may change. Process scheduling and GPU operations affect numerics and the order in which samples arrive, so resuming guarantees a continuous training state, not bit-for-bit reproducibility.

**Changing the architecture means retraining from scratch.** The checkpoint records `arch`, and both `--init-checkpoint` and resume compare it. Checkpoints written before the hybrid existed have no such key, so it is resolved back from `channels`/`blocks` to `legacy-64-6`; the old 64×6 weights therefore cannot be mistaken for initialisation of the new network and instead report `architecture mismatch`. To compare against an old run, read it with `--checkpoint` (inference only) — `Network` rebuilds whatever family that checkpoint's own `arch` names.

**A format-1 checkpoint still works for inference and as a cold start.** `vk/storage.upgrade_format1()` keeps the old single value head as the final head and seeds `mid`/`short` from it (the network starts by predicting the same value at every horizon instead of from noise); the old replay pool was a list of tuples with no schema and is dropped. A format-1 checkpoint therefore **cannot be resumed** (resuming requires format 2 and says so explicitly), but `--init-checkpoint`, `evaluate --checkpoint` and the Web UI all keep reading it. `vk/storage.load_model_state()` is the only place old checkpoints are understood.

Only `.pt` files produced by this project and trusted should be loaded: full recovery uses Python pickle. `best.pt` holds inference weights only, without the optimizer or replay pool, and cannot be used to resume; use `latest` or a complete `checkpoint-*.pt` to resume. An existing training directory requires an explicit `-Resume` and will not be overwritten as a new training run.

## Files and Evaluation

```text
runs/freestyle/              freestyle model, replay, and logs
runs/renju/                  renju model, replay, and logs
  config.json                current configuration
  checkpoint-*.pt            the three most recent complete checkpoints
  best.pt                    first updated model, later replaced by head-to-head results
  metrics.jsonl              time, throughput, loss, search/target/prediction diagnostics, per-color win rates
  games.jsonl                moves and results of completed games
  evaluations.jsonl          periodic evaluations during training
  evaluation-*.json          manual evaluation reports
artifacts/                   benchmarks and short-run tests, not real training output
```

Every 10 rounds the model is evaluated in turn against the historical `best`, a random baseline, and a fixed tactical baseline; both sides use the same search budget and no exploration noise is added. Both sides play the same balanced opening with the colour swapped, so an opening pair is directly comparable. The report gives total wins and losses, per-color results, the score, and a conservative 95% Hoeffding interval based on opening pairing. When the time budget is reached it is marked `complete: false`, and half-finished games are not counted as draws.

`best` is replaced when the comparison against the historical model completes and the score exceeds 55%. The initial `best` is merely the first updated model, and 55% is an engineering promotion threshold, not proof of statistical significance or expert-level strength. By default the confidence interval for 20 games per opponent is wide; `-Pairs` increases the number of independent opening groups. The random and tactical baselines are used for evaluation only and produce no rewards or labels for training.

## Testing and Development

```powershell
python -m pytest -q
python main.py doctor

# short run with a small model / low search budget, for functional acceptance only
python main.py train --rule freestyle --config configs/smoke.json --output artifacts/my-smoke-freestyle --hours 0.04 --max-rounds 1
python main.py train --rule renju --config configs/smoke.json --output artifacts/my-smoke-renju --hours 0.04 --max-rounds 1

# one checkpoint against itself: one side searches, the other plays the raw policy
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent self-policy --pairs 25 --minutes 120
```

Tests cover wins in all four directions, overlines, real and false double-threes, same-direction and crossing double-fours, edges and a full board, legal actions, the sign of search backup, immediate wins and necessary defenses, augmentation alignment, full-game labels, actual parameter updates, checkpoint resumption, stopping, rule isolation, and worker exception propagation, plus eight new pure-function groups added by the refactor: schema, target construction, replay sampling, loss masks, configuration validation, diagnostics, PCR and soft bias, and the search-versus-policy gap.

The implementation is layered, one job per module: `vk/game.py` rules, `vk/candidates.py` hard rules and soft bias, `vk/targets.py` target construction, `vk/search.py` PUCT and PCR budgets, `vk/records.py` the position schema, `vk/selfplay.py` sampling, `vk/replay.py` the structured replay buffer, `vk/network.py` the network and inference, `vk/objective.py` the loss, `vk/training.py` the round loop, `vk/storage.py` checkpoints, `vk/config.py` configuration, `vk/diagnostics.py` the round report, `vk/datasets.py` shards, `vk/pretraining.py` the Rapfi cold start, `vk/teacher.py` teacher generation, `vk/evaluation.py` matches and the search gap, and `vk/cli.py`/`vk/webui.py` the entry points.

Method references: [AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/), [DeepMind OpenSpiel AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md), [KataGo](https://github.com/lightvector/KataGo) (playout cap randomization, surprise weighting, pruned soft policy targets), [PyTorch CUDA installation](https://pytorch.org/get-started/previous-versions/). This project is a simplified AlphaZero system for Gomoku that borrows KataGo's ideas (multi-scale value, PCR, soft policy, surprise sampling) without copying its complexity: no ownership or score head, no playout doubling advantage, no history planes.
