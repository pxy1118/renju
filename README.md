# Visk · 双规则五子棋

简体中文 | [English](README.en.md)

从随机权重开始，以 **MCTS 自我对弈 → 回放池 → 策略价值网络更新** 学习下棋。两种规则共用实现，权重和数据分开保存。项目提供训练系统，尚未包含训练成熟的棋力模型。

## Rapfi 教师蒸馏（第一阶段：freestyle）

推荐的新训练路径是“Rapfi 离线教师 → 监督预训练 → 独立 MCTS 自博弈”。最终模型、命令行和 Web UI 都只依赖本项目的 PyTorch 网络与搜索；[Rapfi](https://github.com/dhbloo/rapfi) 只通过进程协议生成标签或作为评测对手。主项目不链接、不复制到 Python 包、也不发布 Rapfi 二进制或权重；既可以使用下面的独立子模块构建，也可以显式传入自行安装的 `pbrain-rapfi.exe` 和配置/权重目录。

Rapfi 现在作为独立 Git 子模块位于 `external/rapfi`，保留其自身 GPLv3 许可证和提交历史；官方 Networks 子模块中的权重使用其声明的 CC0 许可。首次克隆或更新后执行以下命令，可在被 Git 忽略的 `external/rapfi-runtime` 中构建 exe 并准备配置/权重：

```powershell
git submodule update --init --recursive
.\scripts\build-rapfi.ps1
```

主项目不链接 Rapfi，也不会发布生成的运行时目录。若分发包含 Rapfi 子模块或自行构建的二进制，仍需分别遵守 Rapfi 的 GPLv3 条款。

```powershell
python main.py teacher-generate --rule freestyle --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --output data/teacher/freestyle-v2 --positions 50000
python main.py pretrain --rule freestyle --dataset data/teacher/freestyle-v2 --output runs/freestyle-pretrain-v2 --steps 20000
python main.py train --rule freestyle --config configs/hybrid-freestyle.json --output runs/freestyle-hybrid-lr1e4 --init-checkpoint runs/freestyle-pretrain-v2/best.pt --hours 2
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-hybrid-lr1e4 --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --minutes 60 --pairs 50
# 与旧塌缩模型做交换执色 A/B；64 组（128 局）才能同时计算连续 128 局偏色指标
python main.py evaluate --rule freestyle --checkpoint latest --output runs/freestyle-hybrid --opponent checkpoint --opponent-checkpoint runs/freestyle/checkpoint-old.pt --pairs 64 --minutes 120
```

教师默认运行 4 个进程，每进程 4 线程、256 MB Hash、200,000 节点，读取最后一个完整的 Top-5 深度；单次 5 秒超时，进程最多重启两次。YXBOARD 按当前行棋方编码（`1=当前方`、`2=对手`）。残缺输出、重复/非法落点和持续失败会显式报错，不产生替代标签。数据按整局分到 80/10/10，再仅对训练批次做 D4 增强；NPZ 不允许 pickle，每片最多 4096 条，`manifest.json` 记录生成参数及可执行文件、配置和权重哈希。`data/teacher/` 已被 Git 忽略。

### 数据格式 4：一个局面就是一条显式记录

`vk/records.py` 是唯一真源：一条 self-play / 教师记录是一组**具名字段**，不是 `(x, pi, z)` 这样的元组。生产（`vk/selfplay.py`、`vk/teacher.py`）、存储（`vk/datasets.py`）、回放（`vk/replay.py`）与损失（`vk/objective.py`）都按同一张表读写。

| 字段 | dtype | 含义 |
|---|---|---|
| `state` / `policy` | `uint8(3,15,15)` / `float16(225,)` | 棋盘平面 / 剪枝去噪后的搜索目标 |
| `policy_valid` / `policy_weight` | `uint8` / `float16` | 该行是否监督策略头；PCR 的 cheap 着法权重为 0 |
| `value` / `value_valid` | `float16(3,)` / `uint8` | 三个 value head 的目标（float 无效位为 NaN），位掩码说明哪几个有效 |
| `search_value` / `q_spread` | `float16` | 本局面的根 Q；根节点被访问子节点 Q 的 max−min |
| `policy_surprise` / `value_surprise` | `float16` | KL(目标‖先验)；\|Q − V_net\| |
| `weight` / `simulations` / `full_search` | `float32` / `uint32` / `uint8` | Surprise 采样权重；本着预算；是否深搜 |
| `game_id` / `ply` / `winner` / `source` | `uint32` / `uint16` / `int8` / `uint8` | 对局与手数、绝对胜负、来源（self-play / 教师 / 旧文件归一化） |
| teacher 附加组 | `teacher_best`、`teacher_nodes`、`teacher_topk_actions`、`teacher_topk_winrates` | 引擎分析，self-play 行写哨兵 |

三个 value head 是 **final / mid / short**：`final` 读终局结果，`mid`/`short` 在 10 / 4 手的地平线上截断，未终局时用该局面自己的 `search_value` 做 bootstrap（奇数手符号翻转）。这就是「多时间尺度」：终局标签在小棋盘上必然饱和（`|v|>0.9` 一度占 97%），而地平线标签不是。

旧数据不需要重新生成：`vk/records.normalize_legacy()` 在**读取时**把 v2/v3 的 `value` 映射成 final head 与 `search_value`，地平线 head 标为无效、`source=2`（这些文件从未记录过局面结果）。`vk.datasets.combine_datasets()` 可把 v2/v3/v4 混着合并，重编号时同时改写高位，因此同一局的所有位置始终落在同一个 split。

`artifacts/diag_teacher_regret.py` 用这个尺度测量模型的实际代价：

```powershell
python artifacts/diag_teacher_regret.py --dataset data/teacher/freestyle-v2 --checkpoint runs/freestyle-pretrain/best.pt --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --split test --limit 1500 --hard-rules forced
```

对每个局面它取**一次** Rapfi Top-5 根搜索作为参照：若模型着法在 Top-5 内，`regret = p_top1 - p_模型`（同一次搜索，无需额外查询）；否则再分析模型着法之后的子局面，用 `regret = p_top1 + p_child - 1`。所有数字都是**原始胜率百分点**（`0.05` 就是损失 5 个百分点），报告分桶、分 ply 段、以及“top1 与 top2 差距 > 0.05”的决策关键子集，并输出逐行 CSV。Rapfi 的作答按 `(局面, 引擎, 节点预算)` 缓存，可中断续跑。

### 解耦 policy / value / MCTS / 候选集

`--search`、`--hard-rules`、`--search-bias`、`--opening-mode` 四个开关互相独立，用来判断每个部件的真实贡献（命令行的显式参数优先于配置文件和 checkpoint 内保存的配置）：

| 开关 | 取值 | 含义 |
|---|---|---|
| `--search` | `mcts` / `policy` | MCTS+价值，或直接取策略 argmax（不搜索、不用价值头） |
| `--hard-rules` | `forced` / `none` | 只保留**确定性**战术（成五、必挡），或完全不做硬约束 |
| `--search-bias` | `tactical` / `none` | 对成四威胁与棋子邻域加先验偏置；**永不排除任何合法着法** |
| `--opening-mode` | `sampled` / `teacher` / `book` / `none` | 统一随机平衡开局 / 教师式开局（随机首手后按 Rapfi Top-5 采样） / 均势开局库 / 空盘 |

### 均势开局库（`--opening-mode book`）

无禁手是黑先手必胜的游戏：v3 教师数据 772 局里 **93.4% 是黑胜**，所以"执白胜率"基本是规则常数，而随机落子的开局压不住先手优势。`artifacts/opening_book_generate.py` 把"公平"从随机改成**被验证过**：

```powershell
# 生成：采样候选 → 用 Rapfi 测量黑白双方胜率 → 只保留双侧都在 0.5±gap 内 → D4 去重
python artifacts/opening_book_generate.py --output data/openings/freestyle-balanced.json --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --count 200 --candidates 9000 --gap 0.15 --filter-nodes 200000 --workers 4
# 复验：换一次搜索重测同一批开局，报逐开局漂移
python artifacts/opening_book_generate.py --verify data/openings/freestyle-balanced.json --output artifacts/opening-verify.json --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --filter-nodes 200000 --verify-sample 50
python main.py evaluate --rule freestyle --checkpoint best --output runs/policy-baseline --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --opening-mode book --opening-book data/openings/freestyle-balanced.json
```

实测结果与它的局限（**逐开局平衡在 64 倍节点跨度上摆动 0.12–0.53，只有总体分布可信**）见 [验收记录](ACCEPTANCE.md) 第九节。相同的 `--opening-mode book` 也能用于自博弈：`collect`/`play_game` 会按 seed 在库内轮转。


四种组合的对照（同一 checkpoint、同一 200k 节点 Rapfi、同一开局种子，`--pairs` 是**开局对数**，实际对局数为其两倍）：

```powershell
# A1 policy+nohard   A2 policy+hard   A3 policy+hard+bias   A4 mcts+hard+bias
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --minutes 120 --search policy --hard-rules none --search-bias none
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --minutes 120 --search policy --hard-rules forced --search-bias none
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --minutes 120 --search policy --hard-rules forced --search-bias tactical
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent rapfi --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --minutes 120 --search mcts --hard-rules forced --search-bias tactical
# search-vs-policy：同一个 checkpoint 自己对自己，一侧搜索、一侧纯策略
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent self-policy --pairs 25 --minutes 120
```

每个报告都带 `search_mode`/`hard_rules`/`search_bias`/`opening_mode`/`opening_seed`，以及搜索形状诊断：`root_visited_moves`、`root_max_visit_share`、`root_visit_entropy`、`q_spread`（兄弟 Q 扩散）、`kl_target_prior`（搜索比先验多出的信息）。`--opponent self-policy` 给出 `search_advantage_pp` 与其配对区间：区间覆盖 0 就是「搜索没有带来可测增益」。它们才是“搜索有没有展开”的证据——`节点访问数 / 模拟次数` 恒等于 1，没有信息量。跨臂比较用 `vk.evaluation.paired_delta()` 在同一开局上做成对 bootstrap 区间，而不是并排看两个 Wilson 区间。

模型先手、Rapfi 标注的 DAgger 式数据：

```powershell
python artifacts/dagger_collect.py --checkpoint runs/freestyle-pretrain/best.pt --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --output data/teacher/dagger-v1 --positions 2000 --hard-rules none --search-bias none --search policy
python main.py pretrain --rule freestyle --dataset data/teacher/freestyle-v2 --mix-dataset data/teacher/dagger-v1 --mix-share 0.5 --output runs/freestyle-pretrain-v3 --steps 20000
```

DAgger 输出仍是标准 position schema；每条决策的 `regret` 与 `hard` 只写在 `diagnostics.csv` 里，不进训练 schema。`pretrain --value-weight 0` 训练纯策略模型：此时 loss、best checkpoint 的选择分数、以及战术闸门都会同步切换（闸门改为策略 argmax + 强制着法），`value_mae` 不再作为验收项。

`scripts/compare_arms.py` 一次跑完 A1–A4 并直接给出成对区间（`--pairs` 是开局对数）：

```powershell
python scripts/compare_arms.py --checkpoint runs/freestyle-pretrain/best.pt --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --pairs 25 --output artifacts/compare-arms
```

结果写入 `artifacts/compare-arms/<arm>.json` 与 `comparison.json`。本轮实录见 [验收记录](ACCEPTANCE.md) 的「度量修正与四旋钮解耦」。

### critical-regret 加权与子局面价值标签

教师数据里绝大多数局面「怎么走都差不多」，真正决定棋力的是少数 `top1 - top2 > 0.05` 的决策点。加权训练把预算移到后者，**权重进采样器而不是 loss**，因此目标函数不变：

```powershell
python main.py pretrain --rule freestyle --dataset data/teacher/freestyle-v3 --output runs/critical --steps 20000 --value-weight 0 --critical-weighting
```

默认分桶 `gap<0.01 → 0.25`、`0.01–0.03 → 0.5`、`0.03–0.05 → 1.0`、`0.05–0.10 → 2.0`、`>0.10 → 4.0`（`vk.datasets.DEFAULT_GAP_WEIGHTS`），需要格式 3 的 top-k 胜率；格式 2 数据会显式报错而不是静默退回均匀采样。

**实测结论（阴性）**：在 50k v3 数据上，`--critical-weighting` 确实把 82.5% 的批预算给到 gap>0.03 的 41.9% 局面（gap>0.10 的 9.2% 单独占 38.8%），但同架构、同种子、同步数的对照模型在配对 regret 上没有改善（`A − B = +0.0020`，95% CI `[−0.0042, +0.0084]`），决策关键子集上点估计还略微反向（0.1417 → 0.1501）。原因是采样加权不改变目标函数的最优解。详见 [验收记录](ACCEPTANCE.md) 的「critical 加权实测」。

两个模型在同一批局面上配对比较（根搜索共享，只为 Top-5 之外的着法各查一次子局面）：

```powershell
python artifacts/diag_teacher_regret.py --dataset data/teacher/freestyle-v3 --checkpoint runs/freestyle-pretrain/best.pt --compare-checkpoint runs/critical/best.pt --engine external/rapfi-runtime/pbrain-rapfi.exe --engine-dir external/rapfi-runtime --split test --limit 1500 --candidates legal
```

报告里的 `paired` 段给出 `mean_regret_delta` 与 bootstrap 95% 区间（正数表示第二个模型更省胜率）。

价值标签能不能区分同一父局面的不同着法，是这个工具要回答的问题：

```powershell
python artifacts/value_dataset_diagnostic.py --dataset artifacts/teacher-probe-v3 --split train --limit 0 --output artifacts/value-children-probe
```

默认 `chained` 模式不需要引擎：用父局面的 `1 - W_i` 作为每个子局面的目标，一个父局面展开 k 行。加 `--engine` 则对每个子局面单独搜索。报告包含兄弟标签差的分位数与子局面 `|V|` 分布 —— 历史 value 饱和（叶节点 `|V|` 均值 0.963）的根因是「一个父局面只监督一个标量」，而不是标签里没有相对结构。


`--init-checkpoint` 只读取网络参数和教师来源，不继承优化器或回放池，并与 `--resume` 互斥。由 `pretrain` 生成但未达到 Top-1 45%、Top-5 80%、价值 MAE 0.20 和战术全对门槛的检查点会被正式自博弈拒绝。旧格式 1 checkpoint 仍可推理和按原配置续训。

`configs/hybrid-freestyle.json` 对教师预训练模型使用较保守的 `1e-4` 微调学习率。神经网络之间的 champion 配对赛复用自博弈的多进程/中央批推理模式；断点恢复会以所选完整 checkpoint 的轮次为提交边界，自动移除其后的残缺 JSONL 诊断记录。任务队列使用显式结束标记，避免 Windows 进程刚启动时把暂时不可见的种子误判为任务耗尽。

本机测试结果、GPU 短跑和吞吐限制见 [验收记录](ACCEPTANCE.md)。

## 快速启动

### 浏览器下棋（可与训练同时运行）

双击项目里的 `start-webui.bat`，会在当前窗口启动服务并自动打开浏览器：
**http://127.0.0.1:8765**。之后下棋完全在网页操作，无需在终端输入坐标。
**服务就在这个窗口里运行，按 Ctrl+C 即可停止**（也可以用 `python main.py webui` 直接前台运行）。
重复双击会检测到端口已占用，改为直接打开已运行的界面，不会起第二个服务。也可以执行：

```powershell
.\.venv\Scripts\python.exe main.py webui
# 自定义端口；--output 在这里表示含 freestyle/renju 子目录的模型根目录
.\.venv\Scripts\python.exe main.py webui --port 8766 --output runs
```

支持规则与执色选择、最新模型/Champion 或预训练最佳/指定模型的手动选择、32/64/200 次 MCTS 搜索、手数显示、撤回最近一轮、重新开始及终局提示；终局会高亮获胜的五子，状态行左侧的圆点改为胜方棋子（和棋隐藏），因此“谁赢了”由文案、棋子和连线三处一致表示。模型下拉会扫描 `--output` 下的各个运行目录，列出规则匹配的预训练 `best.pt`、混合训练 Champion 和完整 `checkpoint-*.pt`（运行名、时间、大小），所以 `runs/freestyle-pretrain-v2/best.pt` 与 `runs/freestyle-hybrid/*` 都能直接加载；“最新模型”优先选择最新完整检查点，“Champion / 预训练最佳”选择最新 `best.pt`。连珠合法性复用训练规则；没有检查点时不会拿随机权重冒充训练模型，“开始对弈”会改为“暂无可用模型”并说明原因。

Web UI 固定使用 **CPU、单推理线程**，不申请 CUDA 或训练 GPU 锁，不写训练目录；仍会占用 CPU 和内存，单个服务进程实测常驻约 550 MB 工作集、约 1.5 GB 私有内存（含 CUDA 版 PyTorch 运行时），与训练同时运行会争用内存。新局加载当时已保存的模型，局中固定权重。点击“重新开始”可使用更新后的检查点。服务默认仅监听本机，所有浏览器标签共享一张棋桌，刷新可恢复当前棋局；退出服务后棋局不保留。思考期间请等待本手结束再悔棋或重开。客户端在模型思考时以 250 毫秒、空闲时以 1 秒轮询状态，因此数百毫秒的思考时间也能看到“模型思考中”。直接执行 `python main.py webui` 时日志就打在终端上。模型正在训练不等于已有成熟棋力。

### 临时分享给朋友（默认最多 10 桌）

默认不开分享；加上 `--share` 与 `--host <局域网IP>` 后，服务会打印两条链接：**邀请链接**让朋友入局对弈，**观战链接**让人只读观战。临时分享可以直接双击 `share-webui.bat`（等价于 `start-webui.ps1 -Share`，自动选默认路由的局域网地址），或手写命令：

```powershell
# 查本机局域网地址，然后把它填进 --host
Get-NetIPAddress -AddressFamily IPv4 | Where-Object AddressState -eq Preferred
.\.venv\Scripts\python.exe main.py webui --host 192.168.1.3 --share
# 可选：再加一道口令，链接之外还要输入它
.\.venv\Scripts\python.exe main.py webui --host 192.168.1.3 --share --password <口令>
# 不想填具体 IP 时，绑所有网卡也可（分享链接自动选默认路由地址）
.\.venv\Scripts\python.exe main.py webui --host 0.0.0.0 --share --max-sessions 10
```

启动日志形如：

```text
Visk Web UI: http://127.0.0.1:8765 (CPU inference; training continues)
分享链接（最多 10 桌）: http://192.168.1.3:8765/?k=<随机串>
观战链接（只读，不占棋桌）: http://192.168.1.3:8765/watch?w=<随机串>
```

分享的语义与边界：

- **每人一张独立棋桌**。打开邀请链接的浏览器会拿到一个会话 Cookie，之后自己的棋局互不干扰；页面右侧出现“邀请与观战”面板，显示当前桌数与可复制的链接。
- **观战链接是只读的**。打开 `/watch?w=…` 的人进入一个观战页：镜像服务上所有正在进行的对局（选桌、棋盘、落子记录每秒刷新），但观战者不占用棋桌名额、看不到 CSRF 令牌，所有落子路由对它一律拒绝——观战在结构上就落不了子。观战串与邀请串是两把独立的钥匙，互不通用。
- **链接只在地址栏停留一次**。服务把 `?k=` 换成 Cookie 后重定向到干净的 `/`，所以刷新、后退都不会再把凭据带在地址里；`Set-Cookie` 带 `HttpOnly` 与 `SameSite=Strict`。
- **默认最多 10 桌**（`--max-sessions`，1–16）。名额是资源上限而不是队列：每张棋桌各自持有一份网络与搜索树，所以第 11 个人会看到“名额已满”的 429 页面，而不是把别人的棋局挤掉。下完点“结束我的棋桌”就会释放名额。内存不是瓶颈：实测每桌约 16–30 MB（网络权重只有约 2 MB，占大头的是搜索树），10 桌加起来也就几百 MB；**先撑不住的是 CPU**——每个棋桌各自搜索，同时思考的人越多，每人等待越久。
- **临时性**。邀请串和所有棋桌都只在内存里，服务一停全部失效；`Ctrl+C` 后朋友那边只会在下一次轮询失败，页面提示连接失败。
- **仅限可信网络**。邀请链接等于门票，拿到链接的人可以在你的机器上占用 CPU 和内存（内存随桌数线性增长）。局域网分享没问题；要暴露到公网请用下文的 `--public`，它会自带口令。
- `--share` 但不改 `--host` 时，服务仍只监听 127.0.0.1，启动日志会明确提示“其他机器无法连接”并给出建议的 `--host` 值；`--host` 指定了局域网地址但没加 `--share` 时，服务是单桌且**无凭据**的，只适合你确信网络可信的场合。

### 一条命令暴露到公网（`--public`）

加 `--public` 就够了：服务会自动开分享、绑所有网卡、起一个 Cloudflare quick tunnel（本机需已装 cloudflared），等隧道真正注册成功后再把公网链接和口令一起打出来。

```powershell
.\.venv\Scripts\python.exe main.py webui --public
```

输出形如：

```text
Visk Web UI: http://127.0.0.1:8765 (CPU inference; training continues)
分享链接（最多 10 桌，进入时需输入口令）: http://192.168.1.3:8765/?k=<随机串>
观战链接（只读，不占棋桌）: http://192.168.1.3:8765/watch?w=<随机串>
正在为公网访问启动 Cloudflare 隧道（cloudflared 需能连上外网）……

公网访问口令: <随机口令>
公网邀请链接: https://<随机域名>.trycloudflare.com/?k=<随机串>
公网观战链接: https://<随机域名>.trycloudflare.com/watch?w=<随机串>
```

把**邀请链接和口令**发给对弈的人，把**观战链接**发给只看不下的人即可。要点：

- **口令是自动生成的**。公网等于把棋桌交给陌生人，所以 `--public` 不会开一个无口令的分享；也可以用 `--password <口令>` 指定自己的。口令只挡入口（一次输入换 Cookie），邀请串本身仍是一次性门票，两者拿到任一个都能进，所以两个都要给对人。
- **同一份服务，两种访客**。隧道访客拿到 `https://<隧道域名>/?k=…`（页面按请求的 `Host` 和 `X-Forwarded-Proto` 生成，观战链接同理是 `/watch?w=…`），局域网访客拿到的仍是启动时打印的 `http://192.168.1.3:8765/?k=…`。不需要手写 `--trusted-host`：`--public` 只放行这次隧道真正拿到的域名。
- **隧道连不上就整体退出**，不会留下一个你以为已经公开、其实没公开的服务；这里默认等 40 秒，可用 `--tunnel-timeout` 调整。
- **域名每次重启都会变**，quick tunnel 不保证可用性；Cloudflare 自己也会提示新域名“可能需要一点时间才能访问”（实测本机 DNS 生效约 20 秒）。cloudflared 不在 PATH 时用 `--cloudflared <路径>` 指定，或 `winget install --id Cloudflare.cloudflared`。
- **Ctrl+C 会把隧道一起关掉**，公网链接立刻失效；隧道进程由服务自己托管，不需要另开窗口。
- `--public` 不能和具体网卡地址同用（cloudflared 固定拨 `127.0.0.1`），写了会直接报错；要用别的监听地址就退回下面的手动方式。
- **`--max-sessions` 决定能同时开多少桌**（默认 10，上限 16）。每桌各自持有一份网络与搜索树、约 16–30 MB，所以内存不是问题；真正会顶不住的是 CPU——10 桌同时搜索时每个人都会明显变慢，机器吃力就把它调小。

### 通过 Cloudflare 隧道分享（手动方式）

需要自己控制域名、反向代理或命名隧道（named tunnel）时，仍可手动组合；界面的 Host 校验只接受回环、本机地址和显式放行的域名，所以 DNS 重绑定类攻击仍在门外。

```powershell
# 1) 服务必须同时能被隧道连到：cloudflared 默认连 127.0.0.1:8765，所以绑所有网卡最省事
.\.venv\Scripts\python.exe main.py webui --host 0.0.0.0 --share --password <口令> `
    --trusted-host .trycloudflare.com
# 2) 另一个窗口起隧道，脚本会打印本次的公网地址
.\scripts\tunnel-webui.ps1
```

要点：

- **`--trusted-host` 支持两种写法**：写完整名（`board.example.com`）只放行这一个域名（裸域名与本端口两种形式都接受，因为反代会自己决定端口）；以点开头（`.trycloudflare.com`）放行整棵子域，适合每次重启都会换域名的 quick tunnel。
- **隧道访客拿到的链接会自动换成公网地址**：页面“邀请与观战”面板给出的邀请链接是 `https://<当前访问域名>/?k=…`、观战链接是 `/watch?w=…`（依据请求的 `Host` 与 `X-Forwarded-Proto` 生成），局域网访客看到的仍然是启动时打印的局域网链接。
- **quick tunnel 自带域名随机**，重启 cloudflared 就换地址；`scripts/tunnel-webui.ps1` 会把新地址直接打出来。隧道本身**没有任何鉴权**，所以务必配 `--password`。
- 启动隧道前请确认 8765 上是**分享实例**：如果端口被一个只绑回环、没加 `--share` 的旧实例占用，隧道会连到它，访客只会看到“Local access only”或单桌无口令的界面。


### 停止服务

在运行服务的那个窗口按 **Ctrl+C**。服务随即关闭，当前棋局不保存，下次开局重新读取最新检查点；训练进程和 `runs/` 不受影响。用 `start-webui.bat` 启动时，按 Ctrl+C 后 cmd 可能再问一句“终止批处理操作吗(Y/N)”，按 Y 或直接关掉窗口都可以。

如果窗口已经找不到、但端口仍被占用，先确认是谁在监听再决定是否结束：

```powershell
Get-NetTCPConnection -LocalPort 8765 -State Listen | Select-Object OwningProcess
Get-CimInstance Win32_Process -Filter "ProcessId=<上一步的 PID>" | Select-Object CommandLine
# 命令行是 main.py webui 才是界面服务；含 main.py train 的是训练，别动它
Stop-Process -Id <PID>
```

在项目目录安装依赖并检查 CUDA：

```powershell
Set-Location D:\Workplace\Renju
uv sync --locked
python main.py doctor

# 可选：用自己的机器测量吞吐；不会写入正式训练目录
python main.py benchmark --rule freestyle --minutes 3
python main.py benchmark --rule renju --minutes 3

# 首次训练，每次指定本次运行的时长；顺序执行
python main.py train --rule freestyle --hours 2
python main.py train --rule renju --hours 2

# 后续训练：本次再运行两小时
python main.py train --rule freestyle --hours 2 --resume latest --workers 16
python main.py train --rule renju --hours 2 --resume latest --workers 16

# 与模型下棋（输入行、列，范围 1–15；q 退出）
python main.py play --rule freestyle --checkpoint latest --human-color black
python main.py play --rule renju --checkpoint latest --human-color white

# 每个对手 10 组交换执色对局；60 分钟预算到时输出部分报告
python main.py evaluate --rule freestyle --checkpoint latest --pairs 10 --minutes 60
```

如果尚未激活项目环境，也可以把 `python` 换成 `.\.venv\Scripts\python.exe`。`uv sync --locked` 只安装锁定依赖，不启动训练。锁定版本为 PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、torchaudio 2.11.0+cu130、NumPy 2.5.3 和 Pillow 12.3.0。CUDA 不可用时会报错，不自动使用 CPU。GPU 运行之间使用项目进程锁，不能同时启动两种规则争用显卡。该锁不管理其他软件的 GPU 使用。

## 两种规则

| 选项 | 开局 | 胜负与限制 |
|---|---|---|
| `freestyle` | 黑先，任意空点 | 双方连续五子或以上获胜，无禁手 |
| `renju` | 黑首子固定中心，其后交替自由落点 | 黑方恰好五子获胜；白方五子或以上获胜；黑方三三、四四、长连禁手 |

连珠为**普通开局禁手模式**，不含交换/选点开局、停着、比赛时钟或人工申诉。采用自动终局裁决，禁手落点从动作空间屏蔽；棋盘未满而当前方无合法落点时判负，满盘无胜者判和。这些环境约定不是完整的 RIF 比赛协议。

禁手依据 [RIF 国际规则第 9.1–9.3 条](https://www.renju.net/rifrules/)：同时形成恰好五子时优先获胜；同一个活四的两个端点不重复算成两个四；活三必须能合法延伸成直四，延伸点的三三继续递归检查，不能用简单字符串匹配代替。测试中的固定棋形按这些定义构造，包括同方向双四及延伸点自身为三三的假活三。

## 默认训练配置

`configs/default.json` 列出全部默认值。可复制该文件并通过 `-Config` 传入。新配置应使用新的 `-Output` 目录；续训默认读取检查点原配置，拒绝规则或配置不匹配，防止意外混训。

```powershell
python main.py train --rule renju --hours 4 --config configs/default.json --output runs/renju-experiment
python main.py train --rule renju --hours 2 --output runs/renju-experiment --resume latest
```

- 15×15；输入当前方棋子、对方棋子、当前是否执黑三个平面。
- 网络由 `arch` 选择，`vk/network.py` 的 `ARCHITECTURES` 是唯一真源（宽度 + `R`/`T` 块序字符串），块数与 Transformer 块数都从该字符串派生，名字与实际层不可能不一致。默认 `hybrid-128-10`：128 通道、10 个 block、排列 `RRTRRTRRTR`，即 7 个残差块 + 3 个 Transformer 块，约 **2.80 M** 参数。残差块为 `Conv3×3 → BN → ReLU → Conv3×3 → BN` 加残差后 ReLU；Transformer 块为 Pre-LN + 8 头自注意力 + `128→512→128` GELU MLP，带 **2D 相对位置偏置**（每头一张 29×29 表，按 `(dr, dc)` 取用）且**没有 CLS token**，token 就是 225 个棋盘点。`legacy-64-6` 是改造前的 64 通道 6 残差块网络，按参数名逐字保留，仅用于继续加载旧 checkpoint（见下）。
- 策略与价值共用整个主干；策略输出 225 个 logits（MCTS 侧再取 softmax），价值输出 `tanh` 到 `[-1, 1]`。价值头在主干上做全局平均池化后接 `128→64→1`：原实现把 `1×15×15` 摊平成 225 维再送进线性层，批量大小为 1 时会塌成一维，而 MCTS 正是逐个叶子做单点推理。**现在是三个头**（`final`/`mid`/`short`，见「数据格式 4」），搜索叶值取 `search_value_mix` 的加权（默认 final 0.5 + mid 0.5），因此 Q 不再由同一个饱和标量决定。
- PUCT 系数 2，每步 `simulations` 次新搜索；复用子树和已有访问次数。**硬约束只覆盖确定性战术**：有一步成五就走成五，否则对手有五就必挡，其余全部合法点都可达；成四威胁与棋子邻域改成**加在 logits 上的软偏置**（`search_bias_four=2.0`、`search_bias_neighbour=0.5`），不再把 heuristic 没看上的点永久排除。空盘只把先验压向中心。
- **Playout Cap Randomization**：每个着法按 `cheap_search_prob=0.75` 抽预算——full 用 `simulations`（默认 400），cheap 用 `cheap_search_simulations`（默认 64，且不会超过 full）。只有 full 着法监督策略头（`cheap_search_target_weight=0.0`），cheap 着法只贡献 value 数据；硬规则只剩一个候选点的着法跑 1 次模拟但仍产出 value。
- **策略目标先修正噪声再剪枝**：`policy_noise_correction` 先扣掉 Dirichlet 探索的期望访问量，`policy_target_prune_prop=0.02` / `policy_target_prune_min_count=2` 再丢掉搜索没真正展开的子节点，最后归一化。损失里再加一项 `policy_soft_temperature=2.0` 的软目标（`policy_soft_weight=0.25`），避免策略头塌到单一着法。
- **回放按 surprise 加权采样**：`policy_surprise`/`value_surprise` 在生成时写入记录，采样权重 `w = (1-0.5) + 0.5·clip(s/1.0, 0, 5)`（`surprise_uniform_share`/`surprise_ref`/`surprise_cap`），既有下限也有上限，异常样本无法支配批次。权重、均值与上限都进每轮指标。
- 每局不空盘开始，而是由 `balanced_opening` 采样 8 手平衡开局（前 4 手全盘随机，其后只落在已有棋子的邻域内），自我对弈、champion 晋级赛与 Rapfi 评测共用同一采样器与同一手数，因此评价与训练面对同一分布。空手数（`opening_plies: 0`）可退回空盘开局做对照。
- 自我对弈根节点使用 25% Dirichlet 噪声，总浓度 10.83（每个合法点 alpha = 10.83/候选数）；噪声**只加在 full 着法上**。前 20 手按根访问次数分布采样，之后取最大访问次数。
- 每轮冻结模型，默认 16 个 Windows spawn CPU 对局进程生成 32 局；主进程将请求合并成 GPU 批次，批次等待最多约 3ms。RTX 5070 Ti 实测在 4 worker 时 CPU 与 GPU 都未充分利用，因此提高并发；可用 `--workers` 按机器负载调整。`hybrid-128-10` 单点前向约 2.2 ms（旧 64×6 约 1.0 ms），批次窗口相对变紧，改 `--workers` 或 `simulations` 之前先看每轮的 `average_inference_batch_size`。
- 采样结束再训练 200 步，批大小 256，最近 100,000 个局面回放；Adam 学习率 0.001，L2 系数 0.0001，梯度范数上限 5。回放池达到 `min_replay_size` 前不更新参数，因此该阈值必须**明显低于**单轮产出（含被时间预算切短的轮次），否则训练永远不会开始——每轮指标里的 `trainable` 与 `min_replay_size` 就是给这件事留的观测口。
- 旧式随机启动仍按终局胜负训练；混合配置从 Rapfi 教师预训练权重开始，但正式运行时不调用教师。自博弈始终由已晋级 champion 生成，每 5 轮候选网络先过战术门槛，再以交换执色开局对 champion 做最多 100 组顺序检验；Wilson 95% 下界超过 50% 才晋级，否则恢复 champion 网络及优化器快照。
- 损失由 `vk/objective.py` 统一拼装：`policy_weight=1.0` 的硬目标交叉熵 + 0.25 的软目标项，加三个 value head 的**掩码均方误差**（`value_weight_final=1.0`、`value_weight_mid=0.5`、`value_weight_short=0.25`）。某个 head 在这批数据里没有有效行时它**不贡献梯度**，而不是被拉向 0。训练与预训练调用同一个函数，权重只写在 `vk/config.py` 一处；优化器执行 L2 正则，旋转/镜像同时作用于棋盘和策略，执黑平面不变。
- 每轮 `metrics.jsonl` 都带诊断：各 head 的**目标饱和**（`mean|v|`、`|v|>0.9`、`|v|<0.5`）与**预测饱和**、策略目标熵与网络熵、`kl_target_prior`（搜索比先验多出的信息）与 `kl_target_network`（目标相对当前网络有多陈旧）、`q_spread_mean`（兄弟 Q 扩散）、`policy_valid_share`/`full_search_share`（PCR 生效情况）、`horizon_bootstrap_share`（地平线是否真的截断到游戏内部）、surprise 与采样权重分布。字段清单由 `vk/diagnostics.py` 决定。

超参数是可运行起点，不代表已完成调优。无禁手普通开局的先手优势极大，空盘自我对弈会退化成一色通吃，因此训练从平衡开局开始；即便如此也仍要分别报告黑白成绩，不能只看自我对弈胜率或训练损失判断棋力。

## 时长、停止与恢复

`-Hours` 是本次运行的采样、优化及自动评估总预算，保存文件和关闭进程可能多用数秒。首次 Ctrl+C 发出安全停止请求，完整对局保留，未完成对局丢弃，保存后退出。再次 Ctrl+C 为强制中断，可能只保留上一检查点。

如果时长到达时不足 32 局，已完成的局面仍会保存。如果当前轮有未完成的参数更新，检查点记录 `pending_steps`；下次续训先完成这些更新，再开始下一轮采样。因此短时分段训练不会一直只采样而没有更新。若连一局都没完成，不写入伪造和棋标签。

每轮原子保存一个完整检查点（format 2），保留最近三个。检查点包含网络、优化器、配置、**结构化回放数组**、总对局数、轮数、训练步数、待完成更新及随机数状态。续训允许通过 `--workers` 调整自对弈并发数，其他配置仍需与检查点一致；`vk/config.RESUME_FREE` 是允许变化的全部字段。进程调度和 GPU 运算会影响数值及样本到达顺序，因此恢复保证训练状态连续，不承诺逐位可复现。

**换架构必须从零重训。** 检查点里记录 `arch`，`--init-checkpoint` 与续训都会比对它；改造前写下的检查点没有这个键，会按 `channels`/`blocks` 反推成 `legacy-64-6`，因此旧的 64×6 权重不会被误当成新网络的初始化，而是明确报 `architecture mismatch`。要把旧运行当对照，用 `--checkpoint`（推理）读取即可——`Network` 会按该检查点自己的 `arch` 重建对应网络族。

**format-1 旧检查点仍可推理，也可以做冷启动初始化**：`vk/storage.upgrade_format1()` 把旧的单个 value head 原样当作 final head，并用它的权重初始化 `mid`/`short`（网络一开始在三个地平线上给出同一个值，而不是从噪声开始）；旧的回放池是没有 schema 的元组列表，升级时直接丢弃。因此 format-1 检查点**不能续训**（续训要求 format 2，会明确报错），但 `--init-checkpoint`、`evaluate --checkpoint` 与 Web UI 都能继续读它。`vk/storage.load_model_state()` 是唯一读旧检查点的地方。

只加载本项目生成且可信的 `.pt` 文件：完整恢复使用 Python pickle。`best.pt` 是推理权重，不含优化器和回放池，不能用于续训；续训使用 `latest` 或完整 `checkpoint-*.pt`。已有训练目录要求显式 `-Resume`，不会覆盖为新训练。

## 文件与评估

```text
runs/freestyle/              无禁手模型、回放和日志
runs/renju/                  连珠模型、回放和日志
  config.json               当前配置
  checkpoint-*.pt           最近三个完整检查点
  best.pt                   当前已晋级 champion 的推理权重
  metrics.jsonl             时间、吞吐、损失、搜索/目标/预测诊断、分色胜率、开局统计和晋级结果
  games.jsonl               已完成对局的落点与结果
  evaluations.jsonl         训练中的定期评估
  evaluation-*.json         手动评估报告
artifacts/                  基准和短跑测试，不是正式训练产物
```

默认随机启动配置每 10 轮进行一次 champion 晋级检验；混合配置每 5 轮检验。双方在同一平衡开局上交换执色，报告总胜负、分色结果、配对 Hoeffding 区间和得分 Wilson 区间。达到时间预算时不晋级，也不把半局计为和棋。

候选还必须通过战术门槛，并使交换执色开局的配对得分 Wilson 95% 下界超过 50%，才替换 `best`；未通过会恢复 champion 的网络与优化器。随机和战术基线只用于手动评估，不给训练产生奖励或标签。

## 测试与开发

```powershell
python -m pytest -q
python main.py doctor

# 小模型/低搜索预算的短跑，只用于功能验收
python main.py train --rule freestyle --config configs/smoke.json --output artifacts/my-smoke-freestyle --hours 0.04 --max-rounds 1
python main.py train --rule renju --config configs/smoke.json --output artifacts/my-smoke-renju --hours 0.04 --max-rounds 1

# 同一 checkpoint 自己对自己：一侧搜索、一侧纯策略（搜索到底有没有用的判据）
python main.py evaluate --rule freestyle --checkpoint best --output runs/freestyle-pretrain --opponent self-policy --pairs 25 --minutes 120
```

测试覆盖四方向胜负、长连、真假三三、同方向/交叉双四、边界与满盘、合法动作、搜索回传符号、立即取胜和必要防守、增强对齐、完整对局标签、实际参数更新、断点恢复、停止、规则隔离和 worker 异常传播；重构后又补了 schema/目标构造/回放采样/损失掩码/配置校验/诊断/PCR/软偏置/search-gap 八组纯函数测试。

实现分层（一个模块一件事）：

| 模块 | 职责 |
|---|---|
| `vk/game.py` | 规则与终局裁决 |
| `vk/candidates.py` | 确定性硬约束 + 经验性软偏置 |
| `vk/targets.py` | 纯 numpy 目标构造：剪枝 / 去噪 / 软目标 / 多尺度 value / surprise 权重 |
| `vk/search.py` | PUCT、子树复用、PCR 预算、SearchResult |
| `vk/records.py` | position schema 唯一真源 + D4 增广 + 旧格式归一化 |
| `vk/selfplay.py` | 多进程采样与集中批量推理 |
| `vk/replay.py` | 结构化回放 + surprise 加权采样 |
| `vk/network.py` | 网络结构与批量推理（`Evaluator`） |
| `vk/objective.py` | 损失与权重（训练/预训练共用） |
| `vk/training.py` | 轮循环：采样→更新→晋级→保存 |
| `vk/storage.py` | 检查点与 jsonl；format-1 升级 |
| `vk/config.py` | 配置真源、校验、resume 兼容 |
| `vk/diagnostics.py` | 饱和 / 熵 / KL / Q 扩散 / 回合报告 |
| `vk/datasets.py` | NPZ 分片读写与合并 |
| `vk/pretraining.py` | Rapfi 冷启动监督训练 |
| `vk/teacher.py` | 教师对局生成（Rapfi 进程协议） |
| `vk/evaluation.py` | 对战、配对区间、战术门槛、search-gap |
| `vk/cli.py` / `vk/webui.py` | 命令入口 / 本地 UI |

方法参考：[AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/)、[DeepMind OpenSpiel AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md)、[KataGo](https://github.com/lightvector/KataGo)（PCR、surprise 加权、prune 后的软策略目标）、[PyTorch CUDA 安装](https://pytorch.org/get-started/previous-versions/)。本项目是针对五子棋的简化 AlphaZero 系统，只借 KataGo 的思路（多尺度 value、PCR、soft policy、surprise 采样），不复制它的复杂度：没有 ownership/score head、没有 PDA、没有盘面历史平面。

## 开源协议

本项目以 [MIT 协议](LICENSE) 发布，可自由使用、修改和再分发，请保留版权与许可声明。

欢迎提交 issue 和 pull request。需要 GPU 才能完整运行训练，因此请先看 [验收记录](ACCEPTANCE.md) 中的吞吐数据，再评估在自己机器上的时间成本；只跑单元测试不需要 GPU。

英文说明见 [README.en.md](README.en.md)。
