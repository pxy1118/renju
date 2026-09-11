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

教师默认运行 4 个进程，每进程 4 线程、256 MB Hash、200,000 节点，读取最后一个完整的 Top-5 深度；单次 5 秒超时，进程最多重启两次。YXBOARD 按当前行棋方编码（`1=当前方`、`2=对手`）。残缺输出、重复/非法落点和持续失败会显式报错，不产生替代标签。数据按整局分到 80/10/10，再仅对训练批次做 D4 增强；v2 NPZ 不允许 pickle，每片最多 4096 条，`manifest.json` 记录生成参数及可执行文件、配置和权重哈希。`data/teacher/` 已被 Git 忽略。

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

默认不开分享；加上 `--share` 与 `--host <局域网IP>` 后，服务会打印一条**邀请链接**，朋友在同一局域网打开即可入局。临时分享可以直接双击 `share-webui.bat`（等价于 `start-webui.ps1 -Share`，自动选默认路由的局域网地址），或手写命令：

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
```

分享的语义与边界：

- **每人一张独立棋桌**。打开邀请链接的浏览器会拿到一个会话 Cookie，之后自己的棋局互不干扰；页面右侧出现“邀请棋友”面板，显示当前桌数与可复制的链接。
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
正在为公网访问启动 Cloudflare 隧道（cloudflared 需能连上外网）……

公网访问口令: <随机口令>
公网邀请链接: https://<随机域名>.trycloudflare.com/?k=<随机串>
```

把**链接和口令一起**发给对方即可。要点：

- **口令是自动生成的**。公网等于把棋桌交给陌生人，所以 `--public` 不会开一个无口令的分享；也可以用 `--password <口令>` 指定自己的。口令只挡入口（一次输入换 Cookie），邀请串本身仍是一次性门票，两者拿到任一个都能进，所以两个都要给对人。
- **同一份服务，两种访客**。隧道访客拿到 `https://<隧道域名>/?k=…`（页面按请求的 `Host` 和 `X-Forwarded-Proto` 生成），局域网访客拿到的仍是启动时打印的 `http://192.168.1.3:8765/?k=…`。不需要手写 `--trusted-host`：`--public` 只放行这次隧道真正拿到的域名。
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
- **隧道访客拿到的邀请链接会自动换成公网地址**：页面“邀请棋友”面板给出的是 `https://<当前访问域名>/?k=…`（依据请求的 `Host` 与 `X-Forwarded-Proto` 生成），局域网访客看到的仍然是启动时打印的局域网链接。
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
- 策略与价值共用整个主干；策略输出 225 个 logits（MCTS 侧再取 softmax），价值输出 `tanh` 到 `[-1, 1]`。价值头在主干上做全局平均池化后接 `128→64→1`：原实现把 `1×15×15` 摊平成 225 维再送进线性层，批量大小为 1 时会塌成一维，而 MCTS 正是逐个叶子做单点推理。
- PUCT 系数 2，每步 200 次新搜索；复用子树和已有访问次数。每个节点依次保留全部一步胜、全部强制成四（四子且两端皆空，对手一子堵不住两个成五点）或全部一步防，否则使用 `square3_line4` 邻域候选；空盘只走中心，候选为空才退回全部合法点。三种收窄分别记为 `forced_win`、`strategic`、`forced_defense`。
- 每局不空盘开始，而是由 `balanced_opening` 采样 8 手平衡开局（前 4 手全盘随机，其后只落在已有棋子的邻域内），自我对弈、champion 晋级赛与 Rapfi 评测共用同一采样器与同一手数，因此评价与训练面对同一分布。空手数（`opening_plies: 0`）可退回空盘开局做对照。
- 自我对弈根节点使用 25% Dirichlet 噪声，alpha=0.3；前 20 手按根访问次数分布采样，之后取最大访问次数。
- 每轮冻结模型，默认 16 个 Windows spawn CPU 对局进程生成 32 局；主进程将请求合并成 GPU 批次，批次等待最多约 3ms。RTX 5070 Ti 实测在 4 worker 时 CPU 与 GPU 都未充分利用，因此提高并发；可用 `--workers` 按机器负载调整。`hybrid-128-10` 单点前向约 2.2 ms（旧 64×6 约 1.0 ms），批次窗口相对变紧，改 `--workers` 或 `simulations` 之前先看每轮的 `average_inference_batch_size`。
- 采样结束再训练 200 步，批大小 256，最近 100,000 个局面回放；Adam 学习率 0.001，L2 系数 0.0001，梯度范数上限 5。回放池达到 `min_replay_size` 前不更新参数，因此该阈值必须**明显低于**单轮产出（含被时间预算切短的轮次），否则训练永远不会开始——每轮指标里的 `trainable` 与 `min_replay_size` 就是给这件事留的观测口。
- 旧式随机启动仍按终局胜负训练；混合配置从 Rapfi 教师预训练权重开始，但正式运行时不调用教师。自博弈始终由已晋级 champion 生成，每 5 轮候选网络先过战术门槛，再以交换执色开局对 champion 做最多 100 组顺序检验；Wilson 95% 下界超过 50% 才晋级，否则恢复 champion 网络及优化器快照。
- 优化策略交叉熵加价值均方误差，优化器执行 L2 正则；旋转/镜像同时作用于棋盘和策略，执黑平面不变。

超参数是可运行起点，不代表已完成调优。无禁手普通开局的先手优势极大，空盘自我对弈会退化成一色通吃，因此训练从平衡开局开始；即便如此也仍要分别报告黑白成绩，不能只看自我对弈胜率或训练损失判断棋力。

## 时长、停止与恢复

`-Hours` 是本次运行的采样、优化及自动评估总预算，保存文件和关闭进程可能多用数秒。首次 Ctrl+C 发出安全停止请求，完整对局保留，未完成对局丢弃，保存后退出。再次 Ctrl+C 为强制中断，可能只保留上一检查点。

如果时长到达时不足 32 局，已完成的局面仍会保存。如果当前轮有未完成的参数更新，检查点记录 `pending_steps`；下次续训先完成这些更新，再开始下一轮采样。因此短时分段训练不会一直只采样而没有更新。若连一局都没完成，不写入伪造和棋标签。

每轮原子保存一个完整检查点，保留最近三个。检查点包含网络、优化器、配置、回放池、总对局数、轮数、训练步数、待完成更新及随机数状态。续训允许通过 `--workers` 调整自对弈并发数，其他配置仍需与检查点一致。进程调度和 GPU 运算会影响数值及样本到达顺序，因此恢复保证训练状态连续，不承诺逐位可复现。

**换架构必须从零重训。** 检查点里记录 `arch`，`--init-checkpoint` 与续训都会比对它；改造前写下的检查点没有这个键，会按 `channels`/`blocks` 反推成 `legacy-64-6`，因此旧的 64×6 权重不会被误当成新网络的初始化，而是明确报 `architecture mismatch`。要把旧运行当对照，用 `--checkpoint`（推理）读取即可——`Network` 会按该检查点自己的 `arch` 重建对应网络族。

只加载本项目生成且可信的 `.pt` 文件：完整恢复使用 Python pickle。`best.pt` 是推理权重，不含优化器和回放池，不能用于续训；续训使用 `latest` 或完整 `checkpoint-*.pt`。已有训练目录要求显式 `-Resume`，不会覆盖为新训练。

## 文件与评估

```text
runs/freestyle/              无禁手模型、回放和日志
runs/renju/                  连珠模型、回放和日志
  config.json               当前配置
  checkpoint-*.pt           最近三个完整检查点
  best.pt                   当前已晋级 champion 的推理权重
  metrics.jsonl             时间、吞吐、损失、候选/搜索统计、分色胜率、开局统计和晋级结果
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
```

测试覆盖四方向胜负、长连、真假三三、同方向/交叉双四、边界与满盘、合法动作、搜索回传符号、立即取胜和必要防守、增强对齐、完整对局标签、实际参数更新、断点恢复、停止、规则隔离和 worker 异常传播。

实现分层：`main.py` 是统一命令入口；`vk/game.py` 为规则；`vk/search.py` 为 MCTS；`vk/selfplay.py` 为多进程采样；`vk/network.py` 为网络和推理；`vk/training.py` 为优化与恢复；`vk/evaluation.py` 为对战评估；`vk/cli.py` 负责命令解析。

方法参考：[AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/)、[DeepMind OpenSpiel AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md)、[PyTorch CUDA 安装](https://pytorch.org/get-started/previous-versions/)。本项目是针对五子棋的简化 AlphaZero 系统，不是原论文计算规模或完整实现的复刻。

## 开源协议

本项目以 [MIT 协议](LICENSE) 发布，可自由使用、修改和再分发，请保留版权与许可声明。

欢迎提交 issue 和 pull request。需要 GPU 才能完整运行训练，因此请先看 [验收记录](ACCEPTANCE.md) 中的吞吐数据，再评估在自己机器上的时间成本；只跑单元测试不需要 GPU。

英文说明见 [README.en.md](README.en.md)。
