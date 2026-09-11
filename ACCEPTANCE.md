# 实施与验收记录

## Rapfi 教师蒸馏与战术 MCTS 改造（2026-09-10）

已实现 freestyle 第一阶段的共享战术候选、外部 Rapfi MultiPV 适配器、安全 NPZ 教师数据、监督预训练、仅权重初始化、champion 自博弈/回滚和 Rapfi 对手评测。候选规则在 MCTS 每个节点执行，因此训练、CLI 和 Web UI 一致；日志新增候选数、强制胜防、搜索深度、相对先验 KL、价值绝对均值和晋级结果。

自动测试当前为 **69 passed, 1 skipped**；跳过项是需要设置 `RAPFI_ENGINE` 与 `RAPFI_ENGINE_DIR` 的真实外部引擎用例。新增测试覆盖四方向/间断四/唯一防守/多威胁/八对称等变，以及模拟 Rapfi 的多深度、坐标、超时、崩溃重启、残缺输出和非法着拒绝。另用用户工作区之外临时构建的 Rapfi 实际运行该集成用例（**1 passed**），完成 200,000 节点 Top-5 查询，并生成 2 条真实 NPZ 教师样本；适配器正确选择最后一个完整的 MultiPV 深度。临时二进制、权重和样本均未进入仓库。

尚未执行 50,000 局面生成、20,000 步预训练、正式两小时混合自博弈或最终 A/B，因此棋力阈值与防塌缩结论均为**待实跑验收**，不能由接口测试替代。Renju 教师管线也仍按计划留待第二阶段。

正式首次运行随后暴露并被准入门槛拦截了一个协议语义错误：YXBOARD 的棋子编号应为 `1=当前引擎方、2=对手`，而不是固定黑白。错误数据的相邻局面价值更接近同号而非换号，价值测试 MAE 为 0.2563；策略 Top-1/Top-5 虽达到 50.75%/83.66%，但 `accepted=false`，没有进入自博弈。适配器现已修正并将教师格式升为 v2，预训练显式拒绝旧 v1 数据；旧 50,000 条数据与对应权重只保留为失败证据，不能继续使用。

Rapfi 随后以 `external/rapfi` Git 子模块加入，固定提交 `3c94c2a976f24a0dd1c5517623e9ab6fffe66bd7`，并递归初始化 Networks、Gomocalc 和 Trainer。`scripts/build-rapfi.ps1` 已在本机用 Visual Studio 2022/MSVC 完成 Release AVX2 构建，运行时落在被忽略的 `external/rapfi-runtime`；构建产生若干上游源码代码页及 CRT 链接警告，但退出码为 0，生成的 `pbrain-rapfi.exe` 为 2,073,088 字节。用该产物运行真实集成测试后结果为 **70 passed**。Rapfi 与 Networks 的许可边界记录在 `THIRD_PARTY.md`。

日期：2026-09-10。设备：RTX 5070 Ti 16GB，Windows，项目 Python 3.12.12。最初验收误将用户原有 PyTorch 2.14.0+cu130 环境降为 2.10.0+cu128；后续已恢复原环境，当前验证结果见“环境恢复复验”。

## 实现完成

- 无禁手/普通开局禁手双规则、独立权重及回放池。
- 策略价值残差网络、PUCT、子树复用、自我对弈、八种数据增强和优化。
- Windows CPU 多进程对局与单 GPU 批量推理。
- 时间预算、安全停止、原子保存、最近三个检查点、待完成更新恢复及跨规则校验。
- 固定种子配对评估、历史模型晋级、随机/战术基线、命令行人机对弈。
- `python main.py` 统一提供设备检查、基准、训练、评估和对弈命令；包含锁定依赖和中文说明。

## 自动化验证

最终 `python -m pytest -q`：**46 passed**。包含两种规则实际完整对局、CPU 参数更新与续训；并验证中断后优先完成待更新步骤、无伪造和棋、worker 异常传播、GPU 进程锁与释放，以及续训时安全调整 worker 数量。

原 PowerShell 包装入口验收后已按用户要求移除，项目现统一使用 `python main.py`。此前以 7.2 秒预算执行训练入口，8.41 秒完成保存退出，关闭进程开销约 1.2 秒；共保留 4 局、4 步更新，最后一轮 0 局没有生成和棋样本。`uv lock --check` 通过。

停止路径验证包括真实到时停止和自动化停止回调；没有手工模拟终端 Ctrl+C 按键。

## 真实 GPU 验收

| 项目 | 无禁手 | 连珠 |
|---|---:|---:|
| 小配置首次训练 | 2 局、2 步更新 | 2 局、2 步更新 |
| 首次采样与更新耗时 | 4.59 秒 | 16.31 秒 |
| 续训后累计 | 4 局、4 步更新 | 4 局、4 步更新 |
| 独立评估 | 三类对手各完成 2 局 | 三类对手各完成 2 局 |

小配置为 `configs/smoke.json`，不是默认规模。评估对手包括历史模型、随机和战术基线，输出了分色结果和置信区间；每类只有一个开局组，**仅验证评估流程，不证明棋力**。

默认网络（6 块 × 64 通道）完成 CUDA 前向/反向；使用连珠短跑真实回放数据，批大小 256 的一次更新耗时约 0.50 秒，峰值 PyTorch 已分配显存约 0.54 GiB。这不包含 CUDA 上下文、缓存和其他进程显存，也不是完整训练的显存上限。

人机对弈入口实际加载连珠模型，AI 黑方首子落在 8 行 8 列，随后接收 `q` 正常退出。

## 吞吐与训练量估算

所有基准均为未训练权重、200 次搜索、6 块 × 64 通道、4 个对局进程；只统计完整结束的对局。

| 基准 | 实测 | 可得结论 |
|---|---|---|
| 无禁手，60.50 秒 | 7 局、399 个完整局面、约 1,897 个网络局面/秒 | 此次短测折算约 417 局/小时，32 局采样约 4.61 分钟 |
| 连珠优化前，120.39 秒 | 1 局、约 228 个网络局面/秒 | 旧实现短测，32 局粗算约 64 分钟；不代表最终实现速度 |
| 连珠优化后，60.36 秒 | 38,838 个网络局面，约 643 个/秒；0 局结束 | 搜索运行正常，但对局均未结束并已丢弃；不能据此给出最终局/小时 |

上述不同试跑的随机权重和对局进程调度并不完全相同，不能将这两次连珠数据当成严格的加速比实验。最终基准入口已固定初始化种子，便于今后同配置复测。测试期间 CPU 单元测试也曾运行，吞吐仅供本机启动预算参考。

按无禁手本次短测外推，两小时的**纯采样**约 830 局；真实训练还需扣除网络更新、检查点保存与评估时间，棋力和局长变化也会改变吞吐。连珠最终实现没有足够完整局数，暂不提供可靠整局训练量估算；启动前可手动运行 `python main.py benchmark --rule renju --minutes 3`，若仍没有完整局数，再延长测量。

## 尚未正式训练

**没有启动正式长训，没有创建 `runs/` 下的正式模型。** 验收累计计算时间每种规则均未超过五分钟，所有短跑数据在 `artifacts/` 中。成熟棋力、长期稳定性、正式比赛开局协议和大规模对战胜率不在本次验收结论内。

相关证据：

- `artifacts/acceptance-freestyle/metrics.jsonl` 与对应检查点。
- `artifacts/acceptance-renju/metrics.jsonl` 与对应检查点。
- 两个验收目录中的 `evaluation-*.json`。
- `artifacts/wrapper-timeout-freestyle/metrics.jsonl`。
- `artifacts/benchmark-freestyle/report.json`、`artifacts/benchmark-renju/report.json` 和 `artifacts/benchmark-renju-optimized/report.json`。

正式启动、续训和切换规则的命令见 [README](README.md)。

## 环境恢复复验

项目依赖、锁文件和 `.venv` 已恢复为用户原有组合：PyTorch 2.14.0+cu130、torchvision 0.29.0+cu130、torchaudio 2.11.0+cu130、NumPy 2.5.3、Pillow 12.3.0。五个包均已实际导入；CUDA 13.0 可用，RTX 5070 Ti 上的网络前向/反向通过；恢复后测试通过，`uv lock --check` 通过。上文标注为 2.10.0+cu128 的性能数据只代表误改期间的历史短跑，不能当作当前环境的性能结果。

## 并发加速调整

用户运行中的 4-worker 无禁手训练实测整机 CPU 平均约 12%，Windows 任务管理器显示 GPU 约 31%、显存约 1.7/16 GB。默认自对弈并发已提高到 16 worker，使神经网络推理批次可从最多 4 个局面提高到最多 16 个局面；续训允许只覆盖 worker 数量，不允许借此修改网络或训练语义参数。指标新增平均和最大推理批次大小，用于后续实测调优。

当前已经启动的 Python 进程仍使用启动时读入的 4-worker 配置；需要安全停止并从最新检查点以 `--workers 16` 续训后才会生效。由于该训练仍在占用项目 GPU 锁，本次没有并行启动 16-worker GPU 基准，避免干扰正式训练；加速倍数仍需用下一轮日志实测。

## Web UI 打磨与浏览器实测

日期：2026-09-10（追加）。自动训练仍在运行时完成，未中断训练、未写 `runs/`。

改动三项：

- 客户端轮询改为自适应：模型思考时 250 毫秒、空闲时 1 秒，并为请求加 15 秒超时。此前固定 1 秒轮询，把 0.2–0.7 秒的思考取整成约 1 秒，也让“模型思考中”几乎看不到。
- 无检查点时给出可执行的说明：按钮标签变为“暂无可用模型”，点击后在页面上解释该规则需要先由训练保存检查点，而不是静默无效。
- 模型下拉改为“最新 / 历史最佳 / 最近若干检查点”，列出文件名、写入时间和大小；服务端 `/api/config` 新增 `models`（别名解析为具体文件名）与 `recent`。选中的文件被训练轮换删除时自动退回“最新”，`best.pt` 保持固定别名。

接口收紧：检查点名称含路径分隔符或 `..` 时返回 400；名称只按规则目录内解析，加载前再次校验父目录，越界或文件缺失通过页面错误通道提示（新增用例 `test_missing_file_reports_through_error_channel`）。自动化测试总数由 46 增至 **53 passed**。

浏览器实测（headless Chrome + CDP，脚本与日志在 `artifacts/verification/`）：26/26 通过，包括 225 个交叉点、棋盘正方形无溢出、栅格线绘制、真实点击开局、选中 `checkpoint-00000006-0000001200.pt` 后确认加载的正是该文件（训练 1200 步）、一整个人机往返 654 毫秒可见、思考状态可观测、连珠缺检查点时点击给出原因。同一轮实测发现并修复了一个真实缺陷：按 `latest`/`best` 判断可选性会让手动选中的历史检查点被判为“暂无可用模型”，现已按服务端列出的文件判断。

未验证：`serve()` 的自动打开浏览器与 Ctrl+C 收尾，以及移动端真实触摸设备上的布局，只做了 CSS 静态审查。

## 终局显示修复

日期：2026-09-10（追加）。用户报告“执黑获胜后显示白棋赢了”。逐层复现结果：服务端判定一直正确（`Game.move` 按落子方记录胜者，快照 `winner` 与 `human` 同号，两种规则、双方执色都验证过），问题出在客户端显示时机与可核验性上，已修复两处：

- 人走下制胜一手后，`act()` 仍把 `busy` 置真并向工作线程排一个不会落子的 `reply`，因此已结束的对局要等该任务跑完才显示结果。机器负载高时这个窗口约 1 秒，期间页面显示“模型思考中…”，看不到胜负。现在终局判定在 `act()` 内完成：对局已结束或又轮到人时直接返回 `busy=false`，不再排空任务。
- 页面此前不标出是哪五子连成一线。现在终局会高亮获胜五子（`.piece.win`，金色描边），黑胜显示黑方连线、白胜显示白方连线，结果可在棋盘上直接核对。

浏览器实测：制胜一手点击后 **25 毫秒**即显示“你赢了，这一局下得漂亮。”并高亮 5 子，中英文案与 `winner`/`human` 一致；此前同一测试需等约 1 秒且期间显示“模型思考中…”。回归用例 `test_human_win_is_reported_immediately` 断言制胜返回的 `busy` 为假、无合法落点、胜方等于人的执色；测试总数 54 passed。

同日用户截图复现出第二处：状态行左侧的落子方指示圆点在终局后仍是白点。原因是 `render()` 用 `state.player` 画该圆点，而人获胜后服务端已把 `player` 翻成对手（快照字段本身正确），于是“你赢了”旁边挂着白子，读起来像白方获胜。现在该圆点在终局改为显示**胜方**棋子（黑胜黑点、白胜白点）并加金色外圈、悬停提示“黑方获胜/白方获胜”，和棋时隐藏；同时终局不再显示“上手 Xs”。服务端相应不变量（`player == -winner`）也写进回归用例，避免客户端再被这两个字段误导。浏览器复验：`turn-dot` 类名为 `stone-icon black result`、提示“黑方获胜”、与人的执色一致。

## 启动方式改回前台

日期：2026-09-10（追加）。用户指出正确做法应是“双击启动、Ctrl+C 停止”，而不是后台常驻再配一个停止脚本。原实现用 `pythonw.exe` + `Start-Process` 把服务脱离终端运行，因此必须另写 `stop-webui.ps1` 才能关掉；现改为前台运行：

- `start-webui.bat` → `start-webui.ps1` 用 `& .venv\Scripts\python.exe main.py webui` 在当前控制台前台运行，浏览器仍自动打开，Ctrl+C 即关闭服务。
- 窗口即进程，日志直接打在窗口里，不再重定向到 `artifacts/webui.*.log`；`stop-webui.bat` / `stop-webui.ps1` 已删除。
- 端口已占用时不启动第二个服务，改为直接打开已有界面。

实测发现的关键点（`artifacts/verification/foreground-test.py`、`signal-test.py`）：`pythonw.exe` 的 PE 子系统是 2（GUI），**根本收不到 Ctrl+C**；控制台信号要求目标进程与发送方共享控制台且处于同一进程组，用 `CREATE_NEW_PROCESS_GROUP` 启动的子进程收不到 CTRL_C 但收得到 CTRL_BREAK。启动脚本因此不做进程组隔离，服务器与包装脚本同组同控制台。端到端复验 16/16：双击路径 2 秒内可服务、Ctrl+C 后包装脚本与服务器都退出、端口释放、无 traceback；另一次实测中 Ctrl+C 到达整个进程组时，包装脚本、服务器与调用方一并结束，8765 无残留监听、无遗留进程。`cmd /c` 调用方式下 cmd 会多问一句“终止批处理操作吗(Y/N)”，真实双击路径没有这层中间 cmd。自动化测试总数仍为 54 passed，静态检查 22/22。

## 临时分享（最多 3 桌）

日期：2026-09-11（追加）。用户需求：把界面临时分享给别人玩，并发最多 3。默认行为保持不变——不传任何新参数时仍是单桌、无凭据、只监听 127.0.0.1；分享是显式开关。新增 `--host`、`--share`、`--max-sessions`（1–16，默认 3）、`--password`、`--trusted-host`。

实现：

- `Catalog` 抽出共享的检查点扫描（2 秒 TTL 缓存），`Sessions` 管理每浏览器一张 `Table`。每张棋桌各自持有网络权重与 MCTS 树，因此上限就是内存与 CPU 的上限。
- 邀请链接 `http://<地址>:<端口>/?k=<随机串>`：`hmac.compare_digest` 校验后签发 `HttpOnly; SameSite=Strict` 的会话 Cookie 并 303 重定向到干净的 `/`，凭据不再留在地址栏。带 `--password` 时，无链接的访客看到口令表单（POST，口令走请求体，因此 `&`、`=` 等字符合法且不进浏览器历史），口令不符返回 400“口令不正确”。
- 名额是硬上限而不是队列：满员时第 4 位访客收到 429“名额已满”，不会挤掉正在下棋的人；`/api/leave`（页面上的“结束我的棋桌”）释放名额。
- Host 校验从“只认 127.0.0.1/localhost”改为“绑定地址 + 回环 + 显式 `--trusted-host`”，`0.0.0.0` 绑定时额外接受本机各地址；攻击者可控域名仍被拒（回归用例保留 `Host: evil.example` → 403）。
- 前端新增“邀请棋友”面板（链接、复制、当前桌数、结束棋桌）与徽标“分享对弈 · 最多 3 桌”；未开启分享时该面板隐藏，页面与改动前一致。

实测与踩坑：

- `--host 0.0.0.0 --share` 的邀请地址最初取主机名解析结果，本机第一个结果是 VMware 虚拟网卡（`192.168.142.1`），发给朋友的链接不可达。现在优先用 UDP connect 探测默认路由地址（本机为 `192.168.1.3`），失败才退回主机名解析。
- 本机防火墙/绑定语义：套接字绑到某个具体地址后，`127.0.0.1` 不再应答（`ECONNREFUSED`），用例因此按绑定地址而不是回环发起请求；绑定地址为 `0.0.0.0` 时则回环与局域网地址都通。
- 端口 0 的 Host 白名单必须等 `bind()` 之后才能算，否则集合里是 `127.0.0.1:0`，所有请求 403（已修，`make_server` 在拿到真实端口后填充）。
- `Sessions.open()` 满员返回 `None` 时，`admit()` 最初直接解包导致 `TypeError` 500；改为先判 `None` 再回 429（用例 `test_share_capacity_is_enforced_and_released`）。
- 口令表单从 GET 改 POST 时暴露出两处真实缺陷：`admit()` 回复后 `do_POST` 仍继续读已排空的请求体（旧客户端对格式错误的请求会拿到 400 而非预期的 403），以及口令不符时静默重渲染表单、HTTP 状态仍是 200。现在请求体在 `admit()` 中只读一次并记录 `answered` 标志，落库为 `test_http_local_protection_and_validation`（未授权 POST 仍 403）与口令用例（错误口令 400 + 提示文案）。

验证：`tests/test_webui.py` 由 9 项增至 **14 项**（新增私有棋桌隔离、满员与释放、局域网 Host 与非法 Host、口令入局、`0.0.0.0` 绑定），全量套件 **75 passed, 1 skipped**。另在真实服务上跑端到端脚本 `artifacts/verification/share-live.py`（`--host 192.168.1.3 --share`）：3 位访客各得独立棋桌（`table` 各不相同）、第 4 位收到 429、真实落子后模型 0.16 秒应手、另两桌棋盘与手数不受影响、`/api/leave` 后第 4 位可入局且离开者再访问为 403。渲染复验（headless Chrome `--dump-dom`）确认徽标显示“分享对弈 · 最多 3 桌”、面板显示 `n / 3 桌` 与可复制的邀请链接。

未验证：真实双机（另一台电脑/手机）跨设备点击，仅在本机按局域网地址访问；跨公网隧道与反向代理下的 `--trusted-host` 未实测。

## 反向代理 / 隧道接入（2026-09-11 追加）

用户接上 `cloudflared tunnel --url http://127.0.0.1:8765` 后页面报 `{"error": "Local access only"}`。定位到两个独立原因，都已修：

- **服务是旧参数**：当时 8765 上运行的实例只有 `main.py webui`，既没有 `--share` 也没有 `--trusted-host`，隧道域名当然被 Host 白名单拒绝。
- **白名单匹配方式错**：cloudflared 转发时发的是**不带端口**的裸域名（`<name>.trycloudflare.com`），而白名单里存的是 `域名:端口`，因此即使加了 `--trusted-host <完整域名>` 仍然 403。现在 `--trusted-host` 的裸名同时接受裸域名与本端口两种形式（反代自己决定端口，写 `host:port` 的则按字面匹配）；新增以点开头的写法放行整棵子域（`.trycloudflare.com`），因为 quick tunnel 每次重启都会换域名。`trusted_host_match()` 只做后缀匹配，`board.example.org`、`evil.test` 之类仍然 403。
- **面板邀请链接按访问来源生成**：隧道访客看到的是 `https://<当前域名>/?k=…`（取请求 `Host` 与 `X-Forwarded-Proto`），局域网访客仍看到启动时打印的 `http://<局域网IP>:8765/?k=…`。
- **避免隧道连错实例**：cloudflared 固定拨 `127.0.0.1:8765`，若该端口上是只绑回环且未开启分享的旧实例，访客只会看到单桌无口令界面。因此分享实例建议用 `--host 0.0.0.0`，回环与局域网共用同一个实例（实测同时从两个地址入局、各自拿到独立棋桌）。

验证：`tests/test_webui.py` 增至 **16 项**（新增 `test_trusted_host_supports_a_subdomain_pattern`、代理 Host 与按来源生成邀请链接的断言），全量 **78 passed, 1 skipped**。本机实测：`Host: whatever-quick-tunnel.trycloudflare.com` + `X-Forwarded-Proto: https` 入站 → 口令页 → 面板给出 `https://whatever-quick-tunnel.trycloudflare.com/?k=…`；`Host: evil.example` → 403。

一个**不是本项目**的问题：本次会话中新建的 3 个 quick tunnel，cloudflared 端都报 `Registered tunnel connection`，但 Cloudflare 始终没有生成对应 DNS 记录（Cloudflare DoH 权威查询返回 NXDOMAIN/Status 3，`1.1.1.1` 与 `8.8.8.8` 同样解析不到），而用户 20:29 创建的那个旧域名在隧道进程被杀后仍能解析（返回 502）。即 quick tunnel 的域名分配在 Cloudflare 侧失败，与本项目代码无关；`scripts/tunnel-webui.ps1` 只负责把新地址打出来。

## 混合训练首轮诊断后的恢复与评测加速

日期：2026-09-10（追加）。首个高学习率混合运行在第 20 轮被 champion 门槛拒绝（候选 71 胜 109 负，得分 39.44%），且恢复冷启动时出现一轮只收集 1/64 局。定位并修复：自博弈 actor 不再用 100 ms 空队列超时判断任务结束，而是阻塞读取并消费显式 sentinel；恢复完整 checkpoint 时按提交轮次原子清理 `games.jsonl`、`metrics.jsonl`、`evaluations.jsonl` 中越界或残缺记录；神经网络对神经网络的 arena 改为多进程棋局、父进程按模型分别批推理，保留成对开局与顺序 Wilson 判定；正式 freestyle 混合配置学习率从 `1e-3` 降为 `1e-4`，新运行不复用旧优化器或回放池。

新增回归覆盖 8-worker 冷启动不丢任务、JSONL 恢复边界和双 worker 批量 arena。关闭本机 HTTP 代理对局域网测试的干扰后，全量结果为 **81 passed, 1 skipped**；Rapfi 崩溃重启模拟仍有一个既有的 Python 文本流析构 warning，不影响通过结果。

## 公网一行命令（`--public`，2026-09-10 追加）

需求：`python main.py webui` 增加一个可选参数，决定是否暴露公网，是则给出可分享的链接。此前公网只能靠“一个窗口跑服务 + 另一个窗口跑 `scripts/tunnel-webui.ps1` + 手写 `--trusted-host .trycloudflare.com`”三步拼出来，且第一步必须先知道域名——而 quick tunnel 的域名每次重启都变。

新增 `vk/tunnel.py`（托管 cloudflared 子进程）与 `--public` 参数：

- **域名是发现出来的，不是配置出来的**。cloudflared 每次启动向 Cloudflare 申请一个随机域名，只能从它的日志里读。`Tunnel.start()` 同时等两件事：日志里出现 `https://<name>.trycloudflare.com`，以及 `Registered tunnel connection`（后者才是边缘真正可用的标志，前者只是“已创建”）。
- **日志是不可信输入**。第一版用 `https://[A-Za-z0-9.-]+` 抓“第一个 https 链接”，结果抓到的是 cloudflared 启动横幅里的**服务条款链接** `https://www.cloudflare.com/website-terms/`，于是公网链接变成了 `https://www.cloudflare.com`（实测复现）。现在正则锚定 `.trycloudflare.com` 后缀，再经 `hostname_ok()` 按 RFC 1123 校验，`[::1]`、`host:port`、空格、超长名一律拒绝——这些字符串会进入 Host 白名单并被打进给访客的链接，所以宁可拒绝也不能放行。
- **`--public` 自动开分享、自动绑 `0.0.0.0`、自动生成口令**。公网等于把棋桌交给陌生人，所以不提供“无口令公网分享”这条路：未指定 `--password` 时用 `secrets.token_urlsafe(9)` 生成并随链接一起打印。`--host` 写具体网卡地址会直接报错，因为 cloudflared 固定拨 `127.0.0.1`。
- **隧道连不上就整体退出**（`SystemExit`），不会留下一个“以为已经公开、其实没有”的服务；等待上限 40 秒，`--tunnel-timeout` 可调，`--cloudflared` 可指定路径。
- **Ctrl+C 连带关隧道**（`serve()` 的 `finally` 调 `Tunnel.stop()`，先 `terminate` 再兜底 `kill`），不留公网入口。

实测踩坑（三处真实缺陷，均由端到端脚本发现）：

- **`start_tunnel` 把域名加进 `fence` 会反转访客身份**。`fence` 的语义是“本机、被直接访问”，而邀请链接的生成规则是“Host 命中 fence 就发局域网链接，否则发代理链接”。域名一旦进 fence，隧道访客反而拿到 `http://192.168.1.3:8765/?k=…` 这种他根本打不开的链接。改为只写入 `server.tunnel_host`，放行交给模式匹配。
- **`--public` 下 `--trusted-host` 是空的，没有任何模式能匹配隧道域名**。`proxy_patterns()` 现在让运行中的隧道贡献自己的 `.trycloudflare.com` 子树，因此无需手写 `--trusted-host`；操作者显式给出的模式仍然照常生效（曾一度把通配符限定为“仅在隧道存活时生效”，结果打断了 README 里已记载的 `--trusted-host .trycloudflare.com` 手动用法，故回退）。
- **`proxied_authority()` 拿主机名去比带端口的 `fence`**（`fence` 里都是 `地址:端口`），导致回环访客被误判成代理访客。改为整串比较；cloudflared 转发的是裸域名，永远不会命中 fence，判定因此准确。
- 本次会话最初实测到 quick tunnel 的 DNS 失败（与上一次会话记录的 Cloudflare 侧 NXDOMAIN 一致），但本轮**多次复现均为成功**：新建隧道约 5 秒拿到域名、约 20 秒 DNS 生效，随后公网 GET 返回探针内容（HTTP 200）。可见当时的失败是暂时性的，不是每次必现。

验证：`tests/test_webui.py` 由 16 项增至 **21 项**（新增横幅解析、主机名校验、`proxy_patterns`、`start_tunnel` 不污染 fence、`serve` 退出时必关隧道），全量 **85 passed, 1 skipped**。

另有两份一次性端到端脚本（`artifacts/verification/public-tunnel.py` 直接验证 `vk.tunnel`，`artifacts/verification/public-tunnel-e2e.py` 打真实 `--public` 服务）：真实 cloudflared 在 ~5 秒内给出域名，公网 GET 取回探针内容，`Tunnel.stop()` 幂等且进程确实退出；`--public` 服务上 15 项断言全过——隧道域名入站返回口令页、错误口令 400、正确口令换到 `renju_session`、面板给出 `https://<隧道域名>/?k=…`（而非局域网链接）、隧道访客能读到自己那张棋桌并开新局、第 4 位访客收到 429、局域网访客仍拿局域网链接、`evil.example.test` 等外部域名一律 403。

未验证：**从另一台真实设备/手机**打开公网链接点击落子（本机只能验到边缘回源这一侧）；`--public` 在 Windows 之外的平台未实测。

## 并发上限 3 → 10（2026-09-10 追加）

用户反馈“并发有点少”，把默认桌数从 3 提到 10。改动集中在三处常量与其派生文案：

- `vk/webui.py` 的 `LAN_SESSION_LIMIT` 3 → **10**（`Sessions`、`make_server`、`serve` 的默认值都由它派生）。
- `vk/cli.py` 的 `--max-sessions` 默认值 3 → **10**，上限仍为 16：那是防手滑写出天文数字的护栏，不是对机器能力的断言。（帮助文本里的单桌成本先写成“每桌约 200 MB”，实测后已改为“tens of MB、先撑不住的是 CPU”，见下文。）
- `vk/web/index.html` 的占位文案 `0 / 3 桌` → `0 / 0 桌`。页面上真正的桌数与上限都由 `/api/config` 的 `max_sessions` / `sessions` 填充（`app.js`），占位值只在首帧可见，写死 3 会在下一次调整时再次变成谎言。

**代价要说清楚（这里原先写错了）**：我最初按“每桌约 200 MB、10 桌约 2 GB”改的文案，那是拿“一个完整的 Web UI 进程占用（约 540 MB 工作集）”当成了单桌成本，属于凭空估算。实际用 `artifacts/verification/table-memory.py` 逐桌加压实测（200 次搜索、每桌真下 12 手）：

| 同时棋桌 | 私有内存增量 | 工作集增量 |
|---:|---:|---:|
| 1 | +27.5 MB | +30.6 MB |
| 5 | +86.0 MB | +85.8 MB |
| 9 | +148.6 MB | +144.7 MB |

即**每桌约 16–30 MB**（边际约 16.5 MB）：网络权重只有 0.56 M 参数≈2 MB，占大头的是每桌各自的 MCTS 树。所以 10 桌满员也就几百 MB，**内存根本不是瓶颈，CPU 才是**——每张棋桌独立搜索，同时思考的人越多每人越慢。README 与 `--max-sessions` 的帮助文本都已按实测值改正，并把“别调大”改成“吃力就调小”。

顺带修掉一个隐患：`vk/web/index.html` 里写死的占位文案 `0 / 3 桌` 改为 `0 / 0 桌`。页面上的桌数与上限本来由 `/api/config` 的 `max_sessions` / `sessions` 填充，只有首帧会露出占位值；把它写成具体数字，下一次调整上限时就会再次变成谎话。

**测量过程中的一个教训**：第一次测（空棋桌，不建对局）得到的内存增量是**完全一致的 +0.0 MB**——因为 `Table` 只在开局时才创建网络与搜索树，空表几乎不占内存。若不建对局就下结论，会得出“并发数随便调”的错误答案；这个 −0.0 也提醒我该去核对测量对象，而不是直接采信。

验证：`tests/test_webui.py` 由 21 项增至 **22 项**。`test_share_capacity_is_enforced_and_released` 原先依赖共享 fixture 的 `max_sessions=3`，现在自带一个显式 3 桌的小服务，只验证“拒绝而非驱逐、离开即释放”这一机制，不再让用例数量与默认值绑死；新增 `test_the_default_share_capacity_is_ten` 用 `Sessions` 直接顶到边界（第 10 张必须放行、第 11 张必须被拒，且 10 张互不相同），再断言 `make_server` 报给页面的上限也是 10——避免“页面写着 10 桌、服务只放 3 个”这类前后端不一致。全量 **87 passed, 1 skipped**。真实服务复验：启动日志打印“最多 10 桌”，`/api/config` 返回 `max_sessions: 10`，连开 10 位访客各得独立棋桌、第 11 位收到 429。

## 项目更名：AlphaZero / az → Visk / vk（2026-09-10 追加）

用户认为“AlphaZero + az”不好听，要求换名。选定 **Visk**，包名 **vk**。

命名理由：原名字直接借用了 DeepMind 的项目名，既缺乏辨识度，也与实现不符（README 本身就写明“是简化系统，不是原论文的复刻”）；本项目真正的身份是五子棋/连珠引擎。`Visk` 由 vi(五) + victory 缩合而成，单音节、易记，且 PyPI 上未被占用。两个候选名被否决：`Tesuji` 与 `Sente` 在 PyPI 上已被 Go 语言库占用，`Pente` 是 Parker Brothers 的注册商标棋类游戏，用它命名五子棋引擎会造成混淆。

改动范围（一次性更名，共 56 处模块引用 + 全部对外文案）：

- 目录 `az/` → `vk/`（用 `git mv`，历史按重命名记录而非删除+新增）。
- 导入与 monkeypatch 目标 `az.*` → `vk.*`；`main.py` 入口不变。
- 对外文案：CLI 描述、`Visk Web UI:` 启动行、HTTP `Server` 头（`RenjuWebUI` → `ViskWebUI`）、页面标题与页脚（`RENJU LAB` / `ALPHAZERO` → `VISK`）。
- README 标题改为「Visk · 双规则五子棋」/「Visk — Dual-Rule Gomoku / Renju」；`pyproject.toml` 描述同步。

**刻意保留**：README 末尾对 AlphaGo Zero 与 OpenSpiel AlphaZero 的**方法引用**属于学术出处，不是自身品牌，保持原样；`pyproject.toml` 描述里保留 “AlphaZero” 是为了说明所用算法，而不是项目名。

更名是纯机械替换，但有两处容易踩的坑已避开：`az` 作为子串出现在 `lazy`、`.pytest_cache`、`artifacts` 等词中，因此正则限定为 `\baz` 且后接 `.` 或 `/`；替换前先清掉所有 `__pycache__`，避免旧字节码让 `import az` 仍能命中。

验证：全量 **87 passed, 1 skipped**；新包导入自检通过、`import az` 按预期报 `ModuleNotFoundError`；真实服务复验（`artifacts/verification/rename-smoke.py`）确认页面标题/页脚/Server 头均为 Visk，且随包迁移的静态资源 `/app.js`、`/style.css` 仍返回 200（包目录改名后最容易断的一环）。
