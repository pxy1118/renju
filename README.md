# 双规则五子棋 AlphaZero

简体中文 | [English](README.en.md)

从随机权重开始，以 **MCTS 自我对弈 → 回放池 → 策略价值网络更新** 学习下棋。两种规则共用实现，权重和数据分开保存。项目提供训练系统，尚未包含训练成熟的棋力模型。

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

支持规则与执色选择、最新/历史最佳/最近若干检查点的手动选择、32/64/200 次 MCTS 搜索、手数显示、撤回最近一轮、重新开始及终局提示；终局会高亮获胜的五子，状态行左侧的圆点改为胜方棋子（和棋隐藏），因此“谁赢了”由文案、棋子和连线三处一致表示。模型下拉会列出该规则下最近写入的检查点（名称、时间、大小），可回看训练早期权重；选中的文件被训练轮换删除时自动退回“最新”。连珠合法性复用训练规则；没有检查点时不会拿随机权重冒充训练模型，“开始对弈”会改为“暂无可用模型”并说明原因。

Web UI 固定使用 **CPU、单推理线程**，不申请 CUDA 或训练 GPU 锁，不写训练目录；仍会占用 CPU 和内存，单个服务进程实测常驻约 550 MB 工作集、约 1.5 GB 私有内存（含 CUDA 版 PyTorch 运行时），与训练同时运行会争用内存。新局加载当时已保存的模型，局中固定权重。点击“重新开始”可使用更新后的检查点。服务仅监听本机，所有浏览器标签共享一张棋桌，刷新可恢复当前棋局；退出服务后棋局不保留。思考期间请等待本手结束再悔棋或重开。客户端在模型思考时以 250 毫秒、空闲时以 1 秒轮询状态，因此数百毫秒的思考时间也能看到“模型思考中”。直接执行 `python main.py webui` 时日志就打在终端上。模型正在训练不等于已有成熟棋力。

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
- 6 个残差块、64 通道，共用主干，225 动作策略头和当前行棋方视角价值头。
- PUCT 系数 2，每步 200 次新搜索；复用子树和已有访问次数，不裁剪合法候选区域。
- 自我对弈根节点使用 25% Dirichlet 噪声，alpha=0.3；前 20 手按根访问次数分布采样，之后取最大访问次数。
- 每轮冻结模型，默认 16 个 Windows spawn CPU 对局进程生成 32 局；主进程将请求合并成 GPU 批次，批次等待最多约 3ms。RTX 5070 Ti 实测在 4 worker 时 CPU 与 GPU 都未充分利用，因此提高并发；可用 `--workers` 按机器负载调整。
- 采样结束再训练 200 步，批大小 256，最近 100,000 个局面回放；Adam 学习率 0.001，L2 系数 0.0001，梯度范数上限 5。
- 每局终局之后赋值胜 +1 / 负 -1 / 和 0，标签始终使用局面行棋方视角；不使用中间棋形奖励、人类棋谱或战术教师。
- 优化策略交叉熵加价值均方误差，优化器执行 L2 正则；旋转/镜像同时作用于棋盘和策略，执黑平面不变。

超参数是可运行起点，不代表已完成调优。自我对弈存在先手优势，尤其普通开局不等于公平的比赛开局；因此分别报告黑白成绩，不能只看自我对弈胜率或训练损失判断棋力。

## 时长、停止与恢复

`-Hours` 是本次运行的采样、优化及自动评估总预算，保存文件和关闭进程可能多用数秒。首次 Ctrl+C 发出安全停止请求，完整对局保留，未完成对局丢弃，保存后退出。再次 Ctrl+C 为强制中断，可能只保留上一检查点。

如果时长到达时不足 32 局，已完成的局面仍会保存。如果当前轮有未完成的参数更新，检查点记录 `pending_steps`；下次续训先完成这些更新，再开始下一轮采样。因此短时分段训练不会一直只采样而没有更新。若连一局都没完成，不写入伪造和棋标签。

每轮原子保存一个完整检查点，保留最近三个。检查点包含网络、优化器、配置、回放池、总对局数、轮数、训练步数、待完成更新及随机数状态。续训允许通过 `--workers` 调整自对弈并发数，其他配置仍需与检查点一致。进程调度和 GPU 运算会影响数值及样本到达顺序，因此恢复保证训练状态连续，不承诺逐位可复现。

只加载本项目生成且可信的 `.pt` 文件：完整恢复使用 Python pickle。`best.pt` 是推理权重，不含优化器和回放池，不能用于续训；续训使用 `latest` 或完整 `checkpoint-*.pt`。已有训练目录要求显式 `-Resume`，不会覆盖为新训练。

## 文件与评估

```text
runs/freestyle/              无禁手模型、回放和日志
runs/renju/                  连珠模型、回放和日志
  config.json               当前配置
  checkpoint-*.pt           最近三个完整检查点
  best.pt                   首次更新模型，之后由历史对战结果替换
  metrics.jsonl             时间、吞吐、损失、策略熵、分色胜率
  games.jsonl               已完成对局的落点与结果
  evaluations.jsonl         训练中的定期评估
  evaluation-*.json         手动评估报告
artifacts/                  基准和短跑测试，不是正式训练产物
```

每 10 轮依次对历史 `best`、随机及固定战术基线进行评估；模型双方采用同样搜索预算，不加探索噪声。固定种子生成合法四手开局，每个开局交换模型执色。报告总胜和负、分色结果、得分及基于开局配对的保守 95% Hoeffding 区间。达到时间预算时标注 `complete: false`，不把半局计为和棋。

历史模型比较完成且得分超过 55% 时替换 `best`。初始 `best` 只是首次更新模型，55% 也是工程晋级门槛，并非统计显著或高手棋力证明。默认每个对手 20 局的置信区间很宽，可用 `-Pairs` 增加独立开局组数。随机和战术基线仅用于评估，不给训练产生奖励或标签。

## 测试与开发

```powershell
python -m pytest -q
python main.py doctor

# 小模型/低搜索预算的短跑，只用于功能验收
python main.py train --rule freestyle --config configs/smoke.json --output artifacts/my-smoke-freestyle --hours 0.04 --max-rounds 1
python main.py train --rule renju --config configs/smoke.json --output artifacts/my-smoke-renju --hours 0.04 --max-rounds 1
```

测试覆盖四方向胜负、长连、真假三三、同方向/交叉双四、边界与满盘、合法动作、搜索回传符号、立即取胜和必要防守、增强对齐、完整对局标签、实际参数更新、断点恢复、停止、规则隔离和 worker 异常传播。

实现分层：`main.py` 是统一命令入口；`az/game.py` 为规则；`az/search.py` 为 MCTS；`az/selfplay.py` 为多进程采样；`az/network.py` 为网络和推理；`az/training.py` 为优化与恢复；`az/evaluation.py` 为对战评估；`az/cli.py` 负责命令解析。

方法参考：[AlphaGo Zero](https://deepmind.google/blog/alphago-zero-starting-from-scratch/)、[DeepMind OpenSpiel AlphaZero](https://github.com/google-deepmind/open_spiel/blob/master/docs/alpha_zero.md)、[PyTorch CUDA 安装](https://pytorch.org/get-started/previous-versions/)。本项目是针对五子棋的简化 AlphaZero 系统，不是原论文计算规模或完整实现的复刻。

## 开源协议

本项目以 [MIT 协议](LICENSE) 发布，可自由使用、修改和再分发，请保留版权与许可声明。

欢迎提交 issue 和 pull request。需要 GPU 才能完整运行训练，因此请先看 [验收记录](ACCEPTANCE.md) 中的吞吐数据，再评估在自己机器上的时间成本；只跑单元测试不需要 GPU。

英文说明见 [README.en.md](README.en.md)。
