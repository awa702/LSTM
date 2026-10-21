# 心拍分类与 RV32I 部署交接说明

更新时间：2026-10-21
工程目录：`/home/gary/project/lstm`
状态：训练与导出流程已搭建；当前数据集不完整，整数推理仍是早期脚手架，尚未完成正式评估或板级集成。

## 1. 项目目标

在支持 RV32I 的 CPU 上运行一个轻量 ECG 心拍分类器。先在主机上训练和评估浮点模型，再做定点化、C 参考实现、RV32I 软件推理验证；之后再决定是否以及如何加入硬件 MAC 加速。

一个模型调用处理**一个心拍窗口**并输出该拍类别。它不是对一段时间内多个心拍做序列级诊断。连续演示时可以把连续采样得到的窗口逐个送入模型，并在应用层显示当前类别、最近 N 拍计数或比例。

首选数据源是 MIT-BIH Arrhythmia Database（PhysioNet，360 Hz，48 条双导联记录）。训练集中的 R 峰位置先使用数据库人工标注；这只用于训练样本切分。实际 ADC 连续输入时还需要 QRS/R 峰检测模块，该模块目前不在已完成范围内。R 峰间期（RR interval）可以作为后续附加特征实验，首版模型输入是 R 峰周围的 ECG 波形，不是 RR 间期。

## 2. 已确认的目标 CPU 约束

以下为用户明确给出的平台约束，模型软件和最终固件必须遵守：

- ISA：RV32I、Zicsr、Zmmul。
- 不支持 DIV/REM 指令；不支持 C、F 扩展；不使用分页或 U/S 模式。
- 仅 M-mode；`mtvec` 使用 Direct 模式；`IALIGN=32`。
- 64 KiB RAM，地址范围 `[0x80000000, 0x80010000)`。
- reset 不清 RAM。启动/重启代码不能假定 RAM 已被硬件清零；需要显式初始化其依赖的 `.data`、`.bss`、模型状态和缓冲区，或证明冷启动内容不影响程序。
- 软件推理首阶段优先。后续预计利用 Zmmul 的乘法能力；必须分别评估纯 RV32I（乘法由软件实现）和带 Zmmul 的实现。

应为裸机目标使用类似 `-march=rv32i_zicsr_zmmul -mabi=ilp32` 的工具链参数，并核实编译器工具链是否接受该 ISA 字符串。纯 RV32I 参考构建可以单独用 `-march=rv32i -mabi=ilp32`。最终以实际 CPU 支持列表和链接/反汇编结果为准，不能产生 DIV/REM、压缩指令或浮点指令。Zicsr 只允许 CSR 指令，不代表允许其他特权模式功能。

RAM 核算不能只看模型权重。最终需要结合链接 map 和启动流程列出代码是否位于 RAM、权重、`.data/.bss`、输入/ADC 双缓冲、输出、栈、异常处理及 UART 缓冲；保留余量并实测最坏栈深。若代码由 ROM/指令存储器提供，需要明确区分 ROM 与上述 64 KiB RAM 预算。

## 3. 首版数据与任务方案

### 数据预处理当前实现

实现位置：`scripts/prepare_mitdb.py`

- 采样率：MIT-BIH 原始 360 Hz。
- 导联：优先选 `MLII`，没有时选 `II`，仍没有时退回第一导联。该回退策略需要检查全部 header，确认训练和目标采集导联是否一致。
- 以标注样本位置为 R 峰中心，截取 `[峰-128, 峰+128)`，共 256 点；窗口含峰前 128 点和峰位置及其后 127 点，时长约 0.711 秒。
- 记录选择规则：
  - 二分类 0：符号 `N/L/R/e/j`（AAMI N 类组）。
  - 二分类 1：符号 `A/a/J/S/V/E/F`（当前实验的异常组）。
  - 其他标注一律排除，包含 Q/起搏/未知等；不能将 Q 默认为异常。
- 归一化：仅从训练 split 所有纳入的窗口计算全局均值和标准差；`(x-mean)/std`，裁剪到 `[-5,5]`。该预处理适合先跑通基线，板端等价实现尚未确定。
- 丢弃超出记录边界的窗口。
- 输出 `data/processed/{train,validation,test}.npz`（`x` 形状 `[样本数,256,1]`、float32；`y` 为 0/1）及 `metadata.json`。

### 患者隔离划分

必须按记录/患者分组，不能随机拆分心拍。当前脚本固定测试记录为 `105,111,116,203,221,228,234`，验证记录为 `108,118,124,210,215,223,232`，其余纳入记录进训练集。脚本注释特别要求 201 与 202 属于同一受试者并留在同一 split（当前两者均在训练集）。正式实验需复核 MIT-BIH 的记录-受试者映射及 split 设计，确保数据完整后固定 split 并保存清单，不能为了分数随意调整。

### 任务定义仍需冻结

当前两类映射只是 baseline，不是临床诊断定义。AAMI N 组包含束支传导阻滞等心拍，标签 0 不应称为“医学正常”；输出应称作“当前映射下的 N 组 / S-V-F 组”或类似的技术名称。训练和交接时需明确这是研究/教学分类，不是临床诊断系统。

后续可以在同一患者划分下研究 AAMI 五组 `N/S/V/F/Q`，或加入 RR 特征；但需先核对符号映射、起搏记录处理、类别支持数和每类指标。当前先保持二分类基线，避免同时改变任务、数据和模型。

## 4. 当前模型与实验结果

实现位置：`scripts/train_baseline.py`

- 单向、单层 PyTorch LSTM：输入维度 1，隐藏维度默认 16，取最后时刻输出，再接 `Linear(16,2)`。
- 序列长度固定为 256；训练 Adam，学习率 `1e-3`，带按训练类频率计算的 class weight；随机种子 0。
- 参数量：隐藏维度 16 时 1,250 个参数。训练 checkpoint 为 float32，约 5,000 bytes。
- 训练时当前使用验证集 macro-F1 选取 checkpoint；测试集只在训练结束后评估。

`artifacts/baseline/metrics.json` 是不完整数据子集上的早期 GPU 结果：验证 macro-F1 约 0.5273，测试 macro-F1 约 0.4919。临时测试集 4,691 拍，其中正类仅 42 拍，正类召回率 0.05；该分数**只证明流程可以执行，不是可接受的模型表现或正式结果**。不要对外宣称模型有效。必须在完整数据和冻结的患者划分上重训，报告每类 precision/recall/F1、macro-F1、混淆矩阵、类别数量，并特别检查异常类召回率。

目前路线是针对该数据、标签、窗口重新监督训练一个小模型，不是直接微调一个通用开源模型。论文/开源项目可参考方法和对照，不应跳过患者级划分或数据审查。

## 5. 数据下载状态与网络

数据目录：`data/mitdb/`。完整 MIT-BIH 每条记录的 `.dat` 在 format 212 下通常应为 1,950,000 bytes；脚本会根据 header 中帧数检查预期长度，并要求 `.hea/.atr/.dat` 均齐全。

截至本文时间，数据目录中多数记录完整，但以下文件不完整或缺失：

| 记录 | 当前问题 |
|---|---|
| 116 | `.dat` 只有 229,304 bytes |
| 117 | `.dat` 只有 143,360 bytes |
| 201 | `.dat` 只有 1,875,968 bytes |
| 202 | `.dat` 只有 241,664 bytes |
| 203 | `.dat` 只有 225,280 bytes |
| 205 | `.dat` 只有 94,208 bytes |
| 207 | `.dat` 只有 126,976 bytes |
| 208 | `.dat` 只有 94,208 bytes |
| 209 | `.dat` 只有 339,968 bytes，且缺少 `.atr` |

目录中还留有若干 `.dat.part*` 分块临时文件，不能作为完整记录使用。不要把这些分块直接拼接，除非核实分块偏移、长度和内容校验。PhysioNet 小文件可以 HTTP 200，但近期大文件实测仍只有约 3.3–3.8 KB/s，并在限时探测中超时；曾尝试 Windows/WSL 代理下载也未解决。下载流程可重试：

```sh
cd /home/gary/project/lstm
./.venv/bin/python scripts/prepare_mitdb.py --download
```

该命令默认请求全量记录，速度可能导致长时间下载。执行后必须重新核对每条记录的 header、annotation、data 长度和读取状态，再生成正式 splits。下载或网络恢复尚未发生，不要把部分数据产物当成完整训练数据。

## 6. 环境与运行方式

工程位于 WSL Linux 文件系统 `/home/gary/project/lstm`。当前 `.venv` 是 Python 3.10 环境；曾验证 Python 3.10.12、PyTorch 2.11.0+cu128、CUDA runtime 12.8、RTX 5060 Laptop GPU（8 GiB）。GPU 合成张量上的 LSTM 前向、反向、参数更新检查通过。环境检查脚本：`scripts/check_environment.py`。

```sh
cd /home/gary/project/lstm
./.venv/bin/python scripts/check_environment.py
```

依赖清单在 `requirements.txt`、`requirements-gpu.txt` 和已安装环境锁定文件 `requirements-lock.txt`。GPU 训练只在主机需要 PyTorch/CUDA；裸机 C 推理不需要 Python 或 PyTorch。

数据完整后：

```sh
./.venv/bin/python scripts/prepare_mitdb.py --download
./.venv/bin/python scripts/train_baseline.py --epochs 20 --hidden 16
./.venv/bin/python scripts/export_model.py
./.venv/bin/python scripts/quantize_model.py
```

在数据准备阶段可不带 `--download` 仅处理已齐全的本地记录；缺记录时脚本应明确失败。确认实际命令行参数可用后再运行，不要覆盖当前临时产物前未留存记录。

## 7. 定点化和 RV32I 运行时代码现状

相关文件：

- `scripts/export_model.py`：导出浮点权重、manifest 和 golden 输入/输出。
- `scripts/quantize_model.py`：生成量化权重、manifest、浮点 golden。
- `runtime/lstm_fixed.c/.h`：固定尺寸整数 C 参考实现。
- `runtime/rv32i_mul.c`：无 M 扩展时的乘法辅助代码。
- `runtime/rv32i_mem.c`：小型 freestanding 内存/链接评估辅助代码。

早期量化草案采用输入/权重 int8（Q1.7）、bias int32（脚本注释为 Q7.9）、门输出 int16 Q1.15、状态 int16 Q3.5，舍入为最近值（ties away from zero 的意图），并饱和裁剪。此方案尚未通过误差分析。`runtime/lstm_fixed.c` 中 sigmoid/tanh 是明确的**占位线性/分段近似**，不是精度合格实现；有符号右移行为、移位溢出、累加器上界以及定点格式之间的缩放关系也需逐项审计。生成权重头文件在量化运行后才存在，默认忽略于版本控制。

接手者应先建立 Python 定点 reference，明确定义乘法累加、舍入、饱和、激活函数和状态更新的每一步，再用测试向量比较浮点、Python 定点、C 参考；最后交叉编译 RV32I/Zmmul 并核对机器码和 golden 结果。不要把当前 C 数值输出当作已验证推理结果。

## 8. 内存和计算量现有估算

旧的 README 记录过多组编译统计（不同编译选项/入口下约 2.7–3.7 KiB 的模型核心 footprint、约 272–288 bytes 栈帧、模型量化权重约 1.6 KiB、输入和 logits 约 260 bytes）。这些只是历史参考构建，不是已整合目标固件的最终 RAM 占用；README 表内 RV32I 与 Zmmul 的指令描述应按用户现在确认的 Zmmul 约束重新核对。

粗略计算量约每窗口 290,848 个 32-bit 标量乘法（hidden=16、256 点，含门、状态更新和输出的旧估算）。64 KiB 远大于小模型的静态权重/输入初估，但最终可用空间取决于程序存储位置、运行时、ADC 双缓冲、UART/显示、栈和 reset 初始化。需用目标链接器 map、反汇编和最大栈验证完成结论。Zmmul（只含乘法相关能力）不代表支持除法或取模；训练端/预处理端使用除法不构成目标代码许可。

## 9. Demo 方向

可演示为信号发生器/录制 ECG 数据源 → ADC 连续采样 → R 峰检测 → 以检测到的峰切 256 点窗口 → 分类 → 串口发送原始波形、峰位置和每拍类别 → PC 上位机显示波形与最近 N 拍统计。串口屏也可显示当前类别和计数，但画波形/展示指标时 PC 页面较灵活。展示方式尚未最终决定。

逐拍“测试集回放”需要明确数据通路：数据文件通过 PC/UART/SPI 等送给板端，或使用信号发生器逐段输出并触发采样。后一种方案涉及信号幅度、采样时钟、R 峰检测时延与标签对齐。首版建议先用数字样本离线回放验证分类软件；双缓冲连续采样和发生器自动逐拍控制属于集成设计，尚未实现。

## 10. 下一步工作顺序

1. 确认数据能否稳定下载；补齐所有记录，逐条核验文件长度和 WFDB 可读性。
2. 审核 MIT-BIH 记录-患者关系、导联选择和标注映射；冻结患者级 train/validation/test 清单。
3. 重新生成处理数据，核对每 split 的记录列表、窗口数和每类数；按类别不平衡问题制定训练策略，训练多次/做有意义的基线比较。
4. 以完整 test 集给出可靠指标，保存代码版本、数据清单、随机种子、预处理参数和模型 checkpoint。
5. 生成真正等价且经误差验证的整数 reference，完成 sigmoid/tanh 近似与 RV32I C 对照；补充真实心拍 golden 输入。
6. 分别用 RV32I（乘法软件路径）和 RV32I+Zicsr+Zmmul 交叉编译，确认无非法指令；统计指令/周期、ROM 和 RAM、栈、模型和 I/O 缓冲。
7. 将模型输入接到目标板/仿真环境，确认 reset 不清 RAM 时启动初始化正确，再设计 ADC、UART 或串口屏演示链路。

## 11. 重要文件

| 路径 | 内容 |
|---|---|
| `README.md` | 工程快速介绍、早期结果和运行命令（部分内存数字为历史估算） |
| `scripts/prepare_mitdb.py` | MIT-BIH 下载、患者级划分、窗口与标签生成 |
| `scripts/train_baseline.py` | 小型浮点 LSTM 基线训练和指标 |
| `scripts/export_model.py` | 模型权重与 float golden 导出 |
| `scripts/quantize_model.py` | 早期量化脚手架 |
| `runtime/lstm_fixed.c/.h` | 固定点 C 参考（激活目前为占位实现） |
| `runtime/rv32i_mul.c` | RV32I 软件乘法支持 |
| `data/mitdb/` | 原始记录（部分不完整） |
| `data/processed/metadata.json` | 临时子集预处理元数据，不能代表完整 split |
| `artifacts/baseline/metrics.json` | 临时子集训练分数，不能作为正式结果 |

## 12. 文档与结果的注意事项

仓库没有 Git 元数据（当前目录执行 `git status` 返回非 git repository）。`data/` 和 `artifacts/` 中有本地生成内容；开始大规模重训或重写数据前，先备份想保留的 checkpoint、metrics 和 metadata。README 中有些旧状态陈述可能与实际数据变化不同，本文按 2026-10-21 目录检查和此前网络实测记录说明，接手后先重新核实现状。
