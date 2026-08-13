---
title: 环境搭建：install.sh 与 pip 安装流程
date: 2026-08-07
---

# 环境搭建：install.sh 与 pip 安装流程

<div class="epigraph">
<p>一切性能优化的前提，是先把环境装对。</p>
<footer>—— 面向部署工程师的朴素真理</footer>
</div>

<div class="article-byline">
<p>第四级 · ktransformers（消费级 MoE 推理引擎） ｜ 官方文档 Quick-Start ｜ 2026-08-07</p>
</div>

## 为什么「安装」也值得一篇

前面 23 篇讲完理论，但从理论到「跑起来」隔着一条深沟：ktransformers 依赖 CUDA、PyTorch、llamafile 内核、AMX 指令集，还有推理与微调两套环境。装错了版本，轻则跑不动，重则静默出错。<span class="marginnote">官方 Quick-Start 提供了两条安装路径：<strong>一键脚本 `install.sh`</strong>（适合新手与官方默认配置）与 <strong>pip 安装</strong>（适合想手动控制的进阶用户）。这一节把两条路都走一遍，讲清每一步在干什么。</span>

## 1 安装前的硬件体检

先确认机器够不够格。ktransformers 跑 671B 级模型的最低画像：

| 组件 | 最低要求 | 推荐 |
| --- | --- | --- |
| GPU | 24GB 显存（RTX 4090/4090D/L20） | 同左；16GB 为实践下限 |
| CPU | x86，支持 AVX512 | Intel 第四代 Xeon 及以上（支持 AMX） |
| 内存 | 382GB+（DDR5） | DDR5，勿用 DDR4（带宽不足） |
| 磁盘 | 400GB+ 权重存储 | SSD（加载更快） |

**辨析｜易错点：** 内存用 DDR4 是最常见的部署翻车点——第 1 篇算过：CPU 侧带宽决定吞吐，DDR4 的带宽比 DDR5 低一半以上，解码会明显变慢。**DDR5 不是「推荐」，是「必需」**。

硬件体检里最容易被忽略的一项是「**内存条数量与通道数**」：同样 384GB 内存，插 8 条（八通道）与插 4 条（四通道）的带宽差一倍——而带宽正是 CPU 侧吞吐的决定因素。**「内存够大」与「内存够快」是两件事**：体检时两者都要查。<span class="marginnote">跑 24GB 显存的 DeepSeek-R1（非 671B）不需要这么大内存；大内存专为「671B 权重约 377GB 常驻 DRAM」准备。先算清模型体积，再配内存。</span>

## 2 路线 A：install.sh 一键脚本

官方推荐的快速路径：

```bash
git clone https://github.com/kvcache-ai/ktransformers.git
cd ktransformers
./install.sh
```

脚本依次完成：

1. **创建 conda 环境**：Python 3.11+（版本与 CUDA 运行时对齐）。
2. **安装 PyTorch**：按 CUDA 版本安装对应 wheel（`pip install torch --index-url …`）。
3. **编译/安装 kt-kernel**：C++ 扩展（AMX、llamafile、Marlin 内核）的编译与安装。
4. **安装 transformers-kt 等**：注入框架的 Python 依赖。

第 3 步是「耗时大户」——C++ 内核编译可能耗时十几分钟到几十分钟（取决于 CPU 与内核数量）。**编译慢不是卡死**：看到编译日志滚动是正常的，耐心等待即可。若想缩短编译时间，可以并行编译（`make -j$(nproc)`）或直接用官方为常见 CPU 提供的预编译 wheel——**「省编译时间」与「本机最优」之间的取舍，由你按需选择**。

**为什么可能需要编译**：AMX/AVX-512 内核需要针对本机 CPU 微架构编译（指令集检测与特化），预编译 wheel 未必适配你的 CPU——`install.sh` 里的本地编译正是为了拿到「为你的 CPU 定制」的内核。

「为你的 CPU 定制」可以具体化：同一段矩阵乘代码，用 `-march=native` 编译时，编译器会根据「这台 CPU 具体支持哪些指令」生成最优指令序列；而预编译 wheel 只能按「最保守的通用 CPU」编译（兼容一切，但谁都跑不满）。**「通用兼容」与「本机最优」是编译目标的一对矛盾**——预编译包选前者，源码编译选后者。理解了这一点，你就能接受「为什么 ktransformers 需要编译而不是纯 pip 装完即用」：**它把「编译成本」前置到安装阶段，换取「运行时指令集最优」**——这笔交易对长期运行的推理服务非常划算。<span class="marginnote">这就是「支持 AMX ≠ 自动快」的工程体现：<strong>内核必须为本机指令集编译，预编译包做不到这一点</strong>。本地编译慢但值——换来的是 AMX 的 28× 级 CPU 加速。</span>

## 3 路线 B：pip 安装（进阶）

想更精细控制时，用 pip 分步装：

```bash
# 1. 推理内核（kt-kernel）+ 注入框架
pip install ktransformers

# 2. SGLang 后端（要跑服务时）
pip install sglang-kt

# 3. 微调组件（要跑 SFT/DPO 时）
pip install "ktransformers[sft]"
```

官方建议**把推理环境（kt-kernel）与训练环境（kt-sft）分开建**——两者的依赖（训练要 FlashAttention、加速库，推理不需要）互不干扰，避免版本冲突。

顺着「分离」再补一条实操：**给两个环境起清晰的名字**（如 `kt-infer` / `kt-sft`），避免「分不清在哪个环境」的混乱——环境一多，「我到底在哪个环境跑的」就成了最常见的低级错误源。**「环境命名即文档」**：好名字省去的不是几秒钟，而是一连串「为什么结果不对」的排查。

「环境分离」还引出一个更深的问题：**为什么不能「一套环境通吃」？** 因为推理与训练对依赖的「成熟度」要求不同——推理要「稳定、不折腾」（今天装好的，下周别坏）；训练要「最新、有特性」（新内核、新加速库）。若共用一套环境，升级训练依赖可能意外破坏推理环境——**「稳定」与「前沿」是两种互相冲突的诉求**。分开建环境，本质是把「稳定区」与「前沿区」隔离，各取所需。这也是软件工程里「环境隔离」的通用理由：**不同的工作负载，需要不同的「依赖稳定性」**。<span class="marginnote">工程上的「环境分离」原则：<strong>推理要的是「稳、省内存」，训练要的是「全、新特性」</strong>——两套需求的依赖图不同，混在一起迟早冲突。分开建环境，隔离故障半径。</span>

## 4 公式解析：安装链路 = 一条依赖解析链

安装本质上是在解一组**依赖约束**。设 PyTorch 版本 $T$、CUDA 版本 $C$、Python 版本 $P$、编译器版本 $G$，安装成功当且仅当所有约束同时满足：

$$
\text{ok} = \text{compat}(T, C) \wedge \text{compat}(P, T) \wedge \text{compat}(G, \text{kernel source})
$$

逐项拆解：

- **第一步**：PyTorch 与 CUDA 必须匹配——`torch` 的 wheel 按 CUDA 版本分发，版本错位直接装不上。
- **第二步**：Python 版本决定 torch 轮子可用性（如 Python 3.11 需对应 torch 轮）。
- **第三步**：C++ 内核编译需要编译器支持目标指令集（AMX 需要较新 GCC/Clang）——**编译器太旧，AMX 路径编不出来**。

**一句话**：`install.sh` 的价值是把这条「约束链」自动化——帮你对齐 Python/CUDA/torch/编译器，避免人工踩坑。

「约束链自动化」还引出一个实践建议：**安装前先记录当前的环境状态**（`conda list`、`nvidia-smi`、`gcc --version` 各存一份）——一旦出问题，这些「安装前快照」就是排查的对照基准。**「先留快照、再动手装」是环境工程的职业习惯**：出了问题能快速判断「是我装坏了，还是本来就缺什么」——这比盯着报错信息猜快得多。<span class="marginnote">遇到「装上了但跑不起来」的玄学问题，90% 是这条链上某环不匹配：<strong>先查 torch 与 CUDA 版本、再查编译器与 CPU 微架构</strong>——比看报错信息更接近根因。</span>

## 5 常见安装问题排查

环境搭建是「踩坑高发区」，把最常见的三类问题列成排查表：

| 症状 | 根因 | 解法 |
| --- | --- | --- |
| 装不上 torch | CUDA 版本与 torch 轮不匹配 | 查 `nvidia-smi` 的 CUDA 版本，换对应 index-url |
| 编译内核报错 | 编译器太旧 / 缺依赖 | 升级 GCC/Clang，装 build 依赖 |
| 装上了但 CUDA 不可用 | PyTorch 是 CPU 版 | `python -c "import torch; print(torch.cuda.is_available())"` 验证 |
| AMX 路径没生效 | CPU 不支持 / 编译未开 AMX | `grep amx /proc/cpuinfo`；重编译 |

**排查心态**：环境问题的根因九成在「版本错配」——**先验证每一环的版本（CUDA / torch / Python / 编译器），再动手重装**。无脑重装是最低效的排查方式；「先诊断、再动手」省时百倍。

## 6 小结

把环境搭建浓缩成「先体检、再选路、后排查」：**先体检**（硬件是否符合）、**再选路**（install.sh 或 pip）、**后排查**（版本错配先查链）。环境不是「配好就完」，而是一个「可复现、可排查」的过程。

- 安装前先**硬件体检**：24GB 显存、AVX512/AMX 的 CPU、382GB+ **DDR5** 内存。
- 路线 A：`./install.sh` 一键完成 conda 环境 + PyTorch + kt-kernel 本地编译 + 注入框架。
- 路线 B：`pip install ktransformers` / `sglang-kt` / `ktransformers[sft]` 分步装，**推理与训练环境分开**。
- **本地编译是关键**：AMX/AVX 内核必须针对本机 CPU 微架构编译，预编译 wheel 无法做到。
- 安装 = 解依赖约束链；「装上跑不动」先从 torch-CUDA-编译器链查起。

在下一节，我们把环境真正用起来——**消费级部署案例：24GB 单卡运行 DeepSeek-V3/R1**，从权重准备到跑出第一个 token 的完整流程。
