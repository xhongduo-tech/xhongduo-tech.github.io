## 核心结论

`GitHub Actions` 在 AI 项目里不是“把几条命令放到云上跑”，而是把代码改动、验证结果、审批动作和发布记录连成一条可追溯的门禁链路。这里的“门禁”可以理解为上线前必须经过的一组检查，不满足条件就不能继续往后走。

对普通后端项目，CI 常常只关心“能否编译、单测是否通过”。但对 AI 项目，尤其是改动涉及注意力内核、推理算子、CUDA kernel 或模型服务路径时，这个标准明显不够。因为 AI 改动即使“功能正确”，仍然可能带来三个直接问题：数值误差变大、吞吐下降、峰值显存超预算。三者任何一个出问题，都可能让主分支变得不可发布。

新手可以先这样理解：代码一提交，`GitHub Actions` 就自动做一轮体检。第一层看代码是否基本可用，比如 lint、编译、单测。第二层看模型和算子是否仍然算对。第三层看性能是否回归。第四层在真正发布前接入人工审批，确认这次改动满足上线门槛。这样，仓库里的每一次发布都有输入、有输出、有记录，而不是“某个人本地跑过，看起来没问题”。

下面这张表先给出完整流程的最小视图：

| 输入 | 检查 | 输出 |
|---|---|---|
| PR diff | lint、编译、单测 | 测试报告 |
| 配置参数 | smoke test | 日志与失败样本 |
| 数据版本 | 数值正确性验证 | `max_error`、误差报告 |
| 基准样本 | benchmark | `benchmark.json` |
| GPU 类型 | 性能与显存检查 | `tokens/s`、`p95 latency`、`peak_vram` |
| 发布目标环境 | environment 审批 | deploy 记录、审计日志 |

如果改动的是注意力内核，门禁必须覆盖完整工程语义，而不是停留在工具清单。你需要明确输入是什么，输出是什么，评测口径是什么，回归风险是什么，上线门槛是什么。否则 `Actions` 只是自动跑脚本，不是可靠的 CI/CD。

---

## 问题定义与边界

本文讨论的是 AI 项目的 CI/CD，不是通用 Web 项目的普通流水线。这里的重点不在页面构建，也不在接口单测，而在模型、算子、GPU、benchmark 和上线门槛。

“CI/CD”里，`CI` 是持续集成，意思是代码一合并就自动验证；`CD` 是持续交付或持续部署，意思是验证通过后，把变更稳定地送到下一个环境。对 AI 项目来说，CI/CD 的对象不是单一程序，而是一组紧耦合系统：模型代码、推理图、内核实现、依赖版本、GPU 环境、基准数据和部署配置。

边界先划清楚：

| 目标 | 不负责什么 | 为什么不能省略 |
|---|---|---|
| 编排验证步骤 | 不替代训练框架 | 训练逻辑仍由 PyTorch、JAX 等负责 |
| 设置发布门禁 | 不替代 benchmark 程序本身 | 性能测试要由专门脚本产出 |
| 保留产物与记录 | 不替代线上监控 | 线上问题仍需监控系统发现 |
| 连接审批与部署 | 不替代推理服务 | 部署目标仍是你的 serving 系统 |

一个典型边界例子是“替换 FlashAttention 内核”。如果只看 CI 是否编译通过，结论几乎没有价值。因为真正的问题不是“代码能不能生成二进制”，而是：

1. 输出数值是否仍然在容差范围内。
2. 长序列下吞吐是否比基线更差。
3. 峰值显存是否突破服务预算。
4. 在指定 GPU、指定 dtype、指定输入规模下是否稳定。

这里的“容差”就是允许的小误差范围，因为浮点数计算顺序变化后，结果常常不是逐位完全相同。比如 `bf16` 和 `fp16` 下，很多工程会接受 `max_error <= 1e-3` 或类似阈值，但不能无限放宽。

因此，本文把 GitHub Actions 的职责限定为两件事：

1. 编排验证步骤。
2. 用门禁把结果转化成是否可发布的决定。

它不负责发明新的注意力算法，也不负责替你判断业务是否应该上线某个模型版本。它负责的是：把这些判断所依赖的证据自动、稳定、可追溯地收集出来。

---

## 核心机制与推导

先看标准注意力。设 $Q, K, V \in \mathbb{R}^{n \times d}$，其中 $n$ 是序列长度，$d$ 是每个 token 的向量维度。注意力的标准计算写成：

$$
S = \frac{QK^T}{\sqrt{d}}, \quad P = \text{softmax}(S), \quad O = PV
$$

这里：

- `score matrix`，得分矩阵，指的是每个 query 和每个 key 的相似度表。
- `softmax`，归一化指数函数，作用是把一行得分转成概率分布。
- `HBM`，高带宽显存，可以理解为 GPU 上容量较大但访问代价较高的主显存。

朴素实现的核心问题是：它通常会把整个 $S$ 和 $P$ 都物化，也就是完整写到 HBM 里。因为 $S$ 是 $n \times n$，$P$ 也是 $n \times n$，所以额外的显存占用和读写量都会随序列长度平方增长，即 $Θ(n^2)$ 级别。对短序列这可能还能接受，但当 `seq_len=4096`、`8192` 甚至更高时，HBM 压力会很快成为瓶颈。

可以把两种思路先对比一下：

| 维度 | 朴素实现 | 分块实现 |
|---|---|---|
| 中间结果 | 物化整张 `S`、`P` | 不物化完整 `S`、`P` |
| HBM 读写 | `Θ(n^2)` 级额外开销 | 明显降低 |
| 峰值显存 | 受大矩阵缓冲主导 | 受 tile 和行状态主导 |
| 算术强度 | 较低 | 更高 |
| GPU 利用率 | 易受显存带宽限制 | 更容易让 tensor core 持续忙碌 |

这里的“算术强度”可以理解为“每读一字节数据，能做多少计算”。如果一份数据从 HBM 读出来后可以被 tile 内多次复用，那么单位数据就能服务更多 FLOPs，吞吐更容易上去。

这就是 FlashAttention 一类方法的工程价值。它不是改了注意力的数学定义，而是改了计算顺序和访存策略。做法是把 $K$ 和 $V$ 按块切分成 tile，对每个 query 行在线维护三个状态：

- $m$：当前见过的最大分数。
- $l$：当前归一化分母。
- $o$：当前未归一化输出累积。

在线更新公式为：

$$
m' = \max(m, \text{rowmax}(S_{\text{tile}}))
$$

$$
l' = e^{m - m'} l + \sum e^{S_{\text{tile}} - m'}
$$

$$
o' = e^{m - m'} o + e^{S_{\text{tile}} - m'} V_{\text{tile}}
$$

最后输出：

$$
O = \frac{o}{l}
$$

这里的 `online softmax` 可以白话理解为：不等所有分数都算完再统一做 softmax，而是一边扫描 tile，一边维护“到目前为止正确的归一化状态”。这样就只需要保留行级状态，不需要把完整概率矩阵写回 HBM。

下面给一个玩具例子，证明分块只改顺序，不改数学结果。

设单个 query：

$$
q = [1, 0]
$$

三把 key 分两块：

$$
k_1 = [1, 0], \quad k_2 = [0, 1], \quad k_3 = [1, 1]
$$

取标量 value：

$$
v = [1, 2, 3]
$$

则三个得分为：

$$
[q \cdot k_1, q \cdot k_2, q \cdot k_3] = [1, 0, 1]
$$

直接法先做 softmax。因为

$$
\text{softmax}([1,0,1]) = \frac{[e^1, e^0, e^1]}{2e + 1}
$$

所以输出为：

$$
O = \frac{1 \cdot e + 2 \cdot 1 + 3 \cdot e}{2e + 1}
= \frac{4e + 2}{2e + 1} = 2
$$

分块法：

1. 第一块包含分数 `[1, 0]`，则  
   $m = 1$，  
   $l = 1 + e^{-1} \approx 1.3679$，  
   $o = 1 + 2e^{-1} \approx 1.7358$。

2. 第二块只有分数 `[1]`，新块最大值仍为 `1`，所以  
   $m' = 1$，  
   $l' = l + 1 = 2.3679$，  
   $o' = o + 3 = 4.7358$。

3. 最终输出：

$$
O = \frac{o'}{l'} \approx \frac{4.7358}{2.3679} = 2
$$

结果与直接 softmax 完全一致。这说明在线更新保持了数学等价性。

为什么这件事跟 GitHub Actions 有关？因为一旦你把注意力内核换成分块实现，CI/CD 的检查目标就立刻升级了。你要验证的不再是“接口有没有返回值”，而是“数学结果是否等价、吞吐是否更高、显存是否更省”。这也是 AI 项目和普通业务项目门禁设计差异最大的地方。

真实工程例子可以这样设定：你在仓库里把旧注意力实现替换成 FlashAttention 风格内核，固定输入为 `seq_len=4096`、`head_dim=128`、`dtype=bf16`、`gpu=A100`。Actions 需要产出至少三类结果：

1. `max_error`：与 reference 实现对比的最大误差。
2. `tokens/s` 或 `latency p95`：吞吐或尾延迟是否退化。
3. `peak_vram`：峰值显存是否超预算。

只有当这三类指标都过线，改动才应该具备合并或发布资格。

---

## 代码实现

真正落地时，重点不是“写一个 job”，而是把验证层拆开，让每一层只负责一件事。一个可维护的结构通常是：

1. 编译与单测：判断代码基本可执行。
2. 数值正确性：判断新实现和参考实现是否一致到可接受范围。
3. benchmark：判断性能和显存是否满足门槛。
4. 部署审批：判断是否允许进入目标环境。

先看一个最小可运行的 Python 例子，用来模拟“数值正确性 + 门槛判定”。这里不依赖 GPU，只演示逻辑结构。

```python
import math

def softmax(xs):
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [x / s for x in exps]

def direct_attention_output(scores, values):
    probs = softmax(scores)
    return sum(p * v for p, v in zip(probs, values))

def tiled_attention_output():
    # toy example: scores = [1, 0, 1], values = [1, 2, 3]
    m = float("-inf")
    l = 0.0
    o = 0.0

    # tile 1
    tile_scores = [1.0, 0.0]
    tile_values = [1.0, 2.0]
    new_m = max(m, max(tile_scores))
    scale_old = 0.0 if m == float("-inf") else math.exp(m - new_m)
    exps = [math.exp(s - new_m) for s in tile_scores]
    l = scale_old * l + sum(exps)
    o = scale_old * o + sum(e * v for e, v in zip(exps, tile_values))
    m = new_m

    # tile 2
    tile_scores = [1.0]
    tile_values = [3.0]
    new_m = max(m, max(tile_scores))
    scale_old = math.exp(m - new_m)
    exps = [math.exp(s - new_m) for s in tile_scores]
    l = scale_old * l + sum(exps)
    o = scale_old * o + sum(e * v for e, v in zip(exps, tile_values))
    m = new_m

    return o / l

def gate(metrics, baseline):
    assert metrics["max_error"] <= 1e-3
    assert metrics["tokens_per_s"] >= baseline["tokens_per_s"] * 0.98
    assert metrics["peak_vram_mb"] <= baseline["peak_vram_mb"] * 1.05
    return True

scores = [1.0, 0.0, 1.0]
values = [1.0, 2.0, 3.0]

direct = direct_attention_output(scores, values)
tiled = tiled_attention_output()

assert abs(direct - 2.0) < 1e-9
assert abs(tiled - 2.0) < 1e-9
assert abs(direct - tiled) < 1e-9

baseline = {"tokens_per_s": 10000, "peak_vram_mb": 8000}
candidate = {"max_error": 8e-4, "tokens_per_s": 9850, "peak_vram_mb": 8300}
assert gate(candidate, baseline) is True

print("all checks passed")
```

这个例子体现了一个关键思路：CI/CD 里的“通过”不是布尔判断，而是指标判断。你需要把“合格标准”写进代码或配置，而不是靠口头约定。

下面给一个最小的 GitHub Actions 工作流片段。它展示了 `on: pull_request`、`jobs`、`artifacts`、`environment`、`concurrency` 这些关键字段的基本用法。

```yaml
name: ai-ci

on:
  pull_request:
    branches: [main]
  workflow_dispatch:

concurrency:
  group: ai-ci-${{ github.ref }}
  cancel-in-progress: true

jobs:
  unit-and-smoke:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run unit tests
        run: python -m pytest tests/unit -q
      - name: Run smoke test
        run: python scripts/smoke_test.py --model tiny
      - name: Upload smoke logs
        uses: actions/upload-artifact@v4
        with:
          name: smoke-logs
          path: logs/smoke/

  benchmark:
    if: github.event_name == 'workflow_dispatch'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run benchmark
        run: |
          python scripts/benchmark.py \
            --seq-len 4096 \
            --head-dim 128 \
            --dtype bf16 \
            --gpu a100 \
            --baseline main \
            --output benchmark.json
      - name: Upload benchmark artifact
        uses: actions/upload-artifact@v4
        with:
          name: benchmark-json
          path: benchmark.json

  deploy:
    needs: [unit-and-smoke]
    runs-on: ubuntu-latest
    environment:
      name: production
    steps:
      - uses: actions/checkout@v4
      - name: Deploy
        run: ./scripts/deploy.sh
```

这个 YAML 有几个重要点：

- `pull_request` 触发基础验证，保证每个 PR 都先过基本门槛。
- `workflow_dispatch` 允许手动触发 benchmark，因为 GPU 资源通常更贵，不一定每次 PR 都跑满量性能回归。
- `concurrency` 确保同一个分支的新运行会取消旧运行，避免旧结果覆盖新结果。
- `upload-artifact` 把日志和 `benchmark.json` 保存下来，方便复盘。
- `environment` 把部署挂到受保护环境上，只有审批通过才能继续。

如果把它映射到真实 AI 项目，建议显式固定以下 benchmark 输入：

| 参数 | 作用 | 不固定的风险 |
|---|---|---|
| `seq_len` | 控制序列长度 | 长序列退化被掩盖 |
| `head_dim` | 控制注意力维度 | 不同维度性能差异被混淆 |
| `dtype` | 控制数值类型 | `fp16/bf16/fp32` 结果不可比 |
| `gpu` | 固定硬件环境 | 机器差异导致 benchmark 漂移 |
| `baseline commit` | 固定对照版本 | 无法判断相对回归 |
| `seed` | 固定随机种子 | 误差和延迟波动变大 |

一个新手常见误解是“我已经有 benchmark 脚本了，所以 CI/CD 自动就完善了”。这不对。benchmark 脚本只负责生产指标，Actions 负责把指标接进工程流程。例如你可以规定：

- 单测必须全绿。
- `max_error <= 1e-3`。
- `tokens/s` 不低于基线 `-2%`。
- `peak_vram` 不高于基线 `+5%`。
- 进入 `production` 前必须经过 required reviewers 审批。

当这些条件自动化后，“能跑”才被转化成“可发布”。

---

## 工程权衡与常见坑

工程里最大的问题通常不是不会写工作流，而是不知道该把哪些信号纳入门禁。AI 项目比普通项目多出来的复杂度，本质上来自“正确性”和“资源效率”必须同时成立。

先看常见坑：

| 坑 | 后果 | 规避方式 |
|---|---|---|
| 不固定基线 | benchmark 结果无法比较 | 固定 `baseline commit` |
| 不保留 artifact | 事后无法复盘 | 上传日志、JSON、失败样本 |
| 不启用 concurrency | 旧运行覆盖新结果 | `cancel-in-progress: true` |
| 不设置 environment 审批 | 错误改动直接发版 | 使用 protected environment |
| 只跑单测不跑性能回归 | 正确但更慢的代码进入主分支 | 单独设置 benchmark 门禁 |
| 只看平均值不看 `p95` | 尾延迟恶化被忽略 | 同时记录平均值和 `p95` |
| 不看峰值显存 | 服务在大输入下 OOM | 记录 `peak_vram` 并设预算 |
| 不固定 runner/CUDA/dtype/seed | 结果漂移，难以复现 | 固定环境与输入参数 |

看两个真实会发生的例子。

例子一：新内核单测全绿，但 `tokens/s` 下降 8%。  
如果你的 CI 只检查单测，那么这个 PR 会被合并。之后线上吞吐下降，同样的 GPU 成本只能服务更少请求，单位成本上升。这不是“性能优化没做到”，而是“门禁设计错误”。

例子二：平均延迟不变，但 `p95` 激增。  
`p95` 可以理解为“最慢那 5% 请求的边界延迟”，它比平均值更能反映尾部风险。注意力内核在短序列上可能表现稳定，但在长序列或大 batch 下出现块调度抖动。如果你只看平均延迟，问题可能完全看不出来。

这里还有一个容易被忽略的点：benchmark 本身也有成本。GPU runner 贵，跑一次完整性能回归慢，所以不一定适合每个 PR 都跑全量。常见权衡是：

1. PR 默认跑单测、smoke test、轻量数值校验。
2. 只有涉及关键路径的改动，或由人工手动触发时，才跑重型 benchmark。
3. 发布前对候选 commit 跑正式回归，并保留 artifact。

这种分层方案的本质不是“偷懒”，而是把验证成本和风险等级对齐。低风险改动走快路径，高风险改动走慢路径。

还要注意审批与并发控制。假设两个 PR 都要部署测试环境，如果没有 `concurrency`，旧 PR 的部署可能在新 PR 之后完成，反过来把环境覆盖掉。又比如生产环境如果没有 `environment` 审批，任何有权限触发 workflow 的人都可能把未充分验证的版本推上去。对 AI 服务来说，这种错误会直接影响线上流量和 GPU 成本。

所以，GitHub Actions 的工程价值不只是“自动化”，而是“把自动化变成稳定的发布秩序”。

---

## 替代方案与适用边界

不是所有项目都需要完整的 GitHub Actions 门禁。如果你的项目只是几个 Python 脚本，没有 GPU、没有部署环境、没有性能敏感路径，那么更轻量的方案可能更合适。

但一旦验证目标从“普通后端代码是否正确”升级到“数值正确性 + 吞吐 + 显存”，你就需要能稳定处理四类能力：

1. 固定输入与环境。
2. 保留 benchmark 产物。
3. 控制并发与部署顺序。
4. 接入人工审批。

下面做一个直接对比：

| 方案 | 适合场景 | 缺点 | 是否支持审批/产物/并发控制 |
|---|---|---|---|
| 只用本地脚本 | 个人实验、小规模验证 | 不可追溯，结果依赖本机环境 | 否 / 弱 / 否 |
| 只用普通 CI | 代码检查、基础单测 | 难覆盖 GPU benchmark 和发布门禁 | 弱 / 中 / 中 |
| GitHub Actions + environment + artifact | AI 内核、模型服务、正式发版 | 配置更复杂，运行成本更高 | 是 / 是 / 是 |

这里的“普通 CI”指只把测试放到远端跑，但没有把 benchmark、artifact、审批和并发控制纳入统一流程。它比本地脚本强，但还不够承担 AI 项目的发布职责。

适用边界可以这样理解：

- 如果你只是验证 Python 语法、单元函数和少量 CPU 逻辑，本地脚本就够。
- 如果你需要多人协作下的自动测试，普通 CI 可以解决一部分问题。
- 如果你要改的是模型推理路径、CUDA kernel、注意力实现或部署配置，并且结果会影响线上性能或显存预算，那么需要完整门禁，GitHub Actions 这类平台更合适。

再强调一遍，工具本身不是重点。重点是你的发布标准是否被形式化。没有明确输入、明确输出、明确阈值、明确审批，再好的 CI 平台也只能执行模糊流程。相反，只要这些标准被定义清楚，GitHub Actions 就能把它们稳定地串起来。

如果继续深入，建议先读 GitHub 官方的 workflow syntax、environment、artifacts、concurrency 文档，再看 FlashAttention 的论文和实现。前者解决“怎么把门禁落地”，后者解决“为什么注意力内核需要这种门禁”。

---

## 参考资料

| 分组 | 用途 | 代表资料 |
|---|---|---|
| GitHub Actions 官方文档 | 工作流语法、环境审批、产物、并发控制 | Workflow syntax、Environments、Artifacts、Concurrency |
| FlashAttention 官方实现 | 了解工程实现与接口设计 | `Dao-AILab/flash-attention` README |
| FlashAttention 论文与博客 | 了解分块注意力、online softmax 与性能来源 | FlashAttention 论文、FlashAttention-3 blog |

1. [GitHub Docs: Workflow syntax for GitHub Actions](https://docs.github.com/en/actions/reference/workflows-and-actions/workflow-syntax)
2. [GitHub Docs: Managing environments for deployment](https://docs.github.com/en/actions/reference/environments)
3. [GitHub Docs: Workflow artifacts](https://docs.github.com/en/actions/concepts/workflows-and-actions/workflow-artifacts)
4. [GitHub Docs: Concurrency](https://docs.github.com/en/actions/concepts/workflows-and-actions/concurrency)
5. [Dao-AILab/flash-attention 官方仓库 README](https://github.com/Dao-AILab/flash-attention/blob/main/README.md)
6. [FlashAttention 论文页](https://huggingface.co/papers/2205.14135)
7. [Tri Dao: FlashAttention-3 blog](https://tridao.me/blog/2024/flash3/)
