## 核心结论

训练完成不等于可以部署。训练后处理的核心任务，是把“模型评估、模型选择、模型导出、模型压缩、部署打包”串成一条闭环流水线，保证最后上线的模型体，和你真正评估过的模型体是同一个对象，而不是“训练时一个版本、导出后又变成另一个版本”。

这里有两个最容易混淆的概念。验证集，是“用来排名”的数据，也就是帮助你在多个候选模型里做选择；测试集或评估集，是“用来出最终成绩单”的数据，也就是对外报告泛化能力的数据。论文里反复强调：如果你先在验证集上挑了最优模型，再直接把这个验证分数当最终结果，就会产生选择偏差。直白地说，你是在用一场内部选拔赛成绩，冒充正式比赛成绩。

因此，一个最小可用的训练后处理流程应该满足三件事：

| 环节 | 作用 | 不能替代什么 |
| --- | --- | --- |
| 验证 | 给候选模型排序、调超参数 | 不能替代最终泛化评估 |
| 评估/测试 | 在独立数据上报告最终性能 | 不能参与模型挑选 |
| 导出与优化 | 把模型变成可部署 artifact | 不能跳过回归验证 |

玩具例子可以这样理解。你训练了两个小模型 A 和 B。A 在验证集 F1 是 0.92，B 是 0.90。验证集告诉你 A 更值得继续看，但这不等于 A 一定是部署版本。你还需要把 A，甚至接近 A 的 B，一起放到独立评估集上看最终表现。如果 B 在外部数据上更稳，真正应部署的版本可能反而是 B。

用公式写，就是先在验证集 $\mathcal V$ 上排序，再在独立评估集 $\mathcal E$ 上报告最终结果：

$$
m^*=\arg\max_m \hat{\vartheta}_m(\mathcal V)
$$

然后只把最终候选放到评估集上测：

$$
\hat{\vartheta}_{m^*}(\mathcal E)
$$

如果验证波动很大，还可以用 within-1SE 规则，也就是把“距离验证最优值不超过 1 个标准误”的模型一起送去评估。它的直觉很简单：验证分数接近、波动又不小的时候，不要太早只押一个模型。

---

## 问题定义与边界

训练后处理，指模型参数已经训练完成之后，为了让模型“可比较、可导出、可压缩、可上线、可回滚”而进行的一整套工作。这里的“artifact”是工程术语，白话就是“可保存、可传递、可部署的产物”，比如 `best.ckpt`、`model.onnx`、`quantized_model.onnx`、依赖锁文件、Git tag、评估报告。

它的边界至少包括下面六类输入输出：

| 阶段 | 输入 | 输出 artifact |
| --- | --- | --- |
| 训练完成 | 训练日志、checkpoint | 候选模型集合 |
| 验证排序 | 验证集、指标定义 | 候选排名、标准误、筛选名单 |
| 独立评估 | 测试集或外部评估集 | 最终性能报告 |
| 模型导出 | 最终候选 checkpoint | ONNX、TorchScript 等可部署格式 |
| 模型优化 | 导出模型、校准数据 | 量化模型、剪枝模型、蒸馏学生模型 |
| 部署打包 | 模型文件、代码、依赖、配置 | 可回滚部署包 |

对初学者来说，最重要的边界不是“工具名”，而是“数据有没有串线”。训练集负责学参数，验证集负责选模型，测试集负责看泛化。数据一旦越界，后面所有结论都会变形。比如论文和综述都强调，数据切分要尽早做，最好在任何标准化、降维、特征筛选之前就完成，否则容易产生泄露。

一个常见的真实工程例子是图像分类服务上线。团队手里有 70% 训练数据、10% 验证数据、20% 外部评估数据。流程不是“训练完直接导出 ONNX”，而是：

1. 用训练集训练多个候选。
2. 用验证集比较 Accuracy、F1、延迟预算。
3. 把最优或 within-1SE 的几个候选导出为 ONNX。
4. 用静态 PTQ，也就是 post-training quantization，白话就是“不重新训练，只靠校准数据把浮点模型压成低比特模型”，生成 INT8 版本。
5. 在外部评估集上重新跑精度与延迟。
6. 把最终模型、推理脚本、依赖版本、Git tag、评估报告一起打包。

这里还要区分两种“最终版本”：
第一种是“统计最终版本”，即在独立评估集上确认过性能的模型。
第二种是“工程最终版本”，即真正部署的导出加优化后的模型体。
成熟流程要求这两者一致，至少要能一一对应。

---

## 核心机制与推导

模型选择本质上是一个“先排序，再确认”的过程。排序靠验证集，确认靠独立评估集。为什么不能一步到位？因为你同时比较了多个候选模型，验证集上分数最高的那个，往往多少带一点“运气成分”。候选越多，这种过度乐观越明显，这就是选择偏差。

设候选模型集合为 $\{m_1,m_2,\dots,m_M\}$，验证性能估计为 $\hat{\vartheta}_m(\mathcal V)$。最简单的规则是：

$$
m_{\text{val}}=\arg\max_m \hat{\vartheta}_m(\mathcal V)
$$

但真正对外报告的应是：

$$
\hat{\vartheta}_{m_{\text{val}}}(\mathcal E)
$$

而不是 $\hat{\vartheta}_{m_{\text{val}}}(\mathcal V)$。

如果验证估计存在标准误 $SE_m$，within-1SE 规则可以写成：

$$
\mathcal C=\left\{m:\hat{\vartheta}_m(\mathcal V)\ge \max_j \hat{\vartheta}_j(\mathcal V)-SE_{\text{best}}\right\}
$$

其中 $\mathcal C$ 是进入独立评估的候选集合，$SE_{\text{best}}$ 是验证最优模型的标准误。白话解释：只要一个模型离最优模型足够近，近到差距还没超过统计波动，就别急着淘汰。

看一个玩具例子：

| 模型 | 验证 F1 | 标准误 | 是否进入 within-1SE |
| --- | --- | --- | --- |
| A | 0.92 | 0.01 | 是 |
| B | 0.90 | 0.015 | 是，因为 $0.90 \ge 0.92-0.01$ |
| C | 0.87 | 0.01 | 否 |

这里 B 看起来比 A 差，但差距只有 0.02，而 A 本身的标准误就有 0.01。若数据量不大，这个差距不一定稳定。把 B 也送去独立评估，能降低“错过真正更稳模型”的概率。

再看一个真实工程例子。做推荐系统召回模型时，你可能有三个候选：

| 模型 | 验证 Recall@50 | P99 延迟 | 大小 | 初步结论 |
| --- | --- | --- | --- | --- |
| 双塔模型 V1 | 0.421 | 12ms | 180MB | 精度高但偏大 |
| 双塔模型 V2 | 0.417 | 8ms | 95MB | 接近最优，部署友好 |
| 蒸馏模型 S1 | 0.409 | 5ms | 42MB | 精度略低但非常轻 |

如果 V1 的标准误较大，V2 又处在 within-1SE 范围内，那么只看验证分数就直接上 V1，并不稳妥。正确做法是把 V1 和 V2 都导出、量化、在独立评估集和压测环境上重新跑。最后部署的可能是“验证略低，但量化后更稳、延迟更低、评估损失更小”的 V2。

这说明训练后处理不是“精度单指标竞赛”，而是“泛化、稳定性、延迟、体积”的联合决策过程。

---

## 代码实现

下面给一个可运行的 Python 玩具实现，展示“验证排序 -> within-1SE 筛选 -> 导出 -> 量化 -> 评估 -> 选择最终部署版本”的最小流程。这里不依赖深度学习框架，重点看函数边界和数据流。

```python
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class Candidate:
    name: str
    val_score: float
    val_se: float
    export_format: str = "onnx"
    size_mb: float = 0.0


def pick_within_1se(candidates: List[Candidate]) -> List[Candidate]:
    assert candidates, "candidates cannot be empty"
    best = max(candidates, key=lambda x: x.val_score)
    threshold = best.val_score - best.val_se
    selected = [c for c in candidates if c.val_score >= threshold]
    assert best in selected
    return selected


def export_model(candidate: Candidate) -> Dict:
    artifact = {
        "name": candidate.name,
        "format": candidate.export_format,
        "path": f"artifacts/{candidate.name}.{candidate.export_format}",
    }
    assert artifact["path"].endswith(".onnx")
    return artifact


def quantize_model(exported: Dict, mode: str = "int8") -> Dict:
    assert mode in {"int8", "int4"}
    quantized = {
        **exported,
        "quant_mode": mode,
        "quantized_path": exported["path"].replace(".onnx", f".{mode}.onnx"),
    }
    return quantized


def evaluate_on_eval_set(model_name: str) -> float:
    # 玩具结果：模拟独立评估集分数
    scores = {
        "A": 0.905,
        "B": 0.912,
        "C": 0.870,
    }
    assert model_name in scores
    return scores[model_name]


candidates = [
    Candidate(name="A", val_score=0.92, val_se=0.01, size_mb=180),
    Candidate(name="B", val_score=0.90, val_se=0.015, size_mb=95),
    Candidate(name="C", val_score=0.87, val_se=0.01, size_mb=42),
]

selected = pick_within_1se(candidates)
assert [c.name for c in selected] == ["A", "B"]

eval_results = {}
for c in selected:
    exported = export_model(c)
    quantized = quantize_model(exported, mode="int8")
    score = evaluate_on_eval_set(c.name)
    eval_results[c.name] = {
        "eval_score": score,
        "artifact": quantized["quantized_path"],
    }

final_name = max(eval_results, key=lambda name: eval_results[name]["eval_score"])
assert final_name == "B"
assert eval_results["B"]["artifact"].endswith(".int8.onnx")
print(final_name, eval_results[final_name])
```

这个例子里，A 在验证集更高，但 B 在独立评估集更好，因此最终部署版本应是 B。初学者最该学的是：代码结构里必须显式区分 `validation` 和 `evaluation` 两个阶段，不要写成一个混在一起的大函数。

如果换成真实工程流水线，通常会拆成下面几类模块：

| 模块 | 输入 | 输出 |
| --- | --- | --- |
| `score_on_validation` | checkpoint 列表、验证集 | 验证指标表 |
| `pick_candidates` | 验证指标表 | 候选名单 |
| `export_model` | checkpoint | ONNX/TorchScript |
| `quantize_or_prune` | 导出模型、校准数据 | 优化后模型 |
| `evaluate_deployable_artifact` | 优化后模型、评估集 | 最终报告 |
| `package_release` | 模型、配置、依赖、版本信息 | 部署包 |

真实工程里还要额外保存三类元数据：

1. 数据版本：训练/验证/评估集各自的快照 ID。
2. 代码版本：Git commit、tag、导出脚本版本。
3. 环境版本：Python 版本、CUDA、ONNX Runtime、TensorRT 等依赖版本。

否则你后面即使发现量化后掉点，也很难追溯是数据变了、代码变了，还是运行时变了。

---

## 工程权衡与常见坑

训练后处理真正难的地方，不是“会不会导出 ONNX”，而是要在精度、延迟、显存或内存、吞吐、可维护性之间做折中。量化、剪枝、蒸馏都能让模型更轻，但也都可能改变数值行为，所以优化后必须重新评估。

先看常见手段的工程含义：

| 手段 | 白话解释 | 主要收益 | 主要风险 |
| --- | --- | --- | --- |
| ONNX/TorchScript 导出 | 把训练框架里的模型转成部署格式 | 跨框架、易部署 | 导出前后算子语义差异 |
| INT8 量化 | 用 8 位整数近似浮点 | 降低延迟和体积 | 精度下降、校准不足 |
| INT4 量化 | 用 4 位表示权重 | 更省内存 | 精度更敏感，适用层有限 |
| 剪枝 | 删除不重要参数 | 降低体积和计算量 | 稀疏不一定带来真实加速 |
| 知识蒸馏 | 用大模型教小模型 | 小模型更接近大模型效果 | 训练流程更复杂 |

有几个坑最典型。

| 常见坑 | 现象 | 正确做法 |
| --- | --- | --- |
| 预处理泄露 | 验证分数异常高，上线后大跌 | 每折或每个训练切分内部单独 `fit` 预处理器 |
| 用验证分数当最终分数 | 报告过于乐观 | 最终性能必须来自独立评估集 |
| 只评估浮点模型，不评估量化模型 | 离线好看，线上掉点 | 对部署体重新跑全量回归 |
| 导出版本和训练版本脱节 | 无法复现或回滚 | 绑定 checkpoint、脚本版本、依赖和 tag |
| 只看精度不看时延 | 离线最优，线上超 SLA | 在同一批候选上联合比较精度和时延 |

预处理泄露是新手最容易踩的坑。比如你有 1000 条样本，想做 5 折交叉验证。如果你先在全部 1000 条数据上做 z-score 标准化，再切成 5 折，那么每个验证折都已经“知道了”全体数据的均值和方差。虽然你没用标签，但你仍然把验证分布的信息泄露给了训练流程。正确做法是每一折内只用训练子集去 `fit` 均值和方差，再拿这个变换去处理该折的验证子集。

再看一个真实工程坑。团队把 PyTorch checkpoint 导出为 ONNX，再做 INT8 PTQ，发现离线验证精度只掉了 0.2 个点，于是直接上线。但线上 A/B 测试中 CTR 掉了 1.8%。排查后发现两件事：一是校准集过小，激活范围估计不稳；二是线上使用的算子库版本与离线压测版本不一致。这个问题不是“量化一定不好”，而是“量化后没有把部署体当成新模型重新评估”。

因此，工程上的基本原则是：只要模型体发生变化，就视为新的候选版本，重新走评估和回归流程。

---

## 替代方案与适用边界

within-1SE 不是唯一方案，它适合“候选模型不算太多、验证波动比较明显、希望提高挑中好模型概率”的场景。但如果数据更少、候选更多，或者处在受监管行业，只评估一个模型往往不够稳。

下面是几种常见替代方案：

| 策略 | 核心思路 | 适用场景 | 代价 |
| --- | --- | --- | --- |
| 默认单模型评估 | 只评估验证最优模型 | 候选少、成本敏感 | 容易错过接近最优的模型 |
| within-1SE | 评估与最优足够接近的一组模型 | 验证波动较大 | 评估成本上升 |
| maxT/多重检验 | 同时评估多个模型并控制家族错误率 | 医疗、金融、监管场景 | 统计流程更复杂 |
| Nested CV | 外层评估、内层选模 | 小数据、重视无偏估计 | 训练成本很高 |
| Bootstrap 区间 | 对指标不确定性做重采样估计 | 关注稳定性而非单点最优 | 实现与解释更复杂 |

医疗场景是一个典型例子。数据少、样本获取贵、还要面对监管审查。此时常见做法不是“验证集第一名直接进生产”，而是把前 2 到 3 个候选一起做独立评估，并用多重检验控制误报风险。直白地说，团队宁可多花一些评估成本，也不愿因为一次偶然的验证排名，把错误模型推进正式应用。

另一方面，如果你做的是推荐广告粗排、每天都能拿到大量线上反馈，模型迭代频繁，评估成本又高，那么 within-1SE 往往比 nested CV 更实用。它没有那么“统计最严”，但在工程上更平衡。

还有一个边界要说清楚：INT4 并不是“INT8 的直接替代品”。从 TensorRT 文档看，INT4 更适合权重量化，而且支持边界比 INT8 更窄。白话就是，它很省空间，但不是所有模型、所有层、所有硬件路径都适合直接切过去。对零基础读者来说，最安全的顺序通常是：先把 FP32 部署打通，再做 INT8，最后再考虑 INT4、剪枝、蒸馏这类更激进优化。

---

## 参考资料

- [Westphal, Brannath. Evaluation of multiple prediction models: A novel view on model selection and performance assessment](https://pmc.ncbi.nlm.nih.gov/articles/PMC7270727/)
- [Sollini et al. Key concepts, common pitfalls, and best practices in artificial intelligence and machine learning: focus on radiomics](https://pmc.ncbi.nlm.nih.gov/articles/PMC9682557/)
- [NVIDIA TAO Toolkit: ModelOpt ONNX Backend (Static PTQ)](https://docs.nvidia.com/tao/tao-toolkit/latest/text/tao_quant/backends_modelopt_onnx.html)
- [NVIDIA TensorRT: Working with Quantized Types](https://docs.nvidia.com/deeplearning/tensorrt/10.15.1/inference-library/work-quantized-types.html)
- [Moscovich, Rosset. On the Cross-Validation Bias due to Unsupervised Preprocessing](https://academic.oup.com/jrsssb/article/84/4/1474/7073256)
