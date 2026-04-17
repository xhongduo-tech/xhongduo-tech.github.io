## 核心结论

Makefile 的核心价值，不是“替代脚本”，而是把一组有先后关系的命令写成一张可重复执行的依赖图。规则的基本结构是 `target: prerequisites` 加上一段 `recipe`。`target` 是目标，也就是你想得到的结果；`prerequisites` 是前置条件，也就是生成结果前必须先准备好的文件；`recipe` 是真正执行的命令，也就是 shell 里会跑的那几行。

对 AI 项目来说，这个结构非常实用。训练、评测、导出模型、部署，本质上都是“生成某个产物”的步骤。Make 不关心你是在编译 C 程序还是训练神经网络，它只关心一件事：目标文件是否比依赖文件旧。若旧，就重跑；若新，就跳过。可以把它写成一个判断公式：

$$
\text{run recipe if } t(\text{target}) < \max \{ t(\text{prereq}_1), t(\text{prereq}_2), \dots \}
$$

这里的时间戳，就是文件最后修改时间的记录。这个机制决定了 Make 适合“增量执行”，也就是只重跑必要步骤，而不是每次从头跑完整流水线。

一个直接结论是：在 AI 实验里，用 Makefile 统一训练、评测、部署入口，能把“少记命令、减少重复、固定流程”三件事同时解决。对个人项目，它是实验自动化工具；对小团队，它还是一种轻量的流程约定。

---

## 问题定义与边界

AI 项目的常见问题，不是不会写训练脚本，而是流程散。数据下载一条命令，预处理一条命令，训练一条命令，评测又一条命令。久而久之，项目变成“只有作者自己知道怎么跑”。

这个问题的边界要先说清。Makefile 适合管理“由文件变化驱动的任务链”。如果某一步的输入输出能落到文件上，比如 `data/raw.csv`、`artifacts/model.pt`、`reports/metrics.json`，Make 就能很好地工作。反过来，如果任务完全依赖数据库状态、远程服务响应、消息队列事件，文件时间戳就不够用了，Make 的判断会变得粗糙。

下面这张表可以概括它解决的问题范围：

| 问题 | 表现 | Makefile 能力 |
| --- | --- | --- |
| 重复手动执行命令 | 每次都手敲训练、评测、导出 | 用依赖图串成固定入口 |
| 不知道该先跑什么 | 新同事或未来的自己容易漏步骤 | 依赖链显式声明先后顺序 |
| 全量重跑太慢 | 改了评测脚本却把训练也重跑 | 只在目标过期时触发重算 |
| 非文件任务混在一起 | `clean`、`deploy` 这类命令行为特殊 | 用 `.PHONY` 明确声明“这不是文件目标” |

`.PHONY` 首次出现时要解释一下。它的意思是“伪目标”，白话说就是：这个名字代表一个动作，不代表一个真实文件。比如 `clean` 是删除产物的动作，不应该拿它和磁盘上的 `clean` 文件比时间戳。

一个玩具例子很能说明问题。假设你只有两个步骤：先训练，再评估。

```make
DATA=../data/dataset.csv
MODEL=models/latest.pt

train: $(MODEL)

$(MODEL): $(DATA)
	python train.py --data=$< --out=$@

evaluate: $(MODEL)
	python eval.py --model=$<
```

这里 `$(MODEL)` 依赖 `$(DATA)`。如果数据集没变，`make train` 会直接说目标已是最新；如果数据集更新，训练才会重跑。接着 `evaluate` 依赖模型文件，所以它天然接在训练之后。

---

## 核心机制与推导

Make 的规则语法可以拆成三段：

1. 目标 `target`
2. 依赖 `prerequisites`
3. 命令 `recipe`

例如：

```make
metrics.json: model.pt eval.py
	python eval.py --model model.pt --out metrics.json
```

这句话的含义是：`metrics.json` 这个结果，需要 `model.pt` 和 `eval.py` 先准备好；如果 `metrics.json` 比它们更旧，就执行下面这条命令。

为什么这件事对实验自动化有用？因为实验流程天然是一个有向图。训练要等数据处理完成，评测要等模型产出，部署要等导出完成。Make 用文件依赖替代了“人在脑子里记步骤”。

再往前推一步，Make 还有两个关键机制。

第一是隐式规则和模式规则。模式规则可以理解为“带占位符的规则模板”。`%` 是通配符，白话说就是“名字可变，但结构相同”。

```make
models/%.pt: configs/%.yaml data/processed.csv
	python train.py --config=$< --data=data/processed.csv --out=$@
```

如果你执行 `make models/resnet.pt`，Make 会自动匹配到 `configs/resnet.yaml`。这让“同一类实验的多个变体”可以共用一套规则，而不是把每个模型都手写一遍。

第二是 `.PHONY`。它不参与时间戳判断，而是每次都执行。这适合环境准备、清理、部署这类动作。

```make
.PHONY: clean deploy
clean:
	rm -rf artifacts reports

deploy:
	python scripts/deploy.py --model artifacts/model.pt
```

这里有一个常见误区：新手会把所有目标都写成 `.PHONY`。这会破坏 Make 最有价值的能力，也就是增量执行。训练和评测这类本来可以通过产物文件判断是否过期的步骤，不应该轻易写成 `.PHONY`。

用一个更直观的“玩具推导”看增量执行：

| 目标 | 依赖 | 是否需要重跑 |
| --- | --- | --- |
| `model.pt` | `dataset.csv` | 当 `dataset.csv` 更新后重跑 |
| `metrics.json` | `model.pt`, `eval.py` | 当模型或评测脚本更新后重跑 |
| `clean` | 无 | 每次都执行，因为它是 `.PHONY` |

这意味着，Make 并不是“执行脚本的另一个入口”，而是在维护一条最小重跑路径。

---

## 代码实现

先给一个可运行的 Python 玩具例子，用来模拟 Make 的核心判断逻辑。这里不是真正调用 `make`，而是把“目标是否过期”的规则写出来。

```python
from dataclasses import dataclass

@dataclass
class Node:
    name: str
    ts: int  # 时间戳，数字越大表示越新

def should_run(target: Node, prereqs: list[Node]) -> bool:
    if not prereqs:
        return False
    newest_prereq = max(p.ts for p in prereqs)
    return target.ts < newest_prereq

data = Node("dataset.csv", 10)
model = Node("model.pt", 8)
metrics = Node("metrics.json", 9)
eval_script = Node("eval.py", 11)

assert should_run(model, [data]) is True
assert should_run(metrics, [model, eval_script]) is True

new_model = Node("model.pt", 12)
new_metrics = Node("metrics.json", 13)

assert should_run(new_model, [data]) is False
assert should_run(new_metrics, [new_model, eval_script]) is False
```

这个例子说明了 Make 最重要的语义：是否执行，不由“你想不想再跑一次”决定，而由“目标是否比依赖旧”决定。

下面给一个更贴近 AI 项目的 Makefile。这个例子里，训练、评测、导出被统一成标准入口。

```make
PYTHON=.venv/bin/python
RAW=data/raw.csv
PROC=data/processed.parquet
MODEL=artifacts/model.pt
METRICS=reports/metrics.json
ONNX=artifacts/model.onnx

.PHONY: all env clean
all: $(METRICS) $(ONNX)

env:
	python -m venv .venv
	$(PYTHON) -m pip install -r requirements.txt

$(RAW):
	$(PYTHON) scripts/download.py --out $@

$(PROC): $(RAW) scripts/preprocess.py
	$(PYTHON) scripts/preprocess.py --input $< --out $@

$(MODEL): $(PROC) train.py
	$(PYTHON) train.py --data $< --out $@

$(METRICS): $(MODEL) eval.py
	$(PYTHON) eval.py --model $< --out $@

$(ONNX): $(MODEL) export_onnx.py
	$(PYTHON) export_onnx.py --model $< --out $@

clean:
	rm -rf data artifacts reports
```

真实工程里，这种结构有两个价值。

第一，命令入口统一。团队成员只需要记住 `make all`、`make clean`、`make reports/metrics.json` 这些目标，不必记住一串脚本参数。

第二，步骤的“可审计性”更强。所谓可审计，就是别人能看出流程到底怎么跑。Makefile 把实验顺序和依赖写成了代码，而不是藏在 README 的文字说明里。

一个真实工程例子是视觉问答或大模型微调项目。常见流程包括：下载数据、安装依赖、训练模型、运行评测、导出结果。若把每一步都做成目标，就可以做到“训练是模型文件的生成过程，评测是指标文件的生成过程，部署是导出包的生成过程”。这比把所有动作堆到一个超长 shell 脚本里更清晰，因为每个产物和依赖都可见。

---

## 工程权衡与常见坑

Makefile 好用，但它的边界和坑也很明确。

| 坑 | 现象 | 规避方式 |
| --- | --- | --- |
| 命令前用了空格而不是 TAB | 报 `missing separator` | 配方行必须用 TAB，或显式设置 `.RECIPEPREFIX` |
| 忘记写 `.PHONY` | `make clean` 可能不执行 | 对非文件动作统一声明 `.PHONY` |
| 依赖链写不全 | 评测读到旧模型，或跳过预处理 | 明确写出真实输入输出关系 |
| 目标没有实际产物 | 每次都难以判断是否需要重跑 | 尽量让步骤落到文件上 |
| 把训练目标写成 `.PHONY` | 每次都重新训练，失去增量能力 | 训练目标应对应模型文件 |
| CMake 和 Make 双重管理 CUDA 细节 | 编译参数冲突，架构不一致 | 让 CMake 管编译，Make 管统一入口 |

这里重点说 `.PHONY` 的误用。下面这个写法表面没错，实际有风险：

```make
clean:
	rm -rf results/*.json
```

如果项目目录里刚好有个名为 `clean` 的文件，Make 会以为目标已经存在，于是直接说 “Nothing to be done”。正确写法是：

```make
.PHONY: clean
clean:
	rm -rf results/*.json
```

另一个工程上很重要的权衡，是 Make 和 CMake 的分工。CMake 是跨平台构建系统生成器，白话说，它负责为不同平台生成合适的构建规则，尤其擅长 C++、CUDA 这类编译型项目。AI 项目里如果有自定义 CUDA 扩展，例如 PyTorch extension，通常不应让 Make 直接手写 `nvcc` 参数，而应由 CMake 管理 `CMAKE_CUDA_ARCHITECTURES`、CUDA 标准版本、链接选项等细节。Make 更适合站在外层，调用 `cmake --build` 或上层目标，把“编译扩展、训练模型、跑评测”串成统一入口。

---

## 替代方案与适用边界

Makefile 不是唯一方案，只是小而强的方案。

| 方案 | 优势 | 适用边界 |
| --- | --- | --- |
| 纯 Makefile | 简单、依赖关系可见、学习成本低 | 单机实验、小团队原型、文件驱动流程 |
| CMake + Make/Ninja | 擅长 C++/CUDA、多平台编译 | 有自定义算子、原生扩展、复杂编译链 |
| CI/CD + Makefile | 自动触发、记录可追踪 | 团队协作、持续测试、持续部署 |
| Airflow / Prefect / Dagster | 调度能力强、可管理远程任务 | 数据平台、分布式任务编排 |
| Bash 脚本 | 上手快 | 一次性脚本，难维护的长期项目不推荐 |

适用边界可以一句话概括：如果你的流程主要围绕“文件产物不断生成”，Makefile 很合适；如果你的流程主要围绕“服务状态、分布式调度、云资源编排”，Makefile 就偏轻了。

在 AI 项目里，一个很常见的组合是：

1. CMake 负责编译 CUDA 扩展。
2. Makefile 统一对外入口，例如 `make train`、`make eval`、`make package`。
3. CI 系统调用这些目标，把人手工执行过的流程搬到自动化流水线。

这种组合的好处是分层清楚。CMake 解决“怎么编”，Makefile 解决“先做什么再做什么”，CI 解决“什么时候自动做”。

---

## 参考资料

- GNU Make Manual: https://www.gnu.org/software/make/manual/make.html
- GNU Make Manual, Rule Syntax: https://www.gnu.org/software/make/manual/html_node/Rule-Syntax.html
- GNU Make Manual, Using Implicit Rules: https://www.gnu.org/software/make/manual/html_node/Using-Implicit.html
- Streamlining Data Science Projects: Implementing Code Development Best Practices with Makefile: https://medium.com/%40bhawanamathur_46573/streamlining-data-science-projects-implementing-code-development-best-practices-with-makefile-4e272e01b292
- VT-vision VQA Makefile 示例: https://gist.github.com/wecacuee/e56b6c8bc6c762e1440f
- NVIDIA, Building Cross-Platform CUDA Applications with CMake: https://developer.nvidia.com/blog/building-cuda-applications-cmake/
