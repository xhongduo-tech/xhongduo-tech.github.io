## 核心结论

Shell 脚本的价值，不是“能把几条命令写进一个文件”，而是把实验管理里最重复、最容易出错的部分做成可复用流程。对 ML 实验来说，这个流程通常包括四件事：生成配置、启动训练、并行调度、异常清理。

Bash 里的变量，就是“给一个值起名字”；数组，就是“把一组同类值按顺序放在一起”；条件，就是“根据真假决定是否执行”；循环，就是“把同一动作重复多次”；函数，就是“把一段逻辑包装成可重复调用的命令”。这五类机制组合起来，已经足够覆盖多数单机实验批量运行场景。

一个最小玩具例子如下：

```bash
epochs=(1 2)
for epoch in "${epochs[@]}"; do
  echo "epoch=${epoch}"
done
```

这里数组 `epochs` 表示一组轮次，`for` 循环按顺序展开它。对初学者来说，这就是“超参数列表 -> 逐个执行”的最直接模型。

真正进入工程后，两个能力最关键：

| 能力 | 解决的问题 | 典型工具 |
| --- | --- | --- |
| 异常清理 | 训练失败后释放临时目录、锁文件、日志句柄 | `trap` |
| 并行提交 | 同时跑多组实验，提高 CPU/GPU 利用率 | `xargs -P`、`parallel --jobs` |

如果只记一个结论，可以记这个公式：

$$
\text{稳定实验流水线}=\text{参数化脚本}+\text{并行调度}+\text{统一清理出口}
$$

---

## 问题定义与边界

本文讨论的问题很具体：在没有 Airflow、Kubernetes、Ray、Slurm 这类调度系统的前提下，如何仅用 Bash 把多个实验稳定跑完。

典型目标有三类：

1. 一次性生成多份配置文件。
2. 顺序或并行执行多组训练命令。
3. 在报错、手动中断、脚本退出时，确保清理动作一定执行。

最小问题示例如下：

```bash
configs=("base" "tune")
for cfg in "${configs[@]}"; do
  ./run_train.sh "$cfg"
done
```

这段代码表达的结构很清楚：配置列表 `configs` 是输入，`run_train.sh` 是统一动作，循环负责把输入逐个送进动作中。

本文边界也要说明白：

- 只讨论 shell 层流水线，不讨论集群调度。
- 只讨论单机或单节点上的批量实验管理。
- 重点是脚本结构、异常处理、并发控制，不展开训练框架本身。
- 默认解释器是 Bash，不讨论 `sh`、`zsh`、`fish` 的兼容差异。

这意味着，如果你的需求已经变成“跨多台机器调度 5000 个任务”，Shell 仍能做入口层，但不再是完整解法。

---

## 核心机制与推导

先看脚本为什么能管理实验。原因不是 Bash “强大”，而是实验任务本身可被拆成有限步骤：

1. 用变量保存路径、设备号、输出目录。
2. 用数组保存多个实验配置。
3. 用循环逐个提交。
4. 用条件跳过已经完成的实验。
5. 用函数封装日志、校验、清理。
6. 用 `trap` 把异常出口统一起来。

例如：

```bash
DATA_DIR="./data"
OUTPUT_DIR="./runs"
configs=("base" "tune" "ablation")

run_one() {
  local cfg="$1"
  echo "run $cfg"
}
```

这里 `local` 的意思是“变量只在函数内部有效”，可以避免函数改坏全局状态。

### `trap` 为什么关键

`trap` 的作用是“当某个信号或退出事件发生时，先执行指定命令”。对实验脚本来说，它的核心价值是：不要把清理动作散落在每个失败分支里，而要放到统一出口。

```bash
cleanup() {
  echo "cleanup temp files"
}

trap 'cleanup' ERR EXIT SIGINT
```

这里三个触发项可以先粗略理解成：

| 触发项 | 白话解释 | 常见场景 |
| --- | --- | --- |
| `ERR` | 某条命令失败时触发 | 训练脚本返回非 0 |
| `EXIT` | 脚本结束时触发 | 正常结束或提前 `exit` |
| `SIGINT` | 收到中断信号时触发 | 按 `Ctrl+C` |

这并不意味着任何场景都该把三者绑在一起。因为 `EXIT` 和 `ERR` 可能重复触发同一清理逻辑，所以 `cleanup` 最好设计成幂等的。幂等的意思是“执行一次和执行多次结果一致”。删除临时目录时用 `rm -rf`，就是典型幂等操作。

### 并行为什么能提速

如果单个实验耗时为 $t$，总共有 $N$ 个实验，串行总时间近似为：

$$
T_{\text{serial}} = N \cdot t
$$

若机器可同时运行 $P$ 个相互独立任务，理想并行时间近似为：

$$
T_{\text{parallel}} \approx \left\lceil \frac{N}{P} \right\rceil \cdot t
$$

真实情况会更慢，因为存在 I/O、显存争用、日志写入冲突等开销，但这个式子足够帮助初学者理解为什么并行值得做。

`xargs -P` 和 GNU `parallel` 都能实现并行提交。重点不是“哪一个更高级”，而是理解它们都在做同一件事：从输入列表中取任务，开多个子进程并发执行。

玩具例子：

```bash
printf '%s\n' exp1 exp2 exp3 exp4 | xargs -P 2 -n 1 echo run
```

- `-P 2` 表示最多并发 2 个进程。
- `-n 1` 表示每次只取 1 个参数。
- 如果没有 `-n 1`，多个参数可能被拼到同一条命令里，含义会变掉。

真实工程里，这个差别可能直接导致“一张 GPU 同时被多个任务误占用”。

---

## 代码实现

下面给出一个完整脚本模板，覆盖数组、函数、Here Document、`trap`、并行提交这五个核心点。

```bash
#!/usr/bin/env bash
set -Eeuo pipefail

WORKDIR="./runs"
TMPDIR="$(mktemp -d)"
JOBS=2
configs=("base" "tune" "ablation")

cleanup() {
  rm -rf "$TMPDIR"
}

trap 'cleanup' EXIT ERR SIGINT

make_config() {
  local cfg="$1"
  local file="$WORKDIR/${cfg}.yaml"

  mkdir -p "$WORKDIR"

  cat <<'EOF' > "$file"
train:
  epochs: 3
  batch_size: 32
optimizer:
  lr: 0.001
EOF
}

run_one() {
  local cfg="$1"
  local config_file="$WORKDIR/${cfg}.yaml"
  local log_file="$WORKDIR/${cfg}.log"

  if [[ -f "$log_file.done" ]]; then
    echo "skip $cfg"
    return 0
  fi

  echo "start $cfg"
  bash ./run_train.sh "$config_file" > "$log_file" 2>&1
  touch "$log_file.done"
}

export WORKDIR
export -f run_one

for cfg in "${configs[@]}"; do
  make_config "$cfg"
done

printf '%s\n' "${configs[@]}" | xargs -P "$JOBS" -n 1 bash -lc 'run_one "$@"' _
```

这里有几个实现点要看懂。

第一，`set -Eeuo pipefail` 是 Bash 的严格模式组合，用来尽早暴露问题：
- `-e`：命令失败时退出。
- `-u`：未定义变量时报错。
- `-o pipefail`：管道中任一命令失败都算失败。
- `-E`：让 `ERR` trap 在函数中也能传播。

第二，Here Document 就是“把多行文本直接写进文件”的语法。上面的：

```bash
cat <<'EOF' > "$file"
...
EOF
```

把分隔符写成 `'EOF'`，含义是“正文中变量不要提前展开”。这对生成配置文件非常重要。否则像 `$HOME`、`$TOKEN` 这样的字符串会在写入时被替换，结果可能和你想要的不一致。

第三，真实工程例子通常不是 `echo run`，而是批量生成配置后再统一发给训练脚本。例如一个视觉分类项目可能需要同时跑：

- 不同学习率：`1e-3`、`3e-4`
- 不同随机种子：`42`、`123`
- 不同数据集切分：`fold0`、`fold1`

这时脚本负责把“实验组合”翻译成“配置文件 + 日志路径 + 命令行参数”，训练框架只关心执行单次任务。

为了帮助初学者验证思路，下面给一个可运行的 Python 玩具例子。它不替代 Shell，而是演示“配置列表 -> 笛卡尔积 -> 任务数量校验”的逻辑。

```python
from itertools import product

lrs = [1e-3, 3e-4]
seeds = [42, 123]
folds = ["fold0", "fold1"]

jobs = []
for lr, seed, fold in product(lrs, seeds, folds):
    jobs.append({"lr": lr, "seed": seed, "fold": fold})

assert len(jobs) == 8
assert jobs[0] == {"lr": 1e-3, "seed": 42, "fold": "fold0"}
assert jobs[-1] == {"lr": 3e-4, "seed": 123, "fold": "fold1"}

print(jobs[:2])
```

这个例子对应到 Bash，就是数组展开和循环嵌套。Shell 不擅长复杂数据结构，但很适合作为命令编排层。

如果机器安装了 GNU `parallel`，脚本还能进一步写成：

```bash
parallel --jobs 4 run_train.sh ::: "${configs[@]}"
```

它的读法是：把 `configs[@]` 中每个元素依次代入 `run_train.sh`，同时最多开 4 个任务。

---

## 工程权衡与常见坑

Bash 适合做编排，不适合做复杂业务逻辑。这个边界不清楚，就会把脚本写成一团无法维护的字符串拼接。

下面是实验管理里最常见的坑：

| 坑 | 影响 | 规避 |
| --- | --- | --- |
| `xargs` 不加 `-n 1` | 多个参数被拼成一条命令，任务边界错乱 | 并行提交时默认加 `-n 1` |
| Here Document 不写 `'EOF'` | 变量提前展开，配置内容被污染 | 需要原样输出时使用带引号分隔符 |
| `trap` 里写复杂逻辑 | 中断响应变慢，失败路径更难排查 | `trap` 里只做轻量清理 |
| 未引用变量 | 路径中有空格时命令拆词 | 一律写成 `"$var"` |
| 没有幂等清理 | 重复触发 `cleanup` 时报错 | 删除、解锁等动作设计成可重复执行 |
| 并行数大于资源数 | GPU/CPU 争用，训练变慢甚至 OOM | `JOBS` 与硬件资源对齐 |

还有两个常见误区。

第一个误区是把 `trap 'cleanup; exit 1' ERR` 当成万能模板。`ERR` 本来就是失败路径，再额外写复杂退出逻辑，往往会让错误堆栈更难看。多数时候，把失败交给 `set -e`，把清理交给 `trap`，职责更清晰。

第二个误区是误以为并行越高越快。假设单卡训练任务显存峰值是 18GB，而 GPU 总显存是 24GB，那么同卡并行两个任务大概率只会互相挤压。此时虽然命令层面是并发，系统层面却进入频繁等待，反而更慢。

一个经验公式是：

$$
P \le \min(\text{CPU 可承受并发}, \text{GPU 可承受并发}, \text{I/O 可承受并发})
$$

并行数 `P` 必须由最紧的资源约束决定，而不是由“想同时跑几个”决定。

---

## 替代方案与适用边界

Bash 不是唯一方案。更准确地说，它是“最轻的编排方案”之一。

和 Python 相比，可以这样看：

| 方案 | 优点 | 缺点 | 适合场景 |
| --- | --- | --- | --- |
| Bash | 无额外依赖、直接调用系统命令、部署简单 | 数据结构弱、调试复杂、字符串易出错 | 单机批处理、训练节点临时编排 |
| Python + `subprocess` | 结构更清晰、数据处理强、易写复杂逻辑 | 需要运行环境、脚本入口更重 | 复杂实验组合、状态管理、结果汇总 |
| Makefile | 适合表达依赖关系、增量执行自然 | 动态参数与并发控制不如脚本灵活 | 数据处理管线、固定目标构建 |

如果实验数量少，甚至只需要：

- 生成几个配置
- 指定几张 GPU
- 写日志到不同目录

那么 Bash 往往已经足够，而且更容易审计。所谓审计，就是“你能一眼看出脚本会执行哪些命令”。这在训练节点上尤其重要。

如果需求变成以下任一情况，就应考虑 Python：

1. 实验组合规则复杂，涉及多层条件和数据转换。
2. 需要读取 JSON/YAML 后动态修改配置。
3. 需要失败重试、任务状态持久化、结果聚合。
4. 需要和数据库、消息队列、远程 API 交互。

举个替代思路的玩具例子，Python 可以这样并发执行命令：

```python
from concurrent.futures import ThreadPoolExecutor
import subprocess

configs = ["base", "tune"]

def run(cfg):
    return subprocess.run(["python", "-c", f"print('run {cfg}')"], check=True)

with ThreadPoolExecutor(max_workers=2) as ex:
    list(ex.map(run, configs))
```

这个版本在结构上更清晰，但代价是你必须维护 Python 环境，并处理更多运行时依赖。

所以适用边界可以总结为一句话：如果任务本质是“命令编排”，优先用 Bash；如果任务本质是“程序逻辑”，优先用 Python。

---

## 参考资料

| 资料 | 说明 | 作用 |
| --- | --- | --- |
| GNU Bash Reference Manual | Bash 官方手册 | 查变量、数组、函数、条件、`trap` 语义 |
| GNU Parallel Manual | GNU Parallel 官方文档 | 查 `--jobs`、输入替换、输出控制 |
| Parallel Processing with `find` and `xargs` | `xargs -P` 示例文章 | 理解并行参数与任务拆分 |
| SS64 Here Documents | Here Document 语法说明 | 理解 `<<EOF` 与 `<<'EOF'` 的区别 |
| Bash 初学教程类资料 | 基础语法讲解 | 适合先建立变量、循环、函数直觉 |

查文档时建议直接搜这些关键词：

- `bash trap ERR EXIT SIGINT`
- `bash arrays loop`
- `xargs -P -n 1`
- `gnu parallel jobs`
- `here document quoted delimiter`

如果要验证某条语法，不要只看博客摘要，优先回到官方手册确认边界条件，尤其是 `trap` 继承、严格模式、并行命令参数拆分这三类细节。
