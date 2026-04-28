## 核心结论

`safetensors` 是一种模型权重文件格式。白话说，它专门用来保存“张量数据”，不是用来保存任意 Python 对象。它的核心价值不是压缩模型，也不是让参数变少，而是把加载过程从“反序列化对象”改成“解析头部并映射原始字节”。

对初级工程师最重要的判断标准只有两个：

1. 如果你要分发、下载、部署“模型权重”，`safetensors` 通常比默认的 `pickle`/`torch.load` 路径更安全。
2. 如果你要保存的是“训练时的完整 Python 状态”，例如优化器对象、调度器对象、自定义类实例，它就不是首选。

新手可以先记住这个对照：`torch.load()` 更像“打开文件时，解释器顺手把文件里描述的对象重新构造出来”；`safetensors` 更像“先读目录，再按偏移去拿对应张量的字节”，不会额外执行对象构造逻辑。

文件结构可以抽象成：

$$
F = [8B\ header\_len][H][D]
$$

其中 `H` 是头部元数据，`D` 是连续的原始张量字节区。这个设计把“文件里有哪些张量、每个张量在什么位置、是什么 dtype、shape 是多少”显式写出来，因此更可审计、更容易做按需加载，也更适合多副本部署和冷启动优化。

| 方案 | 安全性 | 是否支持任意对象 | 是否支持 mmap / 按需读 |
|---|---|---|---|
| `pickle` | 弱 | 支持 | 不适合作为核心能力 |
| `torch.load` | 默认仍有反序列化边界问题 | 支持 | 支持 `mmap` 参数，但语义仍基于 PyTorch 序列化 |
| `safetensors` | 强 | 不支持 | 支持，且格式天然适合 |

---

## 问题定义与边界

先把问题讲清楚：模型权重文件的主要任务，不是“保存任意 Python 世界”，而是“安全、高效、稳定地保存张量”。

张量，白话说，就是一块带形状和数据类型的数值数组。模型参数本质上是很多张量。对部署场景来说，我们通常只关心这些张量能否正确读取，而不是把训练脚本里所有对象一起还原。

因此，真正的问题是：

- 文件能否在加载时避免执行未知代码
- 文件能否让加载器快速知道每个张量的位置
- 文件能否支持只读部分张量，而不是一次性把所有内容都拉进内存
- 文件出了问题时，能否容易审计和定位

这里要明确边界：`safetensors` 解决的是**格式安全**和**加载路径可控性**，不是完整供应链安全。

一个必须区分的例子是：

- 如果模型文件里混入恶意对象，`pickle` 路径可能在加载时触发代码执行。
- 如果模型文件本身权重被篡改成错误数值，`safetensors` 也照样能安全打开，只是模型结果会变差，甚至故意被“投毒”。

所以它不能替代这些事情：

- SHA256 校验
- 文件签名
- 制品仓库权限控制
- 来源审计
- 模型行为验证

下面这个表格是边界的核心：

| 能力 | `pickle` / `torch.load` | `safetensors` |
|---|---|---|
| 任意 Python 对象 | 支持 | 不支持 |
| 仅张量 + 字符串元数据 | 支持 | 支持 |
| 加载时执行代码风险 | 有 | 无 |
| 文件完整性验证 | 需外部配合 | 仍需外部配合 |

所以不要把“安全加载”理解成“文件绝对可信”。它更准确的含义是：**加载器不会因为解析文件而执行任意 Python 代码**。

---

## 核心机制与推导

`safetensors` 的格式非常直接。它先放 8 字节头部长度，再放 JSON 头部，最后放连续字节区。头部里每个张量都会写清楚：

- 名称
- `dtype`
- `shape`
- `data_offsets`

`offset`，白话说，就是“这个张量对应的字节从哪里开始，到哪里结束”。

单个张量占多少字节，可以用这个公式：

$$
bytes(T_i) = numel(T_i) \times sizeof(dtype_i)
$$

其中 `numel` 是元素总数，`sizeof(dtype)` 是每个元素的字节数。整个文件大小近似为：

$$
|F| \approx 8 + |H| + \sum_i bytes(T_i)
$$

这说明一个关键事实：**格式不会减少权重本身大小**。如果一个 `float32` 权重需要 4 字节，它在 `safetensors` 里还是 4 字节。变化的是“怎么组织”和“怎么读取”。

先看一个玩具例子。

设有两个张量：

- `A.shape = [2, 2]`，`dtype=float32`
- `B.shape = [2, 2]`，`dtype=float32`

那么：

- `A.numel = 4`，占 `4 × 4 = 16B`
- `B.numel = 4`，占 `16B`

总数据区大小是 `32B`。如果头部 JSON 占 `120B`，则总文件大小约为：

$$
8 + 120 + 32 = 160B
$$

这个例子说明两件事：

1. `safetensors` 不是压缩格式。
2. 它把张量描述信息和原始字节区分开了，因此加载器可以先读目录，再决定读哪些数据。

加载路径通常可以画成这样：

`header_len -> header JSON -> tensor offsets -> mmap -> lazy read -> CPU -> GPU`

这里有三个关键词。

`mmap`，白话说，就是“把文件映射进进程地址空间，按访问再触发实际读盘”，不是一开始把整个文件完整复制到用户态缓冲区。

`lazy read`，白话说，就是“先不真的把所有张量都读出来，只有访问到某个区域时才取那部分数据”。

`zero-copy` 在这里也要严格理解。它更多是指 CPU 侧可以减少一次中间拷贝，或者直接基于文件映射使用底层存储；**不是说数据能零拷贝直接进 GPU**。只要最终张量要进显存，就仍然有 CPU 到 GPU 的传输成本。

真实工程例子可以看多副本推理服务。假设你在 K8s 上滚动发布一个大模型，多个 Pod 同时冷启动。使用 `safetensors` 的好处通常是：

- 不再通过 `pickle` 反序列化路径加载外部权重
- 可以先读头部，再按需加载当前分片需要的张量
- CPU 峰值内存和启动抖动通常更容易控制
- 安全审计更简单，因为文件结构明确，风险边界更清晰

如果模型做张量并行，不同 GPU 进程只需要自己负责的切片，那么“只加载需要的部分”会比“把整个 checkpoint 都反序列化出来再切分”更合理。

从机制上说，它像下面这个伪代码：

```text
header = read_header(file)
index = resolve_offsets(header)
tensor = load_tensor_slice(index["layer0.weight"])
```

整个过程的重点是“解析结构”和“按偏移取字节”，不是“执行对象恢复逻辑”。

---

## 代码实现

最小可用写法是 `safe_open`。它返回的是一个按键访问张量的读取器，而不是一个任意 Python 对象图。

```python
from math import prod

def tensor_nbytes(shape, dtype_size):
    return prod(shape) * dtype_size

a = tensor_nbytes((2, 2), 4)   # float32
b = tensor_nbytes((2, 2), 4)
header_len = 120
file_size = 8 + header_len + a + b

assert a == 16
assert b == 16
assert file_size == 160

print(file_size)
```

上面这段代码可直接运行。它验证了前面的玩具例子：格式变化不会减少参数字节数，只会改变文件布局和读取方式。

下面是 `safetensors` 的典型读取方式：

```python
from safetensors import safe_open

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    keys = list(f.keys())
    first_key = keys[0]
    tensor = f.get_tensor(first_key)

    print("num_keys =", len(keys))
    print("first_key =", first_key)
    print("shape =", tuple(tensor.shape))
    print("dtype =", tensor.dtype)
```

如果你要按需读取，而不是把所有权重一次性放进字典，也可以只取单个张量：

```python
from safetensors import safe_open

target = "model.layers.0.attn.q_proj.weight"

with safe_open("model.safetensors", framework="pt", device="cpu") as f:
    if target in f.keys():
        w = f.get_tensor(target)
        print(target, w.shape)
```

对照写法通常是：

```python
import torch

state = torch.load("model.pt", map_location="cpu")
print(type(state))
```

两者差异不是“API 长相”，而是“边界”：

| 对比项 | `torch.load` | `safe_open` |
|---|---|---|
| 调用方式 | 直接加载对象 | 打开后按 key 取张量 |
| 返回内容 | 可能是任意 Python 对象 | 张量读取接口 |
| 风险点 | 反序列化路径更复杂 | 不执行任意对象构造 |
| 按需读单个 tensor | 取决于保存内容和加载策略 | 天然支持 |
| 审计性 | 对对象图依赖更强 | 头部结构明确 |

还要补一个常被忽略的点：即使继续使用 `torch.load`，也应关注 `weights_only=True`。它的意义是限制 unpickler，只允许更窄的一类内容被恢复，安全边界比无约束反序列化更强。但这仍然不是 `safetensors` 的等价替代，因为它还是站在 PyTorch 序列化语义里，而不是“只存张量字节 + 显式元数据”的格式约束里。

---

## 工程权衡与常见坑

`safetensors` 的收益真实存在，但不要把它神化。

第一，安全不等于可信内容。一个被投毒的权重文件，仍然可能让模型输出异常、偏置结果，或者故意在某些输入上失效。它只是不会在加载阶段顺手执行恶意 Python 代码。

第二，性能收益主要在 CPU 侧启动路径，不在 GPU 神奇加速。数据最终要进显存，仍然要做传输。所以它更适合优化：

- 冷启动时间抖动
- CPU 峰值内存
- 多副本同时拉起时的稳定性
- 多分片加载的可控性

第三，它不适合保存复杂对象状态。比如优化器、调度器、自定义 sampler、训练上下文，这些都不是它要解决的问题。

常见坑可以直接看表：

| 坑 | 现象 | 规避 |
|---|---|---|
| 误以为可存任意对象 | 保存设计失配，后续恢复不了训练上下文 | 只把它用于 tensor + 字符串 metadata |
| 误以为零拷贝到 GPU | 仍然有显存传输成本 | 只把 `mmap` 当 CPU 侧优化 |
| 头部过大 | 加载时报 header 相关错误 | 控制 metadata 规模，不塞大段文本 |
| dtype / 对齐异常 | 读取失败或切片异常 | 优先使用标准 dtype 和常规张量布局 |
| 没做完整性校验 | 文件被篡改难发现 | 配合 SHA256、签名或制品仓库校验 |

还有一个工程上常见误判：有人看到 `.safetensors` 比 `.bin` 加载更稳，就以为它“更快且更小”。前半句可能在启动路径上成立，后半句通常不成立。它不是量化，也不是压缩。真正决定文件大小的是参数量和 dtype，例如 `float32`、`float16`、`bfloat16`。

---

## 替代方案与适用边界

如果你的目标是“推理部署时安全地加载权重”，`safetensors` 往往是默认优先项。如果你的目标是“保存一个复杂训练 checkpoint，未来完整恢复 Python 对象图”，那它不是主角。

可以这样选：

| 方案 | 适合场景 | 优点 | 缺点 |
|---|---|---|---|
| `safetensors` | 权重分发、推理加载、模型仓库 | 安全、可审计、支持按需读与 `mmap` | 不能存任意对象 |
| `torch.save` / `pickle` | 训练检查点、复杂对象恢复 | 灵活，兼容历史代码 | 反序列化风险更高 |
| 外部签名 + 任意格式 | 供应链验证 | 能补文件完整性与来源可信度 | 不能替代格式本身的安全边界 |

一个简单判断法是：

- 你只要“模型权重”：优先 `safetensors`
- 你要“训练现场完整快照”：可能还得用 `torch.save`
- 你担心“文件是否被篡改”：必须再加签名或哈希校验

从团队协作角度看，`safetensors` 特别适合这些边界清晰的场景：

- 开源模型权重分发
- 企业内部模型制品仓库
- 推理服务冷启动优化
- 多 GPU / 多节点按需加载
- 需要做安全审计的部署链路

不太适合的场景则是：

- 强依赖 Python 自定义对象恢复
- 老项目里 checkpoint 结构极其复杂
- 训练恢复流程高度绑定历史 `pickle` 对象图

最后给一个阅读顺序表，方便你继续查官方资料：

| 阅读顺序 | 看什么 | 目的 |
|---|---|---|
| 1 | `safetensors` 官方 README / docs | 先理解格式与目标 |
| 2 | `safe_open` 用法 | 理解按 key 读取和按需加载 |
| 3 | PyTorch `torch.load` 文档 | 对照安全边界与 `mmap` 参数 |
| 4 | PyTorch serialization notes | 理解 `weights_only=True` 的限制与意义 |

---

## 参考资料

1. [Safetensors 官方仓库 README](https://github.com/huggingface/safetensors)
2. [Hugging Face Safetensors 文档](https://huggingface.co/docs/safetensors/index)
3. [PyTorch `torch.load` 文档](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
4. [PyTorch Serialization Semantics](https://docs.pytorch.org/docs/stable/notes/serialization.html)
5. [PyTorch Foundation: Safetensors 项目页](https://pytorch.org/projects/safetensors/)
