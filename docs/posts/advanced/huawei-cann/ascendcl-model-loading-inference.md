---
title: 模型加载与推理执行
date: 2026-08-07
---

# 模型加载与推理执行

<div class="epigraph">
<p>真正的部署，是把知识变成每分钟运行的决策。</p>
<footer>—— 自推理部署格言</footer>
</div>

<div class="article-byline">
<p>第四级 · 华为 CANN 计算架构 ｜ 华为昇腾 CANN 开发指南 ｜ 2026-08-07</p>
</div>

## 为什么从模型加载与推理开始

前两节把设备、上下文、流、内存备齐了，接下来是昇腾程序的「主菜」：**加载模型、执行推理**。这是推理应用的终点——前面的所有准备，都是为了这一刻。昇腾上模型以 **`.om`（Offline Model，离线模型）** 文件形式存在：它由 ATC 在部署前把训练模型编译而成，把融合、布局、算子选择都固化在文件里。AscendCL 用 `aclmdl*` 系列接口负责模型的加载、描述查询与执行。理解这一节，你就拥有了「把一个模型放到昇腾上跑起来」的完整闭环。<span class="marginnote">`.om` 之于昇腾 ≈ TensorRT 的 engine 文件之于 NVIDIA：都是「编译一次、运行多次」的产物。推理部署的性能优化，大部分发生在「生成 .om 的那一步」，而不是运行期——这也是为什么生产推理几乎人手一个 ATC 转换流程。</span>

## 1 模型描述：先问清模型长什么样

执行推理前，必须先知道模型的「接口规格」：几个输入、什么形状、什么 dtype、几个输出。AscendCL 用**模型描述符（`aclmdlDesc`）**来承载这些信息：

**`aclmdlCreateDesc()`**：创建描述对象。
**`aclmdlGetDesc(desc, modelId)`**：从已加载模型取出描述。
**`aclmdlGetNumInputs / aclmdlGetInputDesc`**：查询输入个数与每个输入的描述（形状、类型）。

为什么这么麻烦？因为 `.om` 里的模型形状可能经过 ATC 的固定/动态化处理，**运行期必须以模型描述为准，而不是以「我记得的模型」为准**。写推理代码的第一步永远是「问模型要它的输入规格」——这是避免「形状不匹配」报错的根本方法。<span class="marginnote">「以模型描述为准」是一条重要纪律：模型经过 ATC 转换后，输入顺序、形状、dtype 都可能与训练时不同（比如加了 AIPP 的预处理、或形状被静态固定）。<strong>永远先查 `aclmdlGetDesc`，再按它分配内存</strong>，而不是硬编码你记忆中的形状。</span>

## 2 加载模型：两种方式

AscendCL 提供两种加载方式，对应不同的部署形态：

**`aclmdlLoadFromFile("model.om", &modelId)`**：从文件系统加载 `.om` 文件。最常用，模型文件随应用一起发布。

**`aclmdlLoadFromMem(buffer, size, &modelId)`**：从内存加载模型。适用于「模型以字节流形式分发」（如加密模型、从网络下载后直接加载），避免写临时文件。

两种方式都返回一个 `modelId`，后续执行都靠它。**`modelId` 是运行期唯一的模型句柄**——创建它、用它、最后 `aclmdlUnload(modelId)` 释放它，构成完整的模型生命周期。<span class="marginnote">`LoadFromMem` 在真实生产里并不罕见：安全要求高的场景会把 `.om` 加密存储，运行时解密到内存再加载；或从对象存储拉取模型字节流直接加载。理解「文件 vs 内存」两种入口，部署形态的选择就自由了。</span>

## 3 执行推理：同步与异步

模型加载好、输入输出内存就位后，执行推理有两个入口：

**`aclmdlExecute(modelId, input, output)`**：同步执行——阻塞主机直到推理完成，输出就绪。语义简单，适合首轮验证。

**`aclmdlExecuteAsync(modelId, input, output, stream)`**：异步执行——任务入队到指定流后立即返回。设备后台执行，主机可继续做事。

**易错点：异步执行后直接读输出，读到的是旧数据**。正确做法是读取前对该流同步（`aclrtSynchronizeStream`），或让输出拷贝排队在推理之后（流内有序，拷贝自然在推理后执行）。

同步/异步的选择与性能直接相关：**同步适合「单发、低频、求简单」**；**异步适合「流水线、多批、求吞吐」**——这正是下一节《同步/异步执行与事件同步机制》的主战场。<span class="marginnote">从同步改异步，是昇腾推理提速的第一级台阶：把「搬移、推理、搬回」三个环节拆到不同流上异步执行，重叠后的吞吐通常能翻倍。但异步也把「正确性」的担子压到了你身上——同步的保证被换成了事件的纪律。</span>

## 4 一个完整的推理调用序列

把上面所有环节串起来，一个最小推理调用的骨架是：

```c
aclInit(nullptr);
aclrtSetDevice(0);
aclrtCreateContext(&ctx, 0);
aclrtCreateStream(&stream);

aclmdlDesc* desc = aclmdlCreateDesc();
aclmdlLoadFromFile("resnet50.om", &modelId);
aclmdlGetDesc(desc, modelId);              // 查询输入规格

// 按描述申请输入输出内存（省略形状解析细节）
void *inputBuf = aclrtMalloc(...);
void *outputBuf = aclrtMalloc(...);

aclrtMemcpy(inputBuf, size, hostInput, size,
            ACL_MEMCPY_HOST_TO_DEVICE);    // 输入拷入设备

aclmdlExecute(modelId, inputBuf, outputBuf);   // 同步推理

aclrtMemcpy(hostOutput, size, outputBuf, size,
            ACL_MEMCPY_DEVICE_TO_HOST);    // 结果拷回

aclmdlUnload(modelId);                     // 对称释放
aclrtDestroyStream(stream); aclrtDestroyContext(ctx);
aclrtResetDevice(0); aclFinalize();
```

这个骨架你已经见过一次（编程模型篇），这一节它多了「模型描述」这一步。**新增的那一步，正是推理代码最容易出错的地方**——读模型规格，按规格准备数据。<span class="marginnote">把这套骨架与 CUDA 推理（cuDNN/TensorRT）对照：`LoadFromFile` ≈ `engine->deserialize`，`aclmdlGetDesc` ≈ 查 engine 的输入绑定，`aclmdlExecute` ≈ `enqueueV2`。语义一一对应，只是 API 名换了语言。</span>

## 5 公式解析：吞吐、时延与批处理

推理服务最终要回答「多快、多少」。设一次同步推理的时延为 $L$，批大小为 $B$，则**吞吐（每单位时间完成的样本数）** 与 **时延** 的关系为

$$
\text{吞吐} = \frac{B}{L}
$$

拆三步看：

- **第一步，$L$ 是单次时延**：从输入就绪到输出就绪的墙钟时间，同步执行下用户感知到的就是它。
- **第二步，$B$ 是批大小**：一次推理处理 $B$ 个样本。批量推理摊薄算子启动开销，$L$ 增长通常慢于 $B$ 的增长。
- **第三步，吞吐 = B/L**：当 $B$ 翻倍而 $L$ 只增长 1.5 倍，吞吐提升约 33%——**这是批处理带来的真实收益**。

**这条式子的工程含义是：在线服务优先压时延（小 B），离线批量优先压吞吐（大 B）**。昇腾上批量推理的收益来自「一次搬移、一次启动处理更多数据」——而批大小上限受设备内存约束，这与内存管理篇的公式互相呼应。

## 6 核心术语速查表

本节的术语集中在「模型加载与推理」语境，整理如下：

| 术语 | 含义 |
| --- | --- |
| .om | Offline Model，ATC 生成的离线模型文件 |
| aclmdlDesc | 模型描述符，承载输入输出规格 |
| modelId | 已加载模型的句柄，执行靠它寻址 |
| LoadFromFile | 从文件加载模型 |
| LoadFromMem | 从内存字节流加载模型 |
| aclmdlExecute | 同步推理，阻塞至完成 |
| aclmdlExecuteAsync | 异步推理，入队后立即返回 |
| aclmdlUnload | 卸载模型，释放资源 |
| 批大小 | 一次推理处理的样本数 |
| 时延 | 单次推理的墙钟时间 |
| 吞吐 | 每单位时间完成的样本数 |

## 7 小结

- 模型以 **`.om`** 形式发布；推理前先 `aclmdlGetDesc` 查清输入规格，**以模型描述为准**。
- 加载有两种方式：**`LoadFromFile`**（文件）与 **`LoadFromMem`**（内存字节流），都返回 `modelId`。
- 执行有两个入口：**同步 `aclmdlExecute`**（简单）与 **异步 `aclmdlExecuteAsync`**（高吞吐）；异步读输出前必须同步。
- 推理骨架 = 加载 → 查规格 → 准备数据 → 执行 → 拷回 → 卸载，与 AscendCL 六步骨架对齐。
- 吞吐 = $B/L$：**在线压时延、离线压吞吐**，批大小受内存约束。

在下一节，我们将补上「准备数据」这一环的常见痛点——把输入图像交给模型之前，如何用 **AIPP** 在硬件上完成裁剪、缩放与归一化。
