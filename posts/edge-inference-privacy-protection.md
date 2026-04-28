## 核心结论

边缘推理隐私保护的目标，不是“把模型挪到设备上就安全”，而是把推理链路中的原始输入、关键中间特征、模型参数暴露面压到最小，同时把准确率、时延、能耗维持在可用范围内。

边缘推理，白话说，就是模型的一部分或全部在手机、摄像头、网关这类靠近数据源的设备上执行。隐私保护，白话说，就是让不该看到明文数据的人和系统，看不到或尽量少看到。

最容易被误解的一点是：**不上传原图，不等于隐私已经安全**。即使原图不上传，只上传中间特征，这些特征仍然可能被反推出输入内容，或者被用来推断用户属性。

实际可落地的方案通常不是单一技术，而是组合式设计：本地计算一部分，减少外发数据；对外发特征做最小化；把敏感后半段放进 TEE；只有在威胁模型非常严格时，才对少量关键环节使用加密推理。

| 暴露面 | 典型内容 | 主要风险 | 常见保护手段 |
| --- | --- | --- | --- |
| 原始输入 | 图像、语音、文本、传感器流 | 直接泄漏身份、位置、行为 | 本地预处理、本地推理、传输加密 |
| 中间特征 | 卷积特征、embedding、token hidden state | 被重建输入、属性推断、成员推断 | 切分点优化、降维、扰动、TEE、部分加密 |
| 模型参数 | 权重、结构、量化表 | 模型盗取、反推出训练分布 | 参数加密存储、TEE、访问控制 |
| 输出 | 分类结果、检测框、分数 | 输出反推输入或成员信息 | 输出裁剪、阈值控制、最小化日志 |

---

## 问题定义与边界

这里讨论的是**推理阶段**的隐私保护，不是训练阶段的联邦学习，也不是一般意义上的磁盘加密或数据库脱敏。推理阶段，白话说，就是模型已经训练完成，系统只负责“拿输入，算输出”的这一步。

统一记号如下：

- $x$：原始输入，比如一张人脸图像
- $f$：完整模型
- $z = f_{1:k}(x)$：前 $k$ 层算出的中间特征
- $y = f_{k+1:n}(z)$：后续层产生的最终输出

可以把链路写成：

$$
x \rightarrow f_{1:k}(x)=z \rightarrow f_{k+1:n}(z)=y
$$

这条链路上，$x$、$z$、模型参数、日志、缓存、崩溃转储都可能成为泄漏点。新手版可以这样理解：**边缘推理的隐私问题，不是模型放在哪儿的问题，而是数据在每一跳怎么流动的问题。数据一旦离开设备，原图、特征、日志、调试信息都可能成为泄漏点。**

这个问题本质上是多目标优化，而不是单目标优化。系统通常同时受以下约束：

| 目标 | 含义 | 为什么不能单独优化 |
| --- | --- | --- |
| 隐私 | 减少明文和可逆信息暴露 | 隐私越强，往往计算越重 |
| 准确率 | 输出结果不能明显变差 | 过度压缩特征会伤害效果 |
| 时延 | 端到端响应要达标 | 安全机制可能增加等待 |
| 能耗 | 设备发热和续航要可控 | 深层本地计算会更耗电 |
| 端侧算力 | 手机/NPU/MCU 资源有限 | 不是所有模型都能搬到端上 |

因此更合理的提法不是“怎么做到绝对安全”，而是“在设备能力、网络条件、可信边界已知的前提下，怎么把暴露面压到最小”。

---

## 核心机制与推导

边缘推理隐私保护里最常见的三类机制是 Split Inference、HE 和 TEE。

**Split Inference**，白话说，就是把一个模型切成前后两段，设备先算前半段，再把特征交给服务器。形式化地写：

$$
z = f_{1:k}(x), \quad y = f_{k+1:n}(z)
$$

它的优点是直接、实用、低延迟，尤其适合端上算力不够但又不想上传原始输入的场景。问题在于，低带宽不等于高隐私。因为 $z$ 虽然比 $x$ 小，但如果 $z$ 仍保留大量结构信息，攻击者依然可能重建 $x$，或者从中推断性别、年龄、身份等属性。

**HE（同态加密）**，白话说，就是服务器在看不到明文的情况下直接做计算。基本形式是：

$$
c = Enc_{pk}(x), \quad c_y = Eval(F, c), \quad y = Dec_{sk}(c_y)
$$

这里 `Enc` 是加密，`Eval` 是“在密文上执行函数”，`Dec` 是解密。它保护的是“服务器看不到明文输入”这一点，理论边界最强，但代价通常也最大。很多神经网络结构在 HE 下需要改写成更适合密文计算的形式，比如减少非线性、控制乘法深度，否则时延会明显失控。

**TEE（可信执行环境）**，白话说，就是把敏感代码放进硬件隔离的小房间里运行。可以写成：

$$
y = f(x) \quad \text{inside enclave}
$$

它的核心不是密码学隐藏计算，而是依赖硬件隔离、内存保护和远程证明。远程证明，白话说，就是远端先验证“你真的是那个没被篡改的安全环境”，再把敏感数据交进去。TEE 通常比 HE 更容易落地，但它不是“自动安全”，因为侧信道、调试口、日志和宿主机配置错误都可能成为漏洞。

一个常见的工程目标可以写成：

$$
\min_k \ \alpha \cdot Latency(k) + \beta \cdot Leakage(z_k) + \gamma \cdot Energy(k)
$$

约束是：

$$
Acc(k) \ge A_0
$$

其中 $k$ 是切分点，$Latency$ 是时延，$Leakage$ 是泄漏风险，$Energy$ 是能耗，$A_0$ 是最低准确率要求。这个式子表达的不是精确求解方法，而是工程决策的方向：切分点越浅，设备更轻松，但泄漏更大；切分点越深，泄漏可能下降，但端侧代价上升。

### 玩具例子

假设输入是一张 `224×224×3` 的 `float32` 图像，大小约为：

$$
224 \times 224 \times 3 \times 4 = 602{,}112 \text{ bytes}
$$

如果切分后只上传一个 `256` 维 `float16` 特征，大小约为：

$$
256 \times 2 = 512 \text{ bytes}
$$

带宽下降约：

$$
\frac{602112}{512} \approx 1176
$$

这说明通信成本大幅下降，但不能推出“隐私提升 1176 倍”。因为通信体积衡量的是“传了多少”，不是“泄漏了什么”。

### 真实工程例子

智能门禁摄像头做人脸检测时，端侧 NPU 先做前几层卷积，得到 embedding 或候选框；网关只接收特征或告警结果。如果网关是自家可控设备，Split Inference 往往足够。如果网关是第三方、多租户或边云混部环境，更合理的做法是把后半段模型放进 TEE；如果威胁模型要求服务器完全不可见明文，再考虑 HE 或混合协议，但要接受更高时延。

| 机制 | 谁能看到明文 | 性能代价 | 适用场景 |
| --- | --- | --- | --- |
| Split Inference | 设备看到 $x$，服务器可能看到 $z$ | 低到中 | 低延迟、弱到中等隐私约束 |
| TEE | 明文进入受保护硬件区 | 中 | 第三方环境、需要较强隔离 |
| HE | 服务器看不到输入明文 | 高 | 极高隐私要求、吞吐要求较低 |
| 混合方案 | 明文仅在局部暴露 | 中到高 | 真实工程最常见 |

---

## 代码实现

代码实现的重点不是做一个完整工业系统，而是把“切分点、传输内容、保护边界”表达清楚。下面给一个最小可运行骨架，模拟前半段在设备执行，后半段在服务器执行，并用一个非常粗糙的“泄漏分数”表示特征信息量。

```python
from math import prod

def preprocess(x):
    # 这里假设 x 已经是数值向量；真实系统里会有 resize、normalize 等步骤
    return [float(v) for v in x]

def model_front(x):
    # 模拟前半段模型：做降维并提取简单特征
    mean_v = sum(x) / len(x)
    max_v = max(x)
    min_v = min(x)
    energy = sum(v * v for v in x) / len(x)
    z = [mean_v, max_v, min_v, energy]
    return z

def model_back(z):
    # 模拟后半段模型：根据特征做一个简单二分类
    mean_v, max_v, min_v, energy = z
    score = 0.6 * mean_v + 0.3 * energy + 0.1 * (max_v - min_v)
    return 1 if score >= 0.5 else 0

def estimate_leakage(x, z):
    # 这里只是玩具指标：特征维度 / 输入维度
    # 真实系统需要用重建攻击、属性推断等方式评估
    return len(z) / len(x)

def run_split_inference(x):
    x = preprocess(x)
    z = model_front(x)      # device
    y = model_back(z)       # server or TEE
    leakage = estimate_leakage(x, z)
    return z, y, leakage

def bytes_of_tensor(shape, dtype_bytes):
    return prod(shape) * dtype_bytes

# 玩具输入
x = [0.2, 0.8, 0.4, 0.6, 0.9, 0.1, 0.3, 0.7]
z, y, leakage = run_split_inference(x)

assert len(z) == 4
assert y in (0, 1)
assert 0 < leakage < 1

raw_bytes = bytes_of_tensor((224, 224, 3), 4)   # float32 image
feat_bytes = bytes_of_tensor((256,), 2)         # float16 feature
assert raw_bytes == 602112
assert feat_bytes == 512
assert raw_bytes / feat_bytes == 1176.0
```

这段代码表达了四个边界：

1. `preprocess(x)` 在本地做，避免把未经处理的原始数据直接送出去。
2. `z = model_front(x)` 是切分点，决定外发的到底是什么。
3. `model_back(z)` 可以运行在普通服务器、TEE，或者被改写为密文计算版本。
4. `estimate_leakage` 不能用“特征更小”代替“泄漏更少”，真实系统必须做攻击验证。

最小流程可以概括成：

```text
x = preprocess(x)
z = model_front(x)
send(z)
y = model_back(z)
return y
```

如果换成 TEE 版本，重点变化不在接口，而在执行位置：

```text
x = preprocess(x)
z = model_front(x)
send(z to enclave)
y = enclave_run(model_back, z)
return y
```

如果换成 HE 版本，重点变化在传输对象：

```text
x = preprocess(x)
c = encrypt(x or z)
send(c)
c_y = encrypted_eval(model, c)
y = decrypt(c_y)
return y
```

这也是工程上常见的判断标准：**保护边界的变化，往往比模型结构本身更重要。**

---

## 工程权衡与常见坑

边缘推理隐私保护最容易出错的地方，不是“不会用某项高级技术”，而是评估不完整。

先看常见权衡：

| 方案 | 隐私强度 | 时延 | 能耗 | 适用场景 | 主要风险 |
| --- | --- | --- | --- | --- | --- |
| 全本地推理 | 高 | 低到中 | 高 | 设备算力足够、离线可用 | 模型太大、发热、更新困难 |
| Split Inference | 中 | 低 | 中 | 带宽有限、低延迟优先 | 中间特征泄漏 |
| TEE | 中到高 | 中 | 中 | 服务器不完全可信 | 侧信道、配置错误 |
| HE | 很高 | 高 | 高 | 极高隐私要求 | 性能不可用、模型受限 |
| 混合方案 | 高 | 中 | 中到高 | 大多数真实系统 | 设计复杂、排障更难 |

新手版可以这样理解：**如果你只验证模型还能不能跑，就像只检查门能不能关上，却没看窗户、后门和钥匙有没有被人拿走。**

典型坑主要有六类。

第一，只要不传原图，就默认安全。错误。很多 embedding、feature map、hidden state 都能被重建攻击或属性推断利用。

第二，切分点太浅。前几层特征往往仍保留明显的纹理、边缘和局部结构，泄漏风险高。

第三，切分点太深。设备侧计算过重，导致发热、掉帧、电池消耗和 P95 时延恶化。P95，白话说，就是最慢那 5% 请求的表现，常比平均值更接近真实用户体验。

第四，直接把整个模型搬进 HE。很多 CNN 和 Transformer 在密文下会慢到无法满足实时需求。

第五，以为 TEE 自动解决一切。TEE 保护的是隔离执行，不自动覆盖侧信道、宿主机日志、调试接口、core dump、遥测采集。

第六，只看功能指标，不看攻击指标。隐私保护系统必须做红队验证，而不是只跑 accuracy benchmark。

至少要覆盖三类攻击评测：

| 红队测试 | 要回答的问题 |
| --- | --- |
| 重建攻击 | 能不能从 $z$ 或输出重建出原始输入 $x$ |
| 属性推断 | 能不能从 $z$ 推断性别、年龄、身份等敏感属性 |
| Membership Inference | 能不能判断某条数据是否出现在训练集 |

工程上还要补充两类常被忽略的面：

- 日志与监控：错误日志、A/B 采样、debug dump 可能直接写出明文或可逆特征。
- 回退路径：TEE 初始化失败、密钥服务超时、端侧模型加载失败时，系统是否偷偷回退到“明文上传原图”。

真正稳妥的方案，不是“默认有保护”，而是“失败时也不越过隐私边界”。

---

## 替代方案与适用边界

没有万能解。选择方案时，先问约束，再选技术。

如果设备算力强、输入高度敏感、并且必须离线可用，优先考虑全本地推理。它避免网络暴露，但代价是模型压缩、端侧适配和版本更新更复杂。

如果低延迟优先、网络可用、网关基本可信，Split Inference 往往是性价比最高的方案。它适合先把“原始输入不上云”这一步做扎实，但前提是你真的评估过中间特征泄漏。

如果服务器属于第三方、多租户或跨团队共享环境，TEE 通常比普通 Split Inference 更稳妥。因为它把“谁能看到后半段明文特征”这个问题，从软件承诺转成了硬件隔离边界。

如果威胁模型极其严格，比如服务提供方也不能看到任何明文输入，而且业务允许较高时延与成本，再考虑 HE。它更像“高安全、低通用性”的工具，而不是默认部署选项。

真实场景版可以这样判断：**智能门禁摄像头做人脸检测时，如果网关是自家设备，可以优先用 split inference；如果网关属于第三方或多租户环境，就更适合把敏感后半段放进 TEE；如果威胁模型更严格，再考虑加密推理，但要接受性能下降。**

| 方案 | 低延迟优先 | 第三方不可信 | 算力受限 | 极高隐私要求 | 适用边界 |
| --- | --- | --- | --- | --- | --- |
| Split Inference | 强 | 弱到中 | 强 | 弱到中 | 最实用，但必须测泄漏 |
| TEE | 中 | 强 | 中 | 中到强 | 依赖硬件与平台能力 |
| HE | 弱 | 强 | 弱 | 很强 | 适合高敏感、低实时性 |
| 混合方案 | 中到强 | 强 | 中 | 强 | 工程复杂，但最常见 |

一个实用决策顺序是：

1. 先确定谁不可信：仅网络不可信，还是服务器也不可信。
2. 再确定时延上限：是 `50 ms`、`100 ms` 还是秒级可接受。
3. 再看端侧预算：芯片、内存、功耗、散热是否允许更深本地计算。
4. 最后才决定是切分、TEE、HE，还是混合。

很多系统最终落在“端上做大部分前处理 + 少量特征外发 + 服务端 TEE + 严格日志治理”这个中间地带。原因很简单：它通常不是理论最强，但在真实成本和时延约束下最容易成立。

---

## 参考资料

1. [Intel Software Guard Extensions (SGX)](https://www.intel.com/content/www/us/en/products/docs/accelerator-engines/software-guard-extensions.html)（TEE：理解 enclave、隔离执行与平台边界）
2. [Arm Confidential Compute Architecture](https://www.arm.com/architecture/security-features/arm-confidential-compute-architecture)（TEE：理解 Arm 侧可信计算边界）
3. [CryptoNets: Applying Neural Networks to Encrypted Data](https://www.microsoft.com/en-us/research/publication/cryptonets-applying-neural-networks-to-encrypted-data-with-high-throughput-and-accuracy/?lang=ko-kr)（HE：理解加密推理为何可行以及为什么代价高）
4. [Delphi: A Cryptographic Inference Service for Neural Networks](https://www.usenix.org/conference/usenixsecurity20/presentation/mishra)（HE/混合密码协议：理解更现实的私有推理服务设计）
5. [Neurosurgeon: Collaborative Intelligence Between the Cloud and Mobile Edge](https://nyuscholars.nyu.edu/en/publications/neurosurgeon-collaborative-intelligence-between-the-cloud-and-mob-2)（Split Inference：理解模型切分与端云协同）
6. [Deep Leakage from Gradients](https://arxiv.org/pdf/1901.09546.pdf)（攻击验证：虽然主要讨论梯度泄漏，但有助于理解“中间表示可逆”这类风险思路）
7. [A Survey of Privacy Attacks in Split Learning](https://www.sciencedirect.com/science/article/pii/S0893608023007402)（攻击验证：理解 split learning / split inference 的重建与推断风险）
