## 核心结论

模型权重加密存储，就是把模型的 checkpoint 或权重文件先加密保存，只有在推理或加载时才解密。checkpoint 是训练或部署时保存的一组参数文件，里面本质上就是模型学到的数字。它主要防的是“静态文件被拷走”这一类风险，比如对象存储泄露、磁盘镜像泄露、备份外流，而不是防一切攻击。

可以把它理解成“权重文件先上锁，服务启动时再开锁”。文件存放在 S3、NAS 或本地磁盘时是密文，启动后进程拿到密钥，临时解开，再加载进内存。这个方案的收益很明确：别人拿到静态文件但拿不到密钥时，看到的只是无意义的字节。代价也很明确：启动会更慢，解密会吃 CPU，密钥管理会变复杂。

一个最容易被忽略的事实是：加密存储解决的是存储侧泄露，不是完整访问控制。访问控制就是“谁能拿、谁能解、谁能审计”的规则。如果密钥能被随意调用，或者服务把明文写回 `/tmp`、swap、日志、缓存，那么文件虽然加密了，系统整体仍然不安全。

最常见的工程结论可以先记住三条：

| 结论 | 适用场景 | 代价 |
|---|---|---|
| 整文件加密后启动时一次性解密 | 部署简单、模型不大、冷启动要求不高 | 启动慢，CPU 峰值高，内存峰值高 |
| 分片按需解密 | 大模型、多副本、冷热访问明显 | 缓存、版本、撤销、轮换更复杂 |
| 加密必须配合密钥管理和明文清理 | 几乎所有生产场景 | 运维和故障处理成本上升 |

一个最小机制图可以写成：

```text
密文权重 C_i 存储在磁盘/对象存储
        ↓
启动时向 KMS 请求解包数据密钥 DK_i
        ↓
用 DK_i 解密得到明文权重 W_i
        ↓
加载到进程内存
```

对应公式是：

$$
C_i = Enc(DK_i, W_i)
$$

$$
WrappedDK_i = Enc(K, DK_i)
$$

这里 `K` 是主密钥，主密钥就是“只用来保护其他密钥的上层密钥”；`DK_i` 是数据密钥，也就是“真正拿来加密大文件的工作密钥”。

---

## 问题定义与边界

先定义边界，否则很容易把“加密存储”说成“模型安全”的总方案。这里的威胁模型重点是：攻击者拿到了静态文件副本，但没有拿到解密能力。静态文件就是落在磁盘、对象存储、备份系统里的那份数据。这个场景下，加密非常有效。

玩具例子：你把 `model.safetensors` 放在对象存储里，同时每天做一次备份。某次备份桶配置错误，别人下载到了整份文件。如果文件是明文，对方直接能复制模型；如果文件是密文，对方即使拿到 20GB 文件，也无法直接恢复参数。

但它不能直接防下面这些事情：

| 能防什么 | 不能防什么 |
|---|---|
| 磁盘文件被直接拷走 | 运行中内存被 dump |
| 对象存储里的权重文件泄露 | root 权限进程读取解密后数据 |
| 备份文件外流 | 服务主动把明文写入日志或临时目录 |
| 低权限人员拿到原始存储副本 | 已获得解密权限的人滥用权限 |

这里最重要的边界是“离线泄露”和“在线滥用”要分开看。离线泄露指的是攻击者只拿到文件副本，没有运行环境；在线滥用指的是攻击者已经进入主机、进程或密钥调用路径。前者主要靠加密降低风险，后者必须再加最小权限、审计、主机隔离、进程隔离等机制。

再看一个新手容易踩中的场景：攻击者拿到 S3 备份文件，但没有密钥，所以暂时无法恢复明文；可是你的服务在启动后把解密权重写到 `/tmp/model.bin` 方便复用，那么只要有人再拿到这台机器的临时盘，防护就被自己破坏了。也就是说，真正的问题不是“有没有加密”，而是“解密后的明文生命周期是否可控”。

---

## 核心机制与推导

大权重文件一般不会直接用主密钥去加密。主密钥直接加密几十 GB 文件并不灵活，也不利于轮换。实际更常见的是 envelope encryption，中文常叫信封加密。它的意思是：先生成一个数据密钥 `DK_i`，用它加密分片权重 `W_i`；再用主密钥 `K` 去加密这个 `DK_i`。这样主密钥只负责“包住钥匙”，不直接搬运大货。

公式是：

$$
C_i = Enc(DK_i, W_i)
$$

$$
WrappedDK_i = Enc(K, DK_i)
$$

其中：
- `W_i` 是第 `i` 个明文权重分片。
- `C_i` 是对应密文分片。
- `DK_i` 是分片的数据密钥。
- `K` 是 KMS 或 HSM 管理的主密钥。

为什么要分片？因为大模型通常本来就分成多个 shard。shard 就是切成几块存的文件，方便传输、并行加载和局部访问。分片后可以只解密热分片，也就是“当前更常被访问的那部分权重”。

启动成本可以粗略写成：

$$
T_{boot} \approx \sum_i (T_{read,i} + T_{dec,i} + T_{load,i})
$$

如果做整文件解密，近似成：

$$
T_{boot} \approx \frac{|W|}{B_{io}} + \frac{|W|}{B_{dec}} + T_{framework}
$$

其中 $|W|$ 是权重大小，$B_{io}$ 是读取吞吐，$B_{dec}$ 是解密吞吐，$T_{framework}$ 是框架加载、张量构建、内存映射等附加成本。

数值玩具例子：假设权重文件大小是 1GB，磁盘读取速度是 1GB/s，AES 解密有效吞吐是 500MB/s。那么：

- 读文件约 $1GB / 1GB/s = 1s$
- 解密约 $1GB / 0.5GB/s = 2s$

于是最小启动时间大约是：

$$
T_{boot} \approx 1s + 2s = 3s
$$

但这还不是“服务已可用”的时间。因为你还没算框架反序列化、张量拷贝、显存搬运、模型结构绑定等开销。真实系统里，3 秒只是“数据从密文变成明文”的一部分。

再看分片按需解密。假设 1GB 模型拆成 4 个 256MB 分片，首个请求只需要热分片 1 个，那么首个分片的读取与解密时间大约是：

- 读分片：$256MB / 1GB/s \approx 0.25s$
- 解密分片：$256MB / 0.5GB/s \approx 0.5s$

总计约 0.75 秒。首个请求会更快，但系统复杂度立刻上升，因为你要回答四个问题：哪些分片是热的、缓存放哪里、何时失效、密钥撤销后怎么强制清掉旧缓存。

整文件解密和分片按需解密的对比如下：

| 方案 | 优点 | 缺点 | 适合什么 |
|---|---|---|---|
| 整文件解密 | 逻辑最简单，故障面小 | 冷启动慢，CPU/内存峰值高 | 中小模型、低弹性要求 |
| 分片按需解密 | 首次可用更快，可利用热分片 | 元数据、缓存、失效、轮换复杂 | 大模型、分层加载、热点明显 |
| 分片预热后常驻 | 延迟稳定 | 明文驻留时间长，撤销更难 | 长生命周期服务 |

真实工程例子：一个 13B 模型被拆成多个 `safetensors` shard，密文存放在对象存储。服务启动时先从 KMS 解包 `WrappedDK_i`，把热分片解密到内存或 `tmpfs`，再构建模型。`tmpfs` 是挂在内存上的临时文件系统，优点是不落物理盘。这样能减少明文长期落盘，但你必须同步做缓存版本号、密钥轮换、撤销后驱逐、失败回滚，否则“安全收益”会被运维一致性问题抵消。

---

## 代码实现

实现时有三条基本原则：密钥分离、尽量不落明文盘、优先使用安全权重格式。密钥分离就是主密钥只保管数据密钥；安全权重格式优先选 `safetensors`，因为它比基于 pickle 的老路径更可控。pickle 是 Python 的对象序列化机制，它能恢复对象，但也可能在加载时触发不安全代码路径。

下面用一个可运行的 Python 玩具实现展示“先解包密钥，再解密分片，再加载”的顺序。它不是生产级加密，只是帮助理解流程：

```python
import hashlib

def xor_stream(key: bytes, n: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < n:
        block = hashlib.sha256(key + counter.to_bytes(4, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:n])

def encrypt_bytes(data: bytes, dk: bytes) -> bytes:
    stream = xor_stream(dk, len(data))
    return bytes(a ^ b for a, b in zip(data, stream))

def decrypt_bytes(cipher: bytes, dk: bytes) -> bytes:
    return encrypt_bytes(cipher, dk)

def wrap_dk(master_key: bytes, dk: bytes) -> bytes:
    mask = hashlib.sha256(master_key).digest()[:len(dk)]
    return bytes(a ^ b for a, b in zip(dk, mask))

def unwrap_dk(master_key: bytes, wrapped: bytes) -> bytes:
    return wrap_dk(master_key, wrapped)

master_key = b"master-key-32bytes-demo!!!"
dk = b"data-key-16byte"
shards = [b"layer0:123", b"layer1:456"]

wrapped_dk = wrap_dk(master_key, dk)
cipher_shards = [encrypt_bytes(s, dk) for s in shards]

runtime_dk = unwrap_dk(master_key, wrapped_dk)
plain_shards = [decrypt_bytes(c, runtime_dk) for c in cipher_shards]

assert runtime_dk == dk
assert plain_shards == shards
assert cipher_shards[0] != shards[0]
```

生产系统里的伪代码更接近这样：

```python
dk = kms_decrypt(wrapped_dk)

for shard_meta in manifest.hot_shards():
    ciphertext = read_from_object_store(shard_meta.path)
    plaintext = aes_gcm_decrypt(ciphertext, dk, shard_meta.nonce)
    load_into_model(plaintext)   # 直接进内存或写入 tmpfs
    secure_erase(plaintext)
```

如果做分片缓存，还需要把“版本号”和“TTL”一起考虑。TTL 就是缓存最多活多久的时间限制。

```python
def get_shard(shard_id, key_version):
    cached = cache.get(shard_id)
    if cached and cached.key_version == key_version and not cached.expired():
        return cached.data

    dk = kms_decrypt(fetch_wrapped_dk(shard_id, key_version))
    plaintext = decrypt(fetch_ciphertext(shard_id), dk)
    cache.put(shard_id, plaintext, key_version=key_version, ttl_seconds=300)
    return plaintext
```

这里有两个关键点不能省：

1. 缓存项要带 `key_version`，否则密钥轮换或撤销后，旧明文还可能继续命中。
2. 加载完成后要清理临时明文，尤其是写入 `tmpfs` 的场景，要明确回收时机。

一个简化流程图如下：

```text
拉取密文 shard
    ↓
从 KMS 解包 WrappedDK
    ↓
在内存或 tmpfs 中解密
    ↓
加载到模型
    ↓
清理临时明文与旧缓存
```

---

## 工程权衡与常见坑

这类方案真正难的地方不在“把文件加密”，而在“把系统做对”。整文件解密简单，但冷启动慢；分片按需解密灵活，但一致性复杂。很多系统不是败在密码学，而是败在缓存、临时文件、回滚和轮换流程。

先看常见坑：

| 问题 | 后果 | 规避方式 |
|---|---|---|
| 只轮换 KMS 主密钥，不处理旧密文 | 旧数据仍依赖旧版本密钥 | 做 `ReEncrypt` 或重新包裹数据密钥 |
| 解密后写入临时磁盘或进 swap | 明文重新暴露 | 优先内存或 `tmpfs`，关闭不必要 swap，显式清理 |
| 继续直接用不可信 `torch.load` | 存在反序列化风险 | 优先 `safetensors`，至少限制只加载可信权重 |
| 分片缓存没版本号和 TTL | 撤销后旧明文继续被用 | 缓存绑定 `key_version`，设置 TTL 和强制驱逐 |
| 把加密误当访问控制 | 拿到解密权限的人仍可读全量权重 | 配最小权限、鉴权、审计和调用限额 |

一个典型错误做法是这样的：服务为了减少启动时间，把解密后的权重写到本地缓存目录。后来安全团队撤销了旧密钥，但这台机器上的缓存文件还在，重启后服务甚至还能直接从缓存读取。结果是“密钥撤销成功了，明文却还活着”。这说明撤销密钥和撤销明文不是同一件事，前者是密钥管理动作，后者是缓存和存储清理动作。

另一个常见误区是把“文件加密”理解成“模型不会泄露”。如果部署节点本身可被 root 进入，或者推理进程能被 dump，攻击者仍可能拿到内存中的明文参数。加密存储只缩小了风险面，没有消灭运行态风险。

---

## 替代方案与适用边界

是否要做权重加密存储，要看你防的到底是什么。如果你主要担心备份泄露、对象存储误配置、离线副本被复制，那么静态加密通常很值得做。如果你面对的是高对抗环境，担心租户间攻击、主机入侵、内存抓取，那么只做静态加密远远不够。

下面给出一个选择表：

| 方案 | 能力边界 | 优点 | 不足 |
|---|---|---|---|
| 仅权限控制 | 防止普通未授权访问 | 简单，成本低 | 一旦文件副本外流就失守 |
| 静态加密存储 | 防离线文件泄露 | 对备份、对象存储、磁盘副本有效 | 不解决运行态明文问题 |
| 静态加密 + KMS + 安全加载 + 缓存失效 | 防离线泄露并控制部分解密生命周期 | 生产可用性更强 | 系统复杂度明显上升 |
| TEE/HSM/受控推理环境 | 更强的运行态隔离 | 适合高敏感和强对抗 | 成本高，接入复杂 |

可以用选择题方式理解：

- 内部团队共享模型，但担心备份泄露：静态加密 + KMS 通常足够。
- 面向多环境部署，节点弹性扩缩容频繁：静态加密 + KMS + 安全加载 + 缓存失效更合理。
- 跨租户、高敏感、强对抗环境：要叠加更强隔离、审计、硬件保护，必要时考虑 TEE 或受控推理环境。

因此，模型权重加密存储不是“该不该做”的纯是非题，而是“做到哪一层才匹配你的威胁模型”。如果目标只是降低静态文件泄露风险，它很有效；如果目标是阻止一切模型提取，那它只是基础层，不是终点。

---

## 参考资料

1. [AWS KMS cryptography essentials](https://docs.aws.amazon.com/kms/latest/developerguide/kms-cryptography.html)
2. [Rotate AWS KMS keys](https://docs.aws.amazon.com/kms/latest/developerguide/rotate-keys.html)
3. [Schedule AWS KMS key deletion](https://docs.aws.amazon.com/kms/latest/developerguide/deleting-keys-scheduling-key-deletion.html)
4. [PyTorch `torch.load` Documentation](https://docs.pytorch.org/docs/stable/generated/torch.load.html)
5. [safetensors GitHub Repository](https://github.com/huggingface/safetensors)
6. [Hugging Face Hub Serialization Docs](https://huggingface.co/docs/huggingface_hub/en/package_reference/serialization)
