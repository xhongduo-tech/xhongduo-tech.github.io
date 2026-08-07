---
title: PyTorch 工程实践：数据加载与 Dataset/DataLoader
date: 2026-08-07
---

# PyTorch 工程实践：数据加载与 Dataset/DataLoader

<div class="epigraph">
<p>GPU 在等数据，是最昂贵的等待。</p>
<footer>—— 依据数据加载瓶颈的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 李沐《动手学深度学习》§3.1、PyTorch 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 PyTorch 工程实践开始

模型、损失、优化器都讲完了，但「跑起来」还需要一套「工程骨架」：**数据怎么加载、模型怎么组织、训练循环怎么写、结果怎么存**。PyTorch 提供了这套骨架的标准部件——**Dataset**（数据接口）、**DataLoader**（批量加载器）、**nn.Module**（模型接口）。本节（与下一篇）把「PyTorch 工程实践」系统讲透——它们是「把深度学习的知识变成能跑的代码」的最后一环。

「数据加载」看似琐碎，却是真实训练的「第一瓶颈」：**GPU 计算只需几毫秒，数据加载慢一截，GPU 就「饿」着等数据**。理解 Dataset/DataLoader 的正确用法（批量、多进程预取、增强在数据加载时做），是「训练速度」的隐形优化。本节覆盖：**Dataset 接口、DataLoader 的批量/打乱/多进程、transforms 与数据增强的位置、以及数据加载的性能优化**。<span class="marginnote">「『数据加载』在训练中的角色」：一次训练迭代 =「取数据（数据加载）→ 前向 → 反向 → 更新」。「<strong>数据加载是『流水线的第一环』，慢了就『卡住整条流水线』</strong>」。「<strong>『数据加载速度』与『GPU 计算速度』要匹配</strong>」——「数据加载快于计算，GPU 才不饿」。</span>

## 1 Dataset：数据的「接口」

**Dataset** 是 PyTorch 的数据接口——它定义「怎么访问一个样本」。自定义 Dataset 需要实现两个方法：

```python
class MyDataset(torch.utils.data.Dataset):
    def __len__(self):
        return len(self.data)          # 数据集大小

    def __getitem__(self, idx):
        x = self.data[idx]             # 取第 idx 个样本
        y = self.label[idx]
        return x, y                    # 返回 (样本, 标签)
```

**「Dataset 定义了『单样本怎么取』」**——它不管「批量」「打乱」「多进程」（那是 DataLoader 的事）。Dataset 的「最小职责」让数据加载「模块化」。

**transforms 的位置**：数据增强（图像翻转、标准化）通常在 `__getitem__` 里做——「<strong>增强在『取样本时』做，每个 epoch 生成不同的增强</strong>」（与《数据预处理》的「训练随机增强」一致）。

**易错点：** `__getitem__` 返回的样本要「独立」（不要返回「会被后续修改的共享引用」）。**「Dataset 返回的是『数据』，不是『指针』」**——「共享可变对象会造成数据泄漏」。

## 2 DataLoader：批量、打乱与并行

**DataLoader** 把 Dataset「包装」成「能批量、能打乱、能并行加载」的迭代器：

```python
loader = torch.utils.data.DataLoader(
    dataset, batch_size=64, shuffle=True,   # 批量 + 每 epoch 打乱
    num_workers=4,                          # 4 个进程并行加载
    pin_memory=True,                        # 锁页内存（GPU 传输更快）
    drop_last=True,                         # 丢掉最后的不足 batch
)
```

**DataLoader 的五个关键配置**：

1. **batch_size**：每批样本数——「批次大小的入口」。
2. **shuffle**：每个 epoch 打乱——「训练必须打乱」（见《SGD 的小批量策略》）。
3. **num_workers**：并行加载的进程数——「多进程让 CPU 加载与 GPU 计算并行」。
4. **pin_memory**：锁页内存——「CPU→GPU 传输更快」。
5. **drop_last**：丢掉最后不足 batch——「保证 batch 大小一致（对 BatchNorm 重要）」。

**「DataLoader 把『取样本』升级为『取批次 + 并行』」**——它处理「批量、顺序、并发」三件事。<span class="marginnote">「DataLoader 的『多进程』」：`num_workers > 0` 让「数据加载」在多个进程里并行——「<strong>CPU 一边准备下一批数据，GPU 一边算当前批</strong>」——「数据加载『预取』，GPU 不饿」。「<strong>num_workers 是『数据加载速度』的旋钮</strong>」——「数据加载慢 → 加大 num_workers（但别超过 CPU 核数）」。</span>

**易错点：** `shuffle=True` 只在「训练」用；验证/测试用 `shuffle=False`（顺序无关，可复现）。**「训练打乱、评估不打乱」**是 DataLoader 的标准配置。

## 3 transforms：增强在「数据加载时」做

**transforms**（`torchvision.transforms`）把「数据变换」串成「流水线」：

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),      # 缩放
    transforms.RandomHorizontalFlip(),  # 训练增强：随机翻转
    transforms.ColorJitter(...),        # 训练增强：颜色抖动
    transforms.ToTensor(),              # 转张量
    transforms.Normalize(mean, std),    # 标准化
])
```

**「transforms 在 `__getitem__` 里被调用」**——每个 epoch 取样本时「重新变换」——「<strong>增强每 epoch 随机、数据加载时完成</strong>」。

**训练 vs 验证的 transforms 不同**：

- **训练**：含「随机增强」（翻转、裁剪）——「注入不变性」。
- **验证/测试**：只做「确定性变换」（Resize、ToTensor、Normalize）——「评估要稳定」。

**「训练与验证的 transforms 要分开定义」**——「训练随机增强、评估确定性」是《数据预处理》的纪律在 PyTorch 里的实现。<span class="marginnote">「transforms 的『GPU vs CPU』」：简单变换（标准化）可以在 GPU 上做（张量运算）；但「随机增强」（翻转、裁剪）通常用 `torchvision` 的「CPU 实现」（更快、更标准）——「<strong>增强在 CPU（数据加载时）做，标准化等简单变换可 GPU</strong>」。「增强的『放置』是性能与便利的权衡」。</span>

**易错点：** transforms 的顺序有「讲究」——`ToTensor` 要在 `Normalize` 前（先转张量、再标准化）；`Resize` 要在 `ToTensor` 前（操作 PIL 图像）。**「transforms 的顺序：PIL 操作（Resize/翻转）→ ToTensor → Normalize」**。

## 4 公式解析：数据加载的「吞吐」瓶颈

把「数据加载 vs GPU 计算」的吞吐匹配算清楚。设 GPU 每步计算时间 $T_{\text{gpu}}$、数据加载每步时间 $T_{\text{data}}$：

**不预取**（串行：加载完才算）：

$$
T_{\text{step}} = T_{\text{data}} + T_{\text{gpu}}
$$

**预取**（数据加载与计算并行）：

$$
T_{\text{step}} = \max(T_{\text{data}}, T_{\text{gpu}})
$$

- **第一步，看串行的浪费**：$T_{\text{step}} = T_{\text{data}} + T_{\text{gpu}}$——「GPU 在数据加载时闲着」——「<strong>GPU 利用率 < 100%</strong>」。
- **第二步，看并行的收益**：$T_{\text{step}} = \max(\cdot)$——「数据加载藏在计算里」——「<strong>GPU 利用率接近 100%</strong>」。
- **第三步，读优化方向**：让 $T_{\text{data}} \le T_{\text{gpu}}$（多进程、预取、pin_memory）——「<strong>数据加载不成为瓶颈</strong>」。「<strong>瓶颈分析：先看哪个环节慢，再优化那个环节</strong>」。<span class="marginnote">「『瓶颈的测量』」：训练慢时先问「<strong>GPU 利用率是多少</strong>」（`nvidia-smi` 看 GPU 利用率）——「<strong>利用率低 = 数据加载/通信瓶颈；利用率高 = 计算/模型瓶颈</strong>」。「<strong>先测瓶颈，再优化</strong>」——「GPU 利用率是训练性能的『体温计』」。</span>

## 5 数据加载的工程清单

把「数据加载优化」的实践清单列全：

1. **用 DataLoader 的 num_workers**：多进程预取——「数据加载与计算并行」。
2. **pin_memory=True**：锁页内存——「CPU→GPU 传输更快」。
3. **训练/验证 transforms 分开**：训练随机增强、评估确定性——「增强只用在训练」。
4. **避免「每 epoch 重新读大文件」**：把数据加载到内存/缓存——「磁盘 I/O 是最慢的环节」。
5. **大规模数据的「流式加载」**：`IterableDataset` / 内存映射——「放不下内存的数据流式读」。
6. **测量**：打印「每步时间」分解——「定位是数据慢还是计算慢」。

**「数据加载的『工程成熟度』」**：从「玩具的 numpy 直接喂」到「生产的多进程预取 + 流式加载」——「数据加载是『工程复杂度』随『数据规模』增长最快的一环」。<span class="marginnote">「『IterableDataset』的适用」：数据集大到「放不进内存」（TB 级视频、日志）时，用 `IterableDataset`（流式读取，不索引）——「<strong>『内存放不下』 = 流式加载</strong>」。但流式丢失「随机访问」（shuffle 困难）——「<strong>大规模数据的『打乱』是工程难题</strong>」（常见做法：先分片、片内随机、片间乱序）——「数据规模越大，『随机性』越难保证」。</span>

**易错点：** 「transform 里的随机增强」与「多进程」的配合——每个 worker 进程有独立的随机种子——「<strong>多进程的随机性要『独立且可复现』</strong>」（各 worker 用不同的种子，但整体可复现）。「随机性与并发的『平衡』是 DataLoader 的隐藏细节」。

## 6 小结

- **Dataset**：定义「单样本怎么取」（`__len__` + `__getitem__`）；transforms 在取样本时做。
- **DataLoader**：批量、打乱、多进程预取、pin_memory、drop_last——「取样本 → 取批次 + 并行」。
- **transforms**：训练含随机增强、评估只确定性变换——「增强只在训练、评估要稳定」。
- 吞吐匹配：$T_{\text{step}} = \max(T_{\text{data}}, T_{\text{gpu}})$（预取）——「数据加载藏在计算里，GPU 不饿」。
- 优化清单：num_workers、pin_memory、流式加载、测瓶颈（GPU 利用率）——「先测瓶颈，再优化」。
- 数据加载是「工程复杂度随数据规模增长最快」的一环——「从玩具到生产的最后一公里」。

在下一节，我们完成「PyTorch 工程实践」的下半——模型定义、训练循环与检查点，这就是 **PyTorch 工程实践：模型定义、训练循环与检查点**。
