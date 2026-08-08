---
title: PyTorch 工程实践：模型定义、训练循环与检查点
date: 2026-08-07
---

# PyTorch 工程实践：模型定义、训练循环与检查点

<div class="epigraph">
<p>框架替你算梯度，但训练的「骨架」得你自己搭。</p>
<footer>—— 依据 PyTorch 实践的精神改写</footer>
</div>

<div class="article-byline">
<p>第四级 · 深度学习 ｜ 李沐《动手学深度学习》§3.1–3.3、PyTorch 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么从 PyTorch 工程实践（下）开始

上一节配好了「数据」，本节配齐「模型 + 训练 + 检查点」——**PyTorch 训练的三大件**：`nn.Module`（模型定义）、训练循环（前向/反向/更新五步曲）、检查点（模型保存/加载/恢复）。它们是「让深度学习跑起来」的完整骨架，也是「从教程代码到工程代码」的分水岭——**工程代码要「可复现、可恢复、可监控」**，不只是「能跑」。

「训练循环」看似简单（清梯度 → 前向 → 损失 → 反向 → 更新），但工程化的训练循环还有大量「隐藏环节」：梯度裁剪、学习率调度、早停、日志、检查点……本节把这些「工程细节」讲透——它们是「训练不出错、出错了能恢复」的保障。<span class="marginnote">「『能跑的代码』 vs 『工程的代码』」：教程代码「在单卡上跑通」；工程代码「可复现（种子）、可恢复（检查点）、可监控（日志）、可扩展（多卡/分布式）」——「<strong>工程化的四要素：复现、恢复、监控、扩展</strong>」。「从『能跑』到『可靠』是工程化的距离」。</span>

## 1 nn.Module：模型定义的接口

**`nn.Module`** 是 PyTorch 模型的基类——「把层组织成模型」的接口：

```python
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(      # 子模块：层赋给 self，才会注册参数
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 10),
        )

    def forward(self, x):              # 前向：定义「输入 → 输出」
        return self.net(x)

model = MLP()
```

**`nn.Module` 的关键能力**：

1. **参数管理**：`model.parameters()` 自动收集所有层的参数——「优化器直接拿参数」。
2. **前向定义**：`forward()` 定义「输入 → 输出」——「autograd 自动记录计算图」。
3. **切换模式**：`model.train()` / `model.eval()`——「切换 Dropout/BatchNorm 的行为」。
4. **设备移动**：`.to(device)`——「模型移到 GPU/CPU」。

**「`nn.Module` = 层的容器 + 前向的定义 + 参数与模式的管家」**——它是「模型的『对象化』」。<span class="marginnote">「`nn.Module` 的『子模块』」：`nn.Module` 可以嵌套（一个模块里包含子模块）——「<strong>模型 = 模块的树</strong>」。「`parameters()` 递归收集所有子模块的参数」——「<strong>模块化让『大模型』可以『分层定义』</strong>」。「`<strong>`nn.Sequential` 是『顺序模块』的快捷方式」（一层接一层时用）。</span>

**易错点：** `__init__` 里用的层必须是「`self` 的属性」（赋给 `self`）——「临时变量」（如 `x` 未赋给 `self`）不会被注册进参数管理。「<strong>层要赋给 `self`，否则参数找不到</strong>」是 nn.Module 最常见的坑。

## 2 训练循环：五步曲 + 工程细节

**标准训练循环（每步）**：

```python
for x, y in train_loader:
    optimizer.zero_grad()     # 1. 清梯度
    y_hat = model(x)          # 2. 前向
    loss = loss_fn(y_hat, y)  # 3. 损失
    loss.backward()           # 4. 反向（算梯度）
    optimizer.step()          # 5. 更新参数
```

**「清梯度 → 前向 → 损失 → 反向 → 更新」五步曲**是训练循环的铁律（见《反向传播》的「节拍」）。

**工程化的训练循环还要加**：

**梯度裁剪**：`torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)`——「防爆炸」（见《梯度消失/爆炸》）。
**学习率调度**：`torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma)`——「学习率随时间衰减」（见《学习率策略》）。
**模式切换**：`model.train()`（训练）、`model.eval()`（评估）——「Dropout/BN 的行为切换」。
**梯度累积**（显存不够时）：「攒几步梯度再更新」（见《序列截断》的梯度累积）。

**「五步曲 + 工程附加」**——「训练循环是『固定骨架 + 灵活附件』」。<span class="marginnote">「`torch.no_grad()` 的评估」：评估时「不需要梯度」——用 `torch.no_grad()` 包住评估代码——「<strong>省显存、加速评估</strong>」。「<strong>评估 ≠ 训练：eval 模式 + no_grad</strong>`」是评估循环的两件套。「忘了 `torch.no_grad()` 会让评估『累积计算图』（显存爆炸）」——「评估循环的『不用梯度』是隐性要求」。</span>

**易错点：** 每个 epoch 结束时「评估验证集」要 `model.eval()`，下一个 epoch 开始要 `model.train()`——「<strong>模式切换的『来回』</strong>」是训练循环的常见漏项（评估后忘切回 train，Dropout 一直关着）。

## 3 检查点：保存、加载与恢复

**检查点（checkpoint）**是「训练状态的快照」——「训练中断了能恢复、训练完了能复用」。

**保存**：

```python
checkpoint = {
    "model_state_dict": model.state_dict(),          # 模型参数
    "optimizer_state_dict": optimizer.state_dict(),  # 优化器状态
    "epoch": epoch,                                  # 训练进度
    "best_acc": best_acc,                            # 历史最优指标
}
torch.save(checkpoint, "checkpoint.pt")
```

**「保存『模型参数 + 优化器状态 + 训练进度』」**——不只是模型，还有「能恢复训练」的一切。

**加载（恢复训练）**：

```python
checkpoint = torch.load("checkpoint.pt")
model.load_state_dict(checkpoint["model_state_dict"])        # 恢复模型参数
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])  # 恢复优化器状态
start_epoch = checkpoint["epoch"] + 1                        # 从断点继续
best_acc = checkpoint["best_acc"]
```

**「state_dict 是『参数的名字 → 张量』的字典」**——`load_state_dict()` 按名字对齐——「模型结构不变才能加载」。

**检查点的实践**：

**每 N 个 epoch 存一次**（或验证集最好时存）——「<strong>存『最优验证点』而非『最后点』</strong>」（与《提前终止》的「历史最优快照」一致）。
**原子保存**：先存临时文件再改名——「防保存中断损坏」。

**易错点：** `state_dict` 保存的是「参数的值」，`nn.Module` 本身（结构）不保存——加载时「结构要一致」。**「保存 state_dict，不是 model」**（`torch.save(model)` 是「整对象序列化」，不推荐——「结构变了就加载不了」）。<span class="marginnote">「检查点 vs 推理导出」：检查点（checkpoint）保存「训练状态」（含优化器、epoch——能恢复训练）；推理导出（ONNX/权重）只保存「模型参数」（部署用，不含优化器）——「<strong>检查点给『训练』用、导出给『部署』用</strong>」。「训练与部署的『保存』目的不同、内容不同」。</span>

## 4 公式解析：训练循环的「状态机」

把训练循环理解成「状态机」——每个环节更新「状态」，状态正确才不出错：

```
θ ← 初始化；优化器状态 ← 0；lr ← lr₀；epoch ← 0
for epoch in 1..E:
    model.train()                # 模式状态 → 训练
    for (x, y) in train_loader:
        optimizer.zero_grad()    # 梯度状态清零
        loss = loss_fn(model(x), y)
        loss.backward()          # 梯度状态更新
        optimizer.step()         # 参数状态更新
    model.eval()                 # 模式状态 → 评估
    with torch.no_grad():        # 不累积计算图
        acc = evaluate(model, val_loader)
    lr_scheduler.step()          # 学习率状态更新
    if acc > best_acc:           # 最优快照
        best_acc = acc; save_checkpoint(...)
```

**第一步，看「谁变谁不变」**：模型参数每步变；优化器状态每步变；学习率每 epoch 变；「模式」（train/eval）随阶段变——「<strong>每个状态都有「何时变」的节奏</strong>」。
**第二步，看「状态一致性」**：检查点要保存「所有会变的状态」（θ、优化器、epoch、best）——「<strong>漏存一个，恢复就『对不上』</strong>」。
**第三步，读工程化**：训练循环 =「多个状态按正确节奏更新的『状态机』」——「<strong>理解『每个状态何时变』，才能写对训练循环</strong>」。<span class="marginnote">「训练循环的『可复现性』」：要「可复现」，必须控制「所有随机源」——<strong>数据打乱（shuffle 种子）、模型初始化（种子）、Dropout（种子）</strong>。「<strong>固定种子 ≠ 完全可复现</strong>」（多卡、cudnn 的非确定性）——「<strong>可复现性 = 控制所有随机源 + 固定环境</strong>」。「复现是工程化的『诚信』」。</span>

## 5 工程化训练脚本的完整骨架

把「PyTorch 训练」的工程骨架总结成「一个标准的训练脚本」：

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)                           # 可复现：固定种子

model = MLP().to(device)                       # 模型
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
loss_fn = torch.nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

best_acc = 0.0
for epoch in range(num_epochs):
    model.train()                              # 训练模式
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        optimizer.step()

    model.eval()                               # 评估模式
    with torch.no_grad():
        acc = evaluate(model, val_loader)
    scheduler.step()                           # 学习率调度

    if acc > best_acc:                         # 存最优检查点
        best_acc = acc
        torch.save(model.state_dict(), "best.pt")
```

**「一个训练脚本 = 数据（上节）+ 模型 + 五步曲 + 调度 + 评估 + 检查点」**——「<strong>把每个环节『模块化』，脚本才『可维护』</strong>」——「工程化 = 结构化 + 可复现 + 可恢复」。

**易错点：** 别把「评估」写成「训练模式」——评估要 `model.eval()` + `torch.no_grad()`，训练要 `model.train()`——「<strong>『训练/评估』的模式错位是性能虚高的来源</strong>」（Dropout 开着评估，指标偏乐观）。<span class="marginnote">「『日志』的工程价值」：训练循环里的 `loss`/`acc` 日志不只是「看看」——「<strong>日志是调试、监控、复现的基础</strong>」（见《调试策略》的「可观察性」）。「<strong>结构化日志（每步损失、每 epoch 验证、梯度范数）让你『看得见』训练发生了什么</strong>」。「日志与检查点一样，是训练脚本的『基础设施』」。</span>

## 6 小结

- **nn.Module**：层的容器 + 前向定义 + 参数/模式管家——「模型的对象化」；层要赋给 `self`。
- **训练循环五步曲**：清梯度 → 前向 → 损失 → 反向 → 更新——「铁律节拍」。
- **工程附加**：梯度裁剪、学习率调度、模式切换（train/eval）、梯度累积。
- **检查点**：保存「模型 + 优化器 + epoch + best」——「能恢复训练」；存 state_dict 而非 model。
- **评估循环**：`model.eval()` + `torch.no_grad()`——「评估 ≠ 训练」。
- 工程化四要素：**可复现（种子）、可恢复（检查点）、可监控（日志）、可扩展（多卡）**。

在下一节，我们把「训练好的模型」送上线——导出、优化、部署，这就是**模型部署：ONNX 导出与推理优化**。
