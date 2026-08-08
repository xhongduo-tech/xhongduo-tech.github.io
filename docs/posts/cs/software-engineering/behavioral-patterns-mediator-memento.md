---
title: 行为型模式：中介者与备忘录
date: 2026-08-07
---

# 行为型模式：中介者与备忘录

<div class="epigraph">
<p>中介者让对象之间的对话变成"对着广播说话"，备忘录让状态可以回到从前——一个管交流，一个管时间。</p>
<footer>—— GoF《设计模式》</footer>
</div>

<div class="article-byline">
<p>第三级 · 软件工程 ｜ Pressman《软件工程：实践者的研究方法》第 12 章 ｜ 2026-08-07</p>
</div>

## 为什么从中介者与备忘录开始

行为型模式已讲了责任链、命令、解释器、迭代器。这一节的两个模式，一个解决"**对象之间交互太乱**"（中介者），一个解决"**对象状态需要回溯**"（备忘录）。前者是现代 UI 与消息组件的骨架，后者是快照/撤销的基础。

## 1 中介者：把网状交互收拢成星形

**中介者（Mediator）**：用一个**中介对象**封装一组对象之间的交互，使这些对象**不必显式相互引用**——把"多对多"的网状交互变成"各对象 → 中介"的星形交互。

结构：**Mediator** 接口声明 **notify()**；具体 **ConcreteMediator** 持有并协调各 **Colleague**（同事对象）；同事对象只与中介通信，不直接引用彼此。

```java
// 中介者接口
interface Mediator {
    void notify(Colleague sender, String event);
}

// 同事对象：只认识中介者，不认识其他同事
class Colleague {
    protected Mediator mediator;
    Colleague(Mediator mediator) { this.mediator = mediator; }
    void send(String event) { mediator.notify(this, event); }
}

// 具体中介者：集中协调对象交互
class ConcreteMediator implements Mediator {
    private Colleague a, b;
    public void notify(Colleague sender, String event) {
        if (sender == a) b.onEvent(event);   // A 发消息 → 转发给 B
        else if (sender == b) a.onEvent(event);
    }
}
```

适用场景：**多个对象需要协作、且交互规则复杂易变**——UI 对话框的各控件、聊天室的用户、航班调度。没有中介者，改一处交互要动所有相关对象；有中介者，交互规则集中在中介里，各对象保持独立可复用。<span class="marginnote">中介者的收益：<strong>对象间耦合大降</strong>（彼此不知道对方存在）、交互逻辑集中、符合"高内聚低耦合"。代价：中介可能变成"上帝对象"（所有逻辑汇聚一处）。中介者与外观（Facade）的区别：外观是"单向简化入口"（客户端→子系统），中介者是"双向协调枢纽"（同事互相通过它）。</span>

**辨析｜易错点：** 中介者 vs 观察者：观察者是"一对多广播"（发布者通知订阅者，订阅者间无关系）；中介者是"多对多协调"（所有对象通过中介交流）。UI 里常两者结合：中介者内部用观察者实现消息分发。判断标准：**对象之间是否需要互相知道彼此？** 要，用中介者收拢；不要，用观察者广播。

## 2 备忘录：捕获并恢复对象状态

**备忘录（Memento）**：**在不破坏封装的前提下，捕获一个对象的内部状态**，并在将来使对象恢复到该状态。

结构：**Originator**（发起人，有状态、能创建/恢复备忘录）、**Memento**（备忘录，保存状态的快照，不透明）、**Caretaker**（管理者，保存备忘录但不读其内容）。

```java
// 备忘录：不透明快照，只有 Originator 能读
class Memento {
    private final String state;
    Memento(String state) { this.state = state; }
    String getState() { return state; }
}

// 发起人：创建快照、恢复状态
class Originator {
    private String state;
    void setState(String s) { state = s; }
    Memento save() { return new Memento(state); }
    void restore(Memento m) { state = m.getState(); }
}

// 管理者：只保存快照，不读取内容
class Caretaker {
    private Stack<Memento> history = new Stack<>();
    void push(Memento m) { history.push(m); }
    Memento pop() { return history.pop(); }
}
```

关键设计：**Memento 对外不透明**——Caretaker 只保存、不读；只有 Originator 能读写自己的状态。这保证了"状态恢复"不破坏封装。<span class="marginnote">备忘录的应用：编辑器撤销/重做（快照栈）、游戏存档、数据库的 UNDO 日志（回滚段）。它的权衡：<strong>内存成本</strong>（快照要复制状态）与<strong>不透明性</strong>（Memento 接口要小心设计，别让快照内容泄露）。现代实践中，序列化/不可变对象（快照 = 不可变值）常作为备忘录的轻量替代。</span>

**辨析｜易错点：** 备忘录 vs 命令的撤销：命令模式用"反向操作"撤销（执行 undo 动作）；备忘录用"状态快照"撤销（恢复到过去）。反向操作省内存但复杂（要写反向逻辑）；快照简单但费内存。实际系统常结合：粗粒度快照 + 细粒度命令。判断标准：**撤销靠"重放反向操作"还是"恢复过去状态"？**

## 3 中介者 vs 备忘录

| 维度 | 中介者 | 备忘录 |
| --- | --- | --- |
| 核心 | 收拢对象交互 | 捕获并恢复状态 |
| 结构 | 星形（对象→中介） | Originator + Memento + Caretaker |
| 解决 | 交互过乱 | 状态需回溯 |
| 代价 | 中介上帝化 | 快照内存成本 |
| 现代形态 | 消息总线、UI 协调器 | 不可变快照、UNDO 日志 |

**辨析｜易错点：** 两个模式一个管"对象间的现在"，一个管"对象自己的过去"——方向不同。中介者改变"谁认识谁"，备忘录改变"状态回到何时"。别把它们放进同一张决策表。

## 4 小结

- **中介者**用中枢对象收拢对象交互，把网状变星形，降低对象间耦合。
- 中介者 vs 观察者：多对多协调 vs 一对多广播。
- **备忘录**捕获对象状态快照并支持恢复，且不破坏封装。
- 备忘录撤销靠"恢复状态"，命令撤销靠"反向操作"，可结合使用。
- 中介者管"现在的交流"，备忘录管"过去的状态"，方向不同。

在下一节，我们看行为型里最常用的两个——**观察者与状态**。
