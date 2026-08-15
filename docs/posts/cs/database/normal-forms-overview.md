---
title: 范式概览：1NF、2NF、3NF、BCNF
date: 2026-08-07
---

# 范式概览：1NF、2NF、3NF、BCNF

<div class="epigraph">
<p>每一级范式都是对上一级的修正——逐步剔除依赖中的瑕疵。</p>
<footer>—— 埃德加 · 科德（E. F. Codd，关系模型之父）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数据库 ｜ 《数据库系统概念》 第8章 规范化理论 ｜ 2026-08-07</p>
</div>

## 为什么从范式概览开始

前几节备齐了函数依赖、闭包、候选码这些工具。现在把它们串成**范式（normal form）**的阶梯：1NF、2NF、3NF、BCNF。范式是「关系模式好坏的等级认证」——每一级消除一类依赖瑕疵。这一节把四级范式的定义、相互关系与判定方法整体过一遍，给一张「范式总览图」。细节（分解算法、多值依赖、第四范式）在后续各节展开。<span class="marginnote">范式的思想史：科德 1970 年提出 1NF/2NF/3NF，博伊斯与科德 1974 年修正出 BCNF，之后又陆续有 4NF、5NF。<strong>范式不是越高级越好</strong>——3NF 就能保证「无损 + 依赖保持」，BCNF 可能牺牲依赖保持，实战大多停在 3NF 或 BCNF。</span>

## 1 第一范式（1NF）：原子性

**第一范式**：关系的每个属性都是**原子**的——不可再分，每个值都是单值而非集合、列表。

**违反 1NF 的例子**：`student(id, name, phones)`，其中 `phones` 存逗号分隔的多个电话；或一个单元格放一个集合。**1NF 是关系模型的底层约定**——第 7 章「多值属性拆子表」正是为了满足 1NF。

**辨析｜易错点：** 「原子」是**相对于使用方式**的。一个 JSON 字段存整段结构，若系统从不解析它、只整体存取，也可以算原子；若系统要查「含某 key 的记录」，它就该被拆。**1NF 不是绝对的，是查询需求相对的。**

## 2 第二范式（2NF）：消除部分依赖

**第二范式**：在 1NF 基础上，**每个非主属性都完全函数依赖**于每个候选码——不存在非主属性对候选码的**部分依赖**。

**部分依赖**：非主属性被候选码的**真子集**决定。典型场景是**组合候选码**：

**例如**：`takes(student_id, course_id, student_name, grade)`，候选码 `(student_id, course_id)`。

`student_name` 只依赖 `student_id`（候选码的一部分）——**部分依赖**。结果：学生姓名在每个选课行里重复，更新异常再现。

**解决办法**：拆成 `student(student_id, student_name)` 与 `takes(student_id, course_id, grade)`。

**核心要点：2NF 只处理「组合码下的部分依赖」。** 若候选码都是单属性，不存在部分依赖，模式自动满足 2NF。

## 3 第三范式（3NF）：消除传递依赖

**第三范式**：在 2NF 基础上，**不存在非主属性对候选码的传递依赖**——非主属性不能通过「中间属性」间接依赖候选码。

**传递依赖**：非主属性 $B$ 由非主属性（或非码属性）$A$ 决定，而 $A$ 由候选码决定，形成「码 → A → B」：

**例如**：`account(account_number, branch_name, branch_city)`，候选码 `account_number`。

`branch_name` 依赖 `account_number`，`branch_city` 依赖 `branch_name`——`branch_city` 对候选码**传递依赖**。支行城市信息随账户重复。

**解决办法**：拆成 `account(account_number, branch_name)` 与 `branch(branch_name, branch_city)`。

## 4 BCNF（博伊斯-科德范式）：左侧超码

**BCNF**：对**每个非平凡函数依赖** $\alpha \to \beta$，$\alpha$ **必须是超码**。

**BCNF 与 3NF 的区别**：3NF 只约束「非主属性」对码的依赖，BCNF 约束**所有**依赖（包括主属性之间的依赖）。所以存在「3NF 但非 BCNF」的模式——当两个候选码有重叠时。

**例子**：`dept_advisor(student_id, i_id, dept_id)`，约束为「每个学生最多一位导师，一位导师只在一个系」。候选码有两个：`(student_id, dept_id)` 与 `(student_id, i_id)`。依赖 $i\_id \to dept\_id$ 成立，但 $i\_id$ 不是超码——**违反 BCNF**，但满足 3NF（$dept\_id$ 是主属性，不属于「非主属性传递依赖」）。

**公式解析：BCNF 的判定条件**

模式 $R$ 满足 BCNF，当且仅当对 $F$ 中每个非平凡依赖 $\alpha \to \beta$：

$$
\alpha^+ = R \quad \text{（即 } \alpha \text{ 是超码）}
$$

- **第一步，取依赖**：逐个检查 $F$ 里的非平凡依赖 $\alpha \to \beta$（平凡依赖 $\beta \subseteq \alpha$ 自动满足）。
- **第二步，算闭包**：对每个 $\alpha$ 求 $\alpha^+$。
- **第三步，比较**：若 $\alpha^+ = R$，通过；否则违反 BCNF。
- **第四步，对照 3NF**：3NF 允许「$\beta - \alpha$ 全为主属性」的例外；BCNF 不允许任何例外——所以 BCNF ⊆ 3NF。

## 5 范式金字塔与判定总览

**核心要点（范式包含关系）：**

$$
\text{BCNF} \subset \text{3NF} \subset \text{2NF} \subset \text{1NF}
$$

| 范式 | 消除的瑕疵 | 判定要点 | 工程地位 |
| --- | --- | --- | --- |
| 1NF | 非原子属性 | 无集合/复合值 | 关系模型底线 |
| 2NF | 非主属性部分依赖 | 组合码下无「半码依赖」 | 过渡 |
| 3NF | 非主属性传递依赖 | 码→非码→非码 链 | **默认目标** |
| BCNF | 一切非超码依赖 | 每个依赖左侧是超码 | 最高（可能牺牲依赖保持） |

**辨析｜易错点：** 判定范式先**求候选码**，再分类**主/非主属性**，最后检查「部分/传递依赖」——三步缺一不可。最隐蔽的坑是「存在多个候选码」：漏算候选码会错判主属性，从而误判 3NF/BCNF。

## 6 范式判定的数值算例与术语速查

**把四个范式对同一个模式逐级判定一遍。** 设 `takes(student_id, course_id, student_name, dept_name)`，候选码 `(student_id, course_id)`。

- **1NF**：所有属性原子 ✅。
- **2NF**：`student_name` 只依赖 `student_id`（候选码真子集）——**部分依赖**，违反 2NF ❌。需拆出 `student(student_id, student_name, dept_name)`。
- **3NF**：拆后 `student_name` 依赖 `student_id`、`dept_name` 依赖 `student_name`？若 `student_name` 唯一则 `dept_name` 对 `student_id` 传递依赖——**违反 3NF** ❌，再拆 `student(student_id, student_name)` 与 `dept_student(student_name, dept_name)`。
- **BCNF**：逐检查每个非平凡依赖的左侧闭包是否覆盖全属性——全部满足则 BCNF ✅。

**数值算例：多候选码的判定陷阱** 设 `dept_advisor(s_id, i_id, dept_id)`，候选码 `(s_id, dept_id)` 与 `(s_id, i_id)`。

- 若只看一个候选码 `(s_id, dept_id)`，会漏判 `i_id` 是主属性。
- 正确判定：主属性 = 所有候选码属性的并集 = {s_id, i_id, dept_id}——全是主属性，于是 3NF 的「非主属性传递依赖」检查全部放行——**模式满足 3NF 但违反 BCNF**（`i_id → dept_id` 左侧非超码）。
- 教训：**先求全部候选码，再判主属性**——漏候选码 = 误判范式。

**辨析｜易错点：** 范式的包含关系 BCNF ⊂ 3NF 意味着「满足 BCNF 一定满足 3NF」，但反过来不成立。**判定时从 1NF 逐级向上检查**，每级都要找候选码、分类主/非主属性、检查对应依赖——这是规范化笔试的标准流程。

<span class="marginnote">范式的现实意义不在「追求最高级」，而在「<strong>用范式诊断设计问题的类型</strong>」：1NF 问题 = 原子性坏了，2NF 问题 = 组合码拆得不干净，3NF 问题 = 传递依赖藏了冗余，BCNF 问题 = 主属性之间还有依赖。每种问题对应一种「拆表」的修法。</span>

### 术语速查

| 术语 | 含义 |
| --- | --- |
| 1NF | 属性原子、无集合值 |
| 2NF | 无非主属性部分依赖 |
| 3NF | 无非主属性传递依赖 |
| BCNF | 每个非平凡依赖左侧是超码 |
| 主属性 | 属于某个候选码的属性 |
| 候选码 | 极小的超码 |

## 7 小结

- 范式阶梯：**1NF（原子）→ 2NF（无部分依赖）→ 3NF（无传递依赖）→ BCNF（左侧皆超码）**。
- 2NF 只针对**组合候选码**；单属性码自动满足 2NF。
- 3NF 是**默认工程目标**：总能做到「无损 + 依赖保持」。
- BCNF 约束所有依赖，**可能牺牲依赖保持**；判定靠「每个非平凡依赖左侧闭包 = 全属性」。
- 包含关系 BCNF ⊂ 3NF ⊂ 2NF ⊂ 1NF。

在下一节，我们给闭包加一套推理公理——**Armstrong 公理与正则覆盖**，让依赖推导系统化。
