---
title: DAC / MAC / RBAC
date: 2026-09-08
section: cs
---

# DAC / MAC / RBAC

<div class="epigraph">
<p>自主访问控制让属主改 ACL；强制访问控制用标签策略挡住「自愿泄露」；角色把权限聚成岗位而不是每人一条。</p>
<footer>—— 据 Lampson 保护矩阵；Bell–LaPadula 对照；Sandhu 对 RBAC；Saltzer and Schroeder</footer>
</div>

[上一课](/cs/acl-least-priv)给出主体–客体–操作与最小特权。本课不重画矩阵。缺口是政策谁来写：属主（DAC）、系统标签（MAC）、岗位角色（RBAC）。能力下一课对照矩阵的另一种存放。三者都能表达同一矩阵的子集，失败模式不同。

## 问题

最小特权课留下矩阵。Unix 文件 mode 是 DAC：属主可 chmod 把读权给别人，木马也能借属主去改。MAC：级别与范畴，写不上读下一类规则（教学用 BLP 直觉），用户不能自愿改标签以泄密。RBAC：权限授给角色，人授角色，便于岗位变动。缺口是**三种政策来源**。

<span class="marginnote">直觉类比：DAC 像"门钥匙我自己借人"——属主有权借，也可能被骗着借；木马顶着属主身份跑一段代码，就能名正言顺把密件转发出去。MAC 干的事就是把"借钥匙"这个环节本身没收。</span>

<span class="marginnote">MLS 的形式模型有争议与例外。主干只取「策略不由文件属主随意改」这一层。RBAC 可叠在 DAC 上。</span>

## 方法

实现：DAC 用 ACL 或 mode；MAC 用标签 + 引用监视器；RBAC 用角色表。检查总在引用监视器，与[内核与用户态](/cs/kernel-user) 一致。本课不把 SELinux 类型列表当必背。

<span class="marginnote">引用监视器可以想成"所有门都归同一个保安"：谁想碰客体都必经这一个检查点，且它不可绕过、不可收买（由内核实现）。检查集中在一处，事后审计才有单点可查。</span>

```mermaid
flowchart TD
  MAT["保护矩阵"] --> DAC["属主改 ACL"]
  MAT --> MAC["系统标签策略"]
  MAT --> RBAC["角色聚合权限"]
```

## 机制

DAC 灵活、难防内部误授。MAC 服务机密性轴的强政策，可用性与兼容性代价高。RBAC 降的是管理复杂度，不是新的形式保证。后课能力把「行」发给主体当票。

```mermaid
flowchart TD
  U["普通级用户进程"] -->|"读机密级"| X["拒绝：不上读"]
  U -->|"写公开级"| Y["拒绝：不下写"]
  U -->|"读同级"| OK["放行"]
  X --> EFF["机密信息被标签圈住，流不出格"]
  Y --> EFF
```

<span class="marginnote">数字实例：500 名员工、8 种岗位、40 项权限。逐人授权要维护 500 份个人清单，岗位一变就得逐条找着改；RBAC 只维护 8 份角色定义加一条"某人任某岗"，离职时删一行角色指派就撤干净了。</span>

## 边界

本课不引入基于属性的 ABAC 全貌当主干。矩阵按行发放不可伪造引用，下一课能力对 ACL。

后课默认：政策可来自属主、标签或角色。票与名单是两种表示。

## 小结

- DAC 属主做主；MAC 标签强制；RBAC 按岗位聚合。
- 引用监视器执行，不靠自愿遵守。
- 能力对照下一课。
- 出处：Lampson；Bell–LaPadula 对照；Sandhu RBAC；Saltzer and Schroeder。
