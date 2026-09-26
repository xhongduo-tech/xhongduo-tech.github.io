---
title: Treap
date: 2026-09-08
section: cs
---

# Treap

<div class="epigraph">
<p>键守 BST 序，堆序交给随机优先级；期望高度对数，旋转次数跟随机 BST 同分布。</p>
<footer>—— 据 Seidel and Aragon, Randomized Search Trees, Algorithmica 1996；Cormen, Leiserson, Rivest and Stein 整理</footer>
</div>

[上一课](/cs/sparse-set)收束了序列区间。有序字典主干已有[BST](/cs/bst)、[AVL](/cs/avl-rotate)、[红黑直觉](/cs/rbtree-intuition)，着色与平衡因子都是确定修复。[跳表](/cs/skip-list)用随机层高。本课不重画红黑 case。缺口是 Treap：随机堆优先级 + BST 键，期望 $O(\log n)$，实现是旋转。

## 问题

确定平衡要写一长串旋转分类。随机 BST（按随机插入序）期望高度对数，但不能按任意插入序保证。Treap 给每个键独立均匀随机的优先级 $p(k)$，树同时满足：中序为键序，堆序为优先级（通常小根）。形状唯一：就是按 $p$ 排序插入 BST 的结果。缺口是**用随机优先级代替随机插入序**，从而任意更新序列下期望仍对数。

<span class="marginnote">术语翻译：Treap = Tree + Heap——每个节点带两套规矩：键要满足左小右大的 BST 序，随机优先级要满足父小于子的堆序。两套规矩同时成立且唯一的那棵树，就是 Treap。</span>

<span class="marginnote">Seidel–Aragon 称之为 randomized search tree。split/merge 实现与笛卡尔树插入同构，竞赛里常用无显式旋转的分裂合并。</span>

## 方法

插入：按 BST 挂叶，再沿堆序上旋。删除：转到叶再摘，或先把优先级改成「无穷」再下旋。split($T,k)$：按键切成 $\lt k$ 与 $\ge k$ 两棵 treap；merge 假定左树键全小于右树，按根优先级决定谁当根。期望 $O(\log n)$。

```mermaid
flowchart TD
  KEY["键: BST 序"] --> NODE["节点"]
  PRI["随机优先级: 堆序"] --> NODE
  NODE --> ROT["上旋 / 下旋恢复堆"]
```

与跳表：都是随机期望对数、保序。Treap 是树，中序递归；跳表是多层链表。

<span class="marginnote">直觉类比：Treap 像组织架构图——按中序走时工号从小到大（BST 序），但谁当谁上级由随机抽的资历号（优先级）决定；抽签越随机，层级越接近平衡的对数楼。</span>

## 机制

期望分析与随机 BST 相同：根是优先级最小的键，左右规模均匀于键秩。最坏仍可退化（优先级运气差），合同写期望，与跳表一致。不要用时间戳当优先级除非能证明随机性。

隐式 treap 用下标当键（子树 size），可当可分裂序列，本课点名：键变成位置，机制仍是 split/merge。

<span class="marginnote">常见误区：以为 treap 永不退化。「期望对数」说的是随机优先级下的平均行为——优先级若可预测（比如直接用时间戳），对手能构造出退化成链的更新序列；对抗场景请用确定平衡的 AVL/红黑。</span>

```mermaid
flowchart TD
  SEQ["依次插入键 1,2,3,4,5"] --> BST["普通 BST：<br/>全落右子树，退化成链<br/>查 5 要走 5 步"]
  SEQ --> TREAP["Treap：优先级随机<br/>如 p=[3,9,1,7,5]<br/>键 3 的优先级最小，当根"]
  TREAP --> H["期望高度约 log n<br/>n = 100 万时约 20 层<br/>只有优先级运气极差才退化"]
  BST -.->|"键序不受插入顺序影响<br/>形状由优先级唯一决定"| TREAP
```

## 边界

本课不把并发 treap、或确定 treap（哈希优先级）的对抗分析写完。实时最坏路径用 AVL/红黑。下一课伸展树去掉随机与平衡位，改用摊还。

后课默认：随机平衡 BST 可以是 Treap。无额外随机、靠访问旋转的是 splay。

## 小结

- Treap：BST 键 + 随机堆优先级，期望对数。
- split/merge 与旋转等价。
- 摊还、无随机的自调整是下一课。
- 出处：Seidel and Aragon, *Algorithmica*, 1996；Cormen et al.。
