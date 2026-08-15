---
title: 字典与映射
date: 2026-08-07
---

# 字典与映射

<div class="epigraph">
<p>世界不是一串列表，而是一本字典：每个名字，都有一个含义。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 《Python编程：从入门到实践》（第3版）第6章 ｜ 2026-08-07</p>
</div>

## 为什么从字典开始

列表用下标访问元素，下标是 0、1、2……的整数。但真实数据往往有「名字」而非「编号」：用户 `alice` 的年龄、商品 `A001` 的库存、省份「广东」的 GDP。这种**「键 → 值」**的映射，就是**字典（dictionary）**，简称 `dict`。

字典是 Python 使用频率最高的数据结构之一，也是 JSON 数据、HTTP 参数、配置文件的天然对应物——理解字典，就等于提前理解了后端与数据科学里的半壁江山。

## 1 字典：键值对的容器

**字典（dict）**：用花括号括起的一组「键: 值」对，按键取值、按键修改。

```python
alien_0 = {"color": "green", "points": 5}
print(alien_0["color"])        # 'green'
alien_0["points"] = 10         # 修改
alien_0["speed"] = "medium"    # 新增键
del alien_0["color"]           # 删除键
```

**重点：键必须唯一且不可变。** 字符串、数字、元组可以作为键；列表这类可变对象不能——因为可变键会让「按键查找」失去确定性。<span class="marginnote">为什么键要不可变？字典底层是<strong>哈希表</strong>：用键的哈希值定位存储位置。如果键中途变了，哈希值也变，旧位置就再也找不到它。这个概念在第一级《离散数学》的哈希一节、以及《数据结构》专题里会被彻底展开。</span>

用 `get()` 安全取值，避免键不存在时报错：

```python
print(alien_0.get("color", "unknown"))   # 键不存在时返回默认值
```

`dict.get(key, default)` 与 `setdefault(key, default)`（不存在才写入）是「容忍缺失」的两大利器，比裸用 `dict[key]` 稳健得多。

## 2 遍历字典：keys、values 与 items

字典天然适合遍历。三种遍历角度：

```python
favorite_languages = {
    "jen": "python",
    "sarah": "c",
    "edward": "rust",
    "phil": "python",
}
for name, lang in favorite_languages.items():   # 键值对
    print(f"{name.title()} 喜欢 {lang}")
for name in favorite_languages.keys():          # 只看键
    pass
for lang in favorite_languages.values():        # 只看值
    pass
```

`items()` 返回「键值对」的集合，配合 `for name, lang in ...` 解包，是遍历字典的标准姿势。`keys()`、`values()` 返回视图（view），随字典实时更新。<span class="marginnote">Python 3.7 起，字典<strong>保持插入顺序</strong>——这是语言规范，不是实现巧合。在 JSON 配置、模板渲染这些「顺序有含义」的场景里，这个保证很关键。</span>

字典也有**推导式**：`{k: v for k, v in items if 条件}`，与列表推导同构。

**遍历与排序结合**是常见需求——字典默认无序，想按序输出时用 `sorted()` 包一层：

```python
for name in sorted(favorite_languages.keys()):
    print(name.title())
```

`keys()`、`values()`、`items()` 返回的都是**视图（view）**，可直接传给 `sorted()`、`list()`、`len()`。于是「按字母序列出所有用户」这样的任务，一行 `sorted(favorite_languages)` 就完成——这也是字典在实际脚本里最常见的用法之一。

**字典的构建常是「逐步生长」的**：先建空字典 `{}`，再在循环里不断 `d[key] = value` 填充。导入数据、合并配置、统计分组时，这是标准姿势：

```python
grades = {}
for name, score in student_records:
    grades[name] = score
```

## 3 嵌套字典：字典里的字典

现实数据往往是多层的：一个用户包含多个字段，一个字段又可能是字典。

```python
users = {
    "aeinstein": {"first": "albert", "last": "einstein", "location": "princeton"},
    "mcurie":    {"first": "marie",  "last": "curie",     "location": "paris"},
}
for username, info in users.items():
    print(f"{username}: {info['first']} {info['last']}")
```

**读取嵌套值时要「逐层下探」。** `info['first']` 需要先保证 `info` 是一个字典。对「字段可能缺失」的情况，用 `get()` 提供默认值更稳：

```python
for username, info in users.items():
    lang = info.get("language", "未填写")
    print(f"{username}: {lang}")
```

`info.get("language", "未填写")` 在键不存在时返回默认值，而不是抛 `KeyError`——读取来自外部（如 API 返回的 JSON）的嵌套数据时，这是必须养成的习惯。

**重点：嵌套不是新语法，只是「值可以是任意对象」。** 字典的值可以是列表、另一个字典，乃至后面要学的类实例。JSON 数据的树状结构，就是用嵌套字典 + 列表拼出来的。<span class="marginnote">Python 的 JSON 与字典几乎一一对应：JSON 对象 `{"a": 1}` 就是字典。用 `json.loads()` 把 JSON 文本读成字典、用 `json.dumps()` 再写回去，是数据持久化的第一步（见《输入输出与文件读写》）。</span>

## 4 核心对比表：列表与字典的分工

| 维度 | 列表 `list` | 字典 `dict` |
| --- | --- | --- |
| 访问方式 | 按整数下标 `lst[0]` | 按任意键 `d["name"]` |
| 键 | 隐式下标（0, 1, 2…） | 显式、不可变、唯一 |
| 顺序 | 始终有序 | 3.7 起保插入序 |
| 查找复杂度 | 线性 $O(n)$ | 哈希近似 $O(1)$ |
| 典型用途 | 有序数据、坐标、队列 | 字段表、映射、配置、JSON |

**核心观察：字典用「哈希」换速度。** 列表找元素最坏要逐个比对（$O(n)$），字典却能几乎一次定位（$O(1)$）——代价是键必须可哈希。当你要「按名字查值」，字典总是比列表更合适；当你要「按位置取段」，列表总是更合适。二者互补，而非替代。<span class="marginnote">复杂度记号 $O(1)$、$O(n)$ 来自第二级《算法与复杂度》——现在只需记住：字典查找不随数据量线性变慢，这是它被用作索引的根本原因。</span>

## 5 小结

- 字典 `{}` 存「键: 值」，键**唯一且不可变**，底层是哈希表。
- 用 `d[key]` 取值、`d[key] = v` 增改、`del d[key]` 删除、`get()` 安全取值。
- 遍历用 `items()`（键值对）、`keys()`、`values()`；3.7 起保插入顺序。
- 字典的值可以是任意对象，嵌套字典天然表达 JSON 树形数据。
- 列表按位置、字典按名字；字典查找近似 $O(1)$，适合做「索引」。

在下一节，我们将学习程序如何与用户对话——`input()` 输入与 `while` 循环，让程序在运行中接收指令。
