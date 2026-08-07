---
title: Solidity 语法基础
date: 2026-08-07
---

# Solidity 语法基础

<div class="epigraph">
<p>Solidity 是面向合约的高级语言，它的编译目标是 EVM 字节码——你的代码最终会成为全世界的共同执行规则。</p>
<footer>—— 以太坊官方文档（Solidity Documentation）</footer>
</div>

<div class="article-byline">
<p>第三级 · 区块链 ｜ Solidity 官方文档 · 《区块链技术指南》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Solidity 语法开始

要写智能合约，先学语言。**Solidity** 是部署在 EVM 上的主要高级语言，语法与 JavaScript 相似，但语义完全不同：它有真金白银、有 Gas、有不可变性、有「谁调用、谁付钱」的执行模型。这一节建立 Solidity 的最小心智模型：合约结构、类型系统、函数可见性、数据位置——先能读懂、能改一个最小合约，再在下一节谈完整开发实践。

## 1 一个最小合约的结构

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

contract Counter {
    uint256 public count;

    function increment() external {
        count += 1;
    }
}
```

逐行拆解：

- `pragma solidity ^0.8.20`：声明编译器版本，`^` 表示兼容 `0.8.x`。
- `contract Counter`：合约定义，类似其他语言的类（class），但**实例部署后不可变**。
- `uint256 public count`：状态变量，`public` 自动生成读函数，持久化存储在链上。
- `function increment() external`：函数，`external` 表示只能从合约外部调用。

**核心概念：状态变量 = 区块链状态。** Solidity 里的存储变量不是内存变量，它们被写进世界状态（storage），任何修改都要付 Gas、被全网记录——**这是与普通语言最大的思维差异**。<span class="marginnote">写 Solidity 时，「存储（storage）」是最贵的资源：一次 `SSTORE` 消耗 20000 Gas（约是新地址部署量级）。好的合约设计把「链上存什么」当成架构决策，而不是随手记变量。</span>

## 2 类型系统

Solidity 的类型分**值类型**与**引用类型**：

| 类别 | 类型 | 说明 |
| --- | --- | --- |
| 值类型 | `uint` / `int`（8–256 位） | 无符号/有符号整数，如 `uint256` |
| 值类型 | `bool`、`address`、`bytes32` | 布尔、20 字节地址、定长字节 |
| 值类型 | 枚举 `enum` | 有限取值集合 |
| 引用类型 | `string`、`bytes` | 动态长度数据 |
| 引用类型 | `array`、`mapping` | 数组、键值映射 |

**整数安全**：Solidity 0.8 起内置**溢出检查**——`uint8(255) + 1` 直接 revert，不再默默回绕。这是对历史上著名的整数溢出漏洞（如 2016 年 Parity 多签钱包被黑）的回应。<span class="marginnote">0.8 之前的 Solidity 默认不检查溢出（`unchecked` 才是默认行为），导致大量 DeFi 合约在 2020 年前后被「溢出攻击」掏空。0.8 之后默认检查，但 `unchecked` 块内仍需开发者自担风险。</span>

**mapping 的语法**：

```solidity
mapping(address => uint256) public balances;
```

`mapping` 是 Solidity 的核心容器：键值对存储，不迭代、不遍历、无长度——它被哈希映射到 storage slot 上，读写都是 $O(1)$。

## 3 函数可见性与修饰符

函数有四种可见性：

| 可见性 | 外部调用 | 内部调用 | 说明 |
| --- | --- | --- | --- |
| `public` | ✅ | ✅ | 对外可见，自动生成读函数 |
| `external` | ✅ | ❌ | 仅外部可调，省 Gas |
| `internal` | ❌ | ✅ | 本合约及继承合约 |
| `private` | ❌ | 仅本合约 | 最严格 |

**状态可变性修饰符**：

- `view`：只读状态，不修改（不消耗存储 Gas）。
- `pure`：连状态都不读，纯函数。
- `payable`：允许接收 ETH，是「收款函数」的必备修饰符。
- `nonpayable`（默认）：拒绝接收 ETH。

```solidity
function getBalance(address who) external view returns (uint256) {
    return balances[who];
}
```

## 4 公式解析：msg 与交易上下文

每个函数体内可访问一组全局变量，最核心的是 `msg`：

$$
\text{msg} = \{ \text{msg.sender}, \text{msg.value}, \text{msg.data}, \text{msg.sig} \}
$$

- **第一步**：`msg.sender` 是**直接调用者**的地址——在跨合约调用链中，它逐跳变化，不是「最初发起人」（那是 `tx.origin`）。
- **第二步**：`msg.value` 是随本次调用转入的 wei 数量（1 ETH = $10^{18}$ wei）。
- **第三步**：`payable` 函数才能接收 `msg.value`，非 payable 函数收到 ETH 会 revert。

**gasleft()**：返回剩余 Gas，配合 `require(gasleft() > threshold)` 做 Gas 保护（防「Gas 不足被卡死」的 DoS 类攻击）。

## 5 数据位置：storage / memory / calldata

这是 Solidity 初学者最容易踩的坑。引用类型必须标注数据位置：

| 位置 | 生命周期 | 花费 | 用途 |
| --- | --- | --- | --- |
| `storage` | 持久化在链上 | 最贵 | 状态变量 |
| `memory` | 函数调用期间 | 中等 | 临时变量 |
| `calldata` | 只读、函数参数 | 免费 | 外部传入的参数 |

**辨析｜易错点：赋值不是拷贝的语义差别。** `storage` 变量赋值给 `storage` 变量是**引用**（改一个影响另一个），`memory` 到 `storage` 才是拷贝。例如：

```solidity
// 危险：s2 只是 s1 的引用
SomeStruct storage s2 = s1;
s2.field = 1; // 同时改了 s1
```

另一个易错点：`memory` 数组之间赋值是引用，`memory` 传给 `storage` 才是拷贝——**记不清就检查文档，这类 bug 往往在审计时才暴露**。

## 6 从语法到实践

到这里，你能读懂合约的结构、类型、可见性、数据位置。但「能读」离「能安全部署」还很远——下一节我们走一遍完整的开发流程：写、测、部署、交互，并看看那些让合约安全的工程习惯。

## 7 小结

- 合约 = 状态变量 + 函数；状态变量**持久化在链上**，修改要付 Gas。
- 类型分**值类型**（`uint`/`address`/`bool`）与**引用类型**（`mapping`/`array`/`string`）；0.8 起默认溢出检查。
- 函数可见性（`public`/`external`/`internal`/`private`）× 可变性（`view`/`pure`/`payable`）共同定义调用边界。
- `msg.sender` 是直接调用者，`msg.value` 是转入金额；数据位置 `storage`/`memory`/`calldata` 决定生命周期与成本。

在下一节，我们用这些语法真正开发一个合约：环境搭建、编写、测试、部署与调用——**智能合约开发实践**。
