---
title: Token 标准（ERC-20/721）
date: 2026-08-07
---

# Token 标准（ERC-20/721）

<div class="epigraph">
<p>标准不是给机器看的，是给生态系统看的——当所有人都遵守同一接口，可组合性就诞生了。</p>
<footer>—— 法比安 · 沃格尔施泰勒（Fabian Vogelsteller），ERC-20 作者</footer>
</div>

<div class="article-byline">
<p>第三级 · 区块链 ｜ EIP-20 / EIP-721 · 《区块链技术指南》第7章 ｜ 2026-08-07</p>
</div>

## 为什么从 Token 标准开始

以太坊上最有价值的东西不是 ETH 本身，而是**可编程资产（Token）**。为什么会有「标准」？因为如果每个项目都自定义合约接口，钱包、交易所、DeFi 协议就无法通用交互——**接口统一 = 生态可组合**。**ERC-20** 定义了同质化代币（FT，每枚完全等价），**ERC-721** 定义了非同质化代币（NFT，每枚独一无二）。这两份标准是「Token 经济」的地基，也是理解 DeFi、NFT、稳定币的前提。

## 1 什么是 EIP 与 ERC

**EIP（Ethereum Improvement Proposal）**：以太坊改进提案，社区提议改进的标准文件。**ERC（Ethereum Request for Comments）**：EIP 的一个子类，专门定义**应用层标准**（如 Token 接口）。<span class="marginnote">EIP 的治理流程：作者提出 → 社区评审 → 合入最终状态。ERC-20 由 Fabian Vogelsteller 于 2015 年提出，2017 年正式确立，成了整个 ICO 浪潮与 DeFi 的基础。标准化的力量在于：钱包只要实现 ERC-20 的读取接口，就能显示任何符合标准的代币。</span>

## 2 ERC-20：同质化代币

**ERC-20** 定义了一组必须实现的函数与事件，让任何合约都能成为「可互换代币」。

```solidity
interface IERC20 {
    function totalSupply() external view returns (uint256);
    function balanceOf(address account) external view returns (uint256);
    function transfer(address to, uint256 amount) external returns (bool);
    function allowance(address owner, address spender) external view returns (uint256);
    function approve(address spender, uint256 amount) external returns (bool);
    function transferFrom(address from, address to, uint256 amount) external returns (bool);

    event Transfer(address indexed from, address indexed to, uint256 value);
    event Approval(address indexed owner, address indexed spender, uint256 value);
}
```

关键机制是**授权（allowance）**：代币所有者不用把资产转给合约，只需 `approve` 授权合约代为花费（上限 `allowance`）。这为 DEX、借贷协议打开了「代付」能力。

**公式解析：transferFrom 的双人授权流**

$$
\text{allowance}[owner][spender] \ge \text{amount} \implies \text{spender 可代扣}
$$

- **第一步**：`approve(spender, N)` 让 `owner` 设定 `allowance[owner][spender] = N`。
- **第二步**：`transferFrom(owner, to, amount)` 在扣款后把 `allowance` 减掉 `amount`。
- **第三步**：合约（如 DEX）持有这个授权，就能在交易时**代为划转**用户代币——这是「授权-代付」模式，也是闪电贷等复杂操作的基础。注意 `approve` 的经典坑：`approve` 到非零值后再 `approve` 到另一个非零值，某些实现允许攻击者利用「时间差」双花旧授权（所以 OpenZeppelin 建议先 `approve(0)` 再设置）。

## 3 ERC-721：非同质化代币

**ERC-721（NFT）**：每个 Token 有唯一 ID，不可分割、不可互换。

```solidity
interface IERC721 {
    function balanceOf(address owner) external view returns (uint256);
    function ownerOf(uint256 tokenId) external view returns (address);
    function approve(address to, uint256 tokenId) external;
    function getApproved(uint256 tokenId) external view returns (address);
    function setApprovalForAll(address operator, bool approved) external;
    function isApprovedForAll(address owner, address operator) external view returns (bool);
    function transferFrom(address from, address to, uint256 tokenId) external;
    event Transfer(address indexed from, address indexed to, uint256 indexed tokenId);
}
```

与 ERC-20 的关键差异：

| 维度 | ERC-20 | ERC-721 |
| --- | --- | --- |
| 可互换性 | 同质（1 枚 = 任意 1 枚） | 非同质（每枚唯一） |
| 转账单位 | `amount`（数量） | `tokenId`（具体哪一枚） |
| 授权粒度 | 按 spender 授权额度 | 按 tokenId 授权 + `setApprovalForAll` |
| 典型用途 | 代币、稳定币、治理币 | 数字艺术品、域名、票据、游戏资产 |

**元数据（metadata）**：NFT 通常把描述信息（名称、图片 URL、属性）存在链下，链上只存 `tokenURI(tokenId)` 返回的 JSON 地址。<span class="marginnote">「链上只存指针，链下存数据」是 NFT 的常见架构：`tokenURI` 返回的 JSON 里的 `image` 字段指向一个 IPFS/中心化 URL。真正的「链上 NFT」（完全链上存储）成本极高——这是 2023 年后「链上艺术」运动的技术动机。</span>

## 4 更现代的标准家族

- **ERC-1155**：多代币标准，一个合约同时支持同质与非同质代币，批量转账省 Gas——游戏资产生态的主流。
- **ERC-4626**：收益金库标准，统一了「存入生息资产、按份额记账」的接口，被借贷与收益率协议广泛采用。
- **ERC-20 包装（wrapped）**：把 ETH 包装成 `WETH`（满足 ERC-20 接口），让 ETH 能参与 ERC-20 生态——DEX 里几乎所有交易对都涉及 WETH。

**辨析｜易错点：「NFT」不等于「图片」，更不等于「版权」。** NFT 的链上部分是「唯一 tokenId + 所有权记录」，它证明的是「你拥有这个 tokenId」，至于 tokenId 指向的图片是谁创作的、是否侵权、是否可商用，链上**一概不知**。**「买入 NFT = 拥有版权」是常见的认知误区**。另一个易错点：**ERC-20 的 `transfer` 返回 bool，但不返回值也不一定代表失败**——老式合约不按标准返回时，依赖返回值的钱包会误判，所以现代库（如 OpenZeppelin SafeERC20）会额外检查返回值。

## 5 标准之上：资产的可组合性

Token 标准的真正价值是**可组合性**：一个符合 ERC-20 的代币，天然能被任何 DEX、借贷协议、钱包、交易所接纳；一个 ERC-721，天然能被市场、游戏、身份系统调用。资产从「隔离的应用内对象」变成「生态共享的原语」，这正是 DeFi 乐高积木得以搭建的原因。

## 6 标准实现中的安全细节

「符合标准」不等于「实现安全」——历史上大量 Token 漏洞恰恰出在「标准实现」里。几个必须知道的细节：

**approve 的竞态（race condition）**：经典的「双层 approve」攻击。流程：

1. 用户 `approve(恶意合约, 100)` 授权 100 个代币。
2. 恶意合约立即 `transferFrom(user, 自己, 100)` 转走 100。
3. 用户再次 `approve(恶意合约, 50)`——如果实现不先归零，`allowance` 可能仍是「旧值 + 新值」或「覆盖逻辑有漏洞」，让恶意合约多转。

**防御**：OpenZeppelin 在 0.8 之后的 `approve` 直接覆盖（无竞态），并推荐 `increaseAllowance`/`decreaseAllowance` 替代「从非零改非零」。**凡是需要「改授权额度」的场景，都应该用「增量/减量」而不是「直接覆盖」**。<span class="marginnote">approve 竞态是 2018 年前后 DeFi 被盗的经典漏洞之一。现代最佳实践是「先 approve(0) 再 approve(amount)」，或直接用 `safeApprove`/`forceApprove` 系列——这提醒我们：<strong>ERC-20 的接口「只是接口」，安全性取决于「实现细节」</strong>。标准定义「能做什么」，实现决定「是否安全」。</span>

**transfer 返回值的坑**：标准要求 `transfer` 返回 bool，但**早期大量代币不返回值**（如 USDT 的旧实现）。依赖返回值的合约（`require(token.transfer(...))`）会在「不返回值」的代币上直接 revert。**SafeERC20** 的存在正是为处理这种「标准不统一」的现实——它检查返回字节，没有返回值就「猜一个成功」。

**transferFrom 与「授权-代付」的权限边界**：`transferFrom` 允许 spender 花 owner 的钱，但**spender 不能超过 allowance**。若合约里写 `transferFrom(from, to, max)` 而不检查授权，等于把 owner 的钱「全部划走」——这种「超授权划转」是 2021 年多起 DEX 漏洞的根源。

**ERC-721 的转移钩子**：`transferFrom` 直接转移，而 `safeTransferFrom` 会检查「接收方是否为合约、是否实现了 `onERC721Received`」——**用 `transferFrom` 把 NFT 转给一个不兼容的合约，NFT 会「卡死」在合约里无法取出**。这是 NFT 被「黑洞合约」吞掉的原因，也是「永远用 `safeTransferFrom`」这条铁律的来源。

**元数据与版税**：ERC-721 的 `tokenURI` 指向链下 JSON；若项目方关闭服务器，`tokenURI` 指向 404，NFT「只剩链上一个 ID」。**「链上资产」的「链上价值」往往依赖「链下元数据可用性」**——这是 NFT 的隐蔽风险点。ERC-2981 标准补充了「链上版税信息」，让 NFT 转售时自动向创作者分配版税。

**公式解析：授权链的「信任传递」**

$$
\text{资金风险} = \text{授权给不可信合约的额度} \times \text{该合约被黑的概率}
$$

- **第一步**：用户授权 DEX 无限额度（`approve(DEX, max)`）是便利，也是风险敞口——DEX 被黑，授权额度内的资金全部可被划走。
- **第二步**：最佳实践是「按需授权」——每次交易前只授权「本次所需额度」，交易后归零（`safeApprove` + 归零）。
- **第三步**：授权本质是「把资金支配权委托给合约」——**每次点击「批准」都是在签订一份「可被合约支配」的委托协议**，审慎程度应不亚于转账本身。

**辨析｜易错点：标准「版本」≠「最新最好」。** ERC-20 的「旧实现」在生态里仍然流通（USDT），新协议必须兼容「旧标准的不规范」——「符合 ERC-20」只说明「接口长得像」，不说明「行为正确」。**另一个易错点：NFT 的「稀缺」≠「价值」**——tokenId 的稀缺是协议保证的，但「稀缺」与「市场价值」之间隔着「需求」；「链上稀缺 + 链下无人问津」=「有价无市」。

## 7 小结

- **ERC-20** 定义同质化代币接口：`balanceOf`、`transfer`、`approve`、`transferFrom` + 事件；`allowance` 授权机制支撑「代付」。
- **ERC-721** 定义非同质化代币：按 `tokenId` 唯一寻址，授权分「单枚授权」与「全量授权」。
- 标准的意义在于**生态可组合**；`approve` 非零覆盖是经典安全坑。
- NFT 链上只存**所有权与指针**，元数据多在链下；「NFT = 版权」是常见误区。
- 现代标准家族：**ERC-1155**（多代币）、**ERC-4626**（金库）、**WETH**（包装 ETH）。

在下一节，我们把 Token 组装成完整的金融生态——去中心化交易所、借贷与稳定币：**DeFi 生态（DEX/借贷/稳定币）**。
