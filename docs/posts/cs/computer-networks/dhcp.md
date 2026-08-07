---
title: 动态主机配置协议（DHCP）
date: 2026-08-07
---

# 动态主机配置协议（DHCP）

<div class="epigraph">
<p>新设备接入网络，就像新人入职：DHCP 是那个「一键办理」的行政——自动发工牌（IP）、指办公室（网关）、给通讯录（DNS）。</p>
<footer>—— 网络教材中的通俗说法</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机网络 ｜ 谢希仁《计算机网络》§6.7 ｜ 2026-08-07</p>
</div>

## 为什么从 DHCP 开始

你的电脑、手机接入任何网络（家里、公司、咖啡馆），几乎都不用手动配置 IP——插上线、连上 Wi-Fi 就能上网。背后的功臣是 **DHCP（Dynamic Host Configuration Protocol，动态主机配置协议）**：**自动为主机分配 IP 地址、子网掩码、默认网关、DNS 服务器等网络参数。**<span class="marginnote">DHCP 解决的核心问题：<strong>新设备接入网络时，网络参数从哪来？</strong>手动配置太麻烦（还要懂网络）、容易出错（IP 冲突）、不可扩展。DHCP 让设备「插上就能用」——<strong>「即插即用（plug-and-play）」</strong>是它的设计目标。它是应用层协议，但服务对象是网络配置本身。</span>

这一节讲：**DHCP 的 DORA 四步、租约机制、以及它基于 UDP 广播的特殊性。**

## 1 DORA：DHCP 的「四步曲」

DHCP 客户端获取配置的过程，记作 **DORA**——四个英文单词的首字母：<span class="marginnote"><strong>D</strong>iscover（发现）、<strong>O</strong>ffer（提供）、<strong>R</strong>equest（请求）、<strong>A</strong>ck（确认）。因为客户端还没有 IP，前两步必须用<strong>广播</strong>：客户端广播 <strong>DHCP Discover</strong>（「谁给我发个 IP？」）→ 服务器回 <strong>DHCP Offer</strong>（「给你这个配置」）→ 客户端广播 <strong>DHCP Request</strong>（「我要用它」）→ 服务器回 <strong>DHCP Ack</strong>（「好的，生效」）。<strong>「Discover 广播问、Offer 单播给、Request 广播选、Ack 确认用」</strong>是 DORA 的完整画面。</span>

| 步骤 | 报文 | 方向 | 内容 |
| --- | --- | --- | --- |
| D | DHCP Discover | 客户端→广播 | 「我要 IP，谁提供服务？」 |
| O | DHCP Offer | 服务器→客户端 | 「给你这个 IP 与配置」 |
| R | DHCP Request | 客户端→广播 | 「我选这个服务器/配置」 |
| A | DHCP Ack | 服务器→客户端 | 「确认，生效」 |

**辨析｜易错点：** **Discover 与 Request 都是广播**——因为客户端还没有 IP、不知道服务器在哪。而 **Offer 与 Ack 可以单播**（服务器知道客户端的 MAC）。**「客户端广播，服务器可单播」**是 DORA 的方向规律。另一个易错点：**Request 广播的一个作用是多服务器场景下的「择一」**——客户端广播告诉所有服务器「我选了某台，其余请收回 Offer」。

## 2 租约：IP 不是「永久」，是「租」

DHCP 分配的 IP 有**租期（lease time）**——不是永久占用，而是「租用一段时间」。<span class="marginnote">租约机制的价值：<strong>IP 地址可以回收复用</strong>。设备下线、租约到期，IP 回到地址池，分配给别的设备。DHCP 客户端在租约过半时会主动请求<strong>续租（renewal）</strong>。这套「租约 + 续租 + 回收」让有限地址池服务大量临时设备——咖啡馆的地址池只有几十个 IP，却能服务成百上千的流动顾客。</span>

- **租期**：DHCP 分配的 IP 有效期（如 24 小时）。
- **续租**：租期过半，客户端向 DHCP 服务器请求续租。
- **回收**：租约到期未续，IP 回收到地址池重新分配。

**辨析｜易错点：** **DHCP 是「租」不是「送」**——IP 有租期、会回收。续租发生在「租期过半」时（T1 时间点），而不是快到期才续。**「租约过半续租」**是 DHCP 续租的经典细节。

## 3 DHCP 的技术细节：端口与广播

DHCP 的技术实现有几个值得注意的点：<span class="marginnote"><strong>端口</strong>：DHCP 客户端用 <strong>UDP 68</strong>，服务器用 <strong>UDP 67</strong>。<strong>基于 UDP</strong>——因为发现阶段无法用 TCP（还没有 IP、还没建立连接）。<strong>广播机制</strong>：Discover 是 <code>255.255.255.255</code> 广播（受限广播），Request 也是广播。<strong>跨网段</strong>：DHCP 服务器与客户端可能不在同一网段，靠<strong>中继代理（relay agent）</strong>把广播转成单播转发给服务器。</span>

- **端口**：客户端 UDP 68、服务器 UDP 67。
- **基于 UDP**：发现阶段没法建 TCP。
- **广播**：Discover/Request 用广播。
- **跨网段**：靠 DHCP 中继代理（relay agent）转发。

**辨析｜易错点：** **DHCP 的广播问题**：广播只在同一网段内传播，若 DHCP 服务器在别的网段，客户端广播到不了——所以用**中继代理**（通常是路由器）把广播请求**单播转发**给服务器。**「DHCP 默认广播，跨网段靠中继」**是 DHCP 部署的关键。还有一个易错点：**DHCP 报文里包含「魔饼干（magic cookie）」与各种选项**——用选项字段携带网关、DNS、租期等参数。

## 4 DHCP 与 BOOTP：从「老前辈」到 DHCP

DHCP 的前身是 **BOOTP（Bootstrap Protocol，引导协议）**——用于无盘工作站在启动时获取 IP 并下载操作系统。<span class="marginnote">DHCP 是 BOOTP 的扩展：<strong>BOOTP 静态分配（管理员手工指定每个 MAC 对应固定 IP），DHCP 动态分配（地址池 + 租约）</strong>。DHCP 兼容 BOOTP 报文格式，但增加了「租约」「动态地址池」「选项自动配置」等能力。<strong>「BOOTP 是静态点名，DHCP 是动态摇号」</strong>——DHCP 比老前辈灵活得多。</span>

| 对比维度 | BOOTP | DHCP |
| --- | --- | --- |
| 分配方式 | 静态（固定对应） | 动态（地址池 + 租约） |
| 租约 | 无（永久） | 有（可回收） |
| 配置能力 | 简单 | 丰富（网关、DNS、选项） |
| 现状 | 历史遗产 | 现行标准 |

**辨析｜易错点：** **DHCP 不是凭空发明，而是 BOOTP 的进化**——两者报文格式兼容。考试若问「DHCP 与 BOOTP 关系」，答「DHCP 是 BOOTP 的扩展，增加了动态分配与租约」即可。**「动态」是 DHCP 对 BOOTP 的「static」最核心的改进**。

## 5 小结

- **DHCP 的职责**：自动为主机分配 IP、掩码、网关、DNS——即插即用。
- **DORA 四步**：Discover（广播问）→ Offer（提供）→ Request（广播选）→ Ack（确认）。
- **租约机制**：IP 有租期、过半续租、到期回收——地址池服务海量临时设备。
- **技术细节**：UDP 67/68、广播通信、跨网段靠中继代理。
- **与 BOOTP**：DHCP 是 BOOTP 的动态扩展。
- **即插即用**：DHCP 让网络配置「零人工」，是现代网络的基础设施。

在下一节，我们将认识应用层的「去中心化明星」——**基于 P2P 的应用：文件分发与 BitTorrent**。
