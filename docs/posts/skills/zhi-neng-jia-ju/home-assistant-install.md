---
title: Home Assistant 安装与设备接入
date: 2026-08-07
---

# Home Assistant 安装与设备接入

<div class="epigraph">
<p>让万物归于一，是每个智能家居玩家的执念。</p>
<footer>—— Home Assistant 社区格言</footer>
</div>

<div class="article-byline">
<p>第十级 · 智能家居设备安装与调试 ｜ Home Assistant 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么 HA 是跨生态方案的「万能插座」

前 13 篇的架构，无论米家还是 HomeKit，都有一个天花板：**生态内设备好管，跨生态就断**。Home Assistant（简称 HA）的出现改变了游戏规则——它是一个**开源的本地智能家居平台**，几乎能接入任何设备，然后统一管理、统一自动化。对安装调试从业者来说，HA 是把「零散单子」升级成「全屋集成单」的核心技能。这一篇讲 HA 的安装形态与设备接入，让读者先把「大脑」装起来。

## 1 HA 的四种安装形态

HA 可以跑在各种硬件上，形态决定维护成本与能力上限：

**HAOS（Home Assistant Operating System）**：官方操作系统，装进树莓派、迷你主机（NUC、x86 小主机），自带管理界面与附加组件（Add-ons），**最推荐家庭使用**。树莓派 4/5 与 x86 迷你主机都支持。

**Docker 容器**：在已有 Linux/NAS（群晖、威联通）上用 Docker 跑 HA。适合 NAS 用户、不想独占一台设备的场景；没有 Add-ons 支持，部分功能（如 Zigbee 网卡直通）配置更繁琐。

**虚拟机**：在 PVE、ESXi 上跑 HAOS 虚拟机，适合「一台服务器虚拟化一切」的玩家；硬件直通（USB Zigbee 网卡）配置稍复杂。

**Hass.io / 各类插件集成**：在已有系统里装 HA Core，适合开发调试，不推荐家用。

**重点：家庭部署首选 HAOS，其次是 Docker。** 两者的选择取决于「手上有什么硬件」：有闲置树莓派/迷你主机 → HAOS；已有 NAS → Docker。<span class="marginnote">安装前先想清「HA 装在哪」，比装的过程更重要：<strong>HA 是要 7×24 常开的「家庭服务器」</strong>，用树莓派 SD 卡跑容易因掉电损坏系统，专业方案是配 SSD 或 UPS。</span>

## 2 安装流程（以 HAOS 到迷你主机为例）

**第 1 步：准备硬件**。一台 x86 迷你主机（8G 内存以上为宜）、一块 SSD、U 盘（做启动盘）、网线。

**第 2 步：刷写镜像**。下载 HAOS 官方镜像，用写盘工具（如 balenaEtcher）烧录到 U 盘，插入主机从 U 盘引导启动。

**第 3 步：首次启动**。主机进入安装过程（数分钟），完成后 HA 会以 `http://homeassistant.local:8123` 的地址提供 Web 管理界面——同一局域网内用浏览器打开即可。

**第 4 步：创建账户与家庭**。首次登录创建管理员账户、设置家庭名称、时区（Asia/Shanghai）与地理位置，HA 会据此做日出日落等自动化计算。

**第 5 步：接入设备与固件更新**。引导页会提示接入发现的设备；进入「设置 → 系统 → 更新」把 HA 与 Add-ons 更新到最新稳定版。<span class="marginnote">首次配置的三个细节：<strong>时区设错会导致日出日落场景时间全错；跳过「设备发现」页不代表以后不能加；进入「配置 → 通用」把语言切成中文</strong>——官方有简体中文界面。</span>

## 3 设备接入：集成（Integration）

HA 把每一种设备接入方式称为**集成（Integration）**。常见接入路径：

**自动发现**：HA 会自动发现局域网内的设备（通过 mDNS、UPnP），在「设置 → 设备与服务」里点「添加」，按引导授权即可。HomeKit 设备、部分 Wi-Fi 设备走这条。

**厂商云集成**：米家设备通过 **MIoT/MiHome 集成**（社区插件）接入——绑定小米账号后，米家生态的灯、插座、传感器全部进入 HA。这类接入**依赖小米云**，属于「云集成」。

**本地 API 集成**：支持本地协议的设备（如部分涂鸦、HomeKit 本地、ESPHome 设备）通过 IP 直连接入，**断网可用**，体验最好。

**HACS（Home Assistant Community Store）**：HA 的「应用商店」，装社区集成与前端插件。很多大牌设备（如米家、某些国内设备）都要先装 HACS 再装对应集成。<span class="marginnote">HACS 是 HA 的「解锁钥匙」：<strong>官方集成覆盖不了的大量设备（尤其国内生态），都靠 HACS 里的社区集成</strong>。安装方法在官网 HACS 页有命令行指引，装完就能搜索下载社区集成。</span>

## 4 Zigbee 设备接入：ZHA 与 zigbee2mqtt

这是 HA 最核心也最让新手困惑的部分——让 HA 直接「收编」Zigbee 设备，不再依赖厂商网关：

**需要 USB Zigbee 网卡（Coordinator）**：如 Conbee II、Sonoff Zigbee 3.0 USB、Tube 等。插到 HA 主机上，HA 通过网卡直接组建自己的 Zigbee 网络。**原本在米家网关下的 Zigbee 设备，要先从米家删除，再在 HA 侧重新配对入网**（设备同一时间只能在一个 Zigbee 网络中）。

**两条路线**：

**ZHA**：HA 内置的 Zigbee 协议栈，配置简单、够用，适合入门。在「设备与服务 → 添加集成 → ZHA」选择网卡端口即可。
**zigbee2mqtt**：独立组件，把 Zigbee 数据转成 MQTT 交给 HA，设备兼容性更广、支持调试更细，但要多装一个 MQTT Broker（如 Mosquitto），配置更繁琐。<span class="marginnote">ZHA 与 zigbee2mqtt 的选择口诀：<strong>新手上 ZHA、老手看兼容性上 zigbee2mqtt</strong>。某款设备在 ZHA 下不认，查官方兼容表，往往 zigbee2mqtt 已支持——这就是两条路线并存的意义。</span>

**配对操作**：HA 打开 ZHA/zigbee2mqtt 的「允许加入」→ 长按设备配网键 5 秒 → 设备指示灯闪烁 → HA 自动发现并接入 → 命名归属房间。

## 5 核心对比表：三种安装方式

纯技能主题，用对比表替代公式解析：

| 对比维度 | HAOS | Docker | 虚拟机 |
| --- | --- | --- | --- |
| 推荐硬件 | 树莓派/迷你主机 | 已有 NAS/Linux | 虚拟化服务器 |
| 安装难度 | 低 | 中 | 中高 |
| Add-ons 支持 | 全 | 无（手动配置） | 全 |
| USB 网卡直通 | 简单 | 需映射 | 需直通配置 |
| 维护友好度 | 最好 | 中 | 中 |
| 适合用户 | 家庭首选 | NAS 用户 | 虚拟化玩家 |

## 6 术语速查表

HA 生态有大量专属词汇，接入设备前先对齐：

| 术语 | 含义 |
| --- | --- |
| HAOS | Home Assistant 官方操作系统，家庭部署首选 |
| 集成（Integration） | HA 中接入一类设备的插件或协议适配 |
| Add-on | HA 官方附加组件，如 Mosquitto、文件编辑器 |
| HACS | 社区应用商店，装社区集成与前端插件 |
| ZHA | HA 内置的 Zigbee 协议栈 |
| zigbee2mqtt | 独立 Zigbee 组件，把数据转成 MQTT |
| MQTT | 轻量消息协议，设备间通信的公共语言 |
| 网卡（Coordinator） | USB Zigbee 网卡，HA 借此组建自己的 Zigbee 网络 |
| MIoT/MiHome | 米家设备接入 HA 的社区集成 |
| Supervisor | HAOS 的组件管理器，管理 Add-on 与更新 |

## 7 HA 安装常见误区

HA 安装失败与体验翻车，多源于这几个误区：

| 误区 | 正解 |
| --- | --- |
| 树莓派 SD 卡一插就完 | 掉电易损坏 SD 卡，专业方案用 SSD |
| 时区随便选 | 时区设错，日出日落场景全错 |
| 米家设备要重买 | 米家设备通过 MIoT 集成即可进 HA，无需换设备 |
| Zigbee 设备自动进 HA | 原网关下的 Zigbee 设备需先删除，再用 USB 网卡重新配对 |
| 官方集成覆盖一切 | 国内大量设备要靠 HACS 社区集成 |
| 装完 HA 就不管更新 | 长期不更新会积累安全漏洞与兼容问题 |

## 8 小结

- HA 是**开源本地中枢**，把跨生态设备统一管理；家庭首选 **HAOS**，NAS 用户用 **Docker**。
- 安装五步：**备硬件 → 刷镜像 → 首次启动 → 建账户 → 更新固件**；时区必须设对。
- 设备接入靠**集成**：自动发现、厂商云集成、本地 API、HACS 社区集成四类。
- Zigbee 设备用 **USB 网卡 + ZHA（入门）/ zigbee2mqtt（兼容广）** 直连 HA，脱离厂商网关。
- 原网关下的 Zigbee 设备换网要先删除再配对。

在下一节，我们把 HA 接进苹果生态——讲 HomeKit/米家多生态平台联动。
