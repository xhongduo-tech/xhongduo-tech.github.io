---
title: Magisk 安装、升级与卸载流程
date: 2026-08-07
---

# Magisk 安装、升级与卸载流程

<div class="epigraph">
<p>装 Magisk 最容易出的错，不是命令敲错，而是 boot 镜像与系统版本不匹配。</p>
<footer>—— 刷机社区经验（Android 刷机社区资料）</footer>
</div>

<div class="article-byline">
<p>生活技能树 · 移动设备系统定制与刷机 ｜ Android 刷机社区资料 ｜ 2026-08-07</p>
</div>

## 为什么安装流程值得单独一篇

Magisk 的原理上一节讲清楚了，但「原理懂了」和「装得上」是两回事。安装 Magisk 的难点在于：它改的是 **boot 分区**，而 boot 镜像必须与当前系统版本**严格匹配**——用错版本的 boot 会导致无法开机。这一篇讲 Magisk 的安装、升级、卸载三条主线流程，以及贯穿始终的核心经验：**保住 boot 与系统的版本匹配**。「boot 与系统必须匹配」这条经验，与《操作系统安装与日常维护》里「驱动与系统版本对应」的常识同根同源。

## 1 安装前提与两种主流安装路线

**安装前提**：
- Bootloader 已解锁（改 boot 分区的前提）。
- 已下载与当前系统版本**匹配**的 boot 镜像（或使用「修补当前 boot」的方式，见下文）。
- 数据已备份（虽然安装 Magisk 一般不碰 data，但刷 boot 出错的风险值得备份兜底）。

**两条主流路线**：

**路线一：TWRP 直接刷 Magisk 包**。把 Magisk 的 zip 卡刷包通过 TWRP 的 Install 刷入。TWRP 会调用 Magisk 的安装脚本，脚本**自动修补当前设备的 boot 镜像**并刷回。适合已装 TWRP 的设备，也支持 A/B 设备（TWRP 会处理当前槽位）。<span class="marginnote">TWRP 刷 Magisk 的本质是「脚本替你修补 boot」：<strong>安装脚本读取当前 boot，把 magiskinit 注入，再把修补后的 boot 写回</strong>。好处是不用自己找 boot 镜像；坏处是依赖 TWRP 能正确识别设备与槽位，TWRP 机型不匹配时会翻车。</span>

**路线二：Magisk App 修补 boot 镜像，再 fastboot 刷入**。流程是：提取/下载当前系统的 `boot.img` → Magisk App 里选「Install → Select and Patch a File」→ App 生成修补后的 `magisk_patched-xxx.img` → 传到电脑 → Fastboot 里 `fastboot flash boot magisk_patched-xxx.img`。这条路线**不依赖 TWRP**，是 Pixel 等「没有现成 TWRP」机型的标准做法，也最适合想完全掌控每一步的人。

两条路线的共同点：**都是把「修补后的 boot」刷入 boot 分区**，差别只在「谁来做修补」——TWRP 脚本还是 Magisk App。

## 2 路线一：TWRP 直接刷入 Magisk 包

步骤拆解：

1. 下载**匹配机型与系统版本**的 Magisk zip（官方 GitHub Release 或 App 内下载）。
2. 手机进入 **TWRP**（`adb reboot recovery` 或按键）。
3. 在 TWRP 的 **Install** 里选择 Magisk zip，滑动确认刷入。
4. 刷完后 **重启系统**（`Reboot → System`）。
5. 打开 Magisk App，确认显示「已安装」，root 生效。

**A/B 设备的注意点**：TWRP 一般会自动处理当前槽位——Magisk 刷进当前运行的槽位。若你刚 OTA 升级切了槽，确保 TWRP 看到的是你正在用的槽。

**常见失败与排查**：
刷完无 root：多半是 TWRP 版本与设备不匹配，或 boot 修补失败。**用 Fastboot 刷回原版 boot 即可恢复**——这正是「boot 可回滚」的兜底。
卡开机动画：boot 与系统版本不匹配。刷回正确版本的 boot 重来。

## 3 路线二：修补 boot 镜像后 fastboot 刷入

这条路线对「不想装 TWRP」的用户最友好，步骤：

1. **获取当前系统的 boot 镜像**：从官方线刷包提取，或部分机型可 `adb shell` 备份 `boot` 分区。
2. **用 Magisk App 修补**：Magisk App → 安装 → 选择并修补一个文件 → 选中 `boot.img` → 生成 `magisk_patched-xxx.img`。
3. **把修补镜像传到电脑**：`adb pull magisk_patched-xxx.img .`。
4. **Fastboot 刷入**：
   ```
   adb reboot bootloader
   fastboot flash boot magisk_patched-xxx.img
   fastboot reboot
   ```
5. 验证 root。

**为什么强调「匹配当前系统版本」**：boot 里的内核与 ramdisk 依赖系统版本。**刷入不同版本系统的 boot，内核与 system 不匹配，轻则功能异常、重则无法开机**。所以每次系统升级后，Magisk 往往要**重新修补新版本的 boot**——这引出下面的「升级」问题。<span class="marginnote">「boot 匹配」是 Magisk 使用里反复出现的主题：<strong>系统升级换了 system，旧的修补 boot 就过时了</strong>。Magisk App 的「Direct Install」和「修补 boot」两条升级路径，本质都是「拿新系统的 boot 重新注入 Magisk」——理解了这条主线，升级就永远知道该做什么。</span>

## 4 升级与卸载：保持 boot 匹配的管理之道

**升级 Magisk** 有三条路径：

**Direct Install（直接安装）**：Magisk App 检测到新版本后，选择「直接安装」——App 读取当前 boot、注入新 Magisk、写回。前提是当前 boot 就是「带 Magisk 的版本」，且系统没升级过。
**Select and Patch（修补文件）**：系统刚升级过、boot 是新版本的，先选新的 boot 修补再刷入。
- **安装到未激活槽位（Install to Inactive Slot）**：A/B 设备在 OTA 升级后，可把 Magisk 装到尚未激活的新槽，保持升级后的无缝体验。

**系统升级（OTA）与 Magisk 的配合**：Magisk 保留 OTA 能力，但 OTA 后 boot 会被替换成原版——**root 会消失**。恢复方式：OTA 完成后，在 App 里对**新系统的 boot** 重新修补（或 Direct Install 若 boot 未被替换）。**顺序记牢：先 OTA，再重装 Magisk，别倒过来**。

**卸载 Magisk** 分两层：

- **只移除模块**：Magisk App 里逐个禁用/删除模块即可，不碰 root 本身。
- **完整卸载（Uninstall）**：Magisk App 提供「Uninstall」选项，会**把 boot 恢复成原版**，彻底移除 Magisk。若 App 不可用，手动 `fastboot flash boot <原版boot.img>` 也能达到同样效果——**这是最可靠的卸载兜底**。<span class="marginnote">Magisk 卸载的本质是「把 boot 换回原版」：<strong>因为 Magisk 只在 boot 里留了东西，system 从未被动过，所以卸 boot 即还原</strong>。这比传统 Root 的卸载（要重刷整个 system）轻量得多，也是 systemless 设计的又一红利。</span>

## 5 公式解析：修补 boot 镜像的「patch 链」

无论哪条安装路线，核心都是「修补 boot」。可以把修补看成一个变换：

$$
\text{boot}_{\text{原版}} \xrightarrow{\text{magiskinit 注入}} \underbrace{\text{boot}_{\text{patched}}}_{\text{可启动 + 带 Magisk}} \xrightarrow{\text{flash}} \text{boot 分区}
$$

逐步拆解：

- **boot$_{\text{原版}}$**：与当前系统匹配的原始 boot 镜像（含内核 + ramdisk）。
- **magiskinit 注入**：Magisk 把 `magiskinit` 加进 ramdisk，并调整启动参数，使启动早期执行它。
- **boot$_{\text{patched}}$**：修补后的 boot——**它必须保持与系统兼容**（内核没被换、依赖没破坏），否则启动失败。
- **flash 到 boot 分区**：写入后，下一次启动从修补后的 boot 走，Magisk 随之生效。

这条链揭示了安装成败的关键：**「原版 boot 对不对」决定了「修补后 boot 能不能用」**。boot 不对，修补得再漂亮也是白搭——所以「先拿到正确版本的 boot」永远是第一步。

## 6 核心要点：安装/升级/卸载方法对照表

| 操作 | 方法 | 适用场景 | 关键点 |
| --- | --- | --- | --- |
| 安装 | TWRP 刷 Magisk zip | 已装 TWRP | 脚本自动修补 boot |
| 安装 | App 修补 boot + fastboot | 无 TWRP | 手动控制每一步 |
| 升级 | Direct Install | 系统未换 | App 自动重注入 |
| 升级 | 重新修补新 boot | 系统已 OTA | boot 必须匹配 |
| 升级 | 装到未激活槽 | A/B + OTA 后 | 保持无缝升级 |
| 卸载模块 | App 禁用/删除 | 模块出问题 | 不碰 root |
| 完整卸载 | Uninstall / flash 原版 boot | 彻底移除 | 恢复 boot 即还原 |

## 7 术语速查表

| 术语 | 含义 | 关键点 |
| --- | --- | --- |
| boot.img | 内核+ramdisk 镜像 | 必须与系统匹配 |
| Patch | 修补 boot 注入 Magisk | 安装核心动作 |
| magisk_patched | 修补后的 boot 镜像 | fastboot 刷入对象 |
| Direct Install | App 内直接安装 | 系统未换时用 |
| Install to Inactive Slot | 装到未激活槽 | A/B 设备 |
| Uninstall | 完整卸载 | 恢复原版 boot |
| TWRP 刷包 | 卡刷 Magisk zip | 脚本自动修补 |
| 卡开机动画 | boot 不匹配症状 | 刷回正确 boot |
| root 消失 | OTA 后 boot 被替换 | 需重装 Magisk |
| 版本匹配 | boot 与系统对应 | 成败关键 |

## 8 快速自查清单

安装/升级/卸载 Magisk 前，逐条确认：

- **boot 镜像与当前系统版本是否匹配**？来源是否可靠？
- 安装选的是 **TWRP 刷包还是修补 boot**？对应流程准备好了吗？
- 系统是否刚 OTA 过？若是，**先补 boot 再装 Magisk**。
- 出问题时的**恢复方案**：原版 boot.img 备好了吗？
- 完整卸载走 **Uninstall 或手动刷回原版 boot**，确认 boot 文件在手？

## 9 小结

- Magisk 安装两条路线：**TWRP 刷包**（脚本自动修补 boot）与 **App 修补 boot + fastboot 刷入**（手动可控）。
- 全程核心经验：**boot 镜像必须与当前系统版本严格匹配**，否则卡开机动画。
- 升级三条路径：**Direct Install、重新修补、装到未激活槽**；OTA 后要按「先系统再 Magisk」的顺序重装。
- 卸载的本质是**恢复原版 boot**——systemless 让卸载比传统 Root 轻量得多。

在下一节，我们进入 Magisk 生态的活力所在：**Magisk 模块机制与常用模块推荐**。
