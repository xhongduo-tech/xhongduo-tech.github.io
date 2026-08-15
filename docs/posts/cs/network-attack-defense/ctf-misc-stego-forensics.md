---
title: CTF Misc：隐写术与流量取证
date: 2026-08-07
---

# CTF Misc：隐写术与流量取证

<div class="epigraph">
<p>最好的藏匿之处，是别人以为你已经看过的地方。</p>
<footer>—— 佚名（隐写术社区俗谚）</footer>
</div>

<div class="article-byline">
<p>第三级 · 网络攻防技术（渗透测试/CTF/红蓝对抗/应急响应） ｜ CTF 竞赛资料 ｜ 2026-08-07</p>
</div>

## 为什么从 CTF Misc 开始

CTF 六大题型里，**Misc（杂项）**是入门第一站——它没有固定的知识边界，考的是「细心、工具熟练度、跨领域联想」。

Misc 的两大核心子领域是 **隐写术（steganography）**与 **流量取证（traffic forensics）**：前者把 flag 藏进图片、音频、文本里，后者把 flag 藏进网络流量文件里。为什么先学 Misc？

因为它是「CTF 里最像寻宝的题型」——不需要深厚的数学或汇编功底，需要的是一套**「拿到一个文件先检查什么」的系统流程**。这套「从蛛丝马迹还原隐藏信息」的能力，也正是真实取证（第29篇）与情报分析（第2篇 OSINT）的雏形。

<span class="marginnote">隐写术与加密的区别：<strong>加密</strong>让信息「读不懂」，<strong>隐写</strong>让信息「看不见」。两者常结合——先加密再隐写，或先隐写再加密。CTF 里「先解码再找隐写、先隐写再解码」的嵌套是常见套路。隐写术的历史可以追溯到公元前（希罗多德记载蜡板藏信、剃头送信），如今它已数字化：把数据藏进图片的像素、音频的采样、文件的字节里。</span>

## 1 隐写术的原理：把数据藏进载体

**隐写术（steganography）**：把秘密数据**嵌入一个正常的载体文件**（图片、音频、视频、文本），使秘密「看不见」而载体看起来毫无异常。

**图片隐写（最经典）**。图片由像素组成，每个像素的 RGB 值对视觉极不敏感——把 flag 的每个 bit **写进像素的最低有效位（LSB，least significant bit）**，肉眼完全看不出差异。

`LSB 隐写` 是最基础的图片隐写：`flag` 的二进制被逐 bit 藏进一张风景图里。

**音频隐写**。把数据藏进音频的采样值（LSB 类似）、频谱（Spectrogram，用工具看图谱里是否藏着文字）、或摩尔斯电码。

**文件附加（文件合成）**。把 flag 文件直接**追加**在图片/JPG 之后——`cat flag.txt >> pic.jpg`，图片照样能打开，但文件末尾藏着 flag。这类题用 `binwalk`、`strings` 或文件头检测即可发现。

**核心要点｜图片隐写检查流程：**

| 步骤 | 工具 | 发现什么 |
| --- | --- | --- |
| 看文件类型 | `file` | 真实类型 vs 扩展名 |
| 字符串扫描 | `strings` | 明文 flag |
| 文件分离 | `binwalk` | 隐藏的附加文件 |
| LSB 提取 | `zsteg`/`stegsolve` | LSB 隐写数据 |
| 元数据 | `exiftool` | 注释里的 flag |

<span class="marginnote">CTF Misc 的第一套「起手式」：拿到文件先 `file` 看真实类型、`strings` 扫明文、`binwalk` 看有没有附加文件——<strong>很多时候 flag 就藏在最简单的步骤里</strong>，不用一上来就上高深的 LSB。工具的熟练度决定 Misc 的解题速度：`zsteg`（LSB 自动化）、`stegsolve`（逐通道看位平面）、`exiftool`（元数据）、`binwalk`（文件分离）是常备四件套。</span>

## 2 流量取证：从 pcap 里还原 flag

**流量取证（traffic forensics）**：给你一个 `pcap` 抓包文件（呼应第6篇网络嗅探），从中还原出 flag。它与数字取证同源，但 CTF 版环境干净、目标明确。

**流量取证的标准流程**：

**看协议分布**。`Wireshark` 的 `Statistics → Protocol Hierarchy` 一眼看出流量主要是 HTTP、DNS 还是别的——flag 常藏在「异常」的协议里。

**Follow TCP/UDP Stream**。把一条完整对话拼出来——明文 HTTP 里直接读到 flag；`POST` 请求里藏着上传的文件内容。

**导出对象**。`File → Export Objects → HTTP` 导出流量里传输的文件——flag 文件可能被下载/上传过。

**深挖异常**。DNS 查询里一串奇怪的子域名（DNS 隧道，呼应第18篇）、ICMP 包的数据段（ICMP 隧道）、可疑的邮件附件——flag 藏在最不像正常流量的地方。

**核心要点｜流量取证的异常信号：**

| 信号 | 含义 | 工具 |
| --- | --- | --- |
| 大量 DNS 查询 | DNS 隧道 | tshark 过滤 |
| 明文 HTTP 口令 | 凭证/flag | Follow Stream |
| 传输的文件 | flag 载体 | Export Objects |
| 异常 ICMP 数据 | ICMP 隧道 | 深挖数据段 |

**辨析｜易错点：** 新手常直接开 Wireshark 乱翻，效率极低。**先做「统计」再「过滤」**才是正路：Protocol Hierarchy 定方向、过滤器（`http`、`dns`、`tcp.stream eq N`）定位、Follow Stream 还原。

另一个误区是忽略**编码**——flag 常被 Base64/Hex/URL 编码藏在流量里，抓出来先解码看看（CyberChef 一键解决）。

<span class="marginnote">流量取证与第29篇数字取证、第6篇网络嗅探的技能完全互通：会读 pcap 是渗透、取证、CTF 三线的公共能力。CTF 的流量题是「玩具版」（环境干净、flag 明确），真实取证是「战场版」（流量巨大、痕迹被刻意清理）——但<strong>「统计定方向、过滤定位、跟踪还原」的分析骨架完全一致</strong>。</span>

## 3 Misc 的其他常见考点

Misc 的「杂」还体现在另几类常考题型：

**编码解码**。Base64、Hex、URL 编码、摩尔斯电码、二维码——很多「加密题」其实只是编码（呼应第23篇 Crypto 里的辨析），`CyberChef` 一把梭。

**压缩包与密码**。给一个加密的 zip/rar 压缩包，用弱口令字典（`fcrackzip`、`john`）爆破、或破解 `CRC32` 校验值（短内容时直接爆破还原内容）。

**社工与线索串联**。把多个线索（图片、文本、用户名）串起来找 flag——考「信息联想」，与第2篇 OSINT 同源。

**核心对比表｜Misc 常考类别：**

| 类别 | 形式 | 破解思路 | 工具 |
| --- | --- | --- | --- |
| 编码 | Base64/Hex/摩尔斯 | 识别→解码 | CyberChef |
| 图片隐写 | LSB/附加文件 | 检查流程 | zsteg/binwalk |
| 流量取证 | pcap 藏 flag | 统计→过滤→跟踪 | Wireshark |
| 压缩包 | 加密 zip/rar | 爆破/CRC | fcrackzip/john |

<span class="marginnote">Misc 做题的心态：它考的是<strong>「对已知工具的多走一步」</strong>——看到 `file` 结果与扩展名不符就 `binwalk` 分离；看到一串 Base64 就解码；看到图片就查元数据再查 LSB。CTF 高手做 Misc 快，不是靠冷门知识，而是靠<strong>一套不遗漏的检查流程 + 对常见套路的肌肉记忆</strong>。这套流程正是本节第1节那张检查表的意义所在。</span>

## 4 动手练习：一道综合 Misc 题的思路

把本节知识串成一道「综合 Misc 题」的完整解题思路，体会「多线索串联」：

**题目示例**：得到一个 `capture.pcap` 和一个 `photo.jpg`，flag 在流量里。

**第一步（流量取证）**：Wireshark 打开 pcap，`Statistics → Protocol Hierarchy` 发现大量 HTTP——Follow 一条 HTTP 流，看到一个 `.zip` 文件被下载（`Export Objects → HTTP` 导出）。

**第二步（压缩包）**：zip 有密码——`fcrackzip` 用字典爆破出弱口令，解压得到一串 Base64。

**第三步（编码解码）**：CyberChef 里 `From Base64` 解码，得到一个 PNG；图片里用 `strings` 找到尾部附加的明文 flag——题目解决。

**核心要点｜这道题展示的三个思维：**

- **工具按序尝试**：file → strings → binwalk → 导出 → 爆破 → 解码。
- **线索串联**：流量里导出的文件，往往是下一个线索的入口。
- **编码兜底**：解不出来的数据先丢进 CyberChef 试编码。

<span class="marginnote">Misc 综合题的本质是「<strong>线索链</strong>」：上一个线索的产出，是下一个线索的入口。解题的过程像解连环——所以 Misc 高手的关键能力不是会某个工具，而是<strong>「看到产物，能想到下一步」</strong>。这套能力与 OSINT（第2篇）、取证（第29篇）完全同源。</span>

## 5 小结

- **Misc（杂项）**考「细心 + 工具 + 联想」，两大核心是**隐写术**与**流量取证**。
- **隐写术**把数据藏进载体：图片 LSB、音频频谱、附加文件、元数据；检查流程是 `file → strings → binwalk → LSB → exiftool`。
- **流量取证**从 pcap 还原 flag：**统计定方向 → 过滤定位 → Follow Stream 还原 → 深挖异常**（DNS/ICMP 隧道）。
- 其他常考点：**编码解码、加密压缩包、线索串联**。
- Misc 与真实取证、OSINT 技能互通——「从蛛丝马迹还原信息」是贯穿攻防的元能力。
- 做题靠**不遗漏的流程 + 对套路的肌肉记忆**，而非冷门知识。

在下一节，我们把 Misc 的「找线索」升级为「攻网站」——**CTF Web：Web 题目与漏洞利用**，让前八篇的 Web 技能在赛场上直接变现。