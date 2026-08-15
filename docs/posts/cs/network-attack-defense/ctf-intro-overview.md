---
title: CTF 入门与题型总览（Misc/Web/Crypto/Pwn/Reverse）
date: 2026-08-07
---

# CTF 入门与题型总览（Misc/Web/Crypto/Pwn/Reverse）

<div class="epigraph">
<p>CTF 是最接近真实攻防的沙盘——规则明确、环境安全，而你要像黑客一样思考。</p>
<footer>—— 佚名（CTF Wiki 编者按）</footer>
</div>

<div class="article-byline">
<p>第三级 · 网络攻防技术（渗透测试/CTF/红蓝对抗/应急响应） ｜ CTF 竞赛资料 ｜ 2026-08-07</p>
</div>

## 为什么从 CTF 入门继续

前十五篇我们系统走完了「侦察 → 利用 → 后渗透」的渗透测试主线。

而 **CTF（Capture The Flag，夺旗赛）**是检验与强化这些技能的最佳训练场：它把攻防知识拆成一道道**「拿 flag」的谜题**，环境干净、目标明确、规则公平——你可以放心大胆地试错，而不用担心法律问题（呼应第1篇：授权意识）。

CTF 的最大价值在于**把「知识」变成「动手能力」**：前十五篇的概念，在 CTF 里都变成了「十分钟内要解出来」的实操。本节是 CTF 六篇（第20–25篇）的总纲：讲清 CTF 是什么、怎么玩、六大题型各自考什么，并给出一条从入门到进阶的路径。

<span class="marginnote">CTF 的来源：1996 年 DEF CON 大会上，以黑客攻防对抗的形式首次提出「夺旗」概念，此后演化为两大主流赛制——<strong>解题型（Jeopardy）</strong>：一个榜单、各种难度的题目、解出得 flag 得分；<strong>攻防型（Attack-Defense）</strong>：每队维护自己的服务器同时攻击别人。本专题的六篇主要面向解题型（也是初学者最友好的赛制），攻防型则对应第26篇红蓝对抗。</span>

## 1 CTF 的玩法：flag 与赛制

**CTF（Capture The Flag）**：一种信息安全竞赛，参赛者通过破解题目获取**flag**——一段特定格式的字符串（通常是 `flag{...}` 或 `CTF{...}`），提交给平台即可得分。题目按考察领域分类，难度递进。

**两大主流赛制**：

**解题型（Jeopardy）**：主办方布置一批题目，每道题一个或多个 flag，参赛者解出后提交得分。按 Misc/Web/Crypto/Pwn/Reverse 分类计分，是初学者与大部分线上赛的主流。

**攻防型（Attack-Defense）**：每支队伍运行自己的服务器（含故意植入的漏洞），既要攻击别人、也要防守自己。更贴近真实对抗，难度更高。

**核心要点｜flag 的位置就是漏洞利用的落点：**

| 题型 | flag 藏在哪 | 对应本篇 |
| --- | --- | --- |
| Misc | 隐写、流量、编码里 | 第21篇 |
| Web | 网站的漏洞里 | 第22篇 |
| Crypto | 加密后的密文里 | 第23篇 |
| Pwn | 二进制程序的漏洞里 | 第24篇 |
| Reverse | 程序里藏的逻辑里 | 第25篇 |

<span class="marginnote">CTF 的工具生态：Kali Linux 自带大量 CTF 常用工具（Burp、nmap、john、hashcat）；CTF 竞赛还常用 <strong>CyberChef</strong>（GCHQ 出品的全能编解码工具，一把梭 Base64/Hex/RSA）、<strong>picoCTF</strong>（CMU 面向初学者的平台，题目质量极高）、<strong>buuoj / 攻防世界</strong>（国内常见平台）。入门第一站几乎都是 picoCTF。</span>

## 2 六大题型：各考什么

CTF 题目按考察领域分为几大类（本专题第20–25篇各展开一篇），先总览：

**Misc（杂项）**：最「杂」的题型——**隐写术**（把信息藏进图片/音频）、**流量取证**（从 pcap 里还原 flag）、**编码解码**（Base64/Hex/二维码）、**社工**（从信息里找线索）。考的是「细心 + 常识 + 会用工具」。入门门槛最低，但对敏锐度要求极高。

**Web（Web 漏洞）**：考察 Web 安全——SQL 注入、XSS、文件上传、反序列化等（对应本专题第7–14篇）。给一个网站，找漏洞拿 flag。考的是「Web 攻防的实战」。

**Crypto（密码学）**：古典密码（凯撒/维吉尼亚）与现代密码（RSA/AES/哈希）的**密码分析**——给密文，还原明文。考的是「数学 + 算法理解」。

**Pwn（二进制利用）**：最硬核——栈溢出、堆利用、ROP，控制程序执行流拿 flag（呼应第15篇 Metasploit 与第24篇的底层）。考的是「对程序运行的理解」。

**Reverse（逆向工程）**：把程序**逆向**出逻辑——静态分析（读汇编/反编译）、动态调试（看运行过程），找到程序里藏着的 flag 或算法（对应第25篇）。

**核心对比表｜六大题型的特点：**

| 题型 | 领域 | 难度 | 关键技能 |
| --- | --- | --- | --- |
| Misc | 杂项/取证 | 低 | 隐写工具、细致 |
| Web | Web 攻防 | 中 | 漏洞利用 |
| Crypto | 密码分析 | 中 | 数学、算法 |
| Pwn | 二进制利用 | 高 | 汇编、内存 |
| Reverse | 逆向工程 | 高 | 汇编、调试 |

## 3 从入门到进阶：CTF 学习路径

CTF 是「练出来的」，不是「看出来的」。一条务实的路径：

**第一步：选平台**。picoCTF（中文友好度一般但题目极佳）、攻防世界、CTFshow（国内）、buuoj。先做简单题建立手感。

**第二步：按题型顺序学**。**Misc → Web → Crypto → Reverse → Pwn**，难度递增。Misc 练「细心与工具」，Web 练「攻击思维」，Crypto 练「数学」，Reverse/Pwn 练「底层理解」。每道题做笔记：考点、工具、解法思路。

**第三步：查资料**。CTF Wiki（面向中文化的系统教材）、picoCTF 官方题解、各题目的 writeup（解题报告）。**会查 writeup 也是一种能力**——先自己想，卡住再查，查完必须能独立重做。

**第四步：参加真实比赛**。线上赛（CTFtime 聚合全球赛程）实战，赛后复盘自己的题目。**比赛不是目的，复盘才是**——把没解出的题补会。

**公式解析｜CTF 解题的基本循环：**
$$
\text{理解题目} \rightarrow \text{识别考点} \rightarrow \text{选择工具} \rightarrow \text{执行尝试} \rightarrow \text{分析结果} \rightarrow \text{拿到 flag}
$$
这个循环与前十五篇的渗透测试循环（第7篇）同构——**识别→尝试→分析→迭代**。CTF 训练的本质，就是把这套「攻击性思维循环」练到条件反射。

<span class="marginnote">CTF 与真实渗透的辩证关系：CTF 的环境是「人为设计的干净谜题」，而真实渗透要面对「混乱的生产系统」。所以 CTF 高手不自动等于渗透高手（缺授权、报告、抗挫败的现实性），但 CTF 训练出的<strong>「找线索、试漏洞、拆算法」的思维肌肉</strong>，正是渗透测试最需要的底层能力——这也是为什么本专题把 CTF 六篇放在渗透主线之后，作为「训练营」。</span>

## 4 CTF 常用平台与工具速查

工具与平台的选择，直接决定做题手感。下面这份速查是入门最常用的：

**练习平台**：**picoCTF**（卡内基梅隆大学，题目分梯度、新手最友好）、**CTFtime**（聚合全球赛程与战队榜）、**攻防世界**（国内平台，题库大）、**buuoj**（汇聚历年经典真题）。

**常用工具**：Kali Linux 自带一整套（Burp、nmap、john、hashcat）；**CyberChef** 是编解码全能工具（Base64/Hex/RSA/字符转换一把梭）；浏览器 **DevTools** 在 Web 与 Reverse 题里是「第一调试器」。

**核心要点｜按题型选工具：**

| 题型 | 主力工具 | 辅助 |
| --- | --- | --- |
| Misc | CyberChef、binwalk、zsteg | Wireshark |
| Web | Burp Suite、浏览器 DevTools | sqlmap |
| Crypto | CyberChef、RsaCtfTool | factordb |
| Reverse | IDA/Ghidra、gdb | strings、ltrace |
| Pwn | pwntools、pwndbg | checksec |

<span class="marginnote">做题的「第二外语」是 <strong>writeup（题解）</strong>：每个平台赛后都有大量 writeup，读它们不是抄答案，而是学「别人是怎么想到这一步的」。建议流程：先独立想 30 分钟 → 卡住看提示 → 赛后认真读 writeup 并重做一遍——<strong>把「看懂了」变成「会做了」</strong>，才是 CTF 进步的真谛。</span>

## 5 小结

- **CTF（夺旗赛）**通过解谜题获取 `flag` 得分，是把攻防知识变成动手能力的最佳训练场。
- 两大赛制：**解题型（Jeopardy）**适合入门，**攻防型（Attack-Defense）**贴近真实对抗。
- 六大题型：**Misc、Web、Crypto、Pwn、Reverse**（加上已并入 Misc 的取证），各自考察「细致、攻防、数学、内存、逆向」。
- 学习路径：**选平台 → 按 Misc→Web→Crypto→Reverse→Pwn 顺序练 → 查 writeup → 参加比赛并复盘**。
- CTF 解题循环 = 理解→识别→尝试→分析→迭代，与渗透测试循环同构。
- CTF 训练的是「攻击性思维肌肉」，与真实渗透互补而非替代。

在下一节，我们从最友好的题型开始——**CTF Misc：隐写术与流量取证**，学会从图片、音频、流量里把隐藏的 flag 挖出来。