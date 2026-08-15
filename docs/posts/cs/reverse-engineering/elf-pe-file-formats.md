---
title: ELF 与 PE 文件格式解析
date: 2026-08-07
---

# ELF 与 PE 文件格式解析

<div class="epigraph">
<p>标准的美妙之处在于，可供选择的标准实在太多了。</p>
<footer>—— 安德鲁 · 塔能鲍姆（Andrew Tanenbaum）</footer>
</div>

<div class="article-byline">
<p>第三级 · 逆向工程与二进制分析 ｜ Dang et al.《Practical Binary Analysis》Ch.1 ｜ 2026-08-07</p>
</div>

## 为什么从文件格式开始

反汇编恢复了「执行时」的指令流，但程序在「文件里」是以什么形态存在的？所有逆向工作的第一手原料，都来自可执行文件本身——它的节区布局决定反汇编器从哪儿起步，它的导入表是静态分析的军火库，它的重定位信息能透露编译器的脾气。**读不懂文件格式，就不知道分析工具在帮你做什么、更不知道它们哪里会骗你。** <span class="marginnote">塔能鲍姆的调侃正合用：Windows 的 PE 与 Unix 的 ELF 是两套各自演化了三十多年的格式，细节差异巨大，但骨架惊人地相似——它们都回答了同一个问题：「一堆字节怎样才能被操作系统安全地装进内存并运行起来？」</span>

这也直接呼应第三级《操作系统》的进程与内存管理：可执行文件 = 静态的「蓝图」，进程 = 执行时的「建筑」；文件格式就是那份蓝图的语言。

## 1 可执行文件的通用骨架

无论 ELF 还是 PE，一个可执行文件都要回答四件事：

1. **这是什么？**（魔数/标识、目标架构）；
2. **有哪些块？**（节/段，各自叫什么、多大、权限如何）；
3. **怎么装进内存？**（虚拟地址、对齐规则、入口点）；
4. **和操作系统怎么握手？**（导入什么、导出什么、依赖什么库）。

于是两大格式不约而同地采用了**「头 + 表 + 块」**的分层：一个文件头说明全局属性，若干个「表」（节头表/段头表、导入表、导出表）充当目录索引，真正的代码和数据以「块」的形式平铺在文件里，被表指向。<span class="marginnote">读者若熟悉数据结构，可以把文件格式理解成一个「带索引的序列化结构体」；恶意样本改文件头、伪造节区，本质都是往这个索引结构里塞私货。</span>

## 2 ELF：节与段的双重视角

ELF（Executable and Linkable Format）是 Linux 与多数 Unix 的通用格式。理解 ELF 的关键是**两套视角**：

**链接视角（节，sections）**：由节头表（section header table）描述，服务于链接器与调试器。常见节：`.text`（代码）、`.data`/`.bss`（数据）、`.rodata`（只读常量）、`.symtab`（符号表）、`.strtab`（符号名字符串）、`.debug_*`（调试信息）。<span class="marginnote">`.bss` 节的妙处：它只占文件里 0 字节（内容全为 0），但装载后占据内存——「文件里没有、内存里有」的节，正好是静态分析时需要注意的地方。</span>

**装载视角（段，segments / program headers）**：由程序头表（program header table）描述，服务于装载器与动态链接器。段把若干权限相同的节聚合起来，标注 `R/X/W` 权限与装载地址。典型三段：`R E`（可读可执行的代码段）、`R  `（只读数据）、`RW`（可读写数据）。

**入口点**（ELF 头里的 `e_entry`）告诉内核执行第一条指令的地址；`readelf -h/-S/-l`、`objdump -x` 是查看这些结构的标准命令。

## 3 PE：从 DOS 头到导入表

PE（Portable Executable）是 Windows 的可执行格式。它的「祖辈关系」值得一提：文件以 **DOS 头**（`MZ` 魔数）开头，保证老 DOS 看到它时能优雅地提示「无法在 DOS 模式下运行」；DOS 头末尾的 `e_lfanew` 字段指向真正的 **PE 头**（`PE\0\0` 签名）。<span class="marginnote">PE 的全称是「可移植可执行文件」——设计初衷是让 Windows 的二进制能跨当时不同的 CPU（x86、MIPS、Alpha）使用。这个「可移植」愿望后来没完全实现，但名字留了下来。</span>

PE 头之后是**节头表**与各个节：`.text`、`.data`、`.rdata`、`.rsrc`、`.reloc` 等。对逆向最重要的两个表是：

**导入表（Import Table / IAT 描述表）**：列出每个导入 DLL 及要导入的函数名/序号。上一节《静态分析基础》里「导入表异常精简 ≈ 加壳」的判断，依据就在这里。
- **导出表（Export Table）**：列出本程序导出给其他模块用的函数。DLL 逆向几乎从导出表读起——DLL 的「功能菜单」就写在导出表里。<span class="marginnote">Windows 导出还支持「序号导出」（不导出名字，只给序号），一些恶意 DLL 和某些防破解库刻意用序号导出以增加逆向难度——导出表里 `Name` 为空、只有 `Ordinal` 的就是它。</span>

PE 的查看工具主要是 `dumpbin /headers`、`CFF Explorer`、`pefile`（Python 库）。

### 3.1 资源节与延迟导入

PE 里还有两个静态分析常被忽视的表。**资源节（`.rsrc`）**是个小型文件系统：图标、对话框、版本信息，以及内嵌的载荷文件都以资源的形式存在，用 Resource Hacker 一类工具可以直接浏览和导出——恶意样本把第二阶段的载荷藏在资源里是常规操作，`FindResource` → `LoadResource` → `LockResource` 就是取它的三连 API。

**延迟导入（delay-load）**则是一张「懒加载」的导入表：函数名字先记在 delay import descriptor 里，第一次真正调用时才解析。对静态分析而言，**延迟导入的函数在磁盘上的导入表里看不到**——分析者以为它没调用某 API，实际上运行期会悄悄补上。<span class="marginnote">恶意代码偏爱延迟导入的原因正在于此：让静态导入表「看起来更干净」。CFF Explorer 的 Delayed Imp 标签页、pefile 的 `DIRECTORY_ENTRY_DELAY_IMPORT` 字段能把它挖出来。</span>

![ELF 与 PE 文件格式骨架对照](/images/reverse-engineering/elf-pe-file-formats-1.svg)

## 4 公式解析：虚拟地址换算

文件里「第几个字节」与内存里「哪个地址」是两套坐标，二者换算贯穿所有逆向工作。PE 的核心换算公式：

$$
\text{VA} = \text{ImageBase} + \text{RVA}
$$

其中 **ImageBase** 是程序在内存中的默认基址（PE 通常 0x00400000），**RVA**（Relative Virtual Address）是相对基址的偏移。若要进一步定位到文件里的偏移，还需借助节的对齐：**FileOffset = RVA 所在节的文件偏移 + (RVA − 节RVA)**。分步拆解：

- **第一步，分清三种坐标**：FileOffset（文件里第几个字节）、RVA（相对 ImageBase 的偏移）、VA（进程虚拟地址）。分析工具与调试器频繁在这三者间跳转。
- **第二步，理解 ImageBase**：可执行文件被装载时，默认以 ImageBase 为起点；可执行文件通常固定基址，DLL 可能被重定位（ASLR 下每次运行都变），此时 VA 全靠运行时算。
- **第三步，实践意义**：当你看到崩溃地址 `0x004012ab`，减去 ImageBase 得 RVA `0x12ab`，再查它在哪个节、换算出文件偏移——就能把调试器里的地址对应回文件里的字节。**静态分析与动态调试的「对账」，全靠这一条公式。**

ELF 的对应物是「虚拟地址 = 段装载地址 + 段内偏移」，同样一句「三坐标对账」就能概括。

## 5 对齐、重定位与「信任文件头」的教训

**对齐（alignment）**：文件内节与内存中节的地址都按粒度对齐（PE 常见 0x200 文件对齐、0x1000 内存对齐；ELF 常见 0x1000）。所以「文件里挨着的两个节」在内存里可能并不相邻——静态分析时用文件偏移反推内存地址必须先过对齐这一关。

**重定位（relocation）**：代码里「绝对地址」在装载时往往要按实际基址修正。PE 里 `.reloc` 节存放重定位项，ELF 里由 `.rela.*` 节与动态链接器完成。**辨析｜易错点：** 不要想当然地认为「文件里的值 = 运行时的值」——凡是绝对地址，都可能被重定位改过；逆向结论要拿调试器验证一遍，正是因为它最容易被这类「文件/内存差」误导。

另一个血泪教训是**不要信任文件头**。恶意样本可以伪造节区名、篡改魔数、把代码塞进「数据节」里骗过分析工具；反过来，分析工具信任文件头给出的节区边界，攻击者就能通过「节区头指向文件之外」这类技巧制造解析差异（这就是著名的「文件与内存解析不一致」攻击手法，被无数恶意样本利用）。

## 6 一张 ELF 与 PE 对照表

两大格式的骨架惊人相似，细节处处不同。把它们逐行对照，一眼看清差异：

| 概念 | ELF（Linux） | PE（Windows） |
| --- | --- | --- |
| 文件头魔数 | `\x7FELF` | `MZ`（DOS 头）→ `PE\0\0` |
| 代码节 | `.text` | `.text` |
| 只读数据 | `.rodata` | `.rdata` |
| 可写数据 | `.data` / `.bss` | `.data` / `.bss` |
| 节表 | 节头表（section headers） | 节头表（IMAGE_SECTION_HEADER） |
| 装载视图 | 程序头表（program headers） | 节头表 + 数据目录 |
| 导入描述 | `.dynsym` + `.rela.plt` | 导入表（Import Table） |
| 导出描述 | 动态符号表 | 导出表（Export Table） |
| 重定位 | `.rela.*` 节 + 动态链接器 | `.reloc` 节 |
| 符号表 | `.symtab` / `.strtab` | 调试信息 / COFF 符号 |
| 入口点 | ELF 头 `e_entry` | PE 头 `AddressOfEntryPoint` |
| 查看工具 | `readelf` / `objdump` | `dumpbin` / `CFF Explorer` / `pefile` |

这张表是「文件格式双语的互译词典」：在 Linux 上分析一个概念，到 Windows 上先查它对应什么，分析思路就能无缝迁移。<span class="marginnote">`.bss` 在两边都「文件里 0 字节、内存里占位」；而两边最大的习惯差异是「ELF 用两种表分开讲装载与链接，PE 用一个节头表统管」——理解了这一条，两套格式的骨架就同一了。</span>

## 7 readelf / dumpbin 命令速查

同样的分析动作，Linux 与 Windows 各有一条命令。把高频操作列成对照表，跨平台分析不用查手册：

| 想做什么 | Linux | Windows |
| --- | --- | --- |
| 看文件头 | `readelf -h` | `dumpbin /headers` |
| 看节 | `readelf -S` | `dumpbin /headers` |
| 看程序头（段） | `readelf -l` | —（节头表兼） |
| 反汇编 | `objdump -d` | `dumpbin /disasm` |
| 看导入 | `readelf -d`（动态段） | `dumpbin /imports` |
| 看导出 | `readelf --dyn-syms` | `dumpbin /exports` |
| 看重定位 | `readelf -r` | `dumpbin /relocations` |
| 看资源 | — | `dumpbin /resources` |
| 通用十六进制 | `xxd` / `hexdump` | `pefile`（Python） |

「每条命令对应一个表」是这个环节的通用心法：`readelf -S` 对应节头表、`readelf -l` 对应程序头表、`dumpbin /imports` 对应导入表——命令是表的入口，表是格式的骨架。

## 8 小结

- 两大格式共享骨架：**头 + 表 + 块**，回答「是什么、有哪些块、怎么装载、怎么握手」四问。
- **ELF 双视角**：节（链接/调试视角，`.text/.data/.symtab`）与段（装载视角，R/X/W 权限）；入口点在 `e_entry`。
- **PE 分层**：DOS 头（`MZ`）→ PE 头 → 节头表 → 节；导入表/导出表是逆向的军火库。
- **坐标换算** $\text{VA} = \text{ImageBase} + \text{RVA}$：FileOffset、RVA、VA 三坐标对账，是静态与动态分析互译的桥梁。
- **对齐与重定位**让「文件里」与「内存里」不一致；**不要信任文件头**，解析差异常被恶意样本利用。

在下一节，我们站到对抗的一方——看恶意代码如何反过来利用格式与环境的规则，主动探测分析者的存在，这就是**反调试与反虚拟机技术**。
