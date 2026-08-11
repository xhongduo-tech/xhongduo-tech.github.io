---
pageClass: plain-doc
---

# 逆向工程与二进制分析

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Bruce Dang et al., "Practical Binary Analysis" (2019)
- Michael Sikorski & Andrew Honig, "Practical Malware Analysis" (2012)
- Eldad Eilam, "Reversing: Secrets of Reverse Engineering" (2005)

## 主题规划

<ProgressGrid cat="cs/reverse-engineering" />

### 第1篇

- [x] [静态分析基础 (Sikorski Ch.3)](./static-analysis-basics)
- [x] [动态分析与沙箱 (Sikorski Ch.9)](./dynamic-analysis-sandbox)
- [x] [调试器实战与动态调试（IDA/OllyDbg/gdb） (Eilam Part 2-3)](./debugger-practice)
- [x] [反汇编与控制流恢复 (Dang Ch.2)](./disassembly-control-flow)
- [x] [ELF/PE 文件格式解析 (Dang Ch.1)](./anti-debug-anti-vm)
- [x] [反调试与反虚拟机技术 (Sikorski Ch.16)](./anti-debug-anti-vm)
- [x] [加壳与脱壳 / 混淆与反混淆 (Eilam Part 5)](./api-hook-injection)
- [x] [漏洞挖掘与模糊测试 (Dang Ch.6)](./fuzzing-vulnerability-discovery)

### 第2篇

- [x] [恶意代码行为分析（沙箱/网络） (Sikorski Ch.12)](./malware-behavior-analysis)
- [x] [恶意代码代码分析（IDA 逆向） (Sikorski Ch.12)](./malware-code-analysis)
- [x] [API Hook 与钩子注入 (Eilam Part 5)](./api-hook-injection)
- [x] [固件与内核逆向 (Dang Ch.9)](./firmware-kernel-reversing)
- [x] [二进制插桩与符号执行 (Dang Ch.7)](./binary-instrumentation-symbolic-execution)
