---
title: API Hook 与钩子注入
date: 2026-08-11
---

# API Hook 与钩子注入

<div class="epigraph">
<p>工欲善其事，必先利其器。</p>
<footer>—— 《论语 · 卫灵公》</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机基础 · 逆向工程与二进制分析 ｜ 对标教材 ｜ 2026-08-11</p>
</div>

## 为什么从 Hook 开始

前面一直在「读」程序，现在轮到「改」程序。**API Hook（钩子）** 是在函数入口插入我们自己的代码，从而监控、修改或拦截目标程序的函数调用——它同时是安全分析、恶意代码、调试器、杀软、游戏外挂的通用技术。<span class="marginnote">孔子讲「工欲善其事，必先利其器」：Hook 就是逆向工程工具箱里最锋利的一把——分析者用它做「API 监控」（替代上文的 procmon、ScyllaHide），恶意代码用它做窃密与隐身，调试器与杀软也靠它注入自身逻辑。同一种技术，攻防双方共用。</span>

它也是《反调试与反虚拟机技术》里「ScyllaHide 把检测点整体 hook 掉」的实现原理、《恶意代码代码分析》里「进程注入 API 序列」的机制层展开。学完这一节，你会明白调试器里那些「看起来是自动的功能」究竟是怎么被塞进目标进程的。

## 1 为什么要 hook：三个动机

**监控**：想观察目标进程调用了哪些 API、参数是什么。这是分析的动机——procmon 里那些「文件/注册表访问记录」，底层就是挂了一堆 hook。<span class="marginnote">监控型 hook 的一个经典应用是「API 参数记录」：hook `InternetOpenUrlA` 并打印 URL 参数，恶意样本的 C2 域名瞬间现形——比抓包更快，因为它在调用点抓的是「参数」而非「流量」。</span>

**修改**：想改变目标的行为——改返回值、改参数、跳过某个检查。这是破解、补丁、反调试绕过的动机。

**拦截**：想让目标「调用不到」或「走错路」——杀软拦截危险 API、游戏反外挂拦截可疑注入。

## 2 内联 Hook 与 trampoline

最通用的 hook 形式是**内联 Hook（inline hook）**：修改目标函数开头的指令，让它一进来就跳转到我们的钩子函数。

经典的 x86 实现：把函数前 5 个字节改成一条绝对跳转 `jmp hook_func`（机器码 `E9` + 相对偏移，或 `FF 25` 接地址），原字节备份到别处；钩子函数执行完自己的逻辑后，要么直接接管，要么执行备份的原指令再跳回原函数第 5 个字节之后继续。**被备份的那段原指令就叫 trampoline（蹦床）**。<span class="marginnote">为什么一定是 5 个字节？因为 x86 里 `jmp rel32` 恰好占 5 字节——覆盖一个指令长度的最小值。但如果目标函数开头是一条超过 5 字节的指令（如 `mov rax, imm64`），只覆盖前 5 字节会把指令腰斩成废码，必须选一个「完整指令边界 ≥ 5 字节」的位置下钩——这是内联 hook 最容易出错的地方。</span>

内联 hook 的优势是**只认地址不认符号**，能 hook 任何函数；代价是**改写指令**——会被自校验发现（《加壳、脱壳与混淆》里自校验防的就是这个）、多线程下有竞态（改写瞬间有人执行到半条指令）、对只读内存无效。

### 2.1 反 hook：检测、绕过与平衡

有 hook 就有反 hook，这也是个完整的攻防分支。检测内联 hook 的手段：

**首字节比对**：读函数头 5 字节，和原始字节（从干净的 DLL/镜像拷贝）对比，不一致即有 hook——x86 恶意样本常这么查自己的关键 API；
- **节区权限与写保护**：`VirtualQuery` 看代码节是否被改为可写，可写即可能被 hook 过；
- **导入表比对**：把 IAT 里的地址与真实函数地址比对（从磁盘重载一份模块算）。

绕过或对抗则有三个方向：**直接调 trampoline**（绕过 hook 走原指令）、**恢复现场**（把被改的字节写回去——但这会与「安装 hook 的调试器/杀软」直接冲突）、以及**在更底层 hook**（hook 到 `ntdll` 的 syscall 层甚至 SSDT，让上层 hook 失效）。<span class="marginnote">攻防的天平最终倒在「谁在更底层」：杀软 hook ntdll，恶意代码就 inline syscall 绕过 ntdll；杀软再下沉到内核 SSDT……每一层 hook 都催生更底层的反 hook。这就是为什么《固件与内核逆向》里内核层如此重要——层越低，越难被绕过。</span>

## 3 导入表 Hook：不碰代码只换指针

Windows 上还有一种更优雅的方式：**IAT Hook（导入地址表 hook）**。程序调用导入函数时，先查导入地址表（IAT）拿函数地址——把表里的地址替换成我们函数的地址，程序下次调用时就会直接走进我们的钩子，而**函数体一个字都没被改**。<span class="marginnote">IAT hook 的妙处：不触发自校验、不用处理指令边界、写起来简单。局限也明显——只对「通过 IAT 调用」的 API 有效；`GetProcAddress` 动态解析（恶意代码标配，见《静态分析基础》）绕开了 IAT，hook 就抓不住，得配合内联 hook 才全。</span>

内核态有对应物：**SSDT Hook**（系统服务描述符表，存放内核系统调用分发表）。替换表项后，`NtCreateFile` 一类的系统调用被重定向——rootkit 用它在内核层做文件/进程隐藏，杀软内核驱动用它做强制访问控制。<span class="marginnote">SSDT hook 是把「IAT hook 换地址」的思路搬到内核——但代价更重：改的是全局内核表，一个 bug 就能蓝屏，而且 Windows 内核有 PatchGuard（内核补丁保护）检测这类修改。现代 rootkit 更多转向「inline syscall hook」或直接替换系统调用序号，攻防层次之深由此可见一斑。</span>

## 4 代码注入：把钩子送进别人的进程

hook 是在目标进程里执行我们的代码，所以第一步往往是把代码**注入（injection）**进去。五种经典途径：

1. **DLL 注入（CreateRemoteThread）**：`OpenProcess` → `VirtualAllocEx` 在目标进程分配内存 → `WriteProcessMemory` 写入 `LoadLibraryA` 的参数（DLL 路径）→ `CreateRemoteThread` 让目标进程执行 `LoadLibrary`——这是《恶意代码代码分析》里那个 API 序列的本尊；<span class="marginnote">`CreateRemoteThread` 注入的本质是「让目标进程自己调用 LoadLibrary」——从目标进程视角看，这只是它正常加载了一个 DLL，这就是它难以防御的原因。检测方向是看「线程起始地址是不是 LoadLibrary + 参数是否指向异常 DLL」。</span>
2. **SetWindowsHookEx**：把钩子 DLL 注册为窗口消息钩子，目标进程处理消息时系统自动加载它（GUI 环境特供）；
3. **AppInit_DLLs 注册表**：写入注册表后，凡是加载 `user32.dll` 的进程都会自动加载指定 DLL——最粗暴也最广谱；
4. **进程镂空（process hollowing）**：`CreateProcess` 挂起方式启动一个合法进程，`WriteProcessMemory` 把它的内存镜像替换成恶意代码，`ResumeThread` 后一个「合法外表 + 恶意内容」的进程出现；
5. **线程注入与 APC 注入**：把代码塞进目标进程已有线程的 APC 队列，等线程下次进入可唤醒状态时执行。

### 4.1 一次 DLL 注入的完整解剖

把途径 1 的每一步拆开，你看到的不只是「五个 API 连用」，而是五个「为什么」：

`OpenProcess(PROCESS_ALL_ACCESS, FALSE, pid)`——取得目标进程的句柄。**为什么需要 ALL_ACCESS？** 后续的 `VirtualAllocEx`/`WriteProcessMemory` 都需要对目标进程的写权限；权限给得不够，注入直接失败（这也是系统加固里「限制进程句柄权限」能挡住注入的原因）。

`VirtualAllocEx(hProcess, NULL, size, MEM_COMMIT, PAGE_READWRITE)`——在**目标进程的地址空间**里申请一块内存。注意是 `Ex` 后缀——「在别的进程里做虚拟内存操作」是它的专属语义，也是它与普通 `VirtualAlloc` 的本质区别。

`WriteProcessMemory(hProcess, remoteBuf, dllPath, len, NULL)`——把 DLL 路径字符串写进那块远程内存。**这一步只是写数据，没执行任何东西**——真正的魔法在下一句。

`CreateRemoteThread(hProcess, NULL, 0, LoadLibraryA, remoteBuf, 0, NULL)`——在目标进程里开一条新线程，入口地址是 `LoadLibraryA`，参数是刚才写入的路径。**等价于让目标进程自己调用 `LoadLibraryA(dllPath)`**——从目标进程视角看，它只是在「正常加载一个 DLL」。这就是它难以被察觉的根本原因：动作本身是合法的系统行为，只是参数不怀好意。<span class="marginnote">检测方向于是落在参数与来源：线程起始地址是否指向 `LoadLibraryA`？路径是否指向磁盘上的可疑 DLL？配合《恶意代码行为分析》里「加载了哪些可疑 DLL」的进程观察，这条链条就能被抓住。</span>

读透这一次注入，你就同时读懂了半数恶意加载器的实现——代码注入不是魔法，是一串「在别人家借地址、借线程、借函数入口」的系统调用。

## 5 辨析与小结

**辨析｜易错点：** Hook 有三个技术陷阱。其一，**重入与递归**——你的钩子函数里如果调用了被 hook 的 API，会再次触发钩子，无限递归；标准解法是在钩子函数里直接调用 trampoline 或真实 API 的内部实现，绕开重入。其二，**线程安全**——内联 hook 改写指令是「正在执行的代码被改动」，多线程下必须用 `FlushInstructionCache` 同步指令缓存，否则偶发崩溃极难排查。其三，**hook 顺序**——多个钩子叠加在同一函数上（杀软、调试器、我们的工具同时 hook）时，先来后到决定谁先看到调用，调试时「你的 hook 没生效」很可能是因为被别人抢先了。

### 小结

- **API Hook = 在函数入口插入自己的代码**，动机三：监控、修改、拦截。
- **内联 hook**：改写函数头 5 字节为跳转，trampoline 备份原指令；认地址不认符号、但改指令可被自校验发现。
- **IAT hook**：换导入地址表指针，不碰指令；但被 `GetProcAddress` 动态解析绕过。内核对应物是 **SSDT hook**。
- 注入五途：**CreateRemoteThread、SetWindowsHookEx、AppInit_DLLs、进程镂空、APC 注入**。
- 三大陷阱：重入递归、指令缓存同步、多 hook 叠加顺序。

在下一节，我们把视线从用户态抬到内核与固件——驱动、系统调用、启动固件如何被逆向，这是逆向工程的深水区，**固件与内核逆向**。
