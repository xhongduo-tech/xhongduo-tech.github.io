---
title: SELinux 与强制访问控制 MAC
date: 2026-08-07
---

# SELinux 与强制访问控制 MAC

<div class="epigraph">
<p>当「我的文件我做主」不够安全时，系统站出来说：有些事，谁说了都不算——这就是强制访问控制。</p>
<footer>—— 佚名，操作系统课程讲义</footer>
</div>

<div class="article-byline">
<p>第三级 · 操作系统 ｜ 《操作系统概念》§16.4 与 SELinux ｜ 2026-08-07</p>
</div>

## 为什么从 SELinux 开始

回顾访问控制模型：DAC（Unix chmod）灵活但弱——**属主自己说了算**，防不住进程被攻破后滥用权限。**SELinux（Security-Enhanced Linux）**把 **MAC（强制访问控制）** 落地到 Linux：**系统用全局策略裁决每个访问，即使进程被攻破、即使属主想泄露，策略依然拦截**。它是「最小特权 + 纵深防御」在 Linux 上的工程实现。<span class="marginnote">回顾《访问控制模型》：MAC 是「系统强制裁决」的模型。SELinux 是 MAC 在 Linux 的具体实现——它给每个主体与对象打<strong>安全标签</strong>，按<strong>策略规则</strong>强制判定访问，属主无权更改。<strong>DAC 是「属主自治」，SELinux 是「系统立法」。</strong></span>

## 1 SELinux 的背景与目标

**SELinux（Security-Enhanced Linux）**：美国国家安全局（NSA）开发的 Linux 安全模块（LSM），实现**强制访问控制（MAC）**。

**解决的问题**：

- 传统 Linux 只有 **DAC**——进程（尤其 root）权限过大，一旦被攻破，攻击者获得全部权限。
- **SELinux 的目标**：**即使进程被攻破，也只能按策略做「允许的事」**——把攻击者的破坏限制在最小范围。

**SELinux 的模式**：

- **强制模式（enforcing）**：拒绝一切违反策略的访问。
- **许可模式（permissive）**：记录违反但不拒绝（调试用）。
- **禁用（disabled）**。

**SELinux 与 DAC 的关系**：**SELinux 在 DAC 之上叠加 MAC**——访问必须同时通过 DAC 检查与 MAC 检查。**两道门都开才放行**，安全倍增。

## 2 SELinux 的核心机制：标签 + 策略

**标签（label）**：SELinux 给**每个主体（进程）与每个对象（文件、端口、设备）**打标签。标签格式：user:role:type:level（如 system_u:object_r:httpd_sys_content_t:s0）。

- 主体（进程）标签中的 **type** 称为**域（domain）**——如 httpd_t（Web 服务器域）、sshd_t（SSH 域）。
- 对象标签中的 **type** 称为**类型（type）**——如 httpd_sys_content_t（Web 内容类型）。

**策略规则（policy rule）**：定义「**哪个域 可以对 哪个类型 做什么**」——如：

```
allow httpd_t httpd_sys_content_t : file { read open getattr };
```

规则格式：`allow <域> <类型> : <对象类别> { <操作集> };`

翻译：**允许 Web 服务器域读/打开 Web 内容类型的文件**——它不能读数据库文件（mysqld_db_t）、不能写系统配置（etc_t）。

**访问判定**：进程访问对象时，SELinux 查策略——**「我的域能不能对它的类型做这个操作」**。允许则放行，否则拒绝。

**辨析｜易错点：** 「SELinux 的 type 就是 Unix 的文件类型」是误解。**SELinux 的 type 是「安全标签的类型」**（如 httpd_t、httpd_sys_content_t），与 Unix 的「文件类型」（普通/目录/设备）完全无关。**SELinux 是「标签 + 策略」的强制控制，DAC 是「属主 + 权限位」的自主控制**——两套体系独立叠加。

## 3 为什么 SELinux 强：最小特权 + 纵深防御

SELinux 的价值在**最小特权**（回顾《保护域》）：

- **默认拒绝**：策略里**没写的访问 = 拒绝**（默认 deny）——每个进程只获得策略明确允许的权限。
- **进程隔离**：Web 服务器（httpd_t）即使被攻破，攻击者也只能碰 httpd_sys_content_t 类型的对象——**碰不到数据库、碰不到用户主目录、碰不到内核**。
- **系统进程保护**：即使普通进程是 root，SELinux 仍限制它——**「root 不是万能的」是 SELinux 最颠覆的一点**。

**纵深防御的意义**：DAC 是「第一道门」（文件权限），SELinux 是「第二道门」（强制策略）——**缓冲区溢出攻破了 Web 服务器（第一道门失守），SELinux 仍拦住它的越权访问（第二道门还在）**。

**公式解析：SELinux 对攻击的收缩**

设攻击者攻破进程 $P$（域 $d_P$），$P$ 能访问的对象集合由策略决定 $O_P$，全系统对象 $O_{all}$：

$$\text{攻击者可访问比例} = \frac{|O_P|}{|O_{all}|}$$

- DAC 下：$P$ 若为 root，$O_P \approx O_{all}$——攻击者拿到一切。
- SELinux 下：$O_P$ = 策略允许的对象——**即使 root 也被限制**，$|O_P|/|O_{all}|$ 很小。

**直觉**：SELinux 把「攻破后的战利品」限制在策略圈定的范围——**它不防「被攻破」，它防「被攻破后的扩散」**。这正是纵深防御的核心。<span class="marginnote">SELinux 的哲学是「<strong>防线假设会被突破，所以准备第二道、第三道防线</strong>」。Web 服务器被攻破不可避免，关键是<strong>攻破后它什么都做不了</strong>——这就是为什么 SELinux、AppArmor、容器沙箱（Seccomp）都遵循「默认拒绝 + 最小授权」。</span>

## 4 核心对比表：DAC vs MAC（SELinux）

| 维度 | DAC（Unix） | MAC（SELinux） |
| --- | --- | --- |
| 谁控制权限 | 资源属主 | **系统全局策略** |
| 属主能否更改 | 能（chmod） | **不能** |
| 默认策略 | 允许（有权限就放行） | **默认拒绝** |
| root 特权 | root 全权 | root 也被限制 |
| 粒度 | 属主/组/其他 | 域/类型/操作 |
| 安全强度 | 弱 | **强** |

**辨析｜易错点：** 「SELinux 太麻烦，禁掉算了」是运维常见的短视。**SELinux 的「麻烦」是它的「强制」本质**——配置复杂是代价，换来的是进程被攻破时的隔离。**许多真实攻击（Apache 被入侵后写 WebShell）正是因 SELinux 被禁而成功**。**「安全 vs 便利」的选择要基于资产价值**——高安全环境必须保留强制访问控制。

## 5 数值算例：一条策略规则如何拦截越权

设想攻击者攻破了 Web 服务器进程（域 `httpd_t`），想读 `/etc/shadow`（标签类型 `shadow_t`）。按 DAC 看：若进程恰好以 root 跑（运维误配置）或 shadow 文件权限放得松，DAC 这一关就放行了。但 SELinux 的判定是：

```
# 策略中查找：
allow httpd_t shadow_t : file { read open };   ← 不存在！

# 判定结果：默认拒绝，记录到 audit.log
denied  { read }  scontext=httpd_t  tcontext=shadow_t
```

**没有这条 allow 规则，访问就被强制拒绝**——哪怕进程是 root、哪怕文件权限允许。反观 DAC 下同样的场景，很可能已经放行。这就是两道门的差别：<span class="marginnote">实战里 SELinux 最常见的操作是 `ausearch`/`audit2why` 查被拒记录，再用 `audit2allow` 生成规则放行——但「放行」前要确认是不是真的需要，否则就是给攻击者开了一扇门。<strong>策略的每次加宽，都等于给第二道门装了个插销</strong>。</span>

**更极端的场景**：Web 服务器被攻破后写 WebShell 到 `/var/www/html`（类型 `httpd_sys_content_t`）——策略允许 httpd 读它，所以 WebShell 能读；但 WebShell 想连数据库（`mysqld_db_t`）、想执行系统命令（`bin_t`）、想读用户主目录（`user_home_t`）时，全部被默认拒绝。攻击者困在 `httpd_sys_content_t` 一个类型里——**这正是「被攻破后的扩散」被切断的样子**。

## 6 术语速查表

| 术语 | 含义 | 一句话记忆 |
| --- | --- | --- |
| SELinux | Linux 的 MAC 实现 | 系统立法 |
| 标签 | 主体/对象的 security label | user:role:type:level |
| 域（domain） | 进程标签中的 type | 我是谁 |
| 类型（type） | 对象标签中的 type | 它是什么 |
| 策略规则 | allow 域对类型做什么 | 立法条文 |
| enforcing | 强制模式，拒绝违规 | 真刀真枪 |
| 默认拒绝 | 没写的访问一律拒绝 | 最小授权 |

## 7 小结

- **SELinux** 是 Linux 的 **MAC（强制访问控制）** 实现，在 DAC 之上叠加强制策略。
- 核心机制：**标签（域/类型）+ 策略规则（allow 谁对谁做什么）**。
- **默认拒绝**：策略没写的访问一律拒绝——每个进程只获得最小权限。
- **root 也被限制**：即使进程被攻破、即使拥有 root，也只能做策略允许的事。
- SELinux 防的不是「被攻破」，而是「**被攻破后的扩散**」——纵深防御的核心。

至此，第十四篇「保护与安全」收官。在下一节，我们进入 Linux 专题——**系统调用原理：从用户态到内核态的完整路径**。
