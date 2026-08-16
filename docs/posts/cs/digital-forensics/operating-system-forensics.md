---
title: 操作系统取证
date: 2026-08-07
---

# 操作系统取证

<div class="epigraph">
<p>操作系统是记忆最差的证人也记得最牢：它记下每一次登录、每一个程序、每一支 USB 的插拔。</p>
<footer>—— Eoghan Casey, <em>Digital Evidence and Computer Crime</em> 3e Ch.17–19（意译）</footer>
</div>

<div class="article-byline">
<p>第三级 · 数字取证 ｜ Casey, <em>Digital Evidence and Computer Crime</em> 3e Ch.17–19 ｜ 2026-08-07</p>
</div>

## 为什么从操作系统开始

文件系统告诉我们「文件何时被写、何时被删」，但它回答不了更上层的问题：**谁登录过这台机器？运行过什么程序？插过哪只 U 盘？访问过哪个网站？** 这些问题藏在**操作系统的痕迹（artifacts）**里——注册表、事件日志、临时文件、缓存、快捷方式。Casey 教材第 17–19 章正是按 Windows / UNIX / macOS 三大操作系统分别开列这份「痕迹清单」。<span class="marginnote">操作系统取证的精髓是「默认开启、防不胜防」：绝大多数痕迹是系统为了正常运行而自动留下的，用户根本不知道它们在记。这也让它成为比文件内容更可靠的行为证据——人可以撒谎，痕迹不会。</span>本篇把三张痕迹地图摊开，讲清「在哪找、找到什么、能证明什么」，同时与《内存取证》《网络取证》形成互补：OS 痕迹回答「本机发生了什么」，内存回答「此刻正在发生什么」，网络回答「数据流向了哪里」。

## 1 Windows 注册表：系统的「记忆中枢」

**注册表（Registry）**是 Windows 的配置数据库，由若干**配置单元（hive）**组成，其中与取证最相关的有五类：`SYSTEM`（硬件与启动配置）、`SOFTWARE`（已装软件）、`SAM`（账户与口令散列）、`SECURITY`（安全策略）、以及每个用户自己的 `NTUSER.DAT` 与 `UsrClass.dat`。<span class="marginnote">hive 是「文件」还是「内存」要看时机：关机后它们落地为磁盘文件，运行中有一部分缓存在内存——这就是《内存取证》能从内存镜像里 dump 出注册表的原因。取证师两个来源都要取。</span>注册表分析要用**专门的解析器**（如 Registry Explorer、`regripper`、Autopsy 的内置模块），因为 hive 是二进制格式，直接 `grep` 字符串会漏掉大量结构化信息。

**hive 的 Last Write 时间**是注册表里最容易被忽略的时间戳：每个键都带一个「最后写入时间」。它通常精确到分钟，却能回答「这个 Run 键是什么时候被加的」「这个 USB 设备记录何时写入」——与键里的内容互相印证，是判断「配置何时被改动」的独立时间源。<span class="marginnote">RegRipper 与 Registry Explorer 都会显示每个键的 Last Write；分析「恶意软件何时植入自启动」，先看 Run 键的 Last Write，再看键指向的程序文件的 MFT 时间，两相对照即可锁定植入时刻。</span>

值得优先查的**高价值键**：

**Run 键**（`HKLM\...\Run`、`RunOnce`、`HKCU\...\Run`）：开机自启的程序清单——恶意软件常驻这里，也会暴露「用户刻意设置的持久化」。
**UserAssist**（`HKCU\Software\Microsoft\Windows\CurrentVersion\Explorer\UserAssist`）：记录程序启动次数与时间，经 ROT13 编码，是「用户实际运行过什么」的强证据。<span class="marginnote">UserAssist 是「程序运行次数」这一说法的正规来源：检方说「嫌疑人反复运行销毁工具」，数据就来自这里。解码 ROT13 只是第一步，重点是它带时间戳。</span>
**RecentDocs**（`...\Explorer\RecentDocs` 及 MRUList）：最近打开文档的列表，逐条带时间。
**USBSTOR 与 Setupapi.dev.log**（`SYSTEM\CurrentControlSet\Enum\USBSTOR` + `%SystemRoot%\inf\setupapi.dev.log`）：每支插过的 U 盘的 VID/PID、序列号、首次安装时间——「哪支 U 盘何时被插进这台机器」的铁证，与《存储介质与数据恢复》里 U 盘的采集记录互相印证。<span class="marginnote">USBSTOR 的价值在<strong>序列号</strong>：同一支 U 盘被插入多台电脑，它的序列号会在每台电脑的 USBSTOR 里留下记录——这就能把「嫌疑人的 U 盘」与「案发电脑」跨设备地关联起来。跨设备关联，是 OS 痕迹最被低估的用途。</span>

## 2 Windows 事件日志与其余痕迹

**事件日志（Event Log）**记录系统与应用的重大事件，新版格式是 `.evtx`，存于 `C:\Windows\System32\winevt\Logs`。取证最常查的日志：**Security**（登录成功/失败，事件 ID 4624/4625）、**System**（开关机、服务故障）、**Application**、以及 **PowerShell 操作日志**（ScriptBlock 记录，攻击者常用）。

事件日志的分析有三层由浅入深：**第一层**，按事件 ID 检索——4624 登录成功、4625 登录失败、4688 进程创建、7045 新服务安装，熟记这组 ID 等于拿到了日志的「常用词表」；**第二层**，展开事件的 XML 结构——登录类型（2 为交互式、3 为网络、10 为远程桌面）、源 IP、登录进程等细节藏在结构化字段里，直接决定「这是本地操作还是远程入侵」；**第三层**，跨日志关联——把 Security 与 PowerShell、System 日志按时间对齐，还原「先下马甲→再创建进程→后建服务」的完整攻击链。<span class="marginnote">Windows 10 起默认开启的事件 ID 4688（进程创建）与 4104（PowerShell ScriptBlock），是检测「内存型恶意软件」时最可靠的本地来源——它比杀软更早看到进程。</span><span class="marginnote">Windows 安全日志能回答「谁在何时用哪个账号登进了这台机器」，是《网络取证》里「登录攻击」的本地对应物。注意：安全日志的默认容量有限，被覆盖是常态——所以「日志没了」本身也可能是反取证的信号。</span>

Windows 还有一批「无声的证词」：

**Prefetch**（`C:\Windows\Prefetch`，`.pf` 文件）：程序启动的预读缓存，记录程序名、运行次数与**最近运行时间**——即使程序文件已被删除，Prefetch 仍在。
**回收站**：Windows 回收站用 `$I`（元数据：原始路径与删除时间）+ `$R`（数据）成对文件记录每个被删项目——能还原「被删文件的原路径与删除时刻」。
**LNK 快捷方式**：用户打开过的文件会在跳转列表与最近位置生成 `.lnk`，内嵌目标文件的完整路径与时间。
**卷影副本（Volume Shadow Copy, VSS）**：系统还原与备份留下的历史快照，等于「整盘的历史备份」，能挖出早已删除或覆盖的版本。
**浏览器痕迹**：`Cookies`、`History`、`Cache`、`Login Data`（SQLite 数据库），记录访问过的网址、搜索词、下载与登录凭据。<span class="marginnote">浏览器历史数据库（SQLite）既是宝贝也是坑：删了主页记录，但 SQLite 的 freelist 里往往残留着被删记录——这正是《存储介质与数据恢复》雕刻思路在应用层的重演。</span>

## 3 Linux 的痕迹地图

Linux 没有注册表式的中央数据库，痕迹散落在**明文日志、Shell 历史与配置文件**里，更「看得见」却也更零碎。要按「用户行为 → 系统状态」两条线去找：

**用户行为线**：`~/.bash_history` 记录 Shell 命令历史；`~/.local/share/recently-used.xbel` 记录最近打开的文件；浏览器历史与 Cookies 在 `~/.config` 下；邮件在 `~/Mail` 或 Thunderbird 目录。<span class="marginnote">`~/.bash_history` 是最受欢迎的「自白书」，但它默认只记交互式 Shell 的命令、且可被用户手动清空或 `unset HISTFILE`——取证师把它当作「用户手写的便条」而非「系统强制记录」。</span>
**系统状态线**：`/var/log/auth.log`（认证与登录）、`/var/log/syslog`（系统消息）、`/var/log/wtmp` 与 `lastlog`（登录会话历史）、`/var/log/cron`（计划任务）、`/var/log/apache2/` 或 `/var/log/nginx/`（Web 访问）、`/etc/crontab` 与 `/etc/init.d`（持久化配置）。<span class="marginnote">Linux 的 `auth.log` 与 Windows 的 Security 日志功能相当，但 Linux 默认<strong>不限制</strong>日志体量、也不像 Windows 那样用事件 ID 编号——分析时更依赖 `grep` 与时间范围的熟练操作。</span>

Linux 取证的另一个特色是**一切皆文件的哲学**：系统状态（`/proc`、`/sys`）在运行中只是虚拟文件，关机即失——所以「先取易失数据」在 Linux 上尤其关键，`/proc/`、`ps`、`netstat`、`lsof` 的输出都要在采集阶段就留档，而不是等镜像出来后再补。

Linux 的分析还有两个常被忽略的点。一是**加密主目录**（如 `ecryptfs`、`dm-crypt` 的 `/home`）：登录密码即解密钥匙，取证需要「已登录会话」或用户口令，否则只能拿到密文——这再次指向「现场优先、运行态优先」。二是**包管理器日志**（`/var/log/dpkg.log`、`/var/log/yum.log`、`/var/log/apt/history.log`）：完整记录「哪个软件何时被装、升级、卸载」，对「这台机器上为何会有一款攻击工具」的追问，日志比任何单个文件都权威。<span class="marginnote">`apt`/`dpkg` 日志甚至记录<strong>从哪个源</strong>装的包——结合网络日志，能还原「恶意包是通过官方源还是第三方源混入的」，这是事件响应里定位供应链攻击的线索。</span>

## 4 macOS 的痕迹地图

macOS 的取证介于 Windows 的「集中」与 Linux 的「零散」之间，近年 Apple 把日志统一进 **unified log**（`/private/var/db/diagnostics`），一条命令 `log show` 就能按谓词检索全系统事件。<span class="marginnote">unified log 默认记录海量事件且自带时间，配合 APFS 的纳秒时间戳，macOS 常能还原出「毫秒级的行为序列」。但它有隐私过滤与滚动覆盖，分析时先确认日志起止时间。</span>macOS 特有的痕迹还有：

**应用偏好 pList**（`~/Library/Preferences` 的 `.plist`）：每个应用的配置，常带「最近打开」「窗口位置」等信息。
**Spotlight 索引**（`/.Spotlight-V100` 或 `~/Library/` 的 `.Spotlight-*`）：全盘文件的内容与元数据索引，即使文件被删，索引残片里可能还有名字。<span class="marginnote">Spotlight 索引是被低估的「残留记录库」：索引不会随文件删除立即消失，取证时可从中捞到已删文件名与路径的残迹。</span>
**quarantine 标记**（`~/Library/Preferences/com.apple.LaunchServices.QuarantineEventsV2`）：记录**每个从互联网下载的文件**的来源 URL、下载时间与应用——浏览器「下载了什么都查得到」，是下载型恶意软件分析的利器。
**fseventsd 日志**（`/Volumes/.../.fseventsd`）：文件系统事件流，近似「macOS 版的 USN 日志」，能重建文件被创建/移动/删除的粗粒度时间线。
**钥匙串（Keychain）**：集中保存密码与证书，解密需用户口令，是「凭据在哪」问题的终点站。

macOS 还有一层「历史数据」——**Time Machine**。它把整个系统（含系统文件）按小时/日/周快照到外置盘，本质上是一座「可回溯的系统考古现场」：被删的应用、被改的配置、甚至被清掉的日志，都可能在 Time Machine 里保留旧版。<span class="marginnote">Time Machine 快照用的是 APFS 快照机制（见《文件系统取证》），因此不会额外占用大量空间，却也意味着「旧数据」可能比用户以为的活得更久。取证时先确认外置 Time Machine 盘是否在场，再决定要不要做全量镜像。</span>iOS 设备通过电脑或 iCloud 的备份同理——它们都是「本地已删、备份未删」的典型来源。

## 5 核心对比表：三大操作系统的痕迹地图

把三张地图并排，取证思路的共性与差异立现：

| 维度 | Windows | Linux | macOS |
| --- | --- | --- | --- |
| 中央配置库 | 注册表（hive） | 分散的明文配置 | pList + unified log |
| 登录记录 | Security 事件日志 | auth.log / wtmp | unified log |
| 程序运行痕迹 | Prefetch / UserAssist | shell 历史、进程日志 | unified log / 应用日志 |
| USB 记录 | USBSTOR + setupapi | `dmesg`、udev、`/var/log` | IORegistry 记录 |
| 删除恢复 | 回收站 $I/$R、VSS | 无回收站，靠文件系统 | 回收站、Spotlight 残片 |
| 网络凭据 | 浏览器 SQLite | 浏览器 SQLite | 钥匙串 + 浏览器 |
| 特色金矿 | 卷影副本、事件 ID | 明文日志、proc 虚拟文件 | 纳秒日志、quarantine |

**辨析｜易错点：**「操作系统取证」不等于「把工具的输出打印出来」。注册表键、事件 ID、日志条目都只是**线索**，要变成**证据**必须回答「这条记录对应哪个用户、哪个时刻、哪个进程」三个问题。<span class="marginnote">一句话方法论：<strong>先问「操作系统会为这个行为自动记什么」，再去那个位置找</strong>——而不是把日志目录翻个底朝天。带着「该有什么」的预期去找，效率与准确性都高一个量级。</span>例如查「有没有运行过销毁工具」，就该先想到 Prefetch 与 UserAssist，而不是在事件日志里大海捞针。

## 6 从痕迹到行为：时间与地点的重建

OS 痕迹的真正价值在**跨来源组合**。单条日志说明不了什么，把注册表、事件日志、Prefetch、浏览器历史、USB 记录**按时间对齐**，才能重建一个完整的行为序列：<span class="marginnote">这种「多源时间对齐」正是上一篇时间线分析在 OS 层的延伸：文件系统时间 + OS 日志 + 应用痕迹三方对齐，构成 Casey 所说的「行为重建（reconstruction）」。</span>

举例：侦查员想证明「嫌疑人插了 U 盘拷走了文件」，可从四个来源取证——USBSTOR 记录 U 盘序列号与首次插入时间、Setupapi 日志补上精确时刻、MFT 时间戳显示该时段有批量文件被读、Prefetch 显示同时启动了压缩工具。四个独立来源指向同一件事，这比任何单一日志都难以反驳。<span class="marginnote">反过来，反取证者也会逐个清理这些痕迹——但<strong>清理本身会留下新痕迹</strong>：事件日志被清空、Prefetch 被删除、时间线出现空洞。所以「痕迹被清理」往往比「痕迹存在」更能说明问题。</span>

**重点**：OS 取证与网络、内存取证的边界是流动的。一台运行中的机器，其「OS 痕迹」既在磁盘上也在内存里；Windows 安全日志也记录着来自网络的登录尝试。Casey 教材把三章分列，是为了教学清晰，实操中取证师必须把三张地图拼起来用——这也是本专题从第六篇起进入「内存」「网络」「移动」的衔接点：先有 OS 的行为地图，才能读懂内存在进行中的过程、网络在流向的数据。

还有一个「作业层面」的提醒：OS 痕迹分析高度依赖**工具自动化的流水线**——Autopsy、KAPE（Kroll Artifact Parser/Extractor）、Eric Zimmerman 的工具套件（`Zimmerman` 系列）都能把「提取注册表 → 解析事件日志 → 生成时间线」做成半自动管线。但 Casey 与 Nelson 都警告：**自动化只负责提取，解释永远靠人**。工具跑出的「最近活动」列表只是素材，写进报告的结论必须由取证师手工核对原始证据——这个「人机分工」的边界，正是专业与熟练操作员的分水岭。

## 7 凭据与身份：痕迹背后的「谁」

操作系统痕迹能回答「发生了什么」，但法庭真正要问的是「**谁**做的」。身份问题有三个递进的证据层次：**账号层**、**凭据层**、**行为层**。<span class="marginnote">Casey 反复强调的「身份并非证据终点」：账号可以被共享、密码可以被盗用。取证的职责是列出「谁在技术上可能做了」，而不是替法官判决「谁做了」。</span>

**账号层**最浅：Windows 的 SAM 记录本地账户（口令以散列存储，常用 NTLM 散列）；Linux 的 `/etc/passwd` 与 `/etc/shadow` 记录账户与口令散列；macOS 用户账户在目录服务里。分析账户只能回答「存在哪些账号、口令强度如何」——回答不了「谁在用这个账号」。

**凭据层**更进一步：Windows 的 **LSASS 进程内存**里可能缓存了登录凭据（这就是内存取证 `mimikatz` 插件要 dump 的目标，见下一篇）；Windows 的 DPAPI 密钥保护浏览器与应用的加密数据；macOS 的**钥匙串**保存密码与证书；Linux 的 `~/.ssh/` 保存私钥。凭据的「存在」与「可用」是两回事——拿到 DPAPI 主密钥需要用户口令，密钥环也受口令保护，所以**凭据取证的成败常常卡在口令**，而口令又往往能从其他 OS 痕迹（浏览器保存的密码、备忘录、聊天记录）里找出来。<span class="marginnote">「口令会在别处泄底」是凭据取证的经验法则：用户会在文档、邮件、密码管理器、甚至桌面便签里重复使用口令。OS 取证的价值正在于此——它把散落的凭据线索织成一张「可能的口令候选表」。</span>

**行为层**最扎实也最难：通过时间线把登录、程序运行、文件操作、网络访问与特定用户会话对齐。Windows 的每个会话有 SID，日志里的事件带用户 SID——`who` 式的会话归属让「哪个用户打开了销毁工具」可以被追溯到具体会话。<span class="marginnote">在 Windows 上，「事件日志里事件的安全 ID（SID）+ 登录会话 ID」是身份归属的主线；但别忘了一台机器可能开着多个会话、同一 SID 可能对应多个登录。结论用语要谨慎：说「该 SID 会话下发生了 X」，比说「嫌疑人 X」严谨得多。</span>

身份问题没有完美答案，只有「证据的层层收窄」：账号 → 凭据 → 会话 → 行为，每下一层都更接近「谁」，但每一层都可能被反取证打破。OS 取证在这个链条里承担的是「铺满前两层的记录」——这正与下一篇《内存取证》里「进程与网络的实时快照」、以及《网络取证》里的「远端身份」形成三足鼎立。

## M 小结

- 操作系统自动留下大量**默认开启的痕迹**；Windows 以**注册表 + 事件日志**为核心，Linux 靠**明文日志 + shell 历史**，macOS 靠 **unified log + pList**。
- Windows 高价值键：**Run / UserAssist / RecentDocs / USBSTOR**；无声证词：**Prefetch / 回收站 $I$R / LNK / VSS / 浏览器 SQLite**。
- Linux 分**用户行为线**（`.bash_history` 等）与**系统状态线**（auth.log、syslog、wtmp）两条线找。
- macOS 的金矿：**纳秒级 unified log**、**quarantine 下载记录**、**Spotlight 残片**、**钥匙串**。
- 痕迹是**线索**不是证据，必须回答「哪用户、哪时刻、哪进程」三问；**清理痕迹本身会留痕**。
- OS 痕迹与文件系统、内存、网络三张地图**拼起来**才能重建完整行为，这正是后续三篇的主线。
- 身份证据按 **账号 → 凭据 → 会话 → 行为** 层层收窄；口令常在「别处泄底」，凭据取证要跨源找候选口令。
- 事件日志的三层读法：**按 ID 检索 → 读 XML 结构化字段 → 跨日志关联**；进程创建与 PowerShell 日志是检测内存型恶意软件的本地哨兵。
- 自动化管线（KAPE、Zimmerman、Autopsy）负责提取，**解释与结论永远靠人手工核对原始证据**。

在下一节，我们把镜头转向「正在进行时」的机器内部：**内存取证**——看 Volatility 如何从一段内存镜像里重建进程、网络连接与解密密钥。
