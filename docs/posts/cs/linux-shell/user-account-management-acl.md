---
title: 账号管理与 ACL 权限设定
date: 2026-08-07
---

# 账号管理与 ACL 权限设定

<div class="epigraph">
<p>多用户系统最难的从来不是功能，而是「谁有权做这件事」的边界。</p>
<footer>—— Unix 权限模型的设计共识</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ 鸟哥《Linux私房菜》 第13章 ｜ 2026-08-07</p>
</div>

## 为什么从账号管理开始

前面学权限时，权限都是挂在「属主 / 属组 / 其他」这三类人上。可现实中一个项目组有五个人、三种角色，标准的三组权限根本装不下——于是 Linux 给出两套答案：**账号系统**负责「你是谁、你能登录吗、你的默认组是什么」，**ACL（访问控制列表）** 则把「三组人」扩展成「任意多个指定用户/组的细粒度授权」。本章把它们串起来：先看账号与组的档案文件长什么样，再学增删改账号的命令，最后用 ACL 补上细粒度权限的缺口。

## 1 账号的四个档案

Linux 把用户信息存在四个文本档案里，理解它们就等于理解了账号模型的全部：

**`/etc/passwd`**：用户账号主档，每行一个用户，七个字段以冒号分隔：

```
root:x:0:0:root:/root:/bin/bash
user1:x:1000:1000:user one:/home/user1:/bin/bash
```

从左到右：**用户名、密码占位符、UID、GID、备注、家目录、登录 Shell**。<span class="marginnote">字段二的 `x` 不是密码本身，而是「密码已加密存放于 /etc/shadow」的占位符。`/etc/passwd` 需要被所有程序读取（所以要 644），把真实密码放进去就泄露了——拆到 root-only 的 shadow 里才是安全设计。</span>

**`/etc/shadow`**：加密密码与密码策略，只有 root 可读。每行形如 `user1:$6$salt$hash:19820:0:99999:7:::`——算法前缀 `$6$` 是 SHA-512，`$1$` 是 MD5，`$y$` 是 yescrypt（新版默认）。

**`/etc/group`**：组档案，格式与 passwd 类似，第 4 个字段是该组的**附加成员列表**（成员用逗号分隔）。

**`/etc/skel/`**：不是档案而是目录——新建用户时，这里面的文件会被复制进新家目录，是「每个新用户默认带 .bashrc」的机制来源。

**易错点**：`/etc/passwd` 里 UID 是 0 的只有 root。系统账号（UID 0–999）用于服务，普通用户从 1000 起编号——给新用户手动分配 UID 时别用错区间。

## 2 用户与组的增删改

命令行下管理用户，核心是 `useradd`、`usermod`、`userdel` 与 `groupadd` 一族：

```bash
useradd -m -s /bin/bash alice        # 建用户并建家目录、指定 Shell
useradd -m -G docker alice           # 附加组：把 alice 加进 docker 组
passwd alice                         # 设置/修改密码
usermod -aG sudo alice               # 追加 sudo 组（-a 追加，避免覆盖）
usermod -s /bin/zsh alice            # 改登录 Shell
userdel -r alice                     # 删用户并删其家目录
groupadd devs                        # 建组
```

**易错点**：`usermod -G` 不加 `-a` 会把用户从原有附加组里全部踢出，只保留你新写的组——这是账号管理里最经典的误操作之一。**追加组成员永远写 `-aG`**。<span class="marginnote">`useradd -M` 表示不建家目录（系统账号常用），`-d` 指定家目录路径。而 `adduser` 是 Debian 系的友善封装，会交互式询问密码等信息；`useradd` 更底层、跨发行版一致，脚本里用 `useradd`。</span>

## 3 su 与 sudo：两种提权

日常操作要求「临时以 root 身份做事」，Linux 给出两条路：

**`su`（switch user）**：切换到另一个用户，需要**目标用户密码**。`su -` 切换成 root 且加载 root 的登录环境（`-` 表示 login shell）；`su alice` 切换到 alice 但保留当前环境。

**`sudo`（superuser do）**：以 root（或其他指定用户）权限**执行单条命令**，需要的是**当前用户自己的密码**。能执行哪些命令由 `/etc/sudoers` 决定。

**核心对比表：su 与 sudo**

| 维度 | su | sudo |
| --- | --- | --- |
| 需要的密码 | 目标用户密码 | 当前用户密码 |
| 权限粒度 | 整段 shell 会话 | 单条命令 |
| 审计 | 基本无 | 记录到日志 |
| 配置 | 无需 | 需写 `/etc/sudoers` |

**sudoers 的典型配置**：`visudo` 编辑（它校验语法，避免改坏后 sudo 全部失效）。

```
alice    ALL=(ALL:ALL) ALL        # alice 可以执行任何命令
devs     ALL=(ALL) /usr/bin/systemctl restart *   # 组内只许重启服务
%sudo    ALL=(ALL:ALL) NOPASSWD: ALL    # sudo 组免密（谨慎）
```

**易错点**：`sudo` 是「用你的身份验证、以 root 身份执行」——所以**别用 sudo 去跑可疑脚本**：一旦脚本里写了 `rm -rf`，它是以 root 身份跑的。<span class="marginnote">`sudo -i` 获得一个 root 登录 shell，`sudo -u alice command` 以 alice 身份执行。审计方面，所有 sudo 调用会记入日志（`journalctl -u sudo` 或 `/var/log/auth.log`），出了事查得到是谁干的。</span>

## 4 公式解析：ACL 的精确授权

三组权限不够用时，**ACL（Access Control List）** 登场：它允许给**任意指定的用户或组**单独授权。`getfacl` 查看、`setfacl` 设置：

$$
\text{权限判断} = \text{属主} \;\rightarrow\; \text{ACL 属主} \;\rightarrow\; \text{属组} \;\rightarrow\; \text{ACL 属组} \;\rightarrow\; \text{其他}
$$

拆解这条判断链：

- **第一步**：进程访问文件时，内核按身份优先级逐级匹配——你是属主就停在属主，不是就查 ACL。
- **第二步**：`setfacl -m u:alice:rwx file` 给用户 alice 单独加 rwx，`g:devs:r-x` 给组 devs 加 r-x。
- **第三步**：`getfacl file` 查看完整列表，ACL 生效时 `ls -l` 权限位末尾会多一个 `+`。

常用命令：

```bash
setfacl -m u:alice:rwx /data/project     # 给 alice 读写执行
setfacl -m g:devs:r-x /data/project      # 给组 r-x
setfacl -x u:alice /data/project         # 移除 alice 的 ACL
setfacl -b /data/project                 # 清空全部 ACL
setfacl -R -m u:alice:rwx /data/project  # 递归设置
```

**易错点**：ACL 是**叠加在传统权限之上**的额外层——它不替代 rwx，而是细化它。删掉 ACL 后文件回到传统三组权限。另外 **`chmod` 某些操作会重置 ACL mask**，使「有效权限」变小；`getfacl` 输出里的 `mask::` 就是「所有命名用户/组权限的公共上限」——这是 ACL 里最容易懵的一点。<span class="marginnote">ACL 依赖文件系统支持：ext4、xfs、btrfs 默认开启，挂载时加 `noacl` 才关闭。校验是否生效：`tune2fs -l /dev/sdXN | grep acl`。目录 ACL 默认不传递给新文件，`setfacl -d` 可设默认 ACL 让新建文件自动继承。</span>

## 5 小结

- **四个档案定义账号**：`/etc/passwd`（UID/GID/家目录）、`/etc/shadow`（加密密码）、`/etc/group`（组与成员）、`/etc/skel/`（新家目录模板）。
- **用户命令一族**：`useradd -m` 建用户、`usermod -aG` 追加组（必带 `-a`）、`userdel -r` 删用户。
- **su 要目标密码、sudo 要自己密码**：sudo 单命令授权 + 审计，配 `sudoers` 精确到命令。
- **ACL 是细粒度授权**：`setfacl -m u:用户:权限` 给指定对象授权，`getfacl` 查看、`+` 号提示 ACL 存在。
- **ACL mask 是有效权限上限**：`chmod` 会重置 mask，理解它才不会误判「为什么权限又不够了」。
- **密码安全从 shadow 开始**：密码散列只在 `/etc/shadow`，给账号设强密码并定期轮换是账号管理的基本功。

在下一节，我们从「谁在跑」进入「跑得怎么样」——**程序管理与进程信号**：ps/top/kill 与后台作业。
