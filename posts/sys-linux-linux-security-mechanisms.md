## 核心结论

Linux 安全机制不是单一开关，而是多层叠加的权限收敛链。

第一层是 DAC，Discretionary Access Control，中文常叫“自主访问控制”，白话讲就是“文件所有者和用户组先决定你能不能碰这个对象”。第二层是 ACL，Access Control List，白话讲就是“在 rwx 三段式权限之外，再给某个用户或某个组单独开口子或关口子”。第三层是 Capabilities，白话讲就是“把过去 root 的整包特权拆成很多小特权位，只发需要的那几项”。第四层是 LSM，Linux Security Modules，白话讲就是“内核预留一组安全检查钩子，让 SELinux、AppArmor 这类强制访问控制模块继续收紧权限”。

它们的关系不是替代，而是叠加。一个操作能否成功，通常可以近似写成：

$$
Allow = DAC \land ACL \land Caps \land LSM
$$

只要其中任意一层拒绝，最终访问就失败。

对初学者最重要的理解是：Linux 安全不是“普通用户 vs root”二元模型。现代系统真正有效的做法是最小权限原则，即每个进程只拿完成任务所需的最小权限集合。比如 Web 服务只需要绑定 80 端口，就不该顺带获得挂载文件系统、加载内核模块、修改时钟或管理其他进程的能力。

一个直观场景是读取 `/etc/passwd`。如果文件权限位允许，DAC 可能放行；但若进程所在 SELinux 域与目标文件标签不匹配，LSM 仍可能拒绝。这说明 Linux 的安全增强是在旧 Unix 权限模型上“继续收紧”，而不是“只换一种写法”。

| 层级 | 主要职责 | 典型命令或模块 | 解决的问题 |
| --- | --- | --- | --- |
| DAC/ACL | 先判断主体对对象的基础访问权 | `chmod` `chown` `setfacl` | 传统 Unix 权限控制 |
| Capabilities | 把 root 拆成细粒度特权位 | `setcap` `getcap` `capabilities(7)` | root 权限过粗 |
| LSM | 在内核访问点追加强制策略 | SELinux、AppArmor | 进程被攻破后的横向扩散 |

---

## 问题定义与边界

如果只有 DAC，系统会有两个明显问题。

第一，权限粒度太粗。一个服务如果必须以 root 身份启动，那么它理论上就拿到了大量本不需要的特权。攻击者一旦控制这个进程，提权成本会显著下降。

第二，表达能力有限。DAC 对文件很有效，但对“某个进程是否能访问某类 socket、某段共享内存、某个特定类型文件”这类需求，表达力不够。ACL 虽然细化了对象级授权，但它仍然属于“谁拥有对象，谁分配权限”的思路，无法完整描述“即便文件权限允许，这个服务进程也不该碰”的强制策略。

因此 Linux 的边界可以这样理解：

1. DAC/ACL 解决“这个身份按传统规则是否允许访问”。
2. Capabilities 解决“这个进程是否拥有执行敏感操作的具体特权位”。
3. LSM 解决“即便前面都通过，系统策略是否仍然要求拒绝”。

一个常见公式是：

$$
Allow = DAC(主体,对象) \land ACL(主体,对象) \land [Caps(进程) \supseteq Required] \land Policy(主体标签,对象标签)
$$

这里的“主体”通常是进程，“对象”可能是文件、目录、socket、IPC 对象等。

玩具例子可以先看“绑定 80 端口”这件事。1024 以下端口通常被视为特权端口。传统做法是让服务以 root 启动；更好的做法是只授予 `CAP_NET_BIND_SERVICE`。白话讲，这个 capability 的含义就是“允许绑定低端口”，而不是“允许做所有 root 能做的事”。

所以边界不是“我是不是 root”，而是“我在当前进程上下文里到底持有哪些能力位，并且当前策略是否允许我访问目标对象”。

---

## 核心机制与推导

先看最基础的 DAC 和 ACL。

DAC 的核心是三元组：owner、group、other。比如一个文件权限是 `rw-r-----`，那么所有者可以读写，组用户只能读，其他人无权限。ACL 是对这个模型的补充。它允许你写出“给用户 alice 单独只读，给组 devops 单独读写”这类更精细的规则。

但 ACL 仍然不解决“root 权限过大”的问题，所以引入 Capabilities。

Capabilities 可以理解为“把超级用户拆成几十个开关”。例如：

- `CAP_NET_BIND_SERVICE`：允许绑定低端口
- `CAP_SYS_TIME`：允许修改系统时间
- `CAP_SYS_ADMIN`：能力范围极广，通常被认为接近“半个 root”
- `CAP_DAC_OVERRIDE`：允许绕过 DAC 文件权限检查

这套机制里常见的几个集合如下。

- Effective：当前真正生效的能力位，白话讲就是“此刻能直接拿来用的权限”。
- Permitted：允许进程启用的能力位上限。
- Bounding：执行 `execve()` 后能力可获得范围的硬上限。
- Ambient：在非特权程序 `execve()` 时可以继续继承的一组能力。

几个关键关系是：

$$
Effective \subseteq Permitted
$$

$$
Permitted_{new} = (Inheritable \land FileInheritable) \lor (FilePermitted \land Bounding) \lor Ambient_{new}
$$

如果只抓住工程上最重要的一点，可以记成：

$$
Permitted_{new} \subseteq Bounding
$$

也就是说，Bounding set 像“天花板”，它决定了 `execve()` 之后最多还能拿到什么能力。很多误配置都出在这里：管理员给了文件 capability，却没有认真收紧继承边界，导致后续子进程还保留了不该保留的位。

再看 LSM。

LSM 是内核里的安全钩子框架。白话讲，它不是单独一套策略，而是一排“检查插槽”。SELinux、AppArmor、Smack、TOMOYO 等模块把自己的策略逻辑挂到这些插槽上。能力模块本身也在这套检查序列里，并且在活动 LSM 列表中总是首先出现。其含义不是“后面的模块会覆盖前面的结果”，而是“后面的模块继续追加限制”。

这就是 Linux 安全设计的核心推导：

1. 兼容旧程序，所以保留 Unix 权限模型。
2. root 太粗，所以拆出 Capabilities。
3. 仅靠主体身份还不够，所以再加 LSM 做强制访问控制。
4. 各层默认是收紧，不是放宽。

真实工程例子是 NGINX 或 Apache。它们需要监听 80/443 端口，但并不需要文件系统管理、内核调试、挂载、模块装载等权限。工程上正确做法通常是：

- 用 `setcap` 只赋予绑定低端口能力。
- 用 SELinux 或 AppArmor 限制它只能访问指定站点目录、日志目录和必要 socket。
- 进程再降权到专门服务账号。

这样即便服务进程被利用，攻击面也被压缩在一个较小范围内。

---

## 代码实现

先用一个可运行的 Python 玩具例子，把“多层收紧”逻辑具象化。这个程序不是 Linux 内核源码，而是一个最小模型，用来帮助理解为什么“任意一层拒绝都会失败”。

```python
from dataclasses import dataclass

@dataclass
class Subject:
    uid: int
    groups: set[str]
    caps: set[str]
    label: str

@dataclass
class Object:
    owner_uid: int
    group: str
    mode_owner_read: bool
    mode_group_read: bool
    mode_other_read: bool
    acl_users: dict[int, bool]
    label: str

def dac_allow_read(subj: Subject, obj: Object) -> bool:
    if subj.uid == obj.owner_uid:
        return obj.mode_owner_read
    if obj.group in subj.groups:
        return obj.mode_group_read
    return obj.mode_other_read

def acl_allow_read(subj: Subject, obj: Object) -> bool:
    return obj.acl_users.get(subj.uid, True)

def caps_allow(required: set[str], subj: Subject) -> bool:
    return required.issubset(subj.caps)

def lsm_allow_read(subj: Subject, obj: Object) -> bool:
    # 简化版：只有 web_t 可以读 httpd_sys_content_t
    if subj.label == "web_t" and obj.label == "httpd_sys_content_t":
        return True
    if subj.label == "user_t" and obj.label == "user_home_t":
        return True
    return False

def allow_read(subj: Subject, obj: Object, required_caps: set[str] | None = None) -> bool:
    required_caps = required_caps or set()
    return (
        dac_allow_read(subj, obj)
        and acl_allow_read(subj, obj)
        and caps_allow(required_caps, subj)
        and lsm_allow_read(subj, obj)
    )

web_proc = Subject(uid=1001, groups={"www"}, caps=set(), label="web_t")
site_file = Object(
    owner_uid=2000,
    group="www",
    mode_owner_read=True,
    mode_group_read=True,
    mode_other_read=True,
    acl_users={},
    label="httpd_sys_content_t",
)

passwd_like = Object(
    owner_uid=0,
    group="root",
    mode_owner_read=True,
    mode_group_read=True,
    mode_other_read=True,
    acl_users={},
    label="etc_t",
)

assert allow_read(web_proc, site_file) is True
assert allow_read(web_proc, passwd_like) is False  # DAC 允许也可能被 LSM 拒绝

bind_service_proc = Subject(uid=1001, groups={"www"}, caps={"CAP_NET_BIND_SERVICE"}, label="web_t")
assert caps_allow({"CAP_NET_BIND_SERVICE"}, bind_service_proc) is True
assert caps_allow({"CAP_SYS_ADMIN"}, bind_service_proc) is False
```

这个玩具例子说明两件事。

第一，传统权限允许并不代表最终允许。`passwd_like` 在例子里对 other 可读，但如果标签策略不允许，最终依然失败。

第二，给进程一个能力位，不等于给它全部 root 能力。`CAP_NET_BIND_SERVICE` 只解决低端口绑定，不会自动附带 `CAP_SYS_ADMIN`。

下面看真实工程操作流。假设你把站点内容放在 `/srv/myweb/`，并希望 NGINX 监听 80 端口，同时系统开启 SELinux enforcing。

```bash
sudo setcap 'cap_net_bind_service=+ep' /usr/sbin/nginx
sudo getcap /usr/sbin/nginx

sudo semanage fcontext -a -t httpd_sys_content_t "/srv/myweb(/.*)?"
sudo restorecon -R /srv/myweb

getenforce
sudo setenforce 1
```

如果服务仍然报 `Permission denied`，不要第一反应就关闭 SELinux。更合理的排障流程是：

```bash
sudo setenforce 0
sudo systemctl restart nginx

sudo ausearch -m AVC -ts recent
sudo grep nginx /var/log/audit/audit.log | audit2allow -M mynginx_local
sudo semodule -i mynginx_local.pp

sudo setenforce 1
sudo systemctl restart nginx
```

这里的 `audit2allow` 作用是“根据实际拒绝日志生成最小放行策略建议”。它适合调试，不适合无脑全量导入。正确姿势是看清楚到底放行了什么，再决定是否安装模块。

---

## 工程权衡与常见坑

Linux 安全增强的收益很明确：显著降低服务被攻破后的横向移动和纵向提权风险。但代价也很明确：配置复杂度、排障门槛和少量运维成本会上升。

最常见的误区不是“策略太严”，而是“没有分层理解”。

| 陷阱 | 典型现象 | 根因 | 规避方式 |
| --- | --- | --- | --- |
| 只会改 `chmod 777` | 权限放到最大仍然报拒绝 | 问题不在 DAC，而在 SELinux/AppArmor | 先查 AVC 或 profile 日志 |
| 直接让服务跑 root | 功能正常，但攻击面极大 | 把“能运行”误当“安全” | 用 `setcap` + 降权用户 |
| 给了 file capability 仍失败 | 程序能执行但绑定端口失败 | 运行链路中的继承集合或边界未满足 | 检查 permitted/effective/bounding |
| 自定义目录无法被 httpd 读取 | `/srv` 或 `/opt` 下资源访问失败 | 文件标签不是 `httpd_sys_content_t` | 用 `semanage fcontext` 和 `restorecon` |
| 调试时直接 disabled SELinux | 短期解决，长期更难恢复 | 文件不再正确标记 | 用 permissive 或 permissive domain，而不是 disabled |

有几个工程判断尤其重要。

第一，不要把 `CAP_SYS_ADMIN` 当通用修复键。这个能力覆盖范围过大，很多场景里它已经接近“绕过设计”。如果一个方案只能靠加 `CAP_SYS_ADMIN` 才跑起来，通常说明权限边界设计还没理清。

第二，SELinux 的很多问题本质上不是“服务没权限”，而是“对象标签错了”。比如把站点文件放到 `/srv/myweb/` 后，即使 Unix 权限正确，httpd 也可能因为文件类型不是 `httpd_sys_content_t` 而被拒绝。这种问题靠 `chmod` 根本修不好。

第三，调试应该优先使用 permissive 模式或单域 permissive，而不是 disabled。permissive 的价值在于“仍然记日志，只是不拦截”，这样你还能看到真实被拒绝的访问路径；disabled 则会让标签体系停止工作，后续恢复成本更高。

第四，容器环境里这些机制会叠加得更复杂。namespace 和 cgroup 解决的是资源隔离，seccomp 解决的是系统调用白名单，而 LSM 解决的是对象访问策略。它们不是同一层问题，不能互相替代。

---

## 替代方案与适用边界

SELinux 不是 Linux 安全的唯一实现方式。它只是 LSM 体系中最典型、最严格的一类。

AppArmor 是另一条常见路线。它的白话解释是“按文件路径写限制规则”，对很多团队来说更容易上手，尤其适合 Ubuntu 系环境。相对地，SELinux 更偏标签模型，表达力强，但概念和排障成本更高。

Smack 和 TOMOYO 也属于 LSM 家族，常见于特定发行版或嵌入式场景。它们的定位通常是简化策略或适配特定安全模型。

还要区分几种经常被混为一谈的机制。

- Seccomp：白话讲是“限制进程能调用哪些系统调用”。它擅长减少内核攻击面，但不直接表达“哪个标签的进程能读哪个标签的文件”。
- Namespaces：白话讲是“给进程看见的系统资源做隔离视图”。它解决可见性和隔离，不直接替代访问控制策略。
- cgroups：白话讲是“限制 CPU、内存、IO 等资源配额”。它偏资源治理，不是安全策略主轴。

所以适用边界可以概括为：

| 机制 | 擅长解决的问题 | 不擅长替代的问题 |
| --- | --- | --- |
| DAC/ACL | 基础文件权限与精细对象授权 | 服务被攻破后的强制隔离 |
| Capabilities | root 特权拆分 | 文件/目录标签级强制策略 |
| SELinux/AppArmor | 进程到对象的强制访问控制 | 系统调用级最小化 |
| Seccomp | 缩减系统调用面 | 文件标签、目录标签策略 |
| Namespace/cgroup | 资源隔离与配额 | 精细权限判定 |

如果是面向初级工程师的实战建议，可以这样选：

- 单机服务，先学好 DAC/ACL 和 `setcap`。
- RHEL、Fedora、CentOS Stream 环境，优先理解 SELinux 标签和 AVC 日志。
- Ubuntu 环境，优先理解 AppArmor profile。
- 容器运行时，再把 seccomp、namespace、capabilities 一起看成完整运行时隔离面。

真正成熟的工程体系，不会寄希望于某一个机制“单独保底”，而是把这些机制按层叠加，让每一层都只承担自己最擅长的职责。

---

## 参考资料

- Linux Kernel Documentation, Linux Security Modules: https://www.kernel.org/doc/html/v5.15/security/lsm.html
- Linux Kernel Documentation, LSM usage and active module order: https://www.kernel.org/doc/html/v4.16/admin-guide/LSM/index.html
- `capabilities(7)` manual page: https://man7.org/linux/man-pages/man7/capabilities.7.html
- `setpriv(1)` manual page: https://man7.org/linux/man-pages/man1/setpriv.1.html
- Red Hat, Using SELinux: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/html/using_selinux/
- Red Hat, changing SELinux modes and permissive domains: https://docs.redhat.com/en/documentation/red_hat_enterprise_linux/8/epub/using_selinux/changing-to-enforcing-mode_changing-selinux-states-and-modes
- Linux.com, Overview of Linux Kernel Security Features: https://www.linux.com/training-tutorials/overview-linux-kernel-security-features/
