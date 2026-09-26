---
title: SSH 密钥认证
date: 2026-09-08
section: cs
---

# SSH 密钥认证

<div class="epigraph">
<p>SSH 用主机钥钉服务器身份，用用户钥或证书代替口令。agent 转发、已知主机文件与授权文件的权限，比再选一种分组密码更能决定这次登录是否被掉包。</p>
<footer>—— RFC 4251–4254；Ylonen and Lonvick；对照 OpenSSH 的认证模型</footer>
</div>

上一课[TLS 攻击史](/cs/tls-attack-history)收的是 Web 信道。缺口是**运维信道**：SSH 不走 PKI 浏览器锚，而走 first-contact TOFU 与 `authorized_keys`。不重写 AEAD。

后课默认已经读完本课钉下的合同，只补差，不从该领域第一性原理重开。

## 问题

口令登录把[口令与加盐](/cs/password-salt)问题搬到每个 sshd。公钥认证：用户持签钥，服务器存公钥。主机钥不匹配应中止——TOFU 在第一次无法防中间人。缺口是信任启动与 agent 的委托范围，不是再讲握手字节。

<span class="marginnote">TOFU（Trust On First Use）翻译一下：第一次连接时你没有任何先验指纹可对，只能选择「相信并记住」；从第二次起才有得比对。它防的是「指纹变了」这件事，防不了「第一次就被换」。像第一次接到陌生号码便记下声音，之后靠声音识人——但第一通电话本身无法验证。</span>

### 不要转发 agent 到不可信跳板

agent 转发等于把签名能力借出。跳板被占则用户钥被当签具。

<span class="marginnote">RFC 4252 认证。OpenSSH 证书是另一 CA 模型，与 X.509 平行。本课不写暴力破口令的字表。</span>

<span class="marginnote">agent 转发的常见误区：以为私钥文件被传到了跳板上。实际私钥始终留在你本机，转发出的是一条「远程进程可以请求本地 agent 代签」的通道——但效果等价于把银行卡和密码一起借给同事：对方不需要拿走卡，能让你刷卡就够了。不可信跳板上任何拿到该 socket 的进程都能以你的身份登录下一跳。</span>

## 方法

分主机认证与用户认证。known_hosts 钉指纹；CA 签主机钥可规模化。权限：`.ssh` 过宽则密钥文件被换。对照 WireGuard：双方静态钥，模型更简单。

```mermaid
flowchart TD
  HK["主机钥 TOFU/CA"] --> SESS["会话密钥"]
  UK["用户签钥"] --> AUTH["authorized_keys"]
  AUTH --> SESS
  AGENT["agent"] -.->|"转发=借出签名"| RISK["跳板风险"]
```

## 机制

管理面是高价值会话：一次登录等于 root 路径。密钥口令短语与硬件令牌把签钥关进第二因素。下一课 IPsec：网关到网关的 SA，不是交互式 shell。

```mermaid
flowchart TD
  FIRST["第一次连接：known_hosts 无记录"] --> WINDOW["唯一可能被中间人顶替指纹的窗口"]
  WINDOW --> PIN["记录指纹：钉住"]
  PIN --> LATER["第二次起：比对现连指纹"]
  LATER --> SAME["一致 → 放行"]
  LATER --> DIFF["不一致 → 中止并告警"]
```

<span class="marginnote">口令短语与硬件令牌的分工可以这样记：私钥文件像一张卡，口令短语是「用卡的 PIN」，硬件令牌（如 FIDO 密钥）则更进一步——签名操作必须在物理设备里完成，文件被拷走也签不了。三者叠加后，「偷到文件」不再等于「拿到身份」。</span>

## 边界

本课不把全部 KEX 算法列菜单。IPsec 与 IKE 下一课：UDP 上的协商与 SPD。

## 小结

- TLS 服务万维网；SSH 钉运维身份。
- 主机钥 TOFU 弱在第一次；用户钥代替口令。
- agent 转发是委托，不是免费便利。
- 下一课 IPsec 与 IKE。
- 出处：RFC 4251–4254；OpenSSH 手册；对照 Anderson 对 TOFU。
