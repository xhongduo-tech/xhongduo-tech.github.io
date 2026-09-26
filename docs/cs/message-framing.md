---
title: 粘包与帧定界
date: 2026-09-08
section: cs
---

# 粘包与帧定界

<div class="epigraph">
<p>TCP 是字节流，两次 `write` 可并成一次 `read`；应用必须自己定界：长度前缀、分隔符或自描述编码。</p>
<footer>—— 据 RFC 9293 字节流；Stevens UNP；Protobuf/gRPC 帧对照整理</footer>
</div>

[序列化](/cs/serialization-protobuf) 产出消息。[Reactor](/cs/reactor-proactor) 一次可读任意字节。[上一课](/cs/c10k-concurrency) 不保证消息边界。缺口是**成帧**：粘包、拆包、缓冲。本课不把心跳写完。

## 问题

UDP 有报文边界（仍可能丢/乱序）。TCP 无。JSON 无长度时要括号配对，危险。常见：4 字节长度 + 体；WebSocket 已有帧；gRPC 有 5 字节前缀；HTTP/1.1 用 Content-Length/chunked。TSO 把大缓冲切开，接收 GRO 再粘，应用仍要定界。SSH 与 TLS 记录层自己成帧。

不要用 `sleep` 当定界。

<span class="marginnote">术语翻译：粘包不是 bug，而是 TCP 按「字节流」交付的正常行为——它只保证字节不丢不重，不保留你 `write` 时的条数边界；一次 `write` 可能被拆成多次 `read`，两次 `write` 也可能被并成一次 `read`，所以应用层必须自带定界结构。</span>

<span class="marginnote">常见误区：以为「`sleep` 一秒再 `read` 就能收完整条消息」。延迟不产生边界，网络一抖照样截断，还白加了延迟；定界只能靠长度前缀、分隔符或记录层这类显式结构。</span>

<span class="marginnote">UNP 强调。本课不发明私有协议细节。</span>

### 字节流无消息

长度前缀或记录层定界。sleep 不是帧。最大长度要帽。SCTP/UDP 有边界是对照不是 TCP 的性质。

## 方法

对照三种定界。画：读循环 → 缓冲 → 切消息 → handler。与线路码逗号对齐同构，一层 PHY，一层应用。

```mermaid
flowchart TD
  BYTES["TCP 字节"] --> BUF["应用缓冲"]
  BUF --> LEN["长度前缀切"]
  BUF --> DELIM["分隔符切"]
```

## 机制

部分读：非阻塞下必循环。最大长度要帽，防内存炸弹。QUIC 流有流内偏移，仍要应用帧若多消息。SCTP 消息边界是对照。PMTUD 不解决粘包。

安全：长度字段过大是 DoS。

<span class="marginnote">数字实例：长度前缀若不限帽，攻击者发 4 字节声称「本帧 0xFFFFFFFF 字节」（约 43 亿），服务器一分配就被内存炸弹打爆；先判 `len \gt 上限`（如 16 MiB）就拒绝，是成帧代码的固定动作。</span>

```mermaid
flowchart TD
  READ["read 到任意个字节"] --> APP["追加进应用缓冲"]
  APP --> HAVE{"缓冲里凑齐一条完整消息了吗?"}
  HAVE -->|"不够"| WAIT["回到事件循环等下次可读"]
  HAVE -->|"够了"| CUT["切出一条交给 handler"]
  CUT --> LOOP["剩余字节留在缓冲继续切"]
  LOOP --> HAVE
```

## 边界

本课不引入 ASN.1 TLV 的全部。心跳与超时是下一课。后课默认：字节流必须应用成帧。

把「一条 write 一条 read」当定理，在负载下必碎。

下一课[心跳与超时](/cs/heartbeat-timeout)。

## 小结

- TCP 粘/拆包是正常。
- 长度、分隔或记录层定界。
- 读循环 + 上限缓冲。
- 后课只引用本课钉死的对象，不从该领域总问题重开。
- 出处：RFC 9293；UNP。
