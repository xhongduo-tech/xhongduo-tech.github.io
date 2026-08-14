---
title: 网络编程与套接字
date: 2026-08-07
---

# 网络编程与套接字

<div class="epigraph">
<p>网络编程的本质，是把两个进程之间的「对话」抽象成一对读写的流。</p>
<footer>—— 佚名</footer>
</div>

<div class="article-byline">
<p>第三级 · Java 编程 ｜ 《Java核心技术》第2卷第4章 ｜ 2026-08-07</p>
</div>

## 为什么从网络编程开始

现代程序几乎没有「孤立」的：浏览器请求网页、App 调后端 API、微服务互相调用。**网络编程**让程序跨越进程与机器的边界对话。Java 的网络抽象有两个层次：底层是**套接字（socket）**——直接操作 TCP/UDP，你控制连接与字节流；高层是 `HttpClient`——帮你把 HTTP 请求/响应封装成一行调用。这一篇先讲 TCP/UDP 的差别与套接字模型，再讲 HTTP 客户端——它与你已学的流（InputStream/OutputStream）无缝衔接，因为**套接字本质上就是一对流**。

## 1 TCP 与 UDP：连接的两副面孔

网络传输层的两大协议，服务方式截然相反：

| 维度 | TCP | UDP |
| --- | --- | --- |
| 连接 | 面向连接（先三次握手） | 无连接（直接发） |
| 可靠性 | 可靠：有序、不丢、重传 | 不可靠：可能丢、乱序 |
| 数据边界 | 字节流（无消息边界） | 数据报（有边界） |
| 速度 | 较慢（握手+确认开销） | 快 |
| 典型应用 | HTTP、文件传输、数据库 | 视频通话、DNS、游戏实时位置 |

**重点结论：TCP 保可靠，UDP 保速度。** 需要「每一字节都到达且有序」用 TCP（几乎所有应用协议）；能容忍偶尔丢包、要最低延迟用 UDP（实时音视频）。TCP 的可靠性不是免费的——它用握手、确认、重传换来的，这背后是「可靠传输协议」的教科书内容，见第三级《计算机网络》。<span class="marginnote">Java 里 UDP 用 `DatagramSocket` + `DatagramPacket`，TCP 用 `Socket`/`ServerSocket`。日常开发 95% 是 TCP；UDP 只在低延迟场景出现。先掌握 TCP，UDP 的 API 换一换就通。</span>

**TCP 的三次握手**是「建立连接」的仪式：客户端发 `SYN` → 服务端回 `SYN+ACK` → 客户端回 `ACK`。三次的目的：双方确认「我能发你、你能收、你能发我、我能收」四个方向都通。你不需要在 Java 里手写握手——`new Socket(...)` 内部已经做完了。

## 2 套接字：一对流

**套接字（socket）**是「IP 地址 + 端口号」的组合，标识网络上的一个通信端点。**Java 的 `Socket` 对象内部封装了一对流**：`getInputStream()` 读对端发来的数据，`getOutputStream()` 写数据给对端。

**客户端**的写法：

```java
try (Socket socket = new Socket("example.com", 80);          // 连接对端
     var in = new BufferedReader(new InputStreamReader(
             socket.getInputStream(), StandardCharsets.UTF_8));
     var out = socket.getOutputStream()) {
    out.write("GET / HTTP/1.1\r\nHost: example.com\r\n\r\n".getBytes());
    String line;
    while ((line = in.readLine()) != null) {
        System.out.println(line);
    }
}   // try-with-resources 自动关闭连接
```

**服务端**的写法：`ServerSocket` 监听一个端口，`accept()` **阻塞等待**连接，每来一个连接返回一个新的 `Socket`：

```java
try (ServerSocket server = new ServerSocket(8080)) {
    while (true) {
        try (Socket conn = server.accept()) {   // 阻塞直到有客户端连进来
            handle(conn);                       // 处理这个连接
        }
    }
}
```

**辨析｜易错点：`accept()` 一次只处理一个连接。** 上面的 `while (true)` 串行处理——第二个客户端要等第一个处理完。真实服务必须**每来一个连接就开一个线程**（或提交给线程池）并发处理；串行 accept 是「单线程服务器」，性能天花板极低。这是从「能连」到「能服务并发用户」的分水岭。

**端口**是 0~65535 的整数：0~1023 是特权端口（HTTP 80、HTTPS 443），1024 以上用户可用。**端口冲突**是新手常见报错——`BindException: Address already in use` 说明端口被占用，换一个或先杀掉占用进程。

## 3 半关闭与超时：网络编程的务实细节

网络编程的坑大多藏在细节里。三个高频点：

**读写阻塞**：`readLine()` 在没数据时会**阻塞**当前线程——这不是 bug，是设计。所以服务端每个连接必须独立线程，否则一个慢客户端会卡死所有服务。

**超时设置**：网络可能永远不响应。给 socket 设超时，避免无限期阻塞：

```java
Socket socket = new Socket();
socket.connect(new InetSocketAddress("example.com", 80), 5000);  // 连接超时 5 秒
socket.setSoTimeout(3000);    // 读超时：3 秒没数据就抛 SocketTimeoutException
```

**关闭与半关闭**：TCP 允许**半关闭**——`shutdownOutput()` 告诉对端「我发完了，你还能继续发给我」，常用于「请求-响应」协议（客户端发完请求后半关闭写端，等响应）。`close()` 会关闭整个 socket 的两端。

**辨析｜易错点：`SocketTimeoutException` 要单独捕获。** 设置 `setSoTimeout` 后，读超时抛的是 `SocketTimeoutException`（`IOException` 的子类）——它与连接失败（`ConnectException`）是不同的失败语义，业务上要区分「超时了」与「连不上」。

## 4 公式解析：HTTP 客户端的三行式请求

手写套接字拼 HTTP 报文太原始了。**Java 11 的 `java.net.http.HttpClient`** 把 HTTP 请求封装成声明式 API：

$$

\text{HttpClient} \to \text{HttpRequest} \to \text{send} \to \text{HttpResponse}

$$