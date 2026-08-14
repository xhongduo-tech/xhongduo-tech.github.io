---
title: Fetch 与网络请求
date: 2026-08-07
---

# Fetch 与网络请求

<div class="epigraph">
<p>网络请求是前端与世界的边界：发起请求容易，把错误处理明白才是真功夫。</p>
<footer>—— 尼古拉斯 · 扎卡斯（Nicholas C. Zakas）</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 网络 ｜ 2026-08-07</p>
</div>

## 为什么从 Fetch 开始

静态页面不需要网络，但真实应用几乎都在**与服务器对话**：登录、拉取列表、提交订单、流式加载。浏览器提供 **`fetch()`**——现代、基于 Promise 的网络请求 API——作为这一切的标准入口。它是第19篇 Promise 的最大实战：`fetch` 返回 Promise，async/await 消费它。

这一节把「一次 HTTP 请求」拆成前端视角的完整拼图：**方法、URL、请求头、请求体、响应状态**。还要处理两类完全不同的「错误」——**网络失败**（连不上）与 **HTTP 错误状态**（404/500），以及跨域边界 CORS——这是前端安全（第29篇）与后端协作（将来学 Node/API 设计）之间的第一道闸门。<span class="marginnote">`fetch` 由 WHATWG 定义，基于 `Promise`，是 `XMLHttpRequest`（XHR）的现代替代。XHR 回调式、啰嗦、难组合；fetch 简洁、可链式、与 async/await 无缝。新代码一律 fetch。</span>

## 1 HTTP 请求解剖：一次对话的五个要素

一次 HTTP 请求是一段「对话」，前端通过 `fetch` 描述这五个要素：

| 要素 | 说明 | 示例 |
| --- | --- | --- |
| 方法 | 要做什么 | `GET` 读、`POST` 建、`PUT` 整体改、`DELETE` 删 |
| URL | 发给谁 | `https://api.example.com/users` |
| 请求头 | 元信息 | `Content-Type`、`Authorization` |
| 请求体 | 携带的数据 | JSON、表单、文件 |
| 响应 | 服务器回话 | 状态码 + 响应体 |

**方法与语义**：

| 方法 | 语义 | 请求体 | 幂等 |
| --- | --- | --- | --- |
| `GET` | 读取 | 无 | 是 |
| `POST` | 创建 | 有 | 否 |
| `PUT` | 整体更新 | 有 | 是 |
| `PATCH` | 部分更新 | 有 | 是 |
| `DELETE` | 删除 | 通常无 | 是 |

**幂等（idempotent）**：重复执行效果相同——`GET` 发十次还是读同一份数据；`POST` 发十次可能创建十条记录。选错方法（用 `GET` 删数据、用 `POST` 做纯读）违反 REST 约定，也踩进「浏览器预加载会触发 GET」的安全坑。<span class="marginnote"><strong>REST</strong>（表现层状态转移）把 HTTP 方法当作「对资源的动词」：`/users` + `GET` 读列表、`/users/1` + `DELETE` 删一个。前端工程师至少要知道「方法对应语义」，对接后端接口才不懵。</span>

## 2 fetch 基础：GET 与解析响应

最简单的 GET 请求：

```js
const res = await fetch("https://api.example.com/users");
console.log(res.status);       // 200
console.log(res.ok);           // true（status 在 200-299 区间）
```

**关键认知：`fetch` 只在「网络层失败」时 reject**——连不上、DNS 失败、超时。**HTTP 错误状态（404、500）不会 reject**，`res.ok` 为 false、Promise 照样 resolve。这是新手最大的认知差：

```js
const res = await fetch("/api/users");
if (!res.ok) {
  throw new Error(`请求失败：${res.status}`);   // 必须手动处理 HTTP 错误
}
const users = await res.json();
```

**响应体的解析**取决于 `Content-Type`：

```js
const text = await res.text();        // 纯文本
const json = await res.json();        // JSON（解析失败抛错）
const blob = await res.blob();        // 二进制（图片、文件）
const formData = await res.formData();// 表单数据
```

**辨析｜易错点：** `res.json()` 是**一次性消费**——响应体只能读一次。想「先看文本再转 JSON」行不通：`await res.text()` 后 `await res.json()` 会报「body already used」。按需选一个解析方法，别重复读。<span class="marginnote">fetch 默认 `credentials: "same-origin"`——同源请求带 Cookie，跨域不带。要「登录态跨域传递」需显式 `credentials: "include"`（并配合服务端 CORS 白名单）。Cookie/登录态的细节在第29篇《前端安全基础》展开。</span>

## 3 发送数据：POST 与 JSON

POST 请求要带请求体，关键是**声明 `Content-Type`** 让服务器知道怎么解析：

```js
const res = await fetch("/api/users", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",   // 告诉服务器：body 是 JSON
  },
  body: JSON.stringify({ name: "小明", age: 18 }),  // 序列化成 JSON 字符串
});
```

**三种常见的 body 编码**：

| `Content-Type` | body 形态 | 典型场景 |
| --- | --- | --- |
| `application/json` | `JSON.stringify(obj)` | API 交互，最常用 |
| `application/x-www-form-urlencoded` | URL 编码键值对 | 传统表单 |
| `multipart/form-data` | `FormData` 对象 | 文件上传 |

**`FormData` 直接作 body**（第23篇讲过）——浏览器自动设对 `Content-Type` 并加边界：

```js
const fd = new FormData();
fd.append("file", fileInput.files[0]);
await fetch("/api/upload", { method: "POST", body: fd });
```

**解析响应的统一模式**——「请求 → 判状态 → 解析」三件套，封装成工具函数：

```js
async function request(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.status === 204 ? null : res.json();
}
```

**辨析｜易错点：** `JSON.stringify` 与 `JSON.parse` 是成对出现的——发出去要 `stringify`，读回来要 `parse`。忘了 `stringify`（直接传对象），`fetch` 会把它当字符串（`[object Object]`）发送；忘了 `parse`（拿 `res.json()` 当普通属性），得到的是 Promise。这类「少了半步」的 bug 排起来最费时。<span class="marginnote">`JSON.stringify` 对 `undefined`、函数、Symbol 值会直接跳过；对循环引用的对象抛错。发复杂对象前先在 Console 里 `JSON.stringify(obj)` 验证一遍，是省时间的排查习惯。</span>

## 4 错误处理与超时：AbortController

fetch 的完整错误处理需要分**网络失败**与**HTTP 错误**两层，外加**超时**：

```js
try {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 5000);   // 5 秒超时

  const res = await fetch(url, { signal: controller.signal });
  clearTimeout(timer);

  if (!res.ok) throw new Error(`HTTP ${res.status}`);          // HTTP 错误
  return await res.json();
} catch (err) {
  if (err.name === "AbortError") {
    console.error("请求超时或已取消");
  } else if (err instanceof TypeError) {
    console.error("网络错误：连不上服务器");      // 网络层失败 reject 的多是 TypeError
  } else {
    console.error(err.message);                   // HTTP 错误
  }
}
```

**网络失败**：`fetch` reject，错误多为 `TypeError`（「Failed to fetch」）。
**HTTP 错误**：resolve 但 `!res.ok`，需手动抛。
- **超时/取消**：`AbortController` 是标准解法——`abort()` 触发 `AbortError`，`signal` 传给 fetch 即可取消。<span class="marginnote">`AbortController` 也是第19篇 `Promise.race` 超时的现代替代：它真正<strong>取消</strong>底层请求（省流量），而 race 只是「先到先得」、请求还在跑。用户切换页面/组件卸载时取消在途请求，是避免「卸载后更新已卸载 DOM」警告的正解。</span>

## 5 CORS：跨域的边界

**同源（same-origin）**：协议 + 域名 + 端口三者一致。`https://a.com` 请求 `https://a.com/api` 是同源；请求 `https://b.com/api` 是跨域。

**CORS（Cross-Origin Resource Sharing，跨源资源共享）**：浏览器对跨域请求的**守护机制**——它不阻止请求发出，而是**检查服务器的回应头**：

跨域请求发出后，服务器若在响应头返回 `Access-Control-Allow-Origin: https://a.com`（或 `*`），浏览器放行；**没有这个头，浏览器拦截响应**，JS 拿不到数据（控制台报 CORS 错误）。
简单请求（GET、普通 POST）直接发；复杂请求（自定义头、JSON 之外的 content-type）会先发一个 `OPTIONS` **预检（preflight）**试探服务器是否允许。

**CORS 是浏览器的策略，不是服务器的安全机制**——服务器完全可以收到并处理那个请求（响应被浏览器藏起来了）。它防的是「其他网站读取你的数据」，不防「数据发到你的服务器」。<span class="marginnote"><strong>为什么要有 CORS？</strong> 你在逛恶意网站时，它偷偷 `fetch("https://bank.com/api/balance")`——浏览器不允许恶意网站读取银行接口的响应，但银行服务器其实收到了请求。CORS 让「读取跨域数据」必须经服务器明确授权，同时不阻塞正常请求。开发时用代理（Vite/webpack dev server）或 JSONP（历史方案）绕开。</span>

**辨析｜易错点：** CORS 报错信息里「Response to preflight request doesn't pass access control check」——这是**预检失败**，不是请求失败。常见原因：自定义请求头不在服务器允许列表、或 `Access-Control-Allow-Methods` 没包含用到的动词。排查顺序：先看浏览器 Network 面板里有没有 `OPTIONS` 请求、看它的响应头。

## 6 公式解析：fetch 的 Promise 管道

`fetch` 的一切行为可以收敛成一条**管道公式**——它解释了「什么时候 resolve、什么时候 reject、什么时候要自己判断」：

$$
\text{fetch}(u) \Rightarrow \text{Response} \begin{cases} \text{resolve} & \text{连上了（无论状态码）} \\ \text{reject} & \text{网络层失败} \end{cases} \xrightarrow{\;\; !res.ok \;\;} \text{throw} \xrightarrow{\;\; \text{res.json()}\;\;} \text{data}
$$

**逐步拆解：**

- **第一步，发起**：`fetch(url)` 发出请求，返回 Promise。
- **第二步，网络层**：请求能到达服务器（哪怕返回 404）→ Promise **resolve**；连不上/超时/取消 → Promise **reject**。**状态码不参与 resolve/reject 的判定**——这是与直觉相悖的关键。
- **第三步，检查状态**：`res.ok`（2xx）才继续；否则 `throw new Error` 进入 catch——这步必须**手动写**，fetch 不替你判断。
- **第四步，解析体**：`res.json()` 返回 Promise，解析完成才得到数据；body 已读则抛错。

**代入一个实例（请求一个不存在的接口）：** `fetch("/api/nope")` → 服务器返回 404 → **resolve**（没 reject！）→ `res.ok === false` → 手动 `throw` → 进 catch → 显示「HTTP 404」。若服务器宕机（连不上）→ **reject** → catch 里 `err.name === "AbortError"` 或 `TypeError`。

**直觉是什么？** fetch 把「网络是否成功」与「业务是否成功」**分开**：连上 ≠ 成功，状态码 2xx 才是成功。把「状态码检查」养成肌肉记忆，fetch 的错误处理就不再玄学。<span class="marginnote">这条管道公式也是「封装 request 工具函数」的依据：把「判 ok → 解析 → 抛错」固化进工具，业务代码只管 `try/await`。团队里统一一个 request 封装，能消灭大半「没判 res.ok」的隐藏 bug。</span>

## 7 小结

- HTTP 对话五要素：**方法、URL、请求头、请求体、响应**；方法对应语义，`GET` 幂等、`POST` 非幂等。
- **`fetch` 只在网络层失败时 reject**；404/500 是 resolve + `!res.ok`，必须手动检查。
- 响应体一次性消费：`res.json()`/`res.text()`/`res.blob()` 按 Content-Type 选一个。
- POST 带 JSON 要设 `Content-Type: application/json` + `JSON.stringify`；文件用 `FormData`。
- 错误分层：网络失败（TypeError）、HTTP 错误（手动抛）、超时/取消（`AbortController`）。
- **CORS** 是浏览器守护：服务器响应头授权才放行，复杂请求先 OPTIONS 预检；它不是服务器安全机制。

在下一节，我们把数据「存下来」——**Web Storage 与本地持久化**。刷新页面数据就没了？`localStorage`/`sessionStorage`/`IndexedDB` 让浏览器记住状态。
