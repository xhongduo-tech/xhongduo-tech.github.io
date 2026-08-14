---
title: 表单验证与客户端交互
date: 2026-08-07
---

# 表单验证与客户端交互

<div class="epigraph">
<p>好的表单验证不是拦住用户，而是在用户犯错前轻轻拉他一把。</p>
<footer>—— 卢克 · 沃罗布莱夫斯基（Luke Wroblewski），《Web 表单设计》</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 表单 ｜ 2026-08-07</p>
</div>

## 为什么从表单验证开始

第1篇《列表、表格与表单控件》认识了表单控件，第22篇学会了事件。现在把两者合起来：**如何让表单变得聪明**——输入即时反馈、提交前拦截、错误就地显示。这就是**表单验证与客户端交互**：前端最日常、也最能体现「体验」的工作。

验证有一条必须刻在脑里的红线：**客户端验证是体验，不是安全**。它在浏览器里拦截「明显不对」的输入（空、格式错、超范围），让用户少一次无谓的服务器往返；但攻击者可以绕过一切前端校验直接打后端接口——所以**服务端必须做同等的、真正的校验**。这条「前端体验 + 后端安全」的分工，贯穿第29篇《前端安全基础》。<span class="marginnote">为什么前端验证不是安全？因为「验证」发生在浏览器——而浏览器在攻击者手里就是一个可自由改写的程序。关掉 JS、篡改请求，前端校验形同虚设。安全边界永远在服务端，前端只负责「友好的第一道提醒」。</span>

## 1 约束验证：HTML 自带的校验器

HTML5 引入**约束验证（constraint validation）**——不用写一行 JS，浏览器原生执行校验规则。第1篇见过的 `required` 只是其中之一：

| 属性 | 约束 | 作用于 |
| --- | --- | --- |
| `required` | 必填 | 所有控件 |
| `minlength` / `maxlength` | 长度范围 | text、textarea |
| `pattern` | 正则匹配 | text、search 等 |
| `min` / `max` | 数值范围 | number、date、range |
| `step` | 步进值 | number |
| `type="email"` / `type="url"` | 格式 | input |

```html
<input type="email" required placeholder="you@example.com">
<input type="password" required minlength="8" pattern="(?=.*\d).*">
```

浏览器会在**提交时**自动拦截：不符合规则的控件被标为 `:invalid`，并阻止表单提交，同时显示**本地化错误气泡**。

**CSS 状态选择器**让「错误样式」不写 JS 就能上：

```css
input:valid { border-color: #2e7d32; }
input:invalid { border-color: #c0392b; }
```

`:invalid` 初识（页面刚加载、用户还没碰过）就触发——所以「空输入」也会被标红，通常需要用 `:user-invalid`（用户实际交互过才标）避免「一进来满屏红」的坏体验。<span class="marginnote">`:user-invalid`（用户交互后仍无效）比 `:invalid` 体验好得多：它只在「用户真正试过且没成功」时显示错误，避免了「表单一打开就全部标红」的惊吓。浏览器支持已成熟，新代码优先用它。</span>

## 2 校验 API：checkValidity 与 setCustomValidity

原生校验之外，JS 提供一套**校验 API** 让你精确控制：

```js
const form = document.querySelector("#reg");

form.addEventListener("submit", (e) => {
  e.preventDefault();                 // 先拦下默认提交
  if (form.checkValidity()) {         // 全部通过？
    submitData(new FormData(form));   // 通过才真正提交
  } else {
    form.reportValidity();            // 让浏览器显示错误气泡
  }
});
```

**`form.checkValidity()`**：布尔——所有控件是否都满足约束；不满足时 `false`。
**`form.reportValidity()`**：检查并**显示**错误提示（气泡或错误文本）。
- **`form.noValidate`**：关掉整个表单的原生校验（自定义校验时用）。

**`setCustomValidity(msg)`** 是自定义校验的入口——**设置自定义错误信息**，值非空即视为校验失败：

```js
const confirm = document.querySelector("#confirm");
const password = document.querySelector("#password");

confirm.addEventListener("input", () => {
  if (confirm.value !== password.value) {
    confirm.setCustomValidity("两次密码不一致");
  } else {
    confirm.setCustomValidity("");    // 清空，恢复合法
  }
});
```

`setCustomValidity("")` 是「解除自定义错误」的关键——忘记清空，控件永远校验失败。而 `confirm.validationMessage` 可读取当前的错误信息，用于在自定义 UI 里显示。<span class="marginnote">`setCustomValidity` 本质是「手动把控件塞进 invalid 状态并附上理由」。错误信息就是 `validationMessage`。做「密码一致性」「邀请码校验」这类「控件自身正则表达不了的规则」时，它是唯一入口。</span>

## 3 表单事件流：submit、input、change、blur

表单交互是事件的合集，四个核心事件各有节奏：

| 事件 | 触发时机 | 典型用途 |
| --- | --- | --- |
| `submit` | 点提交/回车 | 拦截提交、最终校验、发请求 |
| `input` | 每次值变化 | 实时预览、实时校验、字数统计 |
| `change` | 失焦且值已变 | 改完再处理（下拉、复选框） |
| `focus` / `blur` | 聚焦/失焦 | 显示/隐藏帮助、标记「已尝试」 |

**submit 的拦截**是表单 JS 的主战场：

```js
form.addEventListener("submit", (e) => {
  e.preventDefault();              // 阻止页面刷新/跳转
  const data = Object.fromEntries(new FormData(form));  // 收集成对象
  fetch("/api/register", { method: "POST", body: JSON.stringify(data) });
});
```

`e.preventDefault()` 是核心——不拦，浏览器会按 `action` 整页刷新跳转（传统表单行为），现代单页交互都要先拦住，改用 `fetch` 异步提交（第24篇）。

**input 的实时校验**要配**防抖**（第15篇的闭包武器）：每次按键都校验很重，等用户停 300ms 再校验：

```js
const check = debounce((value) => {
  fetch(`/api/check-name?q=${encodeURIComponent(value)}`)
    .then(r => r.json())
    .then(res => showHint(res.available ? "可用" : "已被占用"));
}, 300);

username.addEventListener("input", (e) => check(e.target.value));
```

**辨析｜易错点：** `input` 与 `change` 的节奏——`input` 每敲一个字符都触发（用于实时），`change` 只在失焦时触发一次（用于「改完了再算」）。**别把重活绑在 `input` 上**——每次都跑网络请求/复杂计算会卡顿，用防抖或改用 `change`。<span class="marginnote">表单交互的经典体验清单：错误信息放在输入框<strong>旁边</strong>（而非顶部汇总）；出错时 `focus` 到第一个错误控件；成功提交前禁用提交按钮防重复点击；`input` 实时清掉已修正的错误。这些「小细节」决定表单的可用性。</span>

## 4 动态表单：FormData 与字段收集

**`FormData`** 把表单自动收集成键值对——不用手动 `getElementById` 逐个取：

```js
const form = document.querySelector("#reg");
form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const fd = new FormData(form);
  console.log(fd.get("username"));        // 读单个字段
  const obj = Object.fromEntries(fd);     // 转成普通对象
  // fd 直接传给 fetch 的 body，自动设置 multipart 类型
  await fetch("/api/upload", { method: "POST", body: fd });
});
```

`FormData` 会自动收集所有**带 `name` 的控件**（呼应第1篇「没有 name 不参与提交」）。
`fd.get(name)` 读值、`fd.append(name, value)` 追加、`fd.set` 覆盖。
- 直接作为 `fetch` body 时，浏览器自动加 `multipart/form-data` 边界——文件上传用它最省事。

**动态增删字段**是表单交互的常见需求（如「添加多个标签」）：

```js
addBtn.addEventListener("click", () => {
  const input = document.createElement("input");
  input.name = "tag";
  input.required = true;
  tagsBox.append(input);
});
```

新加的字段只要带 `name`，就自动被 `FormData` 收集——「表单是数据契约，name 是键」的模型让动态字段天然成立。<span class="marginnote">`FormData` 收集<strong>当前 DOM 状态</strong>而非初始 HTML——用户输入的值、动态添加的字段都在其中。而第21篇说的 `input.value` 实时值，与 FormData 是同一来源，两者互补：FormData 管「整体收集」，value 管「单个读取」。</span>

## 5 核心对比表：三种验证层次怎么选

验证不是「要么 HTML 要么 JS」，而是分层的组合：

| 层次 | 手段 | 时机 | 定位 |
| --- | --- | --- | --- |
| HTML 约束 | `required`/`pattern`/`type` | 提交时、原生气泡 | 零成本基础校验 |
| CSS 反馈 | `:user-invalid`/`:valid` | 交互时即时 | 视觉反馈 |
| JS 增强 | `setCustomValidity`/`input` 监听 | 任意时刻 | 跨字段/远程校验 |
| 服务端校验 | 后端代码 | 收到请求时 | **安全边界**，不可省略 |

**实践推荐的分层**：先用 HTML 约束覆盖「格式/必填」这类通用规则（零成本）；用 CSS `:user-invalid` 给即时反馈；JS 只负责「HTML 表达不了的规则」（密码一致、异步查重）；后端照常做完整校验——四层各司其职，不互相替代。

**辨析｜易错点：** 「表单没提交成功却看起来成功」的经典 bug——验证通过就 `submit()`，但 `form.submit()` **不触发 submit 事件**（它是原生行为），所以自定义校验不会跑。正确做法：在 submit 事件里 `checkValidity()` 后手动 `fetch` 提交，而不是再调 `form.submit()`。<span class="marginnote">`form.submit()` 绕过 submit 事件、也绕过验证——它直接执行原生提交。所以「代码里点提交」要分两路：模拟用户点 `<button type="submit">`（触发事件+验证）或 `requestSubmit()`（现代 API，会先验证）。`form.requestSubmit()` 是推荐替代。</span>

## 6 公式解析：校验状态机

表单校验的整个流程，是一个**状态机**——每个控件在「合法/非法」之间迁移，UI 与提交动作都由状态驱动：

$$
S = f(\text{input}, \text{constraints}, \text{touched}), \quad S \in \{\text{valid}, \text{invalid}\}
$$

**逐步拆解：**

- **第一步，输入到达**：`input`/`change` 事件携带新值到达控件——校验的「输入信号」。
- **第二步，求约束**：浏览器把 `required`、`pattern`、`min/max`、`setCustomValidity` 全部求值——任一不满足，状态为 `invalid`。
- **第三步，看交互**：`touched`（用户是否碰过）决定错误**何时展示**——`:user-invalid` 只在「碰过且仍非法」时匹配，避免一进场满屏红。
- **第四步，门禁动作**：提交时，`checkValidity()` 汇总所有控件——全 valid 才放行 `submit`；否则 `reportValidity()` 展示错误并聚焦首个错误控件。

**代入一个实例（邮箱输入）：** 用户敲 `"abc"` → `type="email"` 约束不满足 → 状态 `invalid` → 但因用户正在输入、`touched` 但未失焦，`:user-invalid` 条件待定 → 用户失焦 → 错误显示「请输入有效邮箱」→ 用户改正为 `a@b.com` → 约束满足 → 状态 `valid`，错误消失、`:valid` 生效 → 提交时全部 valid → 放行。

**直觉是什么？** 「状态 + 时机」两个维度——**合法性是状态，展示与否是时机**。把「数据对不对」与「什么时候让用户看见」分开建模，正是现代表单库（React Hook Form 等）的设计内核；而 `:user-invalid` 就是「时机」的 CSS 原生表达。<span class="marginnote">这个「状态/时机」二分也解释了为什么「验证消息闪烁」很常见：错误在 `invalid` 与 `valid` 间来回跳，UI 就跟着闪。成熟的方案是「失焦后才锁定错误，之后输入中仍保持」——用状态去抖，而不是让 UI 直接跟随每一次输入。</span>

## 7 小结

- **约束验证**是 HTML 自带的：`required`/`pattern`/`min`/`max`/`type`，零 JS 即拦格式错误。
- 校验 API 三兄弟：`checkValidity()`（判）、`reportValidity()`（判并展示）、`setCustomValidity(msg)`（自定义错误，空串解除）。
- 事件节奏：`submit` 拦截提交、`input` 实时、`change` 失焦处理、`focus/blur` 管时机。
- `FormData` 自动收集带 `name` 的控件，可直接作 `fetch` body；动态字段自动入列。
- 验证分层：**HTML 约束 → CSS 反馈 → JS 增强 → 服务端校验**；前端是体验，后端才是安全边界。
- 校验是状态机：合法性是状态、展示是时机；`:user-invalid` 表达「碰过且仍非法」。

在下一节，我们让页面与服务器对话——**Fetch 与网络请求**。表单数据要发出去、数据要取回来，`fetch` 是现代浏览器给出的标准答案。
