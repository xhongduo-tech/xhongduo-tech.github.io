---
title: 前端自动化测试基础
date: 2026-08-07
---

# 前端自动化测试基础

<div class="epigraph">
<p>测试不是为了证明代码没错，而是为了让你敢在明天改它。</p>
<footer>—— 肯特 · 贝克（Kent Beck），TDD 之父</footer>
</div>

<div class="article-byline">
<p>第三级 · JavaScript 与前端开发（HTML/CSS/JS） ｜ MDN Web Docs 工具 ｜ 2026-08-07</p>
</div>

## 为什么从测试开始

功能写完了、构建也通了，但「改动会不会弄坏旧功能」——这是前端工程里最磨人的焦虑。**自动化测试**把这种焦虑变成可验证的确定性：每次改动跑一遍测试，坏的立刻红、好的安心绿。它是**重构的安全网**：没有测试，改代码就像在雷区走路。

前端测试有自己的金字塔——**单元测试、组件/集成测试、端到端测试**，三层各司其职。测试不是「写完代码后的苦差」，而是一种**设计工具**：写不出来的测试往往暴露了「代码耦合过重、难以独立验证」的问题。从单元到端到端，本节搭起完整的测试认知。<span class="marginnote">「测试金字塔」由 Mike Cohn 提出：<strong>底层单元测试最多（快、便宜、粒度细），顶层 E2E 最少（慢、贵、覆盖真实路径）</strong>。金字塔形状暗示数量分布——只写 E2E 会慢到跑不动，只写单元会漏掉「组件协作」的集成问题。</span>

## 1 测试金字塔：三层测试各自护什么

**单元测试（unit test）**：测试**最小的独立单元**——一个函数、一个工具方法。特点：极快、极多、不碰 DOM 和网络。

```js
// utils.test.js
import { formatPrice } from "./utils";

test("formatPrice 保留两位小数", () => {
  expect(formatPrice(3.14159)).toBe("3.14");
});
```

**集成/组件测试（integration/component test）**：测试**几个单元的协作**——一个组件渲染、交互后产出什么。前端最常见的测试对象是**组件**：渲染它、模拟点击、断言 DOM 与行为。

**端到端测试（E2E test）**：模拟**真实用户在浏览器里操作**——打开页面、点击、填写、断言最终结果。用 Playwright、Cypress 驱动真实浏览器。

| 层次 | 测试什么 | 速度 | 数量 | 稳定性 |
| --- | --- | --- | --- | --- |
| 单元 | 函数/工具 | 毫秒级 | 最多 | 最稳 |
| 组件/集成 | 组件协作、状态 | 秒级 | 中 | 中 |
| E2E | 用户真实路径 | 秒–分级 | 最少 | 易受环境波动 |

**策略**：**金字塔形分布**——大量单元测试打底，中等组件测试，少量 E2E 保关键路径。E2E 最接近真实但最贵，只覆盖「登录、下单、注册」这类核心流程。<span class="marginnote">为什么 E2E 不能全量铺开？它要起真实浏览器、等真实网络、处理真实时序——一次全站 E2E 可能跑几十分钟，还时不时因环境波动「假失败」。把它留给「核心路径 + 发布前回归」是对的。</span>

## 2 测试的基本语法：describe、test 与 expect

现代 JS 测试（Jest、Vitest）语法高度统一，三件套：

```js
import { describe, test, expect } from "vitest";

describe("formatPrice", () => {          // describe：分组
  test("正常数字格式化为两位小数", () => {  // test：一个用例
    expect(formatPrice(3.14159)).toBe("3.14");   // expect：断言
  });

  test("负数处理", () => {
    expect(formatPrice(-1.5)).toBe("-1.50");
  });
});
```

**断言（matcher）**是「期望什么」，常用的一族：

```js
expect(value).toBe(42);            // 严格相等（Object.is）
expect(obj).toEqual({ a: 1 });     // 深比较
expect(arr).toContain("x");        // 数组包含
expect(fn).toHaveBeenCalled();     // 函数被调用过
expect(fn).toHaveBeenCalledWith(1, "a");   // 带参数调用
expect(() => risky()).toThrow();   // 抛错
```

**`toBe`** 用严格相等——对象/数组要 `toEqual`（逐字段深比较）。
`toThrow` 断言「应该抛错」——把调用包进箭头函数，否则错误发生在断言外。

**生命周期钩子**控制测试的环境：

```js
beforeEach(() => resetState());    // 每个用例前跑
afterEach(() => cleanup());        // 每个用例后跑
beforeAll(() => setupOnce());      // 本组第一个用例前跑一次
```

**辨析｜易错点：** 测试必须**互相独立**——每个用例前 `beforeEach` 重置状态，否则用例 A 改了全局、用例 B 跟着错。测试隔离是铁律：**顺序无关、互不影响**的测试才有意义。<span class="marginnote">「测试与顺序无关」意味着：任何一个用例单独跑、倒序跑、打乱跑，结果都一样。依赖「上一个用例留下状态」的测试，是团队里最阴险的假绿——它换台机器、换个顺序就红。</span>

## 3 组件测试：渲染、交互与断言

组件测试测「组件渲染出的 DOM 是否符合预期 + 交互是否改变状态」。主流用 **Testing Library**（配合 Jest/Vitest）：

```js
import { render, screen, fireEvent } from "@testing-library/react";

test("点击按钮后显示欢迎语", () => {
  render(<Greeting name="小明" />);
  fireEvent.click(screen.getByText("打招呼"));
  expect(screen.getByText("你好，小明")).toBeInTheDocument();
});
```

**Testing Library 的哲学**：**按用户视角测试，不按实现细节**——查询用「用户看到什么」（文本、角色、标签）而非「内部类名/结构」：

```js
screen.getByText("保存");          // 按可见文本查
screen.getByRole("button", { name: "保存" });  // 按角色+名称查（推荐）
screen.getByLabelText("用户名");   // 按关联 label 查
```

**`getBy*`**：找不到就抛错（用于「必须存在」）。
**`queryBy*`**：找不到返回 null（用于「应该不存在」）。
- **`findBy*`**：异步等待出现（用于加载后内容）。

**`fireEvent`** 同步触发事件（`click`、`change`）；异步交互用 `userEvent`（更贴近真实：支持键盘、focus、长按时序）——现代推荐 `userEvent`。

**断言「不该出现」**：

```js
expect(screen.queryByText("加载中")).not.toBeInTheDocument();
```

**辨析｜易错点：** 组件测试**别断言内部实现**——查 `className`、查内部 state、查 DOM 结构，都会让测试在「重构样式/内部结构」时脆断。Testing Library 的 `screen.getByRole` 正是为「用户能感知到的」而设计：**测行为，不测实现**——这是组件测试能支撑重构的关键。<span class="marginnote">「测行为不测实现」的底气来自第21篇「DOM 是运行时真相」：用户看到的是渲染后的可访问树，不是源码里的 div 嵌套。按角色/文本/标签查询，恰好就是屏幕阅读器感知页面的方式——测试顺带成了可访问性检查。</span>

## 4 模拟：Mock 与假环境

组件测试要隔离「外部依赖」——网络请求、定时器、随机数，都该**模拟（mock）**，让测试确定、快速、不依赖真实环境：

```js
// 模拟 fetch
vi.stubGlobal("fetch", vi.fn(() =>
  Promise.resolve({ ok: true, json: () => Promise.resolve({ name: "小明" }) })
));

// 模拟模块
vi.mock("./api", () => ({
  fetchUser: vi.fn(() => Promise.resolve({ id: 1 })),
}));

// 模拟定时器
vi.useFakeTimers();
```

**Mock 三兄弟**：

**`vi.fn()`**：造一个「可断言被调用」的函数——记录调用次数、参数。
**`vi.mock(path, factory)`**：整体替换一个模块——隔离网络/依赖。
- **`vi.spyOn(obj, "method")`**：包住真实方法，既能断言调用又保留原行为（或不调原实现）。

**mock 的用处**：

- 单元/组件测试**不碰真实网络**——快、稳、可复现。
- 断言「调用方是否正确地调了依赖」——`expect(fn).toHaveBeenCalledWith("参数")`。

**辨析｜易错点：** mock 是「必要之恶」——mock 越多，测试越「假」。只 mock **跨边界的东西**（网络、时间、随机），业务逻辑本身不要 mock——否则你在「测试自己编的故事」。**mock 边界之外，测真实逻辑**。<span class="marginnote">「mock 边界」的实践：组件内部的计算、格式化、条件分支，让它们真实跑；只有 `fetch`、`Date.now`、`Math.random` 这类「外部世界」才 mock。标准是——测试失败时，你要能相信「是我的逻辑错，不是环境抽风」。</span>

## 5 端到端测试：真实浏览器里的完整路径

**E2E 测试**用 Playwright / Cypress 驱动真实浏览器，覆盖「用户真正做的事」：

```js
import { test, expect } from "@playwright/test";

test("用户能完成登录并看到首页", async ({ page }) => {
  await page.goto("https://app.example.com/login");
  await page.getByLabel("用户名").fill("user1");
  await page.getByLabel("密码").fill("pass123");
  await page.getByRole("button", { name: "登录" }).click();
  await expect(page.getByText("欢迎回来")).toBeVisible();
});
```

**Playwright 的特点**：跨浏览器（Chromium/Firefox/WebKit）、自动等待（元素出现才操作）、可截图/录屏、网络拦截。

**E2E 的价值**：验证**真实集成**——前端 + 后端 + 数据库 + 浏览器，整条链路一起通。单元/组件测试各自为政，可能「每个都对、合起来崩」；E2E 专抓这种「集成裂缝」。

**E2E 的成本**：慢、依赖真实环境（测试数据库、测试账号）、易受时序波动。所以——

只覆盖**核心用户路径**（注册、登录、下单、支付）。
**CI 里跑**（每次提交自动跑，第26篇构建的延伸）。
- 用**测试环境**（独立数据库、mock 支付），不碰生产数据。<span class="marginnote">E2E 的「真实」是双刃剑：它验证了整条链，也因此被整条链的任何一环拖慢/拖挂。现代实践是「E2E 留核心 + 测试替身（test double）管不稳定边界」——例如 mock 支付网关，但真实走前端→后端→数据库。</span>

## 6 公式解析：红—绿—重构循环

测试驱动开发（TDD）把「测试」前移成**写代码的节奏**，浓缩成一条循环公式：

$$
\underbrace{\text{write test}}_{\text{红: 失败}} \xrightarrow{\text{implement}} \underbrace{\text{pass}}_{\text{绿: 通过}} \xrightarrow{\text{refactor}} \underbrace{\text{clean}}_{\text{重构: 保持绿}} \xrightarrow{\text{next}} \cdots
$$