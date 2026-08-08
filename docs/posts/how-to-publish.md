---
pageClass: plain-doc
---

# 如何发布博文

发布一篇博文只需要三步：写 Markdown → 登记链接 → push。GitHub Actions 会自动构建并部署到 GitHub Pages。

## 1. 写 Markdown

在对应分类目录下新建 `xxx.md` 文件。例如写一篇 LoRA 的文章：

```markdown
---
title: LoRA 原理与实现
date: 2026-08-07
---

# LoRA 原理与实现
```

写作要点：

- 数学公式：`$...$` 行内、`$$...$$` 块级（MathJax 渲染，无需插件）
- 化学方程式：公式内使用 `\ce{...}`，如 `\ce{C6H12O6 -> 2 C2H5OH + 2 CO2}`（mhchem 已内置）
- 边注：行内写 <span class="marginnote">…</span>，宽屏排入右侧页边
- 代码块：用三个反引号包裹

## 2. 登记链接

写完文章后，把对应分类的 `index.md` 里该条目改为已勾选链接：

```markdown
- [x] [LoRA 原理与实现](./lora)
```

## 3. 提交推送

```bash
git add .
git commit -m "post: LoRA 原理与实现"
git push
```

推送 `source` 分支后，GitHub Actions 自动构建部署。

## 本地预览

```bash
npm install        # 首次
npm run docs:dev   # 启动本地预览，默认 http://localhost:5173
```
