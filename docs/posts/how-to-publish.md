---
pageClass: plain-doc
---

# 如何发布博文

发布一篇博文只需要三步：写 Markdown → 登记链接 → push。GitHub Actions 会自动构建并部署到 GitHub Pages。

## 1. 写 Markdown

在对应分类目录下新建 `.md` 文件。例如写一篇 LoRA 的文章：

```
docs/posts/advanced/llm-finetuning/lora.md
```

文件头部加上 frontmatter：

```markdown
---
title: LoRA 原理与实现
date: 2026-08-01
---

# LoRA 原理与实现

正文……支持数学公式：$W = W_0 + BA$

$$
\Delta W = BA, \quad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k}
$$
```

- 数学公式：`$...$` 行内、`$$...$$` 块级（MathJax 渲染，无需插件）
- 化学方程式：公式内使用 `\ce{}`，如 `$\ce{2H2 + O2 -> 2H2O}$`（mhchem 已内置）
- 多级标题：文章页的二级/三级/四级标题会自动按教材式编号（1 / 1.1 / 1.1.1）
- 代码块：``` 包裹并标注语言
- 图片：放到 `docs/public/images/`，引用 `/images/xxx.png`
- 全部排版效果见 [样式演示](/posts/style-demo)

## 2. 登记链接

打开该分类目录下的 `index.md`（如 `docs/posts/advanced/llm-finetuning/index.md`），
在「主题规划」中把对应主题从待写改为已发布链接：

```markdown
- [ ] LoRA 原理与实现
```

改为：

```markdown
- [x] [LoRA 原理与实现](./lora)
```

## 3. 推送

```bash
git add .
git commit -m "post: LoRA 原理与实现"
git push
```

push 到 `main` 分支后，GitHub Actions 会自动构建并发布，约 1-2 分钟后线上可见。

## 本地预览

```bash
npm install        # 首次
npm run docs:dev   # 启动本地预览，默认 http://localhost:5173
```
