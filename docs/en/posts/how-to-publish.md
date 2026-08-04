---
pageClass: plain-doc
---

# How to Publish a Post

Publishing a blog post takes only three steps: write the Markdown → register the link → push. GitHub Actions will automatically build and deploy it to GitHub Pages.

## 1. Write the Markdown

Create a new `.md` file in the directory for the corresponding category. For example, for a post about LoRA:

```
docs/posts/advanced/llm-finetuning/lora.md
```

Add the frontmatter at the top of the file:

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

- Math: `$...$` inline, `$$...$$` block-level (rendered by MathJax, no plugin needed)
- Chemical equations: use `\ce{}` inside math, e.g. `$\ce{2H2 + O2 -> 2H2O}$` (mhchem is built in)
- Margin notes: write `<span class="marginnote">content</span>` inline; on wide screens they flow into the right margin
- Code blocks: wrap in ``` and label the language (automatic line numbers and copy button)
- Images: put them in `docs/public/images/`, reference as `/images/xxx.png`
- See the [style demo](/en/posts/style-demo) for all typography effects

## 2. Register the Link

Open the `index.md` in that category's directory (e.g. `docs/posts/advanced/llm-finetuning/index.md`),
and in the "Topic Planning" section change the topic from to-be-written to a published link:

```markdown
- [ ] LoRA 原理与实现
```

Change it to:

```markdown
- [x] [LoRA 原理与实现](./lora)
```

## 3. Push

```bash
git add .
git commit -m "post: LoRA 原理与实现"
git push
```

After pushing to the `main` branch, GitHub Actions will build and publish automatically; the post is live online after about 1-2 minutes.

## Local Preview

```bash
npm install        # first time
npm run docs:dev   # start local preview, default http://localhost:5173
```
