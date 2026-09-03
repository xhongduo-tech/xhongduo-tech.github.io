# LLM & Quant

徐鸿铎的个人站点。VitePress + Tufte，部署于 GitHub Pages。

两栏：**大模型** 与 **量化**（金融量化，不是模型权重量化）。栏目页是该领域的知识树：分支 → 主线 → 技术 → 叶子；每片叶子对应一篇博文。模型压缩、GPTQ、KV 量化等属于大模型分支「压缩与数值」。

## 本地开发

```bash
npm install
npm run docs:dev      # http://localhost:5173
npm run docs:build    # 输出到 docs/.vitepress/dist
```

## 写文章

在 `docs/llm/` 或 `docs/quant/` 新建 `.md`，frontmatter：

```yaml
---
title: 标题
date: 2026-09-03
section: llm   # 或 quant
---
```

文件名用知识树叶子的 `slug`。不要把文章写进栏目的 `index.md`。

## 部署

- 源码在 `source` 分支；GitHub Actions 把 `docs/.vitepress/dist` 推到 `xhongduo-tech/xhongduo-tech.github.io` 的 `main`
- 站点：https://xhongduo-tech.github.io/
