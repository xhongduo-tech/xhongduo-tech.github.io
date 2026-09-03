# 从极限到大模型

徐鸿铎的个人站点。VitePress + Tufte，部署于 GitHub Pages。

写作只分两栏：**部署**（模型、服务器、推理服务）与 **量化**。

## 本地开发

```bash
npm install
npm run docs:dev      # http://localhost:5173
npm run docs:build    # 输出到 docs/.vitepress/dist
```

## 写文章

在 `docs/deploy/` 或 `docs/quant/` 新建 `.md`，frontmatter：

```yaml
---
title: 标题
date: 2026-09-03
section: deploy   # 或 quant
---
```

首页会按日期列出最近文章。不要把文章写进栏目的 `index.md`。

## 部署

- 源码在 `source` 分支；GitHub Actions 把 `docs/.vitepress/dist` 推到 `xhongduo-tech/xhongduo-tech.github.io` 的 `main`
- 站点：https://xhongduo-tech.github.io/
