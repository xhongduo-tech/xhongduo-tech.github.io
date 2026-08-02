# 从极限到大模型

徐鸿铎的个人知识库，基于 VitePress，部署于 GitHub Pages。内容以博文为主轴，覆盖四级递进的体系：

1. **基础科学**：基础数学、基础物理、化学、生物
2. **进阶数理**：高等数学、概率论与数理统计、线性代数、高等物理
3. **计算机基础**：数据结构、计算机组成原理、操作系统、数据库
4. **高阶专题**：机器学习、深度学习、大模型原理、大模型部署、大模型微调、本体论

## 本地开发

```bash
npm install
npm run docs:dev      # 本地预览 http://localhost:5173
npm run docs:build    # 构建到 docs/.vitepress/dist
```

## 写博文

1. 在对应分类目录（如 `docs/posts/advanced/llm-finetuning/`）新建 `.md` 文件
2. 在该目录 `index.md` 的主题规划中把条目改成链接
3. `git push` —— GitHub Actions 自动构建并部署

详见站内页面「如何发布博文」。

## 部署

- 仓库：`xhongduo-tech/xhongduo-tech.github.io`
- 源码在 `source` 分支（默认分支，日常提交到这里），GitHub Actions 构建后把 `docs/.vitepress/dist` 强推到 `main` 分支
- `main` 分支是 GitHub Pages 的发布源（legacy 模式），站点地址：`https://xhongduo-tech.github.io/`
- 日常写作只需 `git push`（当前分支为 `source`），约 1-2 分钟后线上更新
