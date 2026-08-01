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

## 部署到 GitHub

1. 在 GitHub 新建仓库（如 `blog`），把本项目 push 到 `main` 分支
2. 仓库 Settings → Pages → Source 选择 **GitHub Actions**
3. 推送后自动部署，访问 `https://<用户名>.github.io/<仓库名>/`
4. 若仓库名为 `<用户名>.github.io`，则直接部署到根路径（配置已自动处理）

部署前记得修改 `docs/.vitepress/config.mts` 中的站点标题与 GitHub 链接。
