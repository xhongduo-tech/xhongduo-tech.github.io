# 从极限到大模型

徐鸿铎的个人知识库，基于 VitePress，部署于 GitHub Pages。内容分成两份：

**① 我的技术研究（博文）** —— 按四大技术领域组织：

1. **数理基础**：基础数学、基础物理、高等数学、数学分析、线性代数、概率统计，直至量子计算、场论弦论
2. **计算机科学**：数据结构、组成原理、操作系统、数据库、编译原理，及分布式、云原生、高性能计算、安全
3. **AI 与大模型**：机器学习、深度学习、大模型原理/部署/微调、CV、NLP、语音、多模态、智能体
4. **工程技术**：机械、电气、土木、化工、材料、能源、航空航天、环境等全部工科主干学科

**② 全人类知识树** —— 哲学、人文、社科、医学、农学等非技术领域作为知识树结构完整保留，见站内[全人类知识树](/knowledge-tree/)。

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
