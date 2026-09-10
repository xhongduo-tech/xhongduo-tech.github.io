# 徐鸿铎

个人站点。VitePress + Tufte，部署于 GitHub Pages。

五栏：**大模型**、**量化**（金融微观结构与定价，不是模型权重量化）、**金融**（微观、宏观、货币银行与公司金融）、**光刻**（成像、DUV/EUV、计算光刻）、**计算机**（比特到系统栈）。按课程读：课程 → 单元 → 课序 → 课。加深课插在对应基础课之后。后课默认已经读完先修，只补上一课留下的缺口。论文与型号对照放在附录。

金融栏接到量化栏的限价簿为止，不重讲订单簿。大模型从深度学习基础起；量化从随机分析与财务制度起；计算机从程序设计起。模型压缩、GPTQ、KV 量化等属于大模型「压缩与数值」。

## 本地开发

```bash
npm install
npm run docs:dev      # http://localhost:5173
npm run docs:build    # 输出到 docs/.vitepress/dist
```

## 写文章

在 `docs/llm/`、`docs/quant/`、`docs/econ/`、`docs/litho/` 或 `docs/cs/` 新建 `.md`，frontmatter：

```yaml
---
title: 标题
date: 2026-09-03
section: llm   # 或 quant / econ / litho / cs
---
```

文件名用知识树叶子的 `slug`。不要把文章写进栏目的 `index.md`。后课不要把先修的公式再推导一遍；「问题」写上一课留下的缺口。

## 部署

- 源码在 `source` 分支；GitHub Actions 把 `docs/.vitepress/dist` 推到 `xhongduo-tech/xhongduo-tech.github.io` 的 `main`
- 站点：https://xhongduo-tech.github.io/
