# xuhongduo's blog

记录大模型原理、训练、推理与评测的学习过程，用源码和实验验证理解。

VitePress 技术博客。源码在 `source` 分支，GitHub Actions 构建后通过官方 Pages artifact 流程部署，同时替换 `main` 分支的生成文件镜像。[公开站点](https://xhongduo-tech.github.io/)的站名为 `xuhongduo's blog`；站名拼写与 GitHub 账户 `xhongduo-tech` 不同。

## 页面结构

- **首页**：主标题“大模型研习录”、带线性 icon 的主题 tag，以及“页面索引在左、年度 Git 活动点阵在右”的主体。无长描述、Git 最近提交列表或重复阅读入口。
- **文章**：右侧提供“具体博文 / 主题拓扑”切换和今日 Todo，主内容显示对应视图；窄屏将切换放在主内容顶部、Todo 放在下方。
- **项目**：只收录具有真实实现、可复现步骤和实验记录的项目报告；没有成果时显示空状态。
- **关于我**：独立作者页。

顶部导航为文章、项目、关于我。其右侧依次是界面语言下拉、搜索 icon 和明暗切换。下拉提供智能、中文、English，支持方向键与 Esc；搜索保留原快捷键和弹窗。首页左侧索引是三个公开页面的短导航，不恢复旧专题侧栏。

## 公开与私有内容

公开 `docs` 保留上述页面、正式文章和真实项目报告。研究提纲、详细学习地图、项目任务和写作笔记位于 Git 忽略的 `drafts`，不作为公开成果。删除旧公开页面不重写 `source` 历史。

文章可以声明 `prerequisites`，检查器会拒绝缺失前置、自依赖、重复依赖和循环。文章页的**主题拓扑**是公共概念依赖图，不是文章实例 DAG 或每日课程表；`prerequisites` 仍只用于真实文章之间的依赖检查。

## 主题拓扑与领域点亮

`docs/.vitepress/data/topic-topology.mjs` 定义 63 个双语公共知识主题和 91 条前置依赖，按六个领域分组。画布支持拖拽平移、节点拖动、滚轮与按钮缩放、全图和复位；选中主题可查看直接前置与真实关联文章。文章通过 `topicId` 关联主题，没有公开文章的节点只显示“待发布”。这些节点不代表学习进度或掌握状态，也不包含私有年度计划。

`docs/.vitepress/data/domain-progress.mjs` 单独定义六个领域的验收状态：模型基础、训练与后训练、推理系统、评测与复现、应用与多模态、项目实践。

只有完成该领域的人工验收后，才将对应状态改为：

```js
foundations: { completed: true, evidence: '/articles/a-real-acceptance-record' }
```

`evidence` 必须指向已发布文章或公开项目。内容检查会拒绝没有证据或证据未发布的点亮状态。发布一篇文章不自动等于掌握整个领域。

## 中英文与智能模式

中文页面位于根路径，英文界面页面位于 `docs/en`。页面通过 frontmatter 声明对应关系：

```yaml
contentLang: zh
translations:
  en: /en/articles/
```

- **中文 / EN**：保存界面语言偏好并进入对应页面。
- **智能**：根据当前页面 `contentLang` 跟随内容语言，不自动翻译正文。
- 没有对应译文的文章不提供虚假的语言跳转。

新增公开页面时同步提供翻译路径；翻译目标必须真实存在。英文任务数据若尚未提供，Todo 正文仍保留原始计划语言，界面标签使用英文。

## 今日 Todo

本地完整计划在 `drafts/year-plan.json`，部署时通过 `STUDY_PLAN_JSON` Secret 提供。数据加载器只向页面返回当天任务，默认时区为 `Asia/Shanghai`。完成状态只保存在当前浏览器 `localStorage`，不是全局进度或跨设备同步。

工作流在 UTC `16:15` 每日构建，约为上海时间次日 `00:15`。未配置 Secret 时构建成功，但不显示 Todo。

## 本地运行与检查

```sh
npm ci
npm run docs:dev
npm test
npm run docs:check
npm run docs:build
```

`npm test` 覆盖草稿创建、文章依赖、时区与任务选择、领域图结构及公开项目元数据。`npm run docs:check` 核对公开元数据、内部链接、翻译页面、项目结果与领域验收证据。自动检查不能证明技术推理正确或本人掌握。

`npm run docs:preview` 用于检查构建产物；重新构建后应重启预览。构建输出在 `docs/.vitepress/dist`。

## 新建文章与项目

```sh
npm run note:new -- speculative-correctness --area inference --title "投机解码的正确性"
```

草稿默认被 Git 忽略。文章经人工核对后移入 `docs/articles` 并设 `status: published`。公开项目需有 `date` 和真实 `results`，不能用任务书代替成果。不要强制提交含私有信息的草稿：本仓库是公开的。

界面遵循[设计规范](./DESIGN.md)。字体来源与许可证见[字体说明](./docs/public/fonts/README.md)。维护时检查中英文、桌面与移动布局、键盘操作、搜索、Todo、拓扑、动画降级和构建产物隐私；性能结论须来自同条件测量。
