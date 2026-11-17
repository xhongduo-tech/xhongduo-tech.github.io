# xuhongduo's blog

记录模型、训练与推理系统的学习和实践。

VitePress技术博客，围绕大模型原理、推理算法与服务系统、训练后训练、评测复现与实践项目组织。源码在source分支，GitHub Actions构建后通过官方Pages artifact流程部署，同时替换main分支的生成文件镜像。[公开站点](https://xhongduo-tech.github.io/)的站名为`xuhongduo's blog`；站名拼写与GitHub账户`xhongduo-tech`不同。

首页主句为`Token · Attention · Inference`，下方说明为“记录模型、训练与推理系统的学习和实践。”。保留当前单栏文章与研究方向布局，以简洁英文标注和线性icon补充层次；导航保持文章、研究、项目和关于我四组。

## 目录

- docs/foundations：模型、概率、张量与训练基础。
- docs/inference：解码、MTP、DSpark、DeepInfer及推理系统地图。
- docs/training：数据、SFT、LoRA、偏好优化与RL路线。
- docs/evaluation：质量、系统指标和复现规范。
- docs/projects：部署、微调、后训练项目任务与后续报告。
- docs/articles：正式博文，只有article且published进入文章索引。
- docs/writing：写作方法与文章模板。
- drafts：未公开草稿。
- docs/.vitepress：配置、主题与内容索引。

## 本地运行

```sh
npm ci
npm run docs:dev
npm run docs:check
npm run docs:build
```

构建输出在docs/.vitepress/dist。数学公式在构建时生成，Mermaid在包含图表的页面按需加载，搜索采用本地索引；不依赖站外搜索账号。

`npm run docs:preview`用于检查构建产物；重新构建后应重启预览进程，避免旧资源索引与新产物不一致。日常修改使用`docs:dev`。

## 新建博文

```sh
npm run note:new -- speculative-correctness --area inference --title "投机解码的正确性"
```

草稿正文默认被Git忽略，也不进入站点。补齐内容、资料与证据，经人工核对后移到docs/articles，设置status published，再运行检查与构建。不要强制提交含私有信息的草稿：本仓库本身是公开的。正式文章需title、description、kind、status、area、evidence与date；evidence可为mechanism、source、simulation或hardware。源码文章固定commit，硬件实测另记录model、hardware和results。

## 本轮重构

历史Markdown页面与旧导航已移除。原始源码commit为2cfa220011800ed65c9e465d8473df4f4849e01b，Git历史保留；工作目录之外另保存了完整源码快照。当前四个推理专题均为研究提纲，三个项目均待实践，没有用提纲充作正式博文。

后续内容应来自真实学习、源码追踪和实验积累，避免重新生成大量缺乏验证的词条。

## GitHub发布配置

Pages的发布方式设为GitHub Actions。source触发测试、内容检查和构建，上传Pages artifact，再由独立deploy job部署。github-pages环境需要允许source分支部署。main仅保存本轮生成文件镜像，用于同步清除旧HTML，不承担二次构建触发职责。

界面与排版遵循[设计规范](./DESIGN.md)：单栏首页、适合长文的阅读宽度，以及Source Serif 4、Inter和JetBrains Mono的字体分工。英文和icon用于辅助识别，不增加重复模块。字体在本站自托管，来源与许可证见[字体说明](./docs/public/fonts/README.md)。

界面使用CSS实现轻量交互，支持减少动态效果；三款变量字体使用拉丁字符子集，保留现有字型与中文系统字体回退。维护时同时检查首页资源、图表首次加载、深浅模式与窄屏排版。
