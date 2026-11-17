# 徐鸿铎的大模型研究笔记

VitePress技术博客，围绕大模型原理、推理算法与服务系统、训练后训练、评测复现与实践项目组织。源码在source分支，GitHub Actions构建后发布main分支，站点为 https://xhongduo-tech.github.io/ 。

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

构建输出在docs/.vitepress/dist。数学与Mermaid使用本地构建支持，搜索采用本地索引；不依赖站外搜索账号。

`npm run docs:preview`用于检查构建产物；重新构建后应重启预览进程，避免旧资源索引与新产物不一致。日常修改使用`docs:dev`。

## 新建博文

```sh
npm run note:new -- speculative-correctness --area inference --title "投机解码的正确性"
```

草稿正文默认被Git忽略，也不进入站点。补齐内容、资料与证据，经人工核对后移到docs/articles，设置status published，再运行检查与构建。不要强制提交含私有信息的草稿：本仓库本身是公开的。正式文章需title、description、kind、status、area、evidence与date；evidence可为mechanism、source、simulation或hardware。源码文章固定commit，硬件实测另记录model、hardware和results。

## 本轮重构

历史Markdown页面与旧导航已移除。原始源码commit为2cfa220011800ed65c9e465d8473df4f4849e01b，Git历史保留；工作目录之外另保存了完整源码快照。当前四个推理专题均为研究提纲，三个项目均待实践，没有用提纲充作正式博文。

后续内容应来自真实学习、源码追踪和实验积累，避免重新生成大量缺乏验证的词条。
