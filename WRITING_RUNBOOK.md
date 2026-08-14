# 写作工厂运行手册

目标：把全站规划博文全部写完（四级 60 专题 + 第五至第九级新增 56 专题，规划条目见 `docs/.vitepress/data/progress.json`）。
总纲与全人类知识分级方案见 `CURRICULUM.md`。本文档是任何会话（或任何助手）续跑写作流水线的操作手册。**每篇写完即落盘并勾选 index.md，进度天然持久化，跨会话无损续跑。**

## 一、核心设施（已就绪，勿重建）

| 设施 | 位置 | 作用 |
| --- | --- | --- |
| 编辑章程 | `.claude/writing-charter.md` | 文章结构/标注体系/公式解析/配图/SVG 规范/质量底线（最高约束） |
| 60 个专题团队定义 | `.claude/agents/<tier>-<topic>.md` | 每个专题的域简报（对标教材 + 工作方法） |
| 团队生成器 | `scripts/gen-teams.mjs` | 自动扫描全部 index.md 生成团队（新增专题后重跑即可） |
| 新专题脚手架 | `scripts/gen-new-curriculum.mjs` | 由内置表生成第五至九级新专题的 index.md（新增领域后重跑） |
| 进度数据 | `docs/.vitepress/data/progress.json` | 由 `scripts/gen-progress.mjs` 从 index.md 重生成 |
| 风格范本 | `docs/posts/foundations/math/set-concept.md` | 一篇完美博文的样子 |

## 二、批量写作方法（token 最优）

**核心思路**：不为每篇启动一个全新代理（那样每篇都要重读章程/简报/范本，40–80k token）。
改为**每组一个批处理代理，一次读入设置后连续写 4 篇**（实测约 15k token/篇，5 倍省）。

批处理代理的标准提示词模板（每个专题替换 `tier/key/专题名/层级/对标教材`）：

```
你是「从极限到大模型」博客 <专题名> 专题的资深专家写作者（<tier>-<key> 专家小组）。

## 一次性设置（勿重复读）
按序通读：
1. .claude/writing-charter.md（编辑章程，最高约束）
2. .claude/agents/<tier>-<key>.md（小组简报）
3. docs/posts/foundations/math/set-concept.md（风格范本，只读一次）
4. docs/posts/<tier>/<key>/index.md（选题规划，写作中会更新）

## 本轮任务：连续撰写本专题接下来 4 篇博文
方法：
1. 在 index.md 中按出现顺序找出前 4 个仍为 `- [ ]` 的条目。
2. 逐篇撰写并写盘，每篇完成后立刻写下一篇（保持上下文，勿重复读设置文件）：
   - 写入 docs/posts/<tier>/<key>/<slug>.md（slug 用英文 kebab-case）
   - 更新 index.md：该条目改为 `- [x] [<标题>](./<slug>)`
   - frontmatter date: 2026-08-07；byline「第X级 · <专题名> ｜ <教材> <章节> ｜ 2026-08-07」
3. 每篇严格遵循编辑章程：结构模板、公式解析（有公式的主题）、重点加粗、易错辨析、
   ≥2 条 marginnote、小结、结尾引子。
4. 配图吞吐优先：仅当示意图显著提升理解才配 SVG（≤1/篇，遵循章程第4节）。
5. 不改 progress.json、不 commit。每篇写完即落盘。
6. 返回简短报告：4 篇标题/slug/行数 + 剩余条目数。
```

## 三、会话纪律（一专题一 session，防上下文爆墙）

> **经验教训（2026-08-07）**：旧做法是「一个主控 session 跨多波滚动补位」，结果 session 上下文
> 累积到 ~937k token，加上 131k completion 预算即突破 1M 硬上限，触发
> `API Error 400 … maximum context length is 1048576`，整条流水线不可恢复地死掉。
> 根治办法 = **把 session 切小，按专题隔离**。

- **一专题一 session**：每个 Claude Code 会话只负责一个专题（如「基础物理」「化学」「天文学」）。
  该 session 内只用**该专题**的批处理代理（第二节模板），滚动写到该专题 `index.md` 没有 `- [ ]` 为止。
- **上下文预算**：模型 1M 上下文。**主动压缩红线：上下文用到 ~70–80% 就 `/compact`**，
  不要等撞 1M 硬墙——撞墙的 400 不可恢复，只能丢弃上下文重开。判断标准：
  `对话很长/已写多篇` 时先 `/compact` 再继续；或用 `/context` 看用量。
- **最大输出限制**：最大 384K 输出（写作单篇用不满，但 batch 代理一次连写 4 篇需要大预算）。
- **写完一个专题**：该专题无 `- [ ]` 后 → 执行第四节检查点（gen-progress + commit + push）→
  **关闭本 session**，开新 session 接下一个专题。
- **并行度**：批处理代理仍 ≤20 并发（`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`），
  但**只在本专题内部滚动补位**，不跨专题。这个 20 是 Claude Code 的子代理并发上限，
  与 DeepSeek API 的账号并发（数千级）无关——单 session 内 20 个子代理已是实际吞吐极限。
- **填满顺序**：按层级 第一级→第二级→第三级→第四级、组内按 index.md 顺序；推荐每个
  session 从「该层剩余条目最多」的专题开始。

## 四、检查点（防丢失）

写完一个专题（或每 ~100 篇）时，执行：
```bash
node scripts/gen-progress.mjs        # 重生成进度
node scripts/lint-html.mjs           # 静态红线扫描（四类炸构建问题）
npm run docs:build                   # 全量构建（含 dead-link），必须全绿
git add -A && git commit -m "..." && git push
```
推送 `source` 分支后 GitHub Actions 自动构建部署。**红线/构建不过不许提交**。

> **站点体积**：VitePress 会把路由哈希表（`__VP_HASH_MAP__`）内联到每页——10000+ 页时达数百 KB/页，全站 8GB+ 超 GitHub Pages 1GB 上限。
> `docs:build` 末尾的 `scripts/externalize-hashmap.mjs` 已把它外置为共享 `/hash-map.js`，全站降至 ~325MB。
>
> **MathJax 已改客户端渲染（2026-08-08 关键变更）**：10257 页若用服务端内联 SVG，构建需 ~64GB（CI 必挂）。
> 现改为 markdown-it-mathjax3 仅 tokenize、输出 `\(…\)`/`\[…\]`，由浏览器端 MathJax（head 里的 CDN 脚本）排版；
> `enhance.ts` 在路由切换后调 `MathJax.typesetPromise()`。数学内容中的 `< > &` 已转义。
> 本地构建 ~14 分钟（28.5k 页）；CI 默认 runner 已不够用（2026-08-14 曾 OOM 被取消），
> `deploy.yml` 已扩容 swap 至 10G、堆上限 12G、job 超时 60 分钟。

## 五、构建安全红线（代理易犯，主控必查）

**每次检查点前必跑**（本地构建已加大堆，跑起来约 2 分钟）：

```bash
node scripts/lint-html.mjs     # 静态红线扫描：四类会炸 Vue 编译的问题
npm run docs:build             # 全量构建（含 dead-link 检查）
```

`scripts/lint-html.mjs` 覆盖四类代理高频错误（历史翻车记录）：
- **A. 表格行内联代码里的裸 `|`**：GFM 把 `|` 当单元格分隔符，切断代码 span，
  残留 `<A>`/`<B>` 被 Vue 当未闭合标签 → `Element is missing end tag`。表格内要写 `\|`。
- **B. marginnote/sidenote span 内的 markdown `**`**：需强调时改用 `<strong>…</strong>`。
- **C. 代码围栏/数学/行内代码之外未闭合的合法 HTML 标签**（如 `<s>` 句子起始记号写成裸标签）。
- **D. 裸尖括号记号**：`<where>`/`<eos>`/`<pad>`/`<id>` 这类非 HTML 记号必须用反引号包裹 `` `<where>` ``，
  否则被 Vue 当作自定义元素 → 未闭合标签错误。

已修复的历史事故（教训模板）：`bnf-and-ebnf.md`（表格内 `| <B> |`）、`fasttext-subword-oov.md`（裸 `<where>`）、
`earth-science/*`（相对路径 `../../physics/` 应为 `../physics/`，VitePress 会报 dead link）。

## 六、质检要点（抽查）

- 每篇 ≥120 行、≥3 编号分节、≥2 条 marginnote、有公式的主题 ≥1 处公式解析、
  纯概念主题用核心对比表替代并标注。
- byline/frontmatter 日期一致、SVG XML 合法、文章引用的 `/images/...` 路径存在。
- 数值/公式与对标教材一致（拿不准的让代理联网核实）。

## 七、续跑清单（每 session 一次）

1. `node scripts/gen-progress.mjs` 看当前 done 数。
2. 选一个还有 `- [ ]` 条目的专题（优先「当前层级剩余条目最多」），**本 session 只做它**。
3. 按本手册第二节模板启动**该专题**的批处理代理，≤20 并发、专题内滚动补位。
4. 上下文到 ~70–80% 时 `/compact`；写完该专题 → 检查点提交 → **关闭本 session**。
5. 检查点必做：`node scripts/lint-html.mjs` + `npm run docs:build` 全绿后才 `git commit && git push`。
