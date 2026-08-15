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

> **站点体积与构建速度**：VitePress 会把路由哈希表（`__VP_HASH_MAP__`）内联到每页——数万页时每页 ~2MB，
> 构建体量随页数平方膨胀（2.8 万页约 60GB 页面输出，CI 上近 60 分钟、且超 runner 内存被取消）。
> 已在 `config.mts` 开启 `metaChunk: true`：构建期直接把元数据外置为共享 `assets/chunks/metadata.*.js`，
> 本地全量构建 819s → 229s。`scripts/externalize-hashmap.mjs` 保留为兼容兜底（无内联时静默退出）。
>
> **MathJax 已改客户端渲染（2026-08-08 关键变更）**：10257 页若用服务端内联 SVG，构建需 ~64GB（CI 必挂）。
> 现改为 markdown-it-mathjax3 仅 tokenize、输出 `\(…\)`/`\[…\]`，由浏览器端 MathJax（head 里的 CDN 脚本）排版；
> `enhance.ts` 在路由切换后调 `MathJax.typesetPromise()`。数学内容中的 `< > &` 已转义。
> 本地全量构建 ~4 分钟（28.5k 页，metaChunk 外置后）；CI 上预计 ~15–30 分钟。
> `deploy.yml` 保留 10G swap 与 12G 堆上限作为安全垫，job 超时 150 分钟（2026-08-14 曾因渲染超时被取消）。

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

## 八、2026-08-15 批量写博中断恢复现场

> 2026-08-15 08:39 提交 `0d4017425`（增补 233 专题）后启动的大批量写博任务，在上下文压缩时撞 1M 墙中断（错误 `API Error 400 ... maximum context length is 1048576`），**3303 个文件从未提交**。已由新会话恢复：

**已完成（按顺序）**
1. `b23114cda` — checkpoint：3356 文件（3229 md + 126 svg + posts.json）全部入仓，工作现场不再丢失。
2. `f5e6e7297` — 完成 2 个已达标专题：algebraic-k-theory、group-theory-in-physics（仅勾选登记）。
3. `095aa10cc` — 完成 5 专题 56 篇扩写：approximation-theory / uncertainty-quantification / special-functions / geometric-measure-theory / difference-equations。
4. `f57aa0ec4` — 清零全部 29 个 lint 红线问题（多数为 lint 对跨行数学/`$$` 内联代码的误报，语义等价规避；少数真实缺陷：未闭合 `**`、缺 `$`、表格裸 `|`）。
5. `28a17faf1` — 完成 6 专题 68 篇扩写：computational-mechanics / phase-transitions-critical-phenomena / molecular-evolution-phylogenetics / modeling-and-simulation / real-time-systems / nonstandard-analysis。

**经验：许多 100–119 行「草稿」其实是截断稿**（末行断在句中、缺小结 bullet 或收尾句），扩写时要先补完截断处再增实质内容。

**复用设施**
- 勾选脚本：`/tmp/tick_index.py <专题目录> [--apply]` —— 按 frontmatter title ↔ index 条目匹配，仅勾选文件存在且 ≥120 行的条目。标题不一致时先对齐文件 title/H1 到 index 权威标题再勾。
- 扩写 agent 模板：通用 agent，必读 `.claude/writing-charter.md` + 范本 `set-concept.md`，仅扩写指定 <120 行文件，保持 date(2026-08-07)/byline/语气，禁改 index.md，返回「文件名→新行数」。
- 批次闭环：行数复核 → `tick_index.py --apply` → `node scripts/lint-html.mjs`（须全绿）→ `node scripts/gen-progress.mjs` → `git add -A && git commit`。

**2026-08-15 大批量（最大并发 Workflow）已完成**
7. `c6d4c93f5` — Workflow Run1：84 个仅扩写专题 / 986 文件全部 ≥120 行并勾选。
8. `09bce85a4` — Workflow Run2：147 个含空壳专题 / 948 空壳从零撰写 + 短稿扩写，2738 文件全达标、0 空壳；修复 24 处 lint。
9. 已 push 至 `09bce85a4`。累计 **231 专题 / 2738 文件全部达标并勾选**。

**复用设施（Workflow 版，已验证）**
- 编排脚本 `/tmp/write-topics-workflow.mjs`（扩写）与 `/tmp/write-new-posts-workflow.mjs`（新写空壳，含 skills→第十级 byline）。args=专题路径数组，agent 自行发现文件（规避 slug 失配）。
- 勾选：先 `/tmp/tick_index.py --apply`（精确），再 `/tmp/tick_fuzzy.py --apply`（归一化匹配括号/斜杠/分隔变体），最后手工勾剩余含教材注记条目（含章节注记的 index 条目直接 `- [x] [原标题](./slug)`，不改文件 title）。
- 批次闭环：`node scripts/lint-html.mjs`（须全绿）→ `node scripts/gen-progress.mjs` → `git add -A && git commit`。Workflow 子代理会建大量噪声任务条目，批次后需清理。
- ⚠️ 经验：Run2 新写稿引入了 ~24 处 marginnote `**`（应为 `<strong>`）、游离 `</p>`、`<br/>` 等 lint 问题，需修复；另一批 Workflow 完成后必跑 lint。

**⚠️ 重大发现：286 专题 / 4341 空文件（index 已勾选但内容缺失）**
- 这些专题来自 8-14 skills/知识点脚手架批次：index.md 的 `- [x]` 已勾选，但对应文件是 0 字节空壳（如 family-education-parenting 30 篇全空）。
- 因为 index 无未勾项，不在「未勾条目」统计内，早期计算漏掉。**「剩余全新博文」的真实规模 = 231 专题 + 286 专题 / 4341 空文件。**
- Run 3（`wznqnwxlm`）已启动处理这 286 专题 / 4341 空文件，后台最大并发运行中。

**当前状态（Run 3 完成后）**
- Run 3 完成后：验证行数 → 修复 lint → gen-progress → commit → push。index 已是 `- [x]`，无需勾选（但 gen-progress 的 done 数会因内容补全而更真实）。
- 下一个 session 起点：等 Run 3 通知或直接重跑 `find docs/posts -name '*.md' ! -name 'index.md' -size 0c` 看剩余空壳，按本手册纪律继续。
