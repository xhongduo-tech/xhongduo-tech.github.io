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

## 三、并发与节奏

- **并发上限 20**（`CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS`）。一次并行启动 ≤20 个批处理代理。
- **滚动补位**：每完成一个，立即为下一个仍有 `- [ ]` 条目的专题启动新批，保持 20 满载。
- **填满顺序**：按层级 第一级→第二级→第三级→第四级、组内按 index.md 顺序。
- **每波耗时**：20 组 × 4 篇 ≈ 80 篇/波，约 1 小时。
- **全程预估**：约 5540 篇剩余 ≈ 70 波 ≈ 2.5–3 天连续运行（跨多会话）。

## 四、检查点（防丢失）

每完成一批或每 ~100 篇，执行：
```bash
node scripts/gen-progress.mjs        # 重生成进度
git add -A && git commit -m "..." && git push
```
推送 `source` 分支后 GitHub Actions 自动构建部署。

## 五、构建安全红线（代理易犯，主控必查）

- **marginnote/sidenote span 内禁止 markdown `**`**（会导致 `<strong>` 未闭合、构建失败）。
  需强调时用 `<strong>…</strong>`。主控在每波后扫描：
  ```bash
  # 扫描 span 内 **
  node -e '... 检查 <span ...> 到 </span> 之间是否有 ** ...'
  ```
- 每波后 `npm run docs:build` 验证一次。

## 六、质检要点（抽查）

- 每篇 ≥120 行、≥3 编号分节、≥2 条 marginnote、有公式的主题 ≥1 处公式解析、
  纯概念主题用核心对比表替代并标注。
- byline/frontmatter 日期一致、SVG XML 合法、文章引用的 `/images/...` 路径存在。
- 数值/公式与对标教材一致（拿不准的让代理联网核实）。

## 七、续跑清单

1. `node scripts/gen-progress.mjs` 看当前 done 数。
2. 找还有 `- [ ]` 条目的专题（`grep -l -- '- \[ \]' docs/posts/*/*/index.md`）。
3. 按本手册第二节模板启动批处理代理，≤20 并发，滚动补位。
4. 每 ~100 篇做一次检查点提交。
