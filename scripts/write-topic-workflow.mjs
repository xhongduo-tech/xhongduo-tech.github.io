// 写作工作流：每个专题一个专家代理，写满该专题全部待写博文。
// 用法：node 侧把 topics 数组（每项 {key, name, agentType, pending}）通过 args 传入。
// 通过 workflow() 工具调用，args.topics = 专题列表。

export const meta = {
  name: 'write-topics',
  description: '批量撰写待建专题博文（每专题一个专家代理）',
  phases: [{ title: '撰写专题博文', detail: '每个专题由对应专家代理写满全部条目' }],
}

// 每个专题的撰写提示词模板
function promptFor(t) {
  const [tier, ...rest] = t.key.split('/')
  const tierLabel = {
    foundations: '第一级 · 基础科学',
    intermediate: '第二级 · 进阶数理',
    cs: '第三级 · 计算机基础',
    advanced: '第四级 · 高阶专题',
    life: '第五级 · 生命与健康科学',
    engineering: '第六级 · 工程技术',
    humanities: '第七级 · 人文与艺术',
    social: '第八级 · 社会科学',
    frontier: '第九级 · 交叉与前沿',
  }[tier]
  return `你是「从极限到大模型」博客《${t.name}》专题的资深专家写作者（${t.agentType}）。

## 一次性设置（读一次，勿重复读）
按序通读：
1. .claude/writing-charter.md（编辑章程，最高约束）
2. .claude/agents/${t.agentType}.md（小组简报）
3. docs/posts/foundations/math/set-concept.md（风格范本，只读一次）
4. docs/posts/${t.key}/index.md（选题规划，写作中会更新）

## 本轮任务：连续撰写本专题全部 ${t.pending} 篇博文（务必全部完成，不要只写几篇）
方法：
1. 在 index.md 中按出现顺序找出全部仍为 \`- [ ]\` 的条目（共 ${t.pending} 条）。
2. 逐篇撰写并写盘，每篇完成后立刻写下一篇（保持上下文，勿重复读设置文件）：
   - 写入 docs/posts/${t.key}/<slug>.md（slug 用英文 kebab-case，语义化）
   - 更新 index.md：该条目改为 \`- [x] [<标题>](./<slug>)\`（保留标题原文，去掉括号书籍标注）
   - frontmatter date: 2026-08-11；byline「${tierLabel} · ${t.name} ｜ 对标教材 ｜ 2026-08-11」
3. 每篇严格遵循编辑章程：结构模板、公式解析（有公式的主题）、重点加粗、易错辨析、≥2 条 marginnote、小结、结尾引子。
4. 配图吞吐优先：仅当示意图显著提升理解才配 SVG（≤1/篇，遵循章程第4节）。
5. 不改 progress.json、不 commit。每篇写完即落盘。
6. 返回简短报告：实际完成的篇数、每个 slug、剩余未勾选条目数（应尽量为 0）。
7. 若因上下文过长实在写不完，优先保证已写部分落盘勾选，并在报告里如实说明剩余条目数。

警告：frontmatter 必须严格两行 title/date，中间不得混入多余文字（历史事故：矩阵论曾把「定理」混进 frontmatter 导致构建失败）。写完每篇后自查 frontmatter 是否合法。`
}

// 主逻辑：对每个专题启动一个 agent
const results = await pipeline(
  args.topics,
  (t) => agent(promptFor(t), {
    label: `write:${t.key}`,
    phase: '撰写专题博文',
    agentType: t.agentType,
  }),
)

// 汇总
const done = results.filter(Boolean).length
log(`完成 ${done}/${args.topics.length} 个专题`)
return { done, total: args.topics.length }
