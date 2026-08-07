// 自动发现 docs/posts/*/*/index.md，为每个专题生成专家小组定义文件（.claude/agents/<tier>-<topic>.md）。
// 层级标签见 TIER；专题名取 H1；对标教材取首段。运行：node scripts/gen-teams.mjs
import { writeFileSync, mkdirSync, readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'

const TIER = {
  foundations: '第一级 · 基础科学',
  intermediate: '第二级 · 进阶数理',
  cs: '第三级 · 计算机基础',
  advanced: '第四级 · 高阶专题',
  life: '第五级 · 生命与健康科学',
  engineering: '第六级 · 工程技术',
  humanities: '第七级 · 人文与艺术',
  social: '第八级 · 社会科学',
  frontier: '第九级 · 交叉与前沿',
}

function benchmarkOf(md) {
  const lines = md.split('\n')
  for (const ln of lines) {
    if (/^(#|---|##|###|pageClass)/.test(ln)) continue
    if (ln.trim().length > 8) return ln.trim().slice(0, 120)
  }
  return '该学科权威教材体系'
}

function agentFile(tier, key, name, benchmark) {
  const id = `${tier}-${key}`
  const tierLabel = TIER[tier]
  return `---
name: ${id}
description: 专题专家：负责「${name}」（${tierLabel}）分类全部博文的撰写。对标 ${benchmark}。写该专题博文时使用本专家。
tools: Bash, Read, Write, Edit, WebFetch, WebSearch, Glob, Grep
---

# ${name} 专家小组

你是「从极限到大模型」博客 ${tierLabel}《${name}》专题的资深专家写作者，负责把该专题对标教材的体系逐节写成高质量博文。

## 领域坐标
- 专题 key：${tier}/${key}
- 对标教材 / 体系：${benchmark}
- 写作约束：全部博文遵循 \`.claude/writing-charter.md\`（编辑章程），**写作前必须通读**

## 本组工作方法（每篇必走）
1. 读 \`.claude/writing-charter.md\`、本专题规划 \`docs/posts/${tier}/${key}/index.md\`、范本 \`docs/posts/foundations/math/set-concept.md\`
2. 基于对标教材的权威知识撰写（这些教材的经典内容是标准知识）；细节拿不准时用 ≤2 次全网搜索（OpenStax/arXiv/MIT OCW/官方文档）核对
3. 按章程产出 Markdown，写入 \`docs/posts/${tier}/${key}/<slug>.md\`
4. 需要时配 ≤1 张手写 SVG 图，存 \`docs/public/images/${key}/\`，文章以 \`/images/${key}/...\` 引用
5. 更新 \`docs/posts/${tier}/${key}/index.md\` 中对应条目为 \`- [x] [标题](./<slug>)\`
6. 向主控返回简短报告：标题、slug、参考来源、是否配图（**不要**改动 \`progress.json\`，主控统一重生成）
`
}

const root = 'docs/posts'
let count = 0
mkdirSync('.claude/agents', { recursive: true })
for (const tier of readdirSync(root, { withFileTypes: true })) {
  if (!tier.isDirectory()) continue
  const tierLabel = TIER[tier.name]
  if (!tierLabel) continue // 跳过非层级目录（如 index.md 所在目录无子目录，不进入）
  for (const cat of readdirSync(join(root, tier.name), { withFileTypes: true })) {
    if (!cat.isDirectory()) continue
    const file = join(root, tier.name, cat.name, 'index.md')
    let md
    try {
      md = readFileSync(file, 'utf8')
    } catch {
      continue
    }
    const name = (md.match(/^#\s+(.+)/m) || [])[1]?.trim() || cat.name
    const benchmark = benchmarkOf(md)
    writeFileSync(`.claude/agents/${tier.name}-${cat.name}.md`, agentFile(tier.name, cat.name, name, benchmark))
    count++
  }
}
console.log(`已生成 ${count} 个专家小组定义文件`)
