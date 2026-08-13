// 第二轮知识树全面性审查：实用/职业/生活技能 + 科目深度分层
// 针对 8 个类别并行审查，产出「应有但缺失」的专题清单（附权威参考：教材/手册/行业标准）
// 输入：args = [{ id, name, focus }]（或数组）

export const meta = {
  name: 'kt-audit2',
  description: '审查知识树缺失的实用/职业/生活技能科目与科目深度分层',
  phases: [{ title: '审查', detail: '8 类别并行，对照权威行业参考' }],
}

const categories = Array.isArray(args) ? args : args.categories || []

const GAP_SCHEMA = {
  type: 'object',
  required: ['category', 'gaps'],
  properties: {
    category: { type: 'string' },
    gaps: {
      type: 'array',
      items: {
        type: 'object',
        required: ['name', 'level', 'rationale', 'references', 'priority'],
        properties: {
          name: { type: 'string', description: '缺失专题中文名' },
          level: { type: 'string', description: '建议层级，如 基础/中级/高级/专业' },
          rationale: { type: 'string', description: '为什么是重要的人类知识（应有但缺失）' },
          references: { type: 'array', items: { type: 'string' }, description: '权威参考：权威教材/手册/行业标准/官方教材（2-3 个）' },
          priority: { type: 'string', enum: ['high', 'medium', 'low'] },
        },
      },
    },
  },
}

function promptFor(c) {
  return `你是「全人类知识全面性审查」专家，负责审查类别【${c.name}】（${c.focus}）。

背景：本站知识树已覆盖学术学科（哲学/数学/物理/化学/生命/地球空间/计算机AI/工程/医学/农学/社科/人文/交叉），但缺失**实用/职业/生活技能类**与**科目深度分层**。

任务：
1. 系统梳理【${c.name}】类别下，人类日常生活中真实存在、有系统知识体系、应有但当前缺失的专题。
2. 每个专题给出：名称、建议层级、为什么缺失（权威依据）、权威参考（权威教材/手册/行业标准/官方教材，2-3 个，如驾驶教材、维修手册、烹饪经典、职业技能标准、体育总局训练大纲等）。
3. 纪律：
   - 只报真实存在、有权威参考支撑的方向；不编造
   - 不重复知识树已覆盖的（如已有音乐/摄影/美术则乐器/摄影不重复，但可报其细分）
   - 每类别 10-30 个合理；宁缺毋滥
4. 可用 ≤5 次 WebSearch 核对权威参考（人民交通出版社驾考教材、国家职业技能标准、权威菜谱、行业手册等）。

输出 JSON：{ category: '${c.id}', gaps: [{ name, level, rationale, references: [...], priority: 'high|medium|low' }] }`
}

phase('审查')
const results = await parallel(
  categories.map((c) => () =>
    agent(promptFor(c), { label: `audit2:${c.id}`, phase: '审查', schema: GAP_SCHEMA, agentType: 'general-purpose' }),
  ),
)

const clean = results.filter(Boolean)
const byCat = clean.map((r) => ({ category: r.category, gapCount: (r.gaps || []).length }))
const allGaps = clean.flatMap((r) => (r.gaps || []).map((g) => ({ ...g, category: r.category })))
log(`审查完成：${byCat.length} 类别，共 ${allGaps.length} 个缺口`)
return { byCat, gaps: allGaps }
