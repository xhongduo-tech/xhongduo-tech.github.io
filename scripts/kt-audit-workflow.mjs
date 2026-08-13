// 全人类知识树全面性审查工作流
// 每棵树一个专家代理，对照权威学科分类（教育部学科门类/一级学科、国际学科分类、权威教材体系）核对覆盖，
// 产出「缺失学科/分支/专题」缺口清单（附权威依据与建议教材）。
// 输入：args.trees = [{ id, name }]；输出：每树缺口汇总。

export const meta = {
  name: 'kt-audit',
  description: '全面审查全人类知识树 13 棵树 vs 权威学科分类，产出缺口清单',
  phases: [
    { title: '审查', detail: '每树一个专家代理逐树核对' },
    { title: '汇总', detail: '汇总每树缺口，去重排序' },
  ],
}

const GAP_SCHEMA = {
  type: 'object',
  required: ['tree', 'gaps'],
  properties: {
    tree: { type: 'string', description: '树的 id，与输入一致' },
    gaps: {
      type: 'array',
      description: '该树缺失的学科/分支/重要专题清单',
      items: {
        type: 'object',
        required: ['name', 'level', 'rationale', 'textbooks', 'priority'],
        properties: {
          name: { type: 'string', description: '缺失专题的中文名' },
          level: { type: 'string', description: '建议放入的层级，如 基础/核心/进阶/专业/前沿' },
          rationale: { type: 'string', description: '为什么缺失（权威依据：学科门类/一级学科/教材体系）' },
          textbooks: { type: 'array', items: { type: 'string' }, description: '2-3 本权威教材（作者/书名/出版社版本）' },
          priority: { type: 'string', enum: ['high', 'medium', 'low'], description: '优先级：high=主干学科明显缺失；medium=重要分支缺失；low=补充性方向' },
        },
      },
    },
  },
}

function auditPrompt(t) {
  return `你是「全人类知识树全面性审查」专家，负责审查【${t.name}】（树 id：${t.id}）。

任务：
1. 读取 /Users/xuhongduo/Projects/blog/docs/.vitepress/data/knowledge-tree.mjs，定位 id: '${t.id}' 的那棵树，列出它当前已有的全部节点（专题名 + path + 层级）。
2. 以权威参照体系核对覆盖是否全面：
   - 教育部学科门类 / 一级学科 / 二级学科目录（中国）
   - 该领域国际权威学科分类（如 ACM Computing Classification System、IEEE、ACS、APA、IUPAC、学科代码等）
   - 该领域权威教材体系（Springer、Oxford、Cambridge、MIT Press、高等教育出版社等主干教材目录）
3. 找出【缺失】的学科/分支/重要专题：这棵树【应该有但当前没有覆盖】的。
   对每个缺口给出：专题中文名、建议层级、缺失依据（具体到学科门类/一级学科/哪本教材的哪部分）、建议对标教材（2-3 本权威教材）、优先级。
4. 纪律：
   - 只报真实、有权威依据的缺口；已有节点不重复报
   - 不报教材都很少的冷门/边缘方向
   - 每棵树 5-20 个缺口为合理范围；宁缺毋滥，不凑数
5. 可用 ≤5 次 WebSearch 核对学科分类与教材（OpenStax/教育部官网/出版社目录等）。

输出 JSON：{ tree: '${t.id}', gaps: [{ name, level, rationale, textbooks: [...], priority: 'high|medium|low' }] }`
}

// args 兼容两种传入：{ trees: [...] } 或直接是数组
const trees = Array.isArray(args) ? args : args.trees || []

phase('审查')
const results = await parallel(
  trees.map((t) => () =>
    agent(auditPrompt(t), { label: `audit:${t.id}`, phase: '审查', schema: GAP_SCHEMA, agentType: 'general-purpose' }),
  ),
)

phase('汇总')
const clean = results.filter(Boolean)
const byTree = clean.map((r) => ({ tree: r.tree, gapCount: (r.gaps || []).length }))
const allGaps = clean.flatMap((r) => (r.gaps || []).map((g) => ({ ...g, tree: r.tree })))
log(`审查完成：${byTree.length} 棵树，共 ${allGaps.length} 个缺口`)
return { byTree, gaps: allGaps }
