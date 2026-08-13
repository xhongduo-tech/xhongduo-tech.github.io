// 生成技术/非技术专题分类数据 → docs/.vitepress/data/tech-topics.json
//
// 「博文 = 我的技术研究（数理+CS+AI+工程）」的权威分类依据：
// 1. 知识树 4 棵技术树（math/physics/computer-ai/engineering）的全部节点 → 技术专题
// 2. 用户确认的 frontier「数据与计算/AI 相邻」9 个专题 → 额外划入技术
// 其余全部专题 → 非技术（知识树条目）。
//
// 用途：博文索引（posts/index.md）、进度渲染（ProgressOverview.vue）、
//       知识树视角切换（KnowledgeTree.vue）共用此分类。

import { writeFileSync } from 'node:fs'
import { trees } from '../docs/.vitepress/data/knowledge-tree.mjs'

// 4 棵技术树
const TECH_TREES = new Set(['math', 'physics', 'computer-ai', 'engineering'])

// 用户确认额外划入博文的 frontier 专题（数据与计算/AI 相邻）
const FRONTIER_TECH = new Set([
  'frontier/data-science',
  'frontier/explainable-ai',
  'frontier/intelligent-science',
  'frontier/cognitive-computing',
  'frontier/network-science',
  'frontier/complexity-science',
  'frontier/swarm-intelligence',
  'frontier/quantum-information',
  'frontier/brain-computer-interface',
])

// 节点路径 → 专题 key（advanced/computer-vision/resnet → advanced/computer-vision）
function topicOf(path) {
  const parts = path.split('/')
  return parts.slice(0, 2).join('/')
}

const tech = new Set()
for (const tree of trees) {
  if (!TECH_TREES.has(tree.id)) continue
  for (const branch of tree.branches)
    for (const node of branch.nodes) {
      if (node.tag === 'ref') continue // 跨树引用不改变专题主分类
      tech.add(topicOf(node.path))
    }
}
for (const t of FRONTIER_TECH) tech.add(t)

// 专题 → 所在树集合（用于领域归属）
const treeOfTopic = new Map()
for (const tree of trees) {
  for (const branch of tree.branches) {
    for (const node of branch.nodes) {
      const t = topicOf(node.path)
      if (!treeOfTopic.has(t)) treeOfTopic.set(t, new Set())
      treeOfTopic.get(t).add(tree.id)
    }
  }
}

// 4 大技术领域归属（用于博文索引与进度分组）
// 优先级：cs 目录 → 计算机；engineering 目录 → 工程；foundations/intermediate → 数理；
// advanced 按所在树区分（物理树 → 数理，其余 → AI）；frontier → AI；边界个案单独指定。
function domain(topic) {
  const tier = topic.split('/')[0]
  const trees = treeOfTopic.get(topic) || new Set()
  if (tier === 'cs') return 'cs'
  if (tier === 'engineering') return 'engineering'
  if (tier === 'foundations' || tier === 'intermediate') return 'math-physics'
  if (tier === 'advanced') return trees.has('physics') ? 'math-physics' : 'ai'
  if (tier === 'frontier') return 'ai'
  if (topic === 'humanities/history-of-mathematics' || topic === 'humanities/history-of-physics')
    return 'math-physics'
  if (topic === 'social/environmental-policy-governance') return 'engineering'
  return 'other'
}

const domains = {
  'math-physics': [],
  cs: [],
  ai: [],
  engineering: [],
  other: [],
}
for (const t of [...tech].sort()) domains[domain(t)].push(t)

// 全量专题 = 知识树全部节点去重（tier/key）
const all = new Set()
for (const tree of trees)
  for (const branch of tree.branches)
    for (const node of branch.nodes) all.add(topicOf(node.path))
const nontech = [...all].filter((t) => !tech.has(t)).sort()

const out = {
  generated: 'by scripts/gen-tech-topics.mjs',
  tech: [...tech].sort(),
  nontech,
  domains,
  stats: {
    tech: tech.size,
    nontech: nontech.length,
    total: all.size,
    byDomain: Object.fromEntries(Object.entries(domains).map(([k, v]) => [k, v.length])),
  },
}

writeFileSync(
  new URL('../docs/.vitepress/data/tech-topics.json', import.meta.url),
  JSON.stringify(out, null, 2) + '\n',
)
console.log(
  `tech-topics.json 生成完成：技术 ${out.stats.tech} · 非技术 ${out.stats.nontech} · 合计 ${out.stats.total}`,
)
console.log('分领域：', JSON.stringify(out.stats.byDomain))
