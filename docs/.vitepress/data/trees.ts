import { explodeLeaves, leafCount, type Node } from './trees/schema'
import { llmTree } from './trees/llm'
import { quantTree } from './trees/quant'

export type { Node } from './trees/schema'
export { leafCount }

export const trees: Record<'llm' | 'quant', { id: 'llm' | 'quant'; name: string; desc: string; children: Node[] }> = {
  llm: {
    id: 'llm',
    name: '大模型',
    desc: '技术挂在主线上，主线挂在分支上。每个技术点四篇：动机、方法、实现、边界。',
    children: explodeLeaves(llmTree),
  },
  quant: {
    id: 'quant',
    name: '量化',
    desc: '金融量化：微观结构、因子、套利、定价、执行与风险。每个技术点拆成四篇博文。',
    children: explodeLeaves(quantTree),
  },
}
