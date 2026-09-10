import { leafCount, type Node } from './trees/schema'
import { llmTree } from './trees/llm'
import { quantTree } from './trees/quant'
import { econTree } from './trees/econ'
import { lithoTree } from './trees/litho'
import { csTree } from './trees/cs'
import { type SectionId, sectionMeta } from './sections'

export type { Node } from './trees/schema'
export type { SectionId } from './sections'
export { leafCount, sectionMeta }

export const trees: Record<
  SectionId,
  { id: SectionId; name: string; desc: string; children: Node[] }
> = {
  llm: {
    id: 'llm',
    name: sectionMeta.llm.name,
    desc: '按课程读。先深度学习基础，再词表与残差块，然后架构、训练、采样、推理。加深课插在对应基础课之后。附录是论文、型号与世界模型对照。',
    children: llmTree,
  },
  quant: {
    id: 'quant',
    name: sectionMeta.quant.name,
    desc: '按课程读。先随机分析、财务与制度，再订单簿、计量、因子、套利、定价、执行与风险。微观与宏观理论在「金融」栏。',
    children: quantTree,
  },
  econ: {
    id: 'econ',
    name: sectionMeta.econ.name,
    desc: '按课程读。先数学与优化，再选择、厂商、均衡、博弈，然后计量、宏观、货币银行与公司金融，接到限价簿。附录是经典论文对照。',
    children: econTree,
  },
  litho: {
    id: 'litho',
    name: sectionMeta.litho.name,
    desc: '按课程读。先电磁与器件动机，再成像、胶、DUV/EUV 与计算光刻。加深课插在对应基础课之后；后课不重写瑞利公式。',
    children: lithoTree,
  },
  cs: {
    id: 'cs',
    name: sectionMeta.cs.name,
    desc: '按课程读。先程序设计与数值，再比特、组成、理论、数据、算法、编译、操作系统、网络、数据库与安全。加深课紧跟各基础课。本栏不重写 Transformer，也不进入限价簿。',
    children: csTree,
  },
}
