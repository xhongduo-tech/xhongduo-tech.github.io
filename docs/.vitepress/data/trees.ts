import { leafCount, type Node } from './trees/schema'
import { llmTree } from './trees/llm'
import { quantTree } from './trees/quant'
import { econTree } from './trees/econ'
import { lithoTree } from './trees/litho'
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
    desc: '按课程读。主干从词表与残差块起，再注意力、训练、采样、推理。后课只补上一课留下的缺口。附录是论文、型号与世界模型对照。光刻已独立成栏。',
    children: llmTree,
  },
  quant: {
    id: 'quant',
    name: sectionMeta.quant.name,
    desc: '金融量化按课程递进：簿与市场设计、价格形成、因子、套利、定价、执行与风险。微观与宏观理论在「金融」栏；本栏默认那些先修可以按需回看。',
    children: quantTree,
  },
  econ: {
    id: 'econ',
    name: sectionMeta.econ.name,
    desc: '金融理论按课程读：微观选择与均衡、博弈与信息、宏观核算与政策、货币银行与公司金融。最后一课接到量化栏的限价簿。附录是经典论文对照。',
    children: econTree,
  },
  litho: {
    id: 'litho',
    name: sectionMeta.litho.name,
    desc: '光刻按课程读：先波动光学与空中像，再产线判据、抗蚀剂、DUV、EUV、计算光刻与管制。后课不重写瑞利公式。附录是讲义与机台对照。',
    children: lithoTree,
  },
}
