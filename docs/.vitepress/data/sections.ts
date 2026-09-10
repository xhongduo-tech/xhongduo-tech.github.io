export const SECTION_IDS = ['llm', 'quant', 'econ', 'litho', 'cs'] as const

export type SectionId = (typeof SECTION_IDS)[number]

export const sectionMeta: Record<
  SectionId,
  { id: SectionId; name: string; path: string; blurb: string }
> = {
  llm: {
    id: 'llm',
    name: '大模型',
    path: '/llm/',
    blurb: '从深度学习基础到 Transformer、训练与推理。',
  },
  quant: {
    id: 'quant',
    name: '量化',
    path: '/quant/',
    blurb: '随机分析、财务与制度之后，才是限价簿与定价。',
  },
  econ: {
    id: 'econ',
    name: '金融',
    path: '/econ/',
    blurb: '先数学基础，再微观、计量、宏观，接到限价簿。',
  },
  litho: {
    id: 'litho',
    name: '光刻',
    path: '/litho/',
    blurb: '先电磁与器件动机，再成像、产线与计算光刻。',
  },
  cs: {
    id: 'cs',
    name: '计算机',
    path: '/cs/',
    blurb: '先程序与数值，再从比特到系统栈。',
  },
}

export function isSectionId(value: unknown): value is SectionId {
  return typeof value === 'string' && (SECTION_IDS as readonly string[]).includes(value)
}
