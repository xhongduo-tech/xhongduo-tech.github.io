export const SECTION_IDS = ['llm', 'quant', 'econ', 'litho'] as const

export type SectionId = (typeof SECTION_IDS)[number]

export const sectionMeta: Record<
  SectionId,
  { id: SectionId; name: string; path: string }
> = {
  llm: { id: 'llm', name: '大模型', path: '/llm/' },
  quant: { id: 'quant', name: '量化', path: '/quant/' },
  econ: { id: 'econ', name: '金融', path: '/econ/' },
  litho: { id: 'litho', name: '光刻', path: '/litho/' },
}

export function isSectionId(value: unknown): value is SectionId {
  return typeof value === 'string' && (SECTION_IDS as readonly string[]).includes(value)
}
