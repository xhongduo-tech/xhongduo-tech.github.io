export const domains = [
  { id: 'foundations', title: '模型基础', titleEn: 'Model fundamentals', x: 24, y: 116, url: '/articles/#按时间阅读', urlEn: '/en/articles/#reading-by-date' },
  { id: 'training', title: '训练与后训练', titleEn: 'Training & alignment', x: 286, y: 40, url: '/articles/#按时间阅读', urlEn: '/en/articles/#reading-by-date' },
  { id: 'inference', title: '推理系统', titleEn: 'Inference systems', x: 286, y: 196, url: '/articles/#按时间阅读', urlEn: '/en/articles/#reading-by-date' },
  { id: 'evaluation', title: '评测与复现', titleEn: 'Evaluation', x: 548, y: 116, url: '/articles/#按时间阅读', urlEn: '/en/articles/#reading-by-date' },
  { id: 'applications', title: '应用与多模态', titleEn: 'RAG & multimodal', x: 810, y: 40, url: '/articles/#按时间阅读', urlEn: '/en/articles/#reading-by-date' },
  { id: 'projects', title: '项目实践', titleEn: 'Project practice', x: 810, y: 196, url: '/projects/', urlEn: '/en/projects/' },
]

export const domainEdges = [
  ['foundations', 'training'],
  ['foundations', 'inference'],
  ['training', 'evaluation'],
  ['inference', 'evaluation'],
  ['evaluation', 'applications'],
  ['evaluation', 'projects'],
]

// A domain is lit only after manual acceptance and a public evidence link.
// Publishing one article does not automatically complete an entire domain.
export const domainProgress = {
  foundations: { completed: false, evidence: '' },
  training: { completed: false, evidence: '' },
  inference: { completed: false, evidence: '' },
  evaluation: { completed: false, evidence: '' },
  applications: { completed: false, evidence: '' },
  projects: { completed: false, evidence: '' },
}
