export const sections = [
  {
    number: '01',
    title: '模型基础',
    english: 'Foundations',
    description: '从 tokenizer、张量与 Attention 开始，建立理解模型的共同语言。',
    link: '/foundations/',
  },
  {
    number: '02',
    title: '推理与系统',
    english: 'Inference & Systems',
    description: '沿着一次请求，理解生成、缓存、调度与模型服务。',
    link: '/inference/',
  },
  {
    number: '03',
    title: '训练与后训练',
    english: 'Training & Post-training',
    description: '连接数据、损失与参数更新，逐步走进监督微调和偏好优化。',
    link: '/training/',
  },
  {
    number: '04',
    title: '评测与实验',
    english: 'Evaluation',
    description: '明确比较对象、保留集与边界，让模型判断有据可依。',
    link: '/evaluation/',
  },
]

export const focusTopics = [
  { title: '投机采样', english: 'Speculative Decoding', detail: '草稿、验证与生成效率', link: '/inference/speculative-decoding' },
  { title: '多 Token 预测', english: 'Multi-Token Prediction', detail: '预测目标、推理路径与适用边界', link: '/inference/mtp' },
  { title: 'DSpark', english: 'DSpark', detail: '来源、机制与工程链路', link: '/inference/dspark' },
  { title: 'DeepInfer', english: 'DeepInfer', detail: '推理框架的组织与执行', link: '/inference/deepinfer' },
]

export const projects = [
  { number: 'P1', title: '推理算法与服务系统', english: 'Inference & Serving Lab', detail: '从解码、缓存与调度机制出发，用对照实验解释系统取舍。', link: '/projects/serving-lab' },
  { number: 'P2', title: '数据与监督微调', english: 'Finetuning Lab', detail: '从数据处理到小规模微调，保留可比较的模型结果。', link: '/projects/finetuning-lab' },
  { number: 'P3', title: '偏好优化与后训练', english: 'Post-training Lab', detail: '在前置能力具备后，探索偏好数据与模型行为的关系。', link: '/projects/post-training-lab' },
]
