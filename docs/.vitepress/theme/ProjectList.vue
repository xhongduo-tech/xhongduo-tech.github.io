<script setup>
const projects = [
  {
    name: 'BRDC.ai 大模型开放平台',
    period: '2026',
    role: '独立开发',
    desc: '全栈开发的大模型开放平台：14+ 模型在线管理，兼容 OpenAI / Anthropic 接口，基于 GPU KV Cache 实时感知的 API 智能路由，动态优先级调度，Token 梯度路由（8K–128K），自动巡检与运营分析。',
    tags: ['全栈', 'OpenAI 兼容', '智能路由', '动态调度'],
  },
  {
    name: '异构算力集群模型部署与推理优化',
    period: '2026',
    role: '负责人',
    desc: '在完全内网环境统筹 NVIDIA A100 / V100 与华为昇腾 910B3 混合集群，部署 Qwen、Gemma、DeepSeek、GLM 等系列模型。落地 PD 分离、MTP、MPS、TEI、AWQ / GGUF / w8a8 量化，自研梯度算力部署方案支撑近 50 个重点场景。',
    tags: ['vLLM', 'llama.cpp', 'PD 分离', '量化推理', '昇腾'],
  },
  {
    name: '中心大模型 API 统一接入平台',
    period: '2024',
    role: '牵头',
    desc: '对标 DeepSeek 官网的初版大模型 API 统一接入平台：密钥管理、接口文档、调用日志与用量统计。',
    tags: ['API 网关', '密钥管理', '用量统计'],
  },
  {
    name: '内网 PyPI 私有包管理平台',
    period: '2024',
    role: '搭建',
    desc: '内网环境私有 Python 包管理平台，解决离线环境下依赖隔离与版本管理问题。',
    tags: ['PyPI', '内网', '依赖管理'],
  },
  {
    name: 'BRDM 需求项智能检查系统',
    period: '2023',
    role: '牵头',
    desc: '基于 Qwen2-72B 与提示词工程的需求项名称与描述自动检查系统，大模型在银行业务的首次落地。',
    tags: ['Qwen', '提示词工程', '业务落地'],
  },
  {
    name: '员工履历检查系统',
    period: '2023',
    role: '牵头',
    desc: '提示词 + RAG 方案实现非结构化履历信息的精准抽取与合规校验，用于年度轮训检查，测试后移交开发中心。',
    tags: ['RAG', '信息抽取', '合规校验'],
  },
  {
    name: '手部穴位检测（改进 HRNet）',
    period: '2022',
    role: '第二作者 · IJCNN',
    desc: '中科院计算所实习成果：提出改进的高分辨率网络（HRNet）用于手部穴位自动检测，发表于 IJCNN 2022。',
    tags: ['HRNet', '关键点检测', '论文'],
  },
  {
    name: '卫星火情监测与风场可视化',
    period: '2021–2022',
    role: '全栈工程师',
    desc: '中国气象局华云集团：基于风速风向数据的风羽图二三维实时绘制，参与卫星火情监测预测系统与 FY3E / FY4B 卫星数据库管理。',
    tags: ['可视化', '卫星数据', '全栈'],
  },
]
</script>

<template>
  <div class="pj-list">
    <section v-for="p in projects" :key="p.name" class="pj">
      <div class="pj-head">
        <span class="pj-name">{{ p.name }}</span>
        <span class="pj-period">{{ p.period }}</span>
      </div>
      <div class="pj-meta">{{ p.role }} · {{ p.tags.join(' · ') }}</div>
      <p class="pj-desc">{{ p.desc }}</p>
    </section>
  </div>
</template>

<style scoped>
.pj-list {
  border-top: 2px solid var(--tuf-ink);
  margin: 1.5rem 0 2rem;
}
.pj {
  padding: 20px 0;
  border-bottom: 1px solid var(--tuf-rule);
}
.pj-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
}
.pj-name {
  font-size: 1.15rem;
  font-weight: 600;
  color: var(--tuf-ink);
}
.pj-period {
  font-family: var(--vp-font-family-mono);
  font-size: 12.5px;
  color: var(--tuf-faint);
  flex-shrink: 0;
}
.pj-meta {
  margin-top: 4px;
  font-variant: small-caps;
  letter-spacing: 0.05em;
  font-size: 0.85rem;
  color: var(--tuf-accent);
}
.pj-desc {
  margin: 10px 0 0;
  font-size: 0.95rem;
  line-height: 1.8;
  color: var(--tuf-muted);
}
</style>
