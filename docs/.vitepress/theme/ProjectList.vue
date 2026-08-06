<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { page } = useData()
const isEn = computed(() => page.value.relativePath.startsWith('en/'))

const projectsZh = [
  {
    name: 'BRDC.ai 大模型开放平台',
    period: '2026',
    role: '独立开发',
    desc: '全栈开发的大模型开放平台：14+ 模型在线管理，兼容 OpenAI / Anthropic 接口，基于 GPU KV Cache 实时感知的 API 智能路由，动态优先级调度，Token 梯度路由（8K–128K），自动巡检与运营分析。',
    tags: ['全栈', 'OpenAI 兼容', '智能路由', '动态调度'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="12" rx="2"/><path d="M8 20h8M12 16v4"/><circle cx="9" cy="10" r="1.4"/><path d="M13.5 9.5l4 1M13.5 11.5l4-1"/></svg>',
  },
  {
    name: '异构算力集群模型部署与推理优化',
    period: '2026',
    role: '负责人',
    desc: '在完全内网环境统筹 NVIDIA A100 / V100 与华为昇腾 910B3 混合集群，部署 Qwen、Gemma、DeepSeek、GLM 等系列模型。落地 PD 分离、MTP、MPS、TEI、AWQ / GGUF / w8a8 量化，自研梯度算力部署方案支撑近 50 个重点场景。',
    tags: ['vLLM', 'llama.cpp', 'PD 分离', '量化推理', '昇腾'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="5" width="14" height="14" rx="2"/><rect x="9" y="9" width="6" height="6" rx="1"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/></svg>',
  },
  {
    name: '中心大模型 API 统一接入平台',
    period: '2024',
    role: '牵头',
    desc: '对标 DeepSeek 官网的初版大模型 API 统一接入平台：密钥管理、接口文档、调用日志与用量统计。',
    tags: ['API 网关', '密钥管理', '用量统计'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 12l2 2 4-4"/><circle cx="12" cy="12" r="9"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3"/></svg>',
  },
  {
    name: '内网 PyPI 私有包管理平台',
    period: '2024',
    role: '搭建',
    desc: '内网环境私有 Python 包管理平台，解决离线环境下依赖隔离与版本管理问题。',
    tags: ['PyPI', '内网', '依赖管理'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><path d="M3.3 7L12 12l8.7-5M12 22V12"/></svg>',
  },
  {
    name: 'BRDM 需求项智能检查系统',
    period: '2023',
    role: '牵头',
    desc: '基于 Qwen2-72B 与提示词工程的需求项名称与描述自动检查系统，大模型在银行业务的首次落地。',
    tags: ['Qwen', '提示词工程', '业务落地'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
  },
  {
    name: '员工履历检查系统',
    period: '2023',
    role: '牵头',
    desc: '提示词 + RAG 方案实现非结构化履历信息的精准抽取与合规校验，用于年度轮训检查，测试后移交开发中心。',
    tags: ['RAG', '信息抽取', '合规校验'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"/><path d="M21 21l-4.3-4.3"/><path d="M9 11h4M11 9v4"/></svg>',
  },
  {
    name: '手部穴位检测（改进 HRNet）',
    period: '2022',
    role: '第二作者 · IJCNN',
    desc: '中科院计算所实习成果：提出改进的高分辨率网络（HRNet）用于手部穴位自动检测，发表于 IJCNN 2022。',
    tags: ['HRNet', '关键点检测', '论文'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M8 13V5a1.5 1.5 0 0 1 3 0v6"/><path d="M11 11V4a1.5 1.5 0 0 1 3 0v7"/><path d="M14 11V6a1.5 1.5 0 0 1 3 0v9a6 6 0 0 1-6 6h-1a6 6 0 0 1-5.2-3l-1.6-2.8a1.5 1.5 0 0 1 2.6-1.5L8 16"/></svg>',
  },
  {
    name: '卫星火情监测与风场可视化',
    period: '2021–2022',
    role: '全栈工程师',
    desc: '中国气象局华云集团：基于风速风向数据的风羽图二三维实时绘制，参与卫星火情监测预测系统与 FY3E / FY4B 卫星数据库管理。',
    tags: ['可视化', '卫星数据', '全栈'],
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M18 10a4 4 0 0 0-7.5-2A3.5 3.5 0 1 0 9 14.5h9a3 3 0 0 0 0-4.5z"/><path d="M9 18l-2 3M14 18l-1 3M19 18l-2 3"/></svg>',
  },
]

const projectsEn = [
  {
    name: 'BRDC.ai LLM Open Platform',
    period: '2026',
    role: 'Solo',
    desc: 'A full-stack LLM open platform: 14+ models managed online, OpenAI / Anthropic-compatible APIs, GPU KV-cache-aware intelligent routing, dynamic priority scheduling, token-tiered routing (8K–128K), auto-inspection and operations analytics.',
    tags: ['Full-stack', 'OpenAI-compatible', 'Smart routing', 'Dynamic scheduling'],
    icon: projectsZh[0].icon,
  },
  {
    name: 'Model Deployment & Inference Optimization on Heterogeneous Clusters',
    period: '2026',
    role: 'Lead',
    desc: 'Orchestrated mixed NVIDIA A100 / V100 and Huawei Ascend 910B3 clusters in a fully intranet environment to deploy the Qwen, Gemma, DeepSeek and GLM families. Shipped PD disaggregation, MTP, MPS, TEI, and AWQ / GGUF / w8a8 quantization, with a self-designed tiered compute scheme supporting ~50 key scenarios.',
    tags: ['vLLM', 'llama.cpp', 'PD disaggregation', 'Quantized inference', 'Ascend'],
    icon: projectsZh[1].icon,
  },
  {
    name: 'Unified LLM API Gateway',
    period: '2024',
    role: 'Initiated',
    desc: 'Modeled on DeepSeek’s official site, the initial unified LLM API gateway: key management, API documentation, call logs and usage analytics.',
    tags: ['API gateway', 'Key management', 'Usage analytics'],
    icon: projectsZh[2].icon,
  },
  {
    name: 'Intranet PyPI Package Registry',
    period: '2024',
    role: 'Built',
    desc: 'A private Python package registry for the intranet, solving dependency isolation and version management in offline environments.',
    tags: ['PyPI', 'Intranet', 'Dependency management'],
    icon: projectsZh[3].icon,
  },
  {
    name: 'BRDM Requirement-Item Review System',
    period: '2023',
    role: 'Initiated',
    desc: 'Automatic checking of requirement-item names and descriptions powered by Qwen2-72B and prompt engineering — the first LLM deployment in banking business.',
    tags: ['Qwen', 'Prompt engineering', 'Business deployment'],
    icon: projectsZh[4].icon,
  },
  {
    name: 'Resume Compliance Checker',
    period: '2023',
    role: 'Initiated',
    desc: 'A prompt + RAG pipeline for precise extraction and compliance validation of unstructured resumes, used in annual rotation checks before being handed to the development center.',
    tags: ['RAG', 'Information extraction', 'Compliance'],
    icon: projectsZh[5].icon,
  },
  {
    name: 'Hand Acupoint Detection (Improved HRNet)',
    period: '2022',
    role: 'Second author · IJCNN',
    desc: 'Internship work at ICT, CAS: an improved High-Resolution Network (HRNet) for automatic hand-acupoint detection, published at IJCNN 2022.',
    tags: ['HRNet', 'Keypoint detection', 'Paper'],
    icon: projectsZh[6].icon,
  },
  {
    name: 'Satellite Fire Monitoring & Wind-Field Visualization',
    period: '2021–2022',
    role: 'Full-stack Engineer',
    desc: 'At Huayun Group, China Meteorological Administration: real-time 2D/3D wind-barb rendering from wind data, contributing to the satellite fire-monitoring prediction system and FY3E / FY4B satellite database management.',
    tags: ['Visualization', 'Satellite data', 'Full-stack'],
    icon: projectsZh[7].icon,
  },
]

const projects = computed(() => (isEn.value ? projectsEn : projectsZh))
</script>

<template>
  <div class="pj-list">
    <section v-for="p in projects" :key="p.name" class="pj">
      <div class="pj-head">
        <span class="pj-title">
          <span class="pj-icon" v-html="p.icon"></span>
          <span class="pj-name">{{ p.name }}</span>
        </span>
        <span class="pj-period">{{ p.period }}</span>
      </div>
      <div class="pj-meta">{{ p.role }}</div>
      <div class="pj-tags">
        <span v-for="tag in p.tags" :key="tag" class="pj-tag">{{ tag }}</span>
      </div>
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
.pj-list :deep(p.pj-desc) {
  width: 100%;
  max-width: none;
}
.pj-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 16px;
}
.pj-title {
  display: inline-flex;
  align-items: center;
  gap: 0.5em;
}
.pj-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.3em;
  height: 1.3em;
  flex: 0 0 auto;
  color: var(--tuf-ink, #111);
  opacity: 0.8;
}
.pj-icon :deep(svg) {
  width: 1.15em;
  height: 1.15em;
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
  color: var(--tuf-faint);
}
.pj-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
  margin-top: 8px;
}
.pj-tag {
  display: inline-block;
  padding: 2px 9px;
  font-size: 0.74rem;
  letter-spacing: 0.02em;
  color: var(--tuf-muted, #555);
  background: color-mix(in srgb, var(--tuf-ink, #111) 6%, transparent);
  border: 1px solid color-mix(in srgb, var(--tuf-ink, #111) 14%, transparent);
  border-radius: 999px;
  line-height: 1.5;
}
.pj-desc {
  margin: 10px 0 0;
  font-size: 0.95rem;
  line-height: 1.8;
  color: var(--tuf-muted);
}
</style>
