<script setup>
import HomeStats from './HomeStats.vue'

const focus = [
  {
    title: '大模型推理架构',
    desc: 'PD 分离、MTP 多 Token 预测、动态优先级调度、量化推理（AWQ / GGUF / w8a8），在生产环境落地 vLLM、llama.cpp、TEI 等推理引擎。',
  },
  {
    title: '异构算力调度',
    desc: '统筹 NVIDIA A100 / V100 与华为昇腾 910B3 混合集群，自研梯度算力部署方案，支撑近 50 个大模型重点场景。',
  },
  {
    title: '大模型平台工程',
    desc: '独立全栈开发大模型开放平台：14+ 模型在线管理，OpenAI / Anthropic 兼容接口，KV Cache 感知的 API 智能路由。',
  },
  {
    title: 'AI 应用落地',
    desc: '提示词工程 + RAG 驱动的业务系统：需求项检查、履历合规校验、检索与精排、OCR 全链路。',
  },
]

const stack = [
  'Python', 'PyTorch', 'vLLM', 'SGLang', 'llama.cpp', 'TEI',
  'CUDA', 'Ascend CANN', 'Docker', 'Kubernetes',
  'Vue', 'FastAPI', 'RAG', 'Agent', 'RagFlow', 'Dify',
]

const tiers = [
  {
    no: '01', name: '基础科学', link: '/posts/foundations/math/',
    desc: '数学、物理、化学、生物，及天文、地学、认知、心理、逻辑、科哲、经济学 —— 一切的地基。',
  },
  {
    no: '02', name: '进阶数理', link: '/posts/intermediate/advanced-math/',
    desc: '高等数学、概率统计、线性代数、离散数学，直到实变、泛函、拓扑与微分几何。',
  },
  {
    no: '03', name: '计算机基础', link: '/posts/cs/data-structures/',
    desc: '数据结构、组成原理、操作系统、网络、数据库、编译原理、分布式系统 —— CS 核心课全集。',
  },
  {
    no: '04', name: '高阶专题', link: '/posts/advanced/llm-principles/',
    desc: '机器学习、深度学习、强化学习，大模型原理、微调、部署与基础设施，直至 AI 安全与量子计算。',
  },
]
</script>

<template>
  <div class="home">
    <!-- Hero：工程师身份卡 -->
    <section class="hero">
      <p class="hero-eyebrow">徐鸿铎 · Xu Hongduo</p>
      <h1 class="hero-title">从极限到大模型</h1>
      <p class="hero-role">
        大模型架构工程师<span class="hero-at"> @ </span>中国工商银行总行 · 业务研发中心
      </p>
      <p class="hero-tagline">
        白天做大模型推理架构与异构算力调度，晚上写一个自下而上的知识体系 ——
        从高中数理出发，经大学数学物理与计算机科学，抵达 AI 与大模型前沿。
      </p>
      <div class="hero-actions">
        <a href="/posts/" class="btn btn-brand">博文总览</a>
        <a href="/projects/" class="btn">项目</a>
        <a href="/about/" class="btn">关于我</a>
      </div>
    </section>

    <!-- 专注领域 -->
    <section class="section">
      <h2 class="section-title">专注领域</h2>
      <div class="focus-grid">
        <div v-for="f in focus" :key="f.title" class="focus-card">
          <h3 class="focus-title">{{ f.title }}</h3>
          <p class="focus-desc">{{ f.desc }}</p>
        </div>
      </div>
    </section>

    <!-- 技术栈 -->
    <section class="section">
      <h2 class="section-title">技术栈</h2>
      <div class="stack">
        <span v-for="s in stack" :key="s" class="stack-tag">{{ s }}</span>
      </div>
    </section>

    <!-- 写作体系 -->
    <section class="section">
      <h2 class="section-title">写作体系</h2>
      <p class="section-desc">
        60 个学科、5800+ 个选题，对标经典教材逐节写作。这不是博客的目录，是一份学习计划。
      </p>
      <div class="tier-grid">
        <a v-for="t in tiers" :key="t.no" :href="t.link" class="tier-card">
          <div class="tier-head">
            <span class="tier-no">{{ t.no }}</span>
            <span class="tier-name">{{ t.name }}</span>
          </div>
          <p class="tier-desc">{{ t.desc }}</p>
        </a>
      </div>
      <HomeStats />
    </section>
  </div>
</template>

<style scoped>
.home {
  max-width: 960px;
  margin: 0 auto;
  padding: 64px 24px 32px;
}

/* Hero */
.hero {
  padding: 32px 0 48px;
}
.hero-eyebrow {
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  color: var(--vp-c-brand-1);
  letter-spacing: 0.05em;
  margin-bottom: 16px;
}
.hero-title {
  font-size: clamp(2.4rem, 6vw, 3.8rem);
  font-weight: 800;
  letter-spacing: -0.02em;
  line-height: 1.1;
  color: var(--vp-c-text-1);
}
.hero-role {
  margin-top: 16px;
  font-size: clamp(1.05rem, 2.5vw, 1.35rem);
  font-weight: 600;
  color: var(--vp-c-text-2);
}
.hero-at {
  color: var(--vp-c-brand-1);
}
.hero-tagline {
  margin-top: 16px;
  max-width: 680px;
  font-size: 16px;
  line-height: 1.9;
  color: var(--vp-c-text-2);
}
.hero-actions {
  display: flex;
  gap: 12px;
  margin-top: 32px;
  flex-wrap: wrap;
}
.btn {
  display: inline-block;
  padding: 10px 22px;
  border-radius: 8px;
  border: 1px solid var(--vp-c-divider);
  font-size: 14.5px;
  font-weight: 600;
  color: var(--vp-c-text-1);
  text-decoration: none !important;
  transition: border-color 0.2s, background 0.2s;
}
.btn:hover {
  border-color: var(--vp-c-brand-1);
}
.btn-brand {
  background: var(--vp-c-brand-1);
  border-color: var(--vp-c-brand-1);
  color: #fff;
}
.btn-brand:hover {
  background: var(--vp-c-brand-2);
}

/* Sections */
.section {
  padding: 40px 0;
  border-top: 1px solid var(--vp-c-divider);
}
.section-title {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 0.12em;
  color: var(--vp-c-text-3);
  margin-bottom: 24px;
}
.section-desc {
  font-size: 15px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
  max-width: 680px;
  margin-bottom: 24px;
}

/* 专注领域 */
.focus-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.focus-card {
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  transition: border-color 0.2s;
}
.focus-card:hover {
  border-color: var(--vp-c-brand-1);
}
.focus-title {
  font-size: 17px;
  font-weight: 700;
  color: var(--vp-c-text-1);
  margin-bottom: 10px;
}
.focus-desc {
  font-size: 14.5px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
  margin: 0;
}

/* 技术栈 */
.stack {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.stack-tag {
  font-family: var(--vp-font-family-mono);
  font-size: 13px;
  padding: 5px 12px;
  border-radius: 999px;
  border: 1px solid var(--vp-c-divider);
  color: var(--vp-c-text-2);
  transition: border-color 0.2s, color 0.2s;
}
.stack-tag:hover {
  border-color: var(--vp-c-brand-1);
  color: var(--vp-c-brand-1);
}

/* 写作体系 */
.tier-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}
.tier-card {
  display: block;
  border: 1px solid var(--vp-c-divider);
  border-radius: 12px;
  padding: 24px;
  text-decoration: none !important;
  transition: border-color 0.2s, transform 0.2s, box-shadow 0.2s;
}
.tier-card:hover {
  border-color: var(--vp-c-brand-1);
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
}
.dark .tier-card:hover {
  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
}
.tier-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 10px;
}
.tier-no {
  font-family: var(--vp-font-family-mono);
  font-size: 14px;
  color: var(--vp-c-brand-1);
}
.tier-name {
  font-size: 18px;
  font-weight: 700;
  color: var(--vp-c-text-1);
}
.tier-desc {
  font-size: 14.5px;
  line-height: 1.8;
  color: var(--vp-c-text-2);
  margin: 0;
}

@media (max-width: 640px) {
  .focus-grid,
  .tier-grid {
    grid-template-columns: 1fr;
  }
}
</style>
