<script setup>
import { withBase } from 'vitepress'
import techTopics from '../data/tech-topics.json'

const domains = [
  {
    key: 'math-physics',
    name: '数理基础',
    en: 'Mathematics & Physics',
    desc: '数学与物理是一切技术的语言',
    icon: '<path d="M4 19V5"/><path d="M4 19h16"/><path d="M8 16l3-4 3 2 4-6"/>',
    anchor: '数理基础',
  },
  {
    key: 'cs',
    name: '计算机科学',
    en: 'Computer Science',
    desc: '系统与软件工程的完整知识体系',
    icon: '<rect x="4" y="4" width="16" height="16" rx="2"/><path d="M9 9h6v6H9z"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    anchor: '计算机科学',
  },
  {
    key: 'ai',
    name: 'AI 与大模型',
    en: 'AI & Large Models',
    desc: '从机器学习到大模型与智能体',
    icon: '<path d="M12 2a4 4 0 0 0-4 4v1a4 4 0 0 0-3 6.6A4 4 0 0 0 7 21h10a4 4 0 0 0 2-7.4A4 4 0 0 0 16 7V6a4 4 0 0 0-4-4z"/><path d="M12 8v5M12 16h.01"/>',
    anchor: 'ai-与大模型',
  },
  {
    key: 'engineering',
    name: '工程技术',
    en: 'Engineering',
    desc: '全部工科主干学科的全谱',
    icon: '<path d="M14.7 6.3a4 4 0 0 0-5.4 5.4L3 18v3h3l6.3-6.3a4 4 0 0 0 5.4-5.4L14 12l-2-2z"/>',
    anchor: '工程技术',
  },
].map((d) => ({ ...d, count: (techTopics.domains[d.key] || []).length }))
</script>

<template>
  <div class="rd">
    <a
      v-for="d in domains"
      :key="d.key"
      :href="withBase(`/posts/#${d.anchor}`)"
      class="rd-card"
    >
      <span class="rd-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" v-html="d.icon"></svg></span>
      <span class="rd-body">
        <span class="rd-name">{{ d.name }}<em class="rd-en">{{ d.en }}</em></span>
        <span class="rd-desc">{{ d.desc }}</span>
      </span>
      <span class="rd-count">{{ d.count }} 专题</span>
    </a>
  </div>
</template>

<style scoped>
.rd {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(240px, 1fr));
  gap: 0.8rem;
  margin: 1rem 0 1.4rem;
}
.rd-card {
  display: flex;
  align-items: center;
  gap: 0.7rem;
  padding: 0.85rem 0.9rem;
  border: 1px solid var(--tuf-rule);
  border-radius: 6px;
  background: var(--tuf-bg);
  text-decoration: none !important;
  transition: border-color 0.15s, box-shadow 0.15s;
}
.rd-card:hover {
  border-color: var(--tuf-accent);
  box-shadow: 0 1px 6px rgba(0, 0, 0, 0.06);
}
.rd-icon {
  flex-shrink: 0;
  width: 1.7rem;
  height: 1.7rem;
  color: var(--tuf-accent);
}
.rd-icon svg {
  width: 100%;
  height: 100%;
}
.rd-body {
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.rd-name {
  font-size: 1rem;
  font-weight: 600;
  color: var(--tuf-ink);
  display: flex;
  align-items: baseline;
  gap: 0.4rem;
}
.rd-en {
  font-style: normal;
  font-size: 0.68rem;
  color: var(--tuf-muted);
  font-family: var(--vp-font-family-mono);
}
.rd-desc {
  font-size: 0.78rem;
  color: var(--tuf-muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.rd-count {
  margin-left: auto;
  flex-shrink: 0;
  font-size: 0.72rem;
  color: var(--tuf-muted);
  background: color-mix(in srgb, var(--tuf-rule) 60%, transparent);
  padding: 0.1rem 0.4rem;
  border-radius: 3px;
}
</style>
