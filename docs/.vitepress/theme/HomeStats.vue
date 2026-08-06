<script setup>
import { useData } from 'vitepress'
import data from '../data/progress.json'

const { page } = useData()
const isEn = page.value.relativePath.startsWith('en/')

const cats = Object.values(data)
const topics = cats.reduce((s, c) => s + c.total, 0)
const done = cats.reduce((s, c) => s + c.done, 0)
const pct = topics ? Math.round((done / topics) * 100) : 0

const labels = isEn
  ? ['Disciplines', 'Topics', 'Finished', 'Progress']
  : ['学科', '选题', '已完成', '总进度']

const icons = [
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg>',
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>',
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>',
  '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><path d="M7 15l4-4 3 3 5-6"/></svg>',
]

const stats = [
  { label: labels[0], value: cats.length, icon: icons[0] },
  { label: labels[1], value: topics.toLocaleString('en-US'), icon: icons[1] },
  { label: labels[2], value: done, icon: icons[2] },
  { label: labels[3], value: pct + '%', icon: icons[3] },
]
</script>

<template>
  <div class="hs-band">
    <div v-for="s in stats" :key="s.label" class="hs-item">
      <div class="hs-icon" v-html="s.icon"></div>
      <div class="hs-value">{{ s.value }}</div>
      <div class="hs-label">{{ s.label }}</div>
    </div>
  </div>
</template>

<style scoped>
.hs-band {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  max-width: 720px;
  margin: 3rem 0;
  padding: 18px 0;
  border-top: 2px solid var(--tuf-ink);
  border-bottom: 2px solid var(--tuf-ink);
}
.hs-item {
  text-align: center;
}
.hs-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.7em;
  height: 1.7em;
  margin-bottom: 6px;
  color: var(--accent);
  opacity: 0.85;
}
.hs-icon :deep(svg) {
  width: 1.5em;
  height: 1.5em;
}
.hs-value {
  font-size: 2rem;
  font-weight: 600;
  color: var(--tuf-ink);
  font-variant-numeric: tabular-nums;
  line-height: 1.2;
}
.hs-label {
  margin-top: 4px;
  font-variant: small-caps;
  letter-spacing: 0.12em;
  font-size: 0.8rem;
  color: var(--tuf-faint);
}
</style>
