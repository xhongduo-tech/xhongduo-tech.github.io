<script setup>
import data from '../data/progress.json'

const tiers = [
  { key: 'foundations', name: '第一级 · 基础科学' },
  { key: 'intermediate', name: '第二级 · 进阶数理' },
  { key: 'cs', name: '第三级 · 计算机基础' },
  { key: 'advanced', name: '第四级 · 高阶专题' },
]

function catsOf(tier) {
  return Object.entries(data)
    .filter(([k]) => k.startsWith(tier + '/'))
    .map(([k, v]) => ({ path: `/posts/${k}/`, ...v }))
}

function pct(c) {
  return c.total ? Math.round((c.done / c.total) * 100) : 0
}
</script>

<template>
  <div v-for="t in tiers" :key="t.key" class="po-tier">
    <h3>{{ t.name }}</h3>
    <div class="po-grid">
      <a v-for="c in catsOf(t.key)" :key="c.path" :href="c.path" class="po-card">
        <div class="po-row">
          <span class="po-name">{{ c.name }}</span>
          <span class="po-stat">{{ c.done }}/{{ c.total }}</span>
        </div>
        <div class="po-bar">
          <div class="po-bar-fill" :style="{ width: pct(c) + '%' }"></div>
        </div>
      </a>
    </div>
  </div>
</template>

<style scoped>
.po-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 10px;
  margin: 12px 0 8px;
}
.po-card {
  display: block;
  border: 1px solid var(--vp-c-divider);
  border-radius: 8px;
  padding: 10px 14px;
  text-decoration: none !important;
  transition: border-color 0.2s;
}
.po-card:hover {
  border-color: var(--vp-c-brand-1);
}
.po-row {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 8px;
}
.po-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--vp-c-text-1);
}
.po-stat {
  font-size: 12px;
  color: var(--vp-c-text-3);
  font-family: var(--vp-font-family-mono);
}
.po-bar {
  height: 4px;
  background: var(--vp-c-divider);
  border-radius: 2px;
  overflow: hidden;
}
.po-bar-fill {
  height: 100%;
  background: var(--vp-c-brand-1);
  border-radius: 2px;
  transition: width 0.4s ease;
}
</style>
