<script setup>
import { withBase } from 'vitepress'
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
    <div class="po-list">
      <a v-for="c in catsOf(t.key)" :key="c.path" :href="withBase(c.path)" class="po-row">
        <span class="po-name">{{ c.name }}</span>
        <span class="po-bar"><span class="po-bar-fill" :style="{ width: pct(c) + '%' }"></span></span>
        <span class="po-stat">{{ c.done }}/{{ c.total }}</span>
      </a>
    </div>
  </div>
</template>

<style scoped>
.po-list {
  margin: 0.5rem 0 1rem;
  border-top: 1px solid var(--tuf-rule);
}
.po-row {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 7px 0;
  border-bottom: 1px solid var(--tuf-rule);
  text-decoration: none !important;
}
.po-name {
  width: 200px;
  flex-shrink: 0;
  color: var(--tuf-ink);
  font-size: 0.95rem;
}
.po-row:hover .po-name {
  color: var(--tuf-accent);
}
.po-bar {
  flex: 1;
  height: 3px;
  background: var(--tuf-rule);
}
.po-bar-fill {
  display: block;
  height: 100%;
  background: var(--tuf-accent);
}
.po-stat {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--tuf-muted);
  flex-shrink: 0;
}
@media (max-width: 720px) {
  .po-name {
    width: 120px;
  }
}
</style>
