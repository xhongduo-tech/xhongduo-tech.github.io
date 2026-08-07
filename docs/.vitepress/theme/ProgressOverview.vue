<script setup>
import { computed } from 'vue'
import { withBase, useData } from 'vitepress'
import dataZh from '../data/progress.json'
import dataEn from '../data/progress.en.json'

const { page } = useData()
const isEn = computed(() => page.value.relativePath.startsWith('en/'))
const data = computed(() => (isEn.value ? dataEn : dataZh))

const tiers = computed(() =>
  isEn.value
    ? [
        { key: 'foundations', name: 'Level 1 · Foundations' },
        { key: 'intermediate', name: 'Level 2 · Intermediate Mathematics' },
        { key: 'cs', name: 'Level 3 · Computer Science' },
        { key: 'advanced', name: 'Level 4 · Advanced Topics' },
        { key: 'life', name: 'Level 5 · Life & Health' },
        { key: 'engineering', name: 'Level 6 · Engineering' },
        { key: 'humanities', name: 'Level 7 · Humanities & Arts' },
        { key: 'social', name: 'Level 8 · Social Sciences' },
        { key: 'frontier', name: 'Level 9 · Interdisciplinary & Frontier' },
      ]
    : [
        { key: 'foundations', name: '第一级 · 基础科学' },
        { key: 'intermediate', name: '第二级 · 进阶数理' },
        { key: 'cs', name: '第三级 · 计算机基础' },
        { key: 'advanced', name: '第四级 · 高阶专题' },
        { key: 'life', name: '第五级 · 生命与健康' },
        { key: 'engineering', name: '第六级 · 工程技术' },
        { key: 'humanities', name: '第七级 · 人文与艺术' },
        { key: 'social', name: '第八级 · 社会科学' },
        { key: 'frontier', name: '第九级 · 交叉与前沿' },
      ],
)

function catsOf(tier) {
  const prefix = isEn.value ? '/en/posts/' : '/posts/'
  return Object.entries(data.value)
    .filter(([k]) => k.startsWith(tier + '/'))
    .map(([k, v]) => ({ path: prefix + k + '/', ...v }))
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
.po-tier :deep(h3) {
  width: 100%;
  max-width: none;
  margin-top: 1.6rem;
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
