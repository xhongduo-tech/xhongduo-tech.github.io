<script setup>
import { computed } from 'vue'
import { withBase, useData } from 'vitepress'
import dataZh from '../data/progress.json'
import dataEn from '../data/progress.en.json'
import techTopics from '../data/tech-topics.json'

const { page } = useData()
const isEn = computed(() => page.value.relativePath.startsWith('en/'))
const data = computed(() => (isEn.value ? dataEn : dataZh))

// 四大技术领域（博文范围），专题清单来自 tech-topics.json
const domains = computed(() =>
  isEn.value
    ? [
        { key: 'math-physics', name: 'Mathematics & Physics' },
        { key: 'cs', name: 'Computer Science' },
        { key: 'ai', name: 'AI & Large Models' },
        { key: 'engineering', name: 'Engineering' },
      ]
    : [
        { key: 'math-physics', name: '数理基础' },
        { key: 'cs', name: '计算机科学' },
        { key: 'ai', name: 'AI 与大模型' },
        { key: 'engineering', name: '工程技术' },
      ],
)

function catsOf(domainKey) {
  const prefix = isEn.value ? '/en/posts/' : '/posts/'
  return (techTopics.domains[domainKey] || [])
    .map((k) => {
      const v = data.value[k]
      return v ? { path: prefix + k + '/', ...v } : null
    })
    .filter(Boolean)
}

function pct(c) {
  return c.total ? Math.round((c.done / c.total) * 100) : 0
}

// 总进度统计（技术专题）
const summary = computed(() => {
  let total = 0,
    done = 0,
    count = 0
  for (const d of domains.value)
    for (const c of catsOf(d.key)) {
      total += c.total
      done += c.done
      count++
    }
  return { total, done, count, pct: total ? Math.round((done / total) * 100) : 0 }
})
</script>

<template>
  <p class="po-summary">
    技术专题共 <strong>{{ summary.count }}</strong> 个 · 完成
    <strong>{{ summary.done }}/{{ summary.total }}</strong>（{{ summary.pct }}%）
  </p>
  <div v-for="d in domains" :key="d.key" class="po-tier">
    <h3>{{ d.name }}</h3>
    <div class="po-list">
      <a v-for="c in catsOf(d.key)" :key="c.path" :href="withBase(c.path)" class="po-row">
        <span class="po-name">{{ c.name }}</span>
        <span class="po-bar"><span class="po-bar-fill" :style="{ width: pct(c) + '%' }"></span></span>
        <span class="po-stat">{{ c.done }}/{{ c.total }}</span>
      </a>
    </div>
  </div>
</template>

<style scoped>
.po-summary {
  margin: 0.5rem 0 0.8rem;
  color: var(--tuf-muted);
  font-size: 0.92rem;
}
.po-summary strong {
  color: var(--tuf-ink);
}
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
