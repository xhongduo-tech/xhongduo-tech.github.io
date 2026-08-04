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

const stats = [
  { label: labels[0], value: cats.length },
  { label: labels[1], value: topics.toLocaleString('en-US') },
  { label: labels[2], value: done },
  { label: labels[3], value: pct + '%' },
]
</script>

<template>
  <div class="hs-band">
    <div v-for="s in stats" :key="s.label" class="hs-item">
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
