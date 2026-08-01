<script setup>
import data from '../data/progress.json'

const cats = Object.values(data)
const topics = cats.reduce((s, c) => s + c.total, 0)
const done = cats.reduce((s, c) => s + c.done, 0)
const pct = topics ? Math.round((done / topics) * 100) : 0

const stats = [
  { label: '学科', value: cats.length, suffix: '个' },
  { label: '选题', value: topics.toLocaleString('en-US'), suffix: '篇' },
  { label: '已完成', value: done, suffix: '篇' },
  { label: '总进度', value: pct, suffix: '%' },
]
</script>

<template>
  <div class="hs-band">
    <div v-for="s in stats" :key="s.label" class="hs-item">
      <div class="hs-value">
        {{ s.value }}<span class="hs-suffix">{{ s.suffix }}</span>
      </div>
      <div class="hs-label">{{ s.label }}</div>
    </div>
  </div>
</template>

<style scoped>
.hs-band {
  display: grid;
  grid-template-columns: repeat(4, 1fr);
  gap: 16px;
  max-width: 760px;
  margin: 0 auto;
  padding: 32px 0 8px;
  border-top: 1px solid var(--vp-c-divider);
}
.hs-item {
  text-align: center;
}
.hs-value {
  font-family: var(--vp-font-family-mono);
  font-size: clamp(1.6rem, 4vw, 2.4rem);
  font-weight: 700;
  font-variant-numeric: tabular-nums;
  color: var(--vp-c-text-1);
  line-height: 1.2;
}
.hs-suffix {
  font-size: 0.55em;
  font-weight: 500;
  color: var(--vp-c-text-3);
  margin-left: 2px;
}
.hs-label {
  margin-top: 6px;
  font-size: 13px;
  color: var(--vp-c-text-3);
  letter-spacing: 0.08em;
}
@media (max-width: 640px) {
  .hs-band {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
