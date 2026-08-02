<script setup>
import { computed } from 'vue'
import data from '../data/progress.json'

const props = defineProps({
  cat: { type: String, required: true },
})

const cat = computed(() => data[props.cat])
const pct = computed(() =>
  cat.value?.total ? Math.round((cat.value.done / cat.value.total) * 100) : 0,
)

// 连续编号：返回第 chIndex 章第 i 条的全局序号（从 1 开始）
function seq(chIndex, i) {
  let n = 1
  for (let c = 0; c < chIndex; c++) n += cat.value.chapters[c].items.length
  return String(n + i).padStart(3, '0')
}
</script>

<template>
  <figure v-if="cat" class="pg">
    <div class="pg-head">
      <span class="pg-title">写作进度</span>
      <span class="pg-stat">{{ cat.done }} / {{ cat.total }} 篇 · {{ pct }}%</span>
    </div>
    <div class="pg-bar">
      <div class="pg-bar-fill" :style="{ width: pct + '%' }"></div>
    </div>
    <div v-for="(ch, ci) in cat.chapters" :key="ci" class="pg-ch">
      <div v-if="ch.title" class="pg-ch-title">{{ ch.title }}</div>
      <div class="pg-grid">
        <span
          v-for="(it, i) in ch.items"
          :key="i"
          class="pg-item"
          :class="{ done: it.done }"
          :title="it.title"
        >
          <span class="pg-seq">{{ seq(ci, i) }}</span>
          <span class="pg-mark">{{ it.done ? '✓' : '' }}</span>
          <span class="pg-name">{{ it.title }}</span>
        </span>
      </div>
    </div>
  </figure>
</template>

<style scoped>
.pg {
  border-top: 2px solid var(--tuf-ink);
  border-bottom: 2px solid var(--tuf-ink);
  padding: 14px 0 18px;
  margin: 2rem 0;
}
.pg-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  padding-bottom: 10px;
}
.pg-title {
  font-variant: small-caps;
  letter-spacing: 0.1em;
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--tuf-ink);
}
.pg-stat {
  font-family: var(--vp-font-family-mono);
  font-size: 12px;
  color: var(--tuf-muted);
}
.pg-bar {
  height: 3px;
  background: var(--tuf-rule);
  margin-bottom: 14px;
}
.pg-bar-fill {
  height: 100%;
  background: var(--tuf-accent);
  transition: width 0.4s ease;
}
.pg-ch-title {
  font-variant: small-caps;
  letter-spacing: 0.06em;
  font-size: 0.82rem;
  color: var(--tuf-faint);
  margin: 14px 0 4px;
}
.pg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 1px 20px;
}
.pg-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 12.5px;
  line-height: 1.7;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--tuf-faint);
}
.pg-seq {
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  flex-shrink: 0;
}
.pg-mark {
  width: 12px;
  flex-shrink: 0;
  color: var(--tuf-done);
}
.pg-name {
  overflow: hidden;
  text-overflow: ellipsis;
}
.pg-item.done {
  color: var(--tuf-ink);
}
.pg-item.done .pg-seq {
  color: var(--tuf-done);
}
</style>
