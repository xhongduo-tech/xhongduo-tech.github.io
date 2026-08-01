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
  <div v-if="cat" class="pg">
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
  </div>
</template>

<style scoped>
.pg {
  background: #0d1117;
  border: 1px solid #21262d;
  border-radius: 8px;
  padding: 16px 20px;
  margin: 16px 0 24px;
  font-family: var(--vp-font-family-mono);
}
.pg-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  margin-bottom: 10px;
}
.pg-title {
  color: #e6edf3;
  font-size: 14px;
  font-weight: 600;
}
.pg-stat {
  color: #7d8590;
  font-size: 12px;
}
.pg-bar {
  height: 6px;
  background: #21262d;
  border-radius: 3px;
  overflow: hidden;
  margin-bottom: 16px;
}
.pg-bar-fill {
  height: 100%;
  background: #3fb950;
  border-radius: 3px;
  transition: width 0.4s ease;
}
.pg-ch-title {
  color: #7d8590;
  font-size: 12px;
  margin: 14px 0 6px;
  font-family: var(--vp-font-family-base);
}
.pg-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
  gap: 2px 16px;
}
.pg-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
  font-size: 12px;
  line-height: 1.8;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: #6e7681;
}
.pg-seq {
  color: #484f58;
  flex-shrink: 0;
}
.pg-mark {
  width: 12px;
  flex-shrink: 0;
  color: #3fb950;
}
.pg-name {
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--vp-font-family-base);
}
.pg-item.done {
  color: #adbac7;
}
.pg-item.done .pg-seq {
  color: #3fb950;
}
</style>
