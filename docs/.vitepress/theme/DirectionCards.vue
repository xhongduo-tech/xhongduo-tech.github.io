<script setup>
import { computed, ref, onMounted } from 'vue'
import { withBase } from 'vitepress'
import { learnPaths } from '../data/paths'

const rows = computed(() =>
  learnPaths.map((p) => ({
    id: p.id,
    name: p.name,
    goal: p.goal,
    href: withBase(`/path/${p.id}/`),
    total: p.stages.reduce((n, s) => n + s.items.length, 0),
    done: 0,
  })),
)

const loaded = ref(false)

onMounted(() => {
  for (const row of rows.value) {
    try {
      const raw = localStorage.getItem(`path-progress:${row.id}`)
      row.done = raw ? JSON.parse(raw).length : 0
    } catch {
      row.done = 0
    }
  }
  loaded.value = true
})
</script>

<template>
  <ul class="direction-cards">
    <li v-for="row in rows" :key="row.id" class="direction-card">
      <a class="direction-name" :href="row.href">{{ row.name }}</a>
      <p class="direction-goal">{{ row.goal }}</p>
      <p class="direction-progress">
        <template v-if="loaded">已读 {{ row.done }} / {{ row.total }} 课</template>
        <template v-else>{{ row.total }} 课 · 含勾选进度</template>
      </p>
      <div class="direction-bar">
        <span :style="{ width: (row.total ? (100 * row.done) / row.total : 0) + '%' }"></span>
      </div>
    </li>
  </ul>
</template>
