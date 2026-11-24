<script setup lang="ts">
defineProps<{
  value: 'articles' | 'topology'
  english: boolean
}>()

const emit = defineEmits<{
  change: [value: 'articles' | 'topology']
}>()
</script>

<template>
  <div class="article-view-switch" role="group" :aria-label="english ? 'Choose article view' : '选择文章视图'">
    <button type="button" data-view="articles" :aria-pressed="value === 'articles'" aria-controls="article-view-panel" @click="emit('change', 'articles')">
      <svg viewBox="0 0 18 18" aria-hidden="true"><path d="M4 4.5h10M4 9h10M4 13.5h6" /></svg>
      {{ english ? 'Articles' : '具体博文' }}
    </button>
    <button type="button" data-view="topology" :aria-pressed="value === 'topology'" aria-controls="article-view-panel" @click="emit('change', 'topology')">
      <svg viewBox="0 0 18 18" aria-hidden="true"><circle cx="4" cy="5" r="1.8" /><circle cx="14" cy="4" r="1.8" /><circle cx="11" cy="14" r="1.8" /><path d="m5.7 4.8 6.5-.6M5.1 6.3l4.8 6M13 5.7l-1.3 6.5" /></svg>
      {{ english ? 'Topic map' : '主题拓扑' }}
    </button>
  </div>
</template>
