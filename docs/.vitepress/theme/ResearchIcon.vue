<script setup lang="ts">
defineProps<{ name: string }>()

const paths: Record<string, string[]> = {
  foundations: ['M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z'],
  inference: ['M8 6h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2Z', 'M9 3v3m6-3v3M9 18v3m6-3v3M3 9h3m-3 6h3M18 9h3m-3 6h3M10 10h4v4h-4z'],
  training: ['M4 7h4m4 0h8M4 17h8m4 0h4', 'M8 4v6h4V4zM12 14v6h4v-6z'],
  evaluation: ['M4 6l2 2 3-3m3 2h8M4 15l2 2 3-3m3 2h8'],
}
</script>

<template>
  <svg class="research-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">
    <path v-for="(path, index) in paths[name] || paths.foundations" :key="index" :d="path" />
  </svg>
</template>
