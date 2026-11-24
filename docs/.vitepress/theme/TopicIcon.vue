<script setup lang="ts">
withDefaults(defineProps<{ name: string; size?: number }>(), { size: 20 })
</script>

<template>
  <svg :width="size" :height="size" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.35" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">
    <template v-if="name === 'foundations' || name === 'network'">
      <circle cx="5" cy="6" r="2.25" /><circle cx="19" cy="5" r="2.25" /><circle cx="12" cy="18" r="2.25" /><circle cx="12" cy="11" r="2.25" />
      <path d="m7 6.8 3 2.7m4-1 3-2.3M5.8 8l4.4 8M13 13l-.7 2.8M10 11H7" />
    </template>
    <template v-else-if="name === 'training'">
      <path d="m12 3 9 5-9 5-9-5 9-5Z" /><path d="m3 12 9 5 9-5M3 16l9 5 9-5" /><path d="M17 4v5" />
    </template>
    <template v-else-if="name === 'inference' || name === 'chip'">
      <rect x="5" y="5" width="14" height="14" rx="2" /><rect x="9" y="9" width="6" height="6" rx="1" />
      <path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" />
    </template>
    <template v-else-if="name === 'evaluation'">
      <path d="M4 20V5m0 15h16" /><path d="m7 15 4-4 3 2 5-6" /><path d="m16 7 3-1 1 3" />
    </template>
    <template v-else-if="name === 'applications'">
      <rect x="3" y="3" width="7" height="7" rx="1.5" /><rect x="14" y="3" width="7" height="7" rx="1.5" /><rect x="8.5" y="14" width="7" height="7" rx="1.5" />
      <path d="M6.5 10v2.5H12V14m5.5-4v2.5H15" />
    </template>
    <template v-else-if="name === 'projects' || name === 'experiment'">
      <path d="M9 3h6m-5 0v6l-5.5 9.5A1.8 1.8 0 0 0 6 21h12a1.8 1.8 0 0 0 1.5-2.5L14 9V3" /><path d="M7.5 15h9M9 12h6" /><circle cx="12" cy="17.5" r=".7" />
    </template>
    <template v-else-if="name === 'math'">
      <path d="M5 4h14l-8 8 8 8H5M5 12h5" /><path d="M4 2v20" opacity=".45" />
    </template>
    <template v-else-if="name === 'code'">
      <path d="m8 6-6 6 6 6m8-12 6 6-6 6M14 4l-4 16" />
    </template>
    <template v-else-if="name === 'retrieval'">
      <ellipse cx="9" cy="5.5" rx="6" ry="3" /><path d="M3 5.5v6c0 1.7 2.7 3 6 3 1.2 0 2.3-.1 3.2-.4M3 11.5v6c0 1.7 2.7 3 6 3h1" />
      <circle cx="16.5" cy="16.5" r="3.5" /><path d="m19 19 3 3" />
    </template>
    <template v-else-if="name === 'database'">
      <ellipse cx="12" cy="5" rx="8" ry="3" /><path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
    </template>
    <template v-else-if="name === 'decode'">
      <path d="M3 12h13m-5-5 5 5-5 5" /><path d="m18 3 .8 2.2L21 6l-2.2.8L18 9l-.8-2.2L15 6l2.2-.8L18 3Z" /><path d="M4 5h5M4 19h5" />
    </template>
    <template v-else-if="name === 'agent'">
      <circle cx="12" cy="12" r="4" /><circle cx="4" cy="5" r="2" /><circle cx="20" cy="5" r="2" /><circle cx="4" cy="20" r="2" /><circle cx="20" cy="20" r="2" />
      <path d="m5.5 6.4 3.2 3M18.5 6.4l-3.2 3M5.5 18.5l3.2-3.1M18.5 18.5l-3.2-3.1" />
    </template>
    <template v-else>
      <circle cx="12" cy="12" r="8" /><path d="M4 12h16M12 4c2.2 2.3 3.3 5 3.3 8s-1.1 5.7-3.3 8c-2.2-2.3-3.3-5-3.3-8S9.8 6.3 12 4Z" />
    </template>
  </svg>
</template>
