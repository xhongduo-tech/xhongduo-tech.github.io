<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import recent from '../data/recent-posts.json'

const posts = computed(() => recent.posts || [])
</script>

<template>
  <div class="rr">
    <ul class="rr-list">
      <li v-for="p in posts" :key="p.path" class="rr-item">
        <span class="rr-date">{{ p.date }}</span>
        <a :href="withBase(p.path)" class="rr-link">{{ p.title }}</a>
        <span class="rr-domain">{{ p.domain }}</span>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.rr-list {
  margin: 0.4rem 0 1rem;
  padding: 0;
  list-style: none;
  border-top: 1px solid var(--tuf-rule);
}
.rr-item {
  display: flex;
  align-items: baseline;
  gap: 0.8rem;
  padding: 6px 0;
  border-bottom: 1px solid var(--tuf-rule);
  font-size: 0.94rem;
}
.rr-date {
  font-family: var(--vp-font-family-mono);
  font-size: 0.78rem;
  color: var(--tuf-muted);
  flex-shrink: 0;
}
.rr-link {
  color: var(--tuf-ink);
  text-decoration: none;
  border-bottom: 1px dotted #9fb7d0;
}
.rr-link:hover {
  color: var(--tuf-accent);
  border-bottom-style: solid;
}
.rr-domain {
  margin-left: auto;
  font-size: 0.74rem;
  color: var(--tuf-muted);
  flex-shrink: 0;
}
@media (max-width: 720px) {
  .rr-domain {
    display: none;
  }
}
</style>
