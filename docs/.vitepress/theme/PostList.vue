<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as posts } from '../posts.data'

const props = defineProps({
  section: { type: String, default: '' },
  limit: { type: Number, default: 0 },
})

const list = computed(() => {
  let rows = posts
  if (props.section) rows = rows.filter((p) => p.section === props.section)
  if (props.limit > 0) rows = rows.slice(0, props.limit)
  return rows
})
</script>

<template>
  <p v-if="!list.length" class="post-empty">尚无</p>
  <ol v-else class="post-list">
    <li v-for="post in list" :key="post.url" class="blog-entry">
      <a :href="withBase(post.url)">{{ post.title }}</a>
      <time v-if="post.date" :datetime="post.date">{{ post.date }}</time>
    </li>
  </ol>
</template>
