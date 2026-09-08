<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { data as posts } from '../posts.data'
import { curriculumOrder } from '../data/curriculum'
import { isSectionId } from '../data/sections'

const props = defineProps({
  section: { type: String, default: '' },
  limit: { type: Number, default: 0 },
})

function slugOf(post) {
  return post.url.replace(/\/$/, '').split('/').filter(Boolean).pop()
}

const list = computed(() => {
  let rows = posts
  if (props.section) {
    rows = rows.filter((p) => p.section === props.section)
    const order = isSectionId(props.section) ? curriculumOrder(props.section) : []
    const rank = new Map(order.map((slug, i) => [slug, i]))
    rows = [...rows].sort((a, b) => {
      const ra = rank.has(slugOf(a)) ? rank.get(slugOf(a)) : 1e9
      const rb = rank.has(slugOf(b)) ? rank.get(slugOf(b)) : 1e9
      if (ra !== rb) return ra - rb
      return a.title.localeCompare(b.title, 'zh')
    })
  }
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
