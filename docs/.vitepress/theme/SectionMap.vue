<script setup>
import { computed } from 'vue'
import { withBase } from 'vitepress'
import { trees, leafCount } from '../data/trees'
import { SECTION_IDS, sectionMeta } from '../data/sections'
import { data as posts } from '../posts.data'

const written = computed(() => {
  const bySection = Object.fromEntries(SECTION_IDS.map((id) => [id, new Set()]))
  for (const p of posts) {
    const slug = p.url.replace(/\/$/, '').split('/').filter(Boolean).pop()
    if (slug && bySection[p.section]) bySection[p.section].add(slug)
  }
  return bySection
})

function writtenCount(id) {
  let n = 0
  const slugs = written.value[id]
  const walk = (nodes) => {
    for (const node of nodes) {
      if (node.kind === 'leaf' && node.slug && slugs.has(node.slug)) n++
      if (node.children) walk(node.children)
    }
  }
  walk(trees[id].children)
  return n
}

const rows = computed(() =>
  SECTION_IDS.map((id) => ({
    id,
    name: sectionMeta[id].name,
    path: withBase(sectionMeta[id].path),
    blurb: sectionMeta[id].blurb,
    have: writtenCount(id),
    total: leafCount(trees[id].children),
  })),
)
</script>

<template>
  <ul class="section-map">
    <li v-for="row in rows" :key="row.id">
      <a class="section-map-name" :href="row.path">{{ row.name }}</a>
      <p class="section-map-blurb">{{ row.blurb }}</p>
      <span class="section-map-count">{{ row.have }} / {{ row.total }} 课</span>
    </li>
  </ul>
</template>
