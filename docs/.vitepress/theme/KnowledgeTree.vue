<script setup>
import { computed } from 'vue'
import { trees, leafCount } from '../data/trees'
import { data as posts } from '../posts.data'
import TreeList from './TreeList.vue'

const props = defineProps({
  section: { type: String, required: true },
})

const tree = computed(() => trees[props.section])

const written = computed(() => {
  const slugs = new Set()
  for (const p of posts) {
    if (p.section !== props.section) continue
    const slug = p.url.replace(/\/$/, '').split('/').filter(Boolean).pop()
    if (slug) slugs.add(slug)
  }
  return slugs
})

const total = computed(() => (tree.value ? leafCount(tree.value.children) : 0))
const have = computed(() => {
  let n = 0
  const walk = (nodes) => {
    for (const node of nodes) {
      if (node.kind === 'leaf' && node.slug && written.value.has(node.slug)) n++
      if (node.children) walk(node.children)
    }
  }
  if (tree.value) walk(tree.value.children)
  return n
})
</script>

<template>
  <div v-if="tree" class="kt">
    <p class="kt-desc">
      {{ tree.desc }}
      <span class="kt-count">{{ have }} / {{ total }} 篇</span>
    </p>
    <TreeList :nodes="tree.children" :section="section" :written="written" />
  </div>
</template>
