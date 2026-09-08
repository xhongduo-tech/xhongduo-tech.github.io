<script>
import { defineComponent, reactive } from 'vue'
import { withBase } from 'vitepress'
import { leafCount } from '../data/trees/schema'
import { getLesson } from '../data/curriculum'

const KIND = { branch: '课程', mainline: '单元', group: '课序' }

export default defineComponent({
  name: 'TreeList',
  props: {
    nodes: { type: Array, required: true },
    section: { type: String, required: true },
    written: { type: Object, required: true },
    depth: { type: Number, default: 0 },
  },
  setup(props) {
    const open = reactive({})

    function keyOf(node, i) {
      return (node.slug || node.name) + ':' + props.depth + ':' + i
    }

    function isOpen(node, i) {
      const k = keyOf(node, i)
      if (open[k] === undefined) return node.kind === 'branch'
      return open[k]
    }

    function toggle(node, i) {
      const k = keyOf(node, i)
      open[k] = !isOpen(node, i)
    }

    function href(slug) {
      if (!props.written.has(slug)) return null
      return withBase(`/${props.section}/${slug}/`)
    }

    function kindLabel(node) {
      if (node.appendix && node.kind === 'branch') return '附录'
      return KIND[node.kind]
    }

    function lessonNum(slug) {
      const lesson = getLesson(props.section, slug)
      return lesson ? lesson.indexInSequence : 0
    }

    return { KIND, leafCount, isOpen, toggle, href, kindLabel, lessonNum }
  },
})
</script>

<template>
  <div class="kt-list" :data-depth="depth">
    <div
      v-for="(node, i) in nodes"
      :key="(node.slug || node.name) + '-' + i"
      :class="['kt-item', 'kt-' + node.kind, { 'kt-appendix': node.appendix }]"
    >
      <template v-if="node.kind === 'leaf'">
        <a v-if="href(node.slug)" class="kt-leaf" :href="href(node.slug)"
          ><span v-if="lessonNum(node.slug)" class="kt-num">{{ lessonNum(node.slug) }}.</span
          >{{ node.name }}</a
        >
        <span v-else class="kt-pending"
          ><span v-if="lessonNum(node.slug)" class="kt-num">{{ lessonNum(node.slug) }}.</span
          >{{ node.name }}</span
        >
        <span v-if="node.alsoIn?.length" class="kt-also">亦见 {{ node.alsoIn.join('、') }}</span>
      </template>
      <template v-else>
        <button class="kt-head" type="button" @click="toggle(node, i)">
          <span class="kt-caret" aria-hidden="true">{{ isOpen(node, i) ? '▾' : '▸' }}</span>
          <span class="kt-kind">{{ kindLabel(node) }}</span>{{ ' ' }}<span class="kt-name">{{ node.name }}</span
          >{{ ' ' }}<span class="kt-n">{{ leafCount(node.children || []) }}</span>
        </button>
        <TreeList
          v-if="isOpen(node, i)"
          :nodes="node.children || []"
          :section="section"
          :written="written"
          :depth="depth + 1"
        />
      </template>
    </div>
  </div>
</template>
