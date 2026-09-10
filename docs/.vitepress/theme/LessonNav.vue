<script setup>
import { computed } from 'vue'
import { useRoute, useData, withBase } from 'vitepress'
import { getLesson } from '../data/curriculum'
import { isSectionId } from '../data/sections'
import { data as posts } from '../posts.data'

const props = defineProps({
  variant: { type: String, default: 'header' },
})

const route = useRoute()
const { page } = useData()

const lesson = computed(() => {
  const section = page.value.frontmatter?.section
  if (!isSectionId(section)) return null
  const segs = route.path.replace(/\.html$/, '').replace(/\/$/, '').split('/').filter(Boolean)
  const slug = segs[segs.length - 1]
  if (!slug || slug === section || slug === 'index') return null
  return getLesson(section, slug)
})

const written = computed(() => {
  const slugs = new Set()
  if (!lesson.value) return slugs
  for (const p of posts) {
    if (p.section !== lesson.value.section) continue
    const slug = p.url.replace(/\/$/, '').split('/').filter(Boolean).pop()
    if (slug) slugs.add(slug)
  }
  return slugs
})

function isWritten(ref) {
  return Boolean(ref && written.value.has(ref.slug))
}

function href(ref) {
  if (!ref || !lesson.value) return ''
  return withBase(`/${lesson.value.section}/${ref.slug}/`)
}
</script>

<template>
  <nav v-if="lesson" class="lesson-nav" :class="'lesson-nav--' + variant" aria-label="课程序列">
    <p class="lesson-path">
      {{ lesson.appendix ? '附录' : '课程' }} {{ lesson.course
      }}<template v-if="lesson.unit"> · 单元 {{ lesson.unit }}</template
      ><template v-if="lesson.sequence"> · 课序 {{ lesson.sequence }}</template>
    </p>
    <p class="lesson-progress">第 {{ lesson.indexInCourse }} / {{ lesson.courseSize }} 课</p>
    <p v-if="variant === 'header' && lesson.prereq" class="lesson-prereq">
      先修：<a v-if="isWritten(lesson.prereq)" :href="href(lesson.prereq)">{{ lesson.prereq.title }}</a
      ><span v-else>{{ lesson.prereq.title }}（待写）</span>
    </p>
    <p v-else-if="variant === 'header'" class="lesson-prereq">本课为该课程第一课，后课默认已经读完这里。</p>
    <p v-if="lesson.next" class="lesson-next">
      下一课：<a v-if="isWritten(lesson.next)" :href="href(lesson.next)">{{ lesson.next.title }}</a
      ><span v-else>{{ lesson.next.title }}（待写）</span>
    </p>
    <p v-else-if="variant === 'footer'" class="lesson-next">
      {{ lesson.appendix ? '本附录到此结束。' : '主干课序到此结束。' }}
    </p>
  </nav>
</template>
