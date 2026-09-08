<script setup>
import { computed } from 'vue'
import { useRoute, useData, withBase } from 'vitepress'
import { getLesson } from '../data/curriculum'
import { isSectionId } from '../data/sections'

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
      先修：<a :href="href(lesson.prereq)">{{ lesson.prereq.title }}</a>
    </p>
    <p v-else-if="variant === 'header'" class="lesson-prereq">本课为该课程第一课，后课默认已经读完这里。</p>
    <p v-if="lesson.next" class="lesson-next">
      下一课：<a :href="href(lesson.next)">{{ lesson.next.title }}</a>
    </p>
    <p v-else-if="variant === 'footer'" class="lesson-next">
      {{ lesson.appendix ? '本附录到此结束。' : '主干课序到此结束。' }}
    </p>
  </nav>
</template>
