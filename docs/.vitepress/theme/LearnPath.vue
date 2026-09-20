<script setup>
import { computed, ref, onMounted, watch } from 'vue'
import { withBase, useRoute } from 'vitepress'
import { getPath } from '../data/paths'
import { getLesson } from '../data/curriculum'

const props = defineProps({
  pathId: { type: String, required: true },
})

const route = useRoute()
const path = computed(() => getPath(props.pathId))

function resolve(section, slug) {
  const lesson = getLesson(section, slug)
  return {
    section,
    slug,
    title: lesson ? lesson.title : slug,
    path: lesson ? [lesson.course, lesson.unit, lesson.sequence].filter(Boolean).join(' · ') : '',
    progress: lesson && lesson.courseSize ? `第 ${lesson.indexInCourse} / ${lesson.courseSize} 课` : '',
    href: withBase(`/${section}/${slug}/`),
  }
}

const rows = computed(() =>
  (path.value?.stages || []).map((stage) => ({
    name: stage.name,
    note: stage.note,
    lessons: stage.items.map(([s, slug]) => resolve(s, slug)),
  })),
)
const flat = computed(() => rows.value.flatMap((r) => r.lessons))

const STORAGE = 'path-progress'
const done = ref(new Set())

function storageKey() {
  return `${STORAGE}:${props.pathId}`
}

function load() {
  try {
    const raw = localStorage.getItem(storageKey())
    done.value = new Set(raw ? JSON.parse(raw) : [])
  } catch {
    done.value = new Set()
  }
}

function persist() {
  try {
    localStorage.setItem(storageKey(), JSON.stringify([...done.value]))
  } catch {}
}

function toggle(slug) {
  const next = new Set(done.value)
  if (next.has(slug)) next.delete(slug)
  else next.add(slug)
  done.value = next
  persist()
}

function keyOf(item) {
  return item.section + ':' + item.slug
}

onMounted(load)
watch(() => route.path, load)

const doneCount = computed(() => flat.value.filter((l) => done.value.has(keyOf(l))).length)
const total = computed(() => flat.value.length)
const pct = computed(() => (total.value ? Math.round((100 * doneCount.value) / total.value) : 0))
</script>

<template>
  <div v-if="path" class="learn-path">
    <div class="lp-head">
      <p class="lp-goal">{{ path.goal }}</p>
      <p class="lp-progress-line">
        我的进度：<strong>{{ doneCount }}</strong> / {{ total }} 课（{{ pct }}%）
        <span class="lp-progress-hint">勾选只保存在本机浏览器。</span>
      </p>
      <div class="lp-bar"><span :style="{ width: pct + '%' }"></span></div>
    </div>

    <section v-for="stage in rows" :key="stage.name" class="lp-stage">
      <h2>{{ stage.name }}</h2>
      <p class="lp-stage-note">{{ stage.note }}</p>
      <ol class="lp-lessons">
        <li
          v-for="lesson in stage.lessons"
          :key="lesson.slug"
          :class="{ done: done.has(keyOf(lesson)) }"
        >
          <button
            class="lp-check"
            type="button"
            :aria-label="done.has(keyOf(lesson)) ? '标记未读' : '标记已读'"
            @click="toggle(keyOf(lesson))"
          >
            {{ done.has(keyOf(lesson)) ? '✓' : '' }}
          </button>
          <a class="lp-link" :href="lesson.href">{{ lesson.title }}</a>
          <span class="lp-meta">{{ lesson.path }}<template v-if="lesson.progress"> · {{ lesson.progress }}</template></span>
        </li>
      </ol>
    </section>
  </div>
</template>
