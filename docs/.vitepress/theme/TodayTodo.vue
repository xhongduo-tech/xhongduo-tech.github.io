<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useData } from 'vitepress'
import { data as plan } from '../data/study-plan.data'

const STORAGE_KEY = 'xhd-study-todos-v1'
const { lang } = useData()
const completed = ref<string[]>([])
const doneCount = computed(() => plan.tasks.filter(task => completed.value.includes(task.id)).length)
const hours = computed(() => (plan.totalMinutes ?? 0) / 60)
const english = computed(() => lang.value === 'en-US')

function save() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(completed.value))
  } catch {
    // The checkbox still works for this page view when storage is unavailable.
  }
}

function toggle(taskId: string) {
  completed.value = completed.value.includes(taskId)
    ? completed.value.filter(id => id !== taskId)
    : [...completed.value, taskId]
  save()
}

onMounted(() => {
  try {
    const stored = JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
    if (Array.isArray(stored)) {
      const currentIds = new Set(plan.tasks.map(task => task.id))
      completed.value = stored.filter(id => typeof id === 'string' && currentIds.has(id))
    }
  } catch {
    completed.value = []
  }
})
</script>

<template>
  <section v-if="plan.available" class="today-plan" aria-labelledby="today-title">
    <div class="today-heading">
      <div>
        <p class="today-eyebrow" lang="en">TODAY / {{ String(plan.week).padStart(2, '0') }}</p>
        <h2 id="today-title">{{ english ? "Today's plan" : '今日计划' }}</h2>
      </div>
      <div class="today-meta">
        <time :datetime="plan.date">{{ plan.date }}</time>
        <span>{{ plan.topic }} · {{ english ? `${hours} h` : `${hours} 小时` }}</span>
      </div>
    </div>

    <div class="today-progress">
      <progress :value="doneCount" :max="plan.tasks.length" :aria-label="english ? `${doneCount} of ${plan.tasks.length} tasks completed` : `今日已完成 ${doneCount} 项，共 ${plan.tasks.length} 项`" />
      <span>{{ doneCount }} / {{ plan.tasks.length }}</span>
    </div>

    <ul class="today-tasks">
      <li v-for="task in plan.tasks" :key="task.id" :class="{ completed: completed.includes(task.id) }">
        <label>
          <input type="checkbox" :checked="completed.includes(task.id)" @change="toggle(task.id)">
          <span>{{ task.title }}</span>
          <small>{{ task.minutes }} min</small>
        </label>
      </li>
    </ul>
    <p class="today-note">{{ english ? 'Saved only in this browser. Only today’s tasks are shown.' : '完成状态仅保存在当前浏览器；每天只展示当天任务。' }}</p>
  </section>
</template>
