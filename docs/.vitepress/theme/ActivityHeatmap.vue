<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { useData } from 'vitepress'
import { data as activity } from '../data/activity.data'
import type { ActivityDay } from '../data/activity.data'

interface ActivityWeek {
  days: ActivityDay[]
  startDate: string
}

defineProps<{ title: string }>()
const { lang } = useData()
const english = computed(() => lang.value === 'en-US')
const scrollContainer = ref<HTMLElement>()
const tooltip = ref({ visible: false, text: '', x: 0, y: 0 })

const weeks = computed<ActivityWeek[]>(() => {
  const groups: ActivityWeek[] = []
  for (let index = 0; index < activity.days.length; index += 7) {
    const days = activity.days.slice(index, index + 7)
    groups.push({ days, startDate: days[0].date })
  }
  return groups
})
const monthLabels = computed(() => {
  const labels: { index: number; text: string }[] = []
  weeks.value.forEach((week, index) => {
    const marker = index === 0 ? week.days[0] : week.days.find(day => day.date.endsWith('-01'))
    if (marker) labels.push({
      index,
      text: english.value
        ? new Intl.DateTimeFormat('en-US', { month: 'short', timeZone: 'UTC' }).format(new Date(`${marker.date}T00:00:00Z`))
        : `${Number(marker.date.slice(5, 7))}月`,
    })
  })
  return labels.filter((label, index) => index === labels.length - 1 || labels[index + 1].index - label.index >= 3)
})

function level(count: number): number {
  if (count === 0) return 0
  if (count === 1) return 1
  if (count <= 3) return 2
  if (count <= 7) return 3
  return 4
}
function formatDate(date: string): string {
  return english.value
    ? new Intl.DateTimeFormat('en-US', { year: 'numeric', month: 'short', day: 'numeric', timeZone: 'UTC' }).format(new Date(`${date}T00:00:00Z`))
    : `${Number(date.slice(0, 4))}年${Number(date.slice(5, 7))}月${Number(date.slice(8, 10))}日`
}
function dayLabel(day: ActivityDay): string {
  return `${formatDate(day.date)} · ${day.count} ${english.value ? 'commits' : '次更新'}`
}
function showTooltip(event: MouseEvent | FocusEvent, text: string) {
  const target = event.currentTarget as HTMLElement
  const bounds = target.getBoundingClientRect()
  tooltip.value = {
    visible: true,
    text,
    x: Math.min(Math.max(bounds.left + bounds.width / 2, 112), window.innerWidth - 112),
    y: bounds.top - 9,
  }
}
function hideTooltip() { tooltip.value.visible = false }
function scrollToLatest() {
  const container = scrollContainer.value
  if (container) container.scrollLeft = container.scrollWidth
}
onMounted(scrollToLatest)
</script>

<template>
  <div class="activity-log">
    <div class="activity-heading">
      <div>
        <p class="activity-eyebrow" lang="en">ACTIVITY LOG</p>
        <h2 id="activity-title">{{ title }}</h2>
      </div>
      <dl class="activity-stats" :aria-label="english ? 'Repository activity' : '累计站点更新'">
        <div><dt>{{ english ? 'Commits' : '累计提交' }}</dt><dd>{{ activity.total }}<span>{{ english ? '' : '次' }}</span></dd></div>
        <div><dt>{{ english ? 'Active days' : '活跃天数' }}</dt><dd>{{ activity.activeDays }}<span>{{ english ? '' : '天' }}</span></dd></div>
      </dl>
    </div>

    <div ref="scrollContainer" class="activity-scroll" @scroll.passive="hideTooltip">
      <div class="activity-calendar">
        <div class="activity-months" aria-hidden="true">
          <span v-for="month in monthLabels" :key="`${month.index}-${month.text}`" :style="{ gridColumn: month.index + 1 }">{{ month.text }}</span>
        </div>
        <div class="activity-weekdays" aria-hidden="true">
          <span style="grid-row: 2">{{ english ? 'M' : '一' }}</span><span style="grid-row: 4">{{ english ? 'W' : '三' }}</span><span style="grid-row: 6">{{ english ? 'F' : '五' }}</span>
        </div>
        <div class="activity-grid" :aria-label="english ? 'Daily repository activity for the past year' : '过去一年每日站点更新点阵图'">
          <div v-for="week in weeks" :key="week.startDate" class="activity-week">
            <span
              v-for="day in week.days" :key="day.date" class="activity-cell"
              :class="[`level-${level(day.count)}`, { future: day.future }]"
              :role="day.future ? undefined : 'img'" :aria-label="day.future ? undefined : dayLabel(day)"
              :aria-hidden="day.future ? 'true' : undefined" :tabindex="!day.future && day.count ? 0 : undefined"
              @mouseenter="!day.future && showTooltip($event, dayLabel(day))" @mouseleave="hideTooltip"
              @focus="!day.future && showTooltip($event, dayLabel(day))" @blur="hideTooltip"
            />
          </div>
        </div>
      </div>
    </div>

    <div class="activity-footer">
      <p>{{ english ? 'Past 52 weeks · Git commits on source; not a measure of study time or mastery.' : '过去 52 周 · source 分支 Git 提交；不代表学习时长或掌握程度。' }}</p>
      <div class="activity-legend" :aria-label="english ? 'Darker colors represent more commits' : '颜色由浅到深表示更新次数由少到多'"><span>{{ english ? 'Less' : '少' }}</span><i v-for="step in 5" :key="step" :class="`level-${step - 1}`" aria-hidden="true" /><span>{{ english ? 'More' : '多' }}</span></div>
    </div>
    <Teleport to="body">
      <div v-if="tooltip.visible" class="activity-tooltip" role="tooltip" :style="{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }">{{ tooltip.text }}</div>
    </Teleport>
  </div>
</template>
