<script setup>
import { ref, computed, watch, nextTick } from 'vue'
import { useRouter, withBase } from 'vitepress'

const props = defineProps({
  open: { type: Boolean, required: true },
})
const emit = defineEmits(['close'])

const router = useRouter()
const query = ref('')
const active = ref(0)
const listEl = ref(null)
const inputEl = ref(null)
const entries = ref(null)
let loading = null

const results = computed(() => {
  const items = entries.value
  const q = query.value.trim().toLowerCase()
  if (!items) return []
  if (!q) return items.slice(0, 12)
  const tokens = q.split(/\s+/)
  const scored = []
  for (let i = 0; i < items.length; i++) {
    const e = items[i]
    const title = e.title.toLowerCase()
    const path = (e.path || '').toLowerCase()
    const slug = e.url.toLowerCase()
    let ok = true
    let score = 0
    for (const t of tokens) {
      if (title.startsWith(t)) score += 8
      else if (title.includes(t)) score += 5
      else if (path.includes(t)) score += 2
      else if (slug.includes(t)) score += 1
      else { ok = false; break }
    }
    if (ok) scored.push([score, i])
  }
  scored.sort((a, b) => b[0] - a[0] || a[1] - b[1])
  return scored.slice(0, 24).map(([, i]) => ({ ...items[i], _i: i }))
})

watch(results, () => {
  active.value = 0
})

watch(
  () => props.open,
  async (open) => {
    if (!open) return
    query.value = ''
    active.value = 0
    if (!entries.value && !loading) loading = import('../search.data').then((m) => (entries.value = m.data))
    await loading
    nextTick(() => inputEl.value?.focus())
  },
)

function onKeydown(e) {
  if (e.key === 'Escape') {
    emit('close')
    return
  }
  if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
    e.preventDefault()
    const n = results.value.length
    if (!n) return
    active.value = (active.value + (e.key === 'ArrowDown' ? 1 : n - 1)) % n
    nextTick(() => {
      listEl.value?.querySelector('.search-item.active')?.scrollIntoView({ block: 'nearest' })
    })
    return
  }
  if (e.key === 'Enter') {
    const hit = results.value[active.value]
    if (hit) go(hit)
  }
}

function go(hit) {
  emit('close')
  router.go(withBase(hit.url))
}
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="search-overlay" @click.self="emit('close')">
      <div class="search-modal" role="dialog" aria-label="全站搜索">
        <div class="search-head">
          <input
            ref="inputEl"
            v-model="query"
            class="search-input"
            type="text"
            placeholder="搜课名、课程、单元、课序…（Esc 关闭）"
            @keydown="onKeydown"
          />
          <button class="search-close" type="button" aria-label="关闭" @click="emit('close')">Esc</button>
        </div>
        <ol ref="listEl" class="search-list">
          <li
            v-for="(hit, i) in results"
            :key="hit.url"
            :class="['search-item', { active: i === active }]"
            @mouseenter="active = i"
            @click="go(hit)"
          >
            <p class="search-item-title">
              <span class="search-item-section">{{ hit.section }}</span>
              <span v-if="hit.appendix" class="search-item-appendix">附录</span>
              {{ hit.title }}
            </p>
            <p class="search-item-meta">
              {{ hit.path }}
              <template v-if="hit.courseSize">（第 {{ hit.indexInCourse }} / {{ hit.courseSize }} 课）</template>
            </p>
          </li>
          <li v-if="!results.length" class="search-empty">没有匹配的课。试试更短的关键词。</li>
        </ol>
        <p class="search-foot">↑↓ 选择 · Enter 打开 · Esc 关闭</p>
      </div>
    </div>
  </Teleport>
</template>
