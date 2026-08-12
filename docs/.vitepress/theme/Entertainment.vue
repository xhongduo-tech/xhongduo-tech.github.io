<script setup>
// 娱乐大典：数据驱动的明快卡片浏览页。
// 数据来自 ../data/entertainment/{movies,games,music,photography}.json 与 meta.json（均由
// scripts/gen-entertainment.mjs 生成）。语言检测沿用全站 relativePath.startsWith('en/')。
// SSR 安全：setup 无 window/document，全静态 JSON + ref/reactive/computed。
import { computed, reactive, ref } from 'vue'
import { useData } from 'vitepress'
import EntertainmentCard from './EntertainmentCard.vue'
import meta from '../data/entertainment/meta.json'
import movies from '../data/entertainment/movies.json'
import games from '../data/entertainment/games.json'
import music from '../data/entertainment/music.json'
import photography from '../data/entertainment/photography.json'

const { page } = useData()
const isEn = computed(() => page.value.relativePath.startsWith('en/'))

const DATA = { movies, games, music, photography }
const PAGE_SIZE = 24

const catList = Object.entries(meta.categories).map(([key, m]) => ({ key, ...m }))

// 每类独立筛选态，切换 tab 时保留
function emptyFilter() {
  return { q: '', genres: [], decade: null, sort: 'yearDesc', onlyAwards: false, page: 1 }
}
const filters = reactive({
  movies: emptyFilter(),
  games: emptyFilter(),
  music: emptyFilter(),
  photography: emptyFilter(),
})
const active = ref('movies')

// ---------- 每类筛选 ----------
function regionLabel(region) {
  const r = meta.regions[region]
  if (!r) return region
  return isEn.value ? r.en : r.zh
}

function filtered(cat) {
  const f = filters[cat]
  const q = f.q.trim().toLowerCase()
  const genreSet = new Set(f.genres)
  const items = DATA[cat].items.filter((it) => {
    if (f.onlyAwards && !it.awards.length) return false
    if (genreSet.size && !it.genres.some((g) => genreSet.has(g))) return false
    if (f.decade !== null && (it.year < f.decade || it.year >= f.decade + 10)) return false
    if (q) {
      const hay = `${it.title} ${it.en} ${it.creator} ${regionLabel(it.region)}`.toLowerCase()
      if (!hay.includes(q)) return false
    }
    return true
  })
  return sortItems(cat, items, f.sort)
}

function sortItems(cat, items, sort) {
  const arr = [...items]
  if (sort === 'yearAsc') {
    arr.sort((a, b) => a.year - b.year || a.en.localeCompare(b.en))
  } else if (sort === 'yearDesc') {
    arr.sort((a, b) => b.year - a.year || a.en.localeCompare(b.en))
  } else if (sort === 'rating') {
    arr.sort((a, b) => (bestRating(cat, b) ?? -1) - (bestRating(cat, a) ?? -1) || b.year - a.year)
  }
  return arr
}

// 取该类最佳评分源（按 meta 顺序第一个存在的），统一到 0–10 便于跨源比较
function bestRating(cat, it) {
  for (const s of meta.categories[cat].ratingSources) {
    const v = it.rating?.[s.key]
    if (typeof v === 'number') return s.max === 10 ? v : v / 10
  }
  return null
}

function decadeKeys(cat) {
  return Object.keys(DATA[cat].stats.decades)
    .map(Number)
    .sort((a, b) => a - b)
}

function visible(cat) {
  return filtered(cat).slice(0, filters[cat].page * PAGE_SIZE)
}
function hasMore(cat) {
  return filtered(cat).length > filters[cat].page * PAGE_SIZE
}
function remaining(cat) {
  return filtered(cat).length - filters[cat].page * PAGE_SIZE
}
function more(cat) {
  filters[cat].page += 1
}

function toggleGenre(cat, g) {
  const i = filters[cat].genres.indexOf(g)
  if (i >= 0) filters[cat].genres.splice(i, 1)
  else filters[cat].genres.push(g)
  resetPage(cat)
}
function setDecade(cat, d) {
  filters[cat].decade = d
  resetPage(cat)
}
function setSort(cat, s) {
  filters[cat].sort = s
  resetPage(cat)
}
function toggleAwards(cat) {
  filters[cat].onlyAwards = !filters[cat].onlyAwards
  resetPage(cat)
}
function resetPage(cat) {
  filters[cat].page = 1
}

// ---------- Hero 统计 ----------
const totalEntries = computed(() =>
  catList.reduce((s, c) => s + (DATA[c.key].count || 0), 0),
)
const totalAwards = computed(() =>
  catList.reduce((s, c) => s + (DATA[c.key].stats?.totalAwards || 0), 0),
)
const genresCovered = computed(() =>
  Object.values(meta.genres).reduce((s, g) => s + Object.keys(g).length, 0),
)
const yearRange = computed(() => {
  const ranges = catList
    .map((c) => DATA[c.key].stats?.yearRange)
    .filter((r) => r && r.length === 2)
  if (!ranges.length) return null
  return [Math.min(...ranges.map((r) => r[0])), Math.max(...ranges.map((r) => r[1]))]
})

// ---------- 选项卡键盘导航（ARIA tablist） ----------
function onTabKeydown(e) {
  const idx = catList.findIndex((c) => c.key === active.value)
  let next = null
  if (e.key === 'ArrowRight') next = (idx + 1) % catList.length
  else if (e.key === 'ArrowLeft') next = (idx - 1 + catList.length) % catList.length
  else if (e.key === 'Home') next = 0
  else if (e.key === 'End') next = catList.length - 1
  if (next !== null) {
    e.preventDefault()
    active.value = catList[next].key
    // 让焦点跟随激活 tab
    const tab = e.currentTarget.querySelector(`[data-key="${catList[next].key}"]`)
    tab?.focus()
  }
}

function t(key) {
  return isEn.value ? key.en : key.zh
}
</script>

<template>
  <div class="ent">
    <!-- Hero -->
    <section class="ent-hero">
      <div class="ent-hero-title" role="heading" aria-level="1">
        {{ isEn ? '🎬 Entertainment Compendium' : '🎬 娱乐大典' }}
        <span class="ent-hero-en">{{ isEn ? 'FILM · MUSIC · GAMES · PHOTO' : '电影 · 音乐 · 游戏 · 摄影' }}</span>
      </div>
      <p class="ent-hero-sub">
        {{
          isEn
            ? 'Every film, album and game here earned its place — a worldwide compendium of the award-winning and the beloved, sorted by the years that shaped them.'
            : '这里收录全世界历届大奖得主与高分经典：按流派分类、按年份排序，每条附一句它为什么值得被记住。'
        }}
      </p>
      <div class="ent-hero-stats">
        <div class="ent-hs"><b>{{ totalEntries }}</b><span>{{ isEn ? 'works curated' : '收录作品' }}</span></div>
        <div class="ent-hs"><b>{{ totalAwards }}</b><span>{{ isEn ? 'major awards' : '大奖累计' }}</span></div>
        <div class="ent-hs"><b>{{ genresCovered }}</b><span>{{ isEn ? 'sub-genres' : '细分流派' }}</span></div>
        <div class="ent-hs" v-if="yearRange"><b>{{ yearRange[0] }}–{{ yearRange[1] }}</b><span>{{ isEn ? 'years covered' : '年代跨度' }}</span></div>
      </div>
    </section>

    <!-- 吸顶选项卡 -->
    <nav class="ent-tabs" role="tablist" :aria-label="isEn ? 'Categories' : '分类'" :data-cat-active="active" @keydown="onTabKeydown">
      <button
        v-for="c in catList"
        :key="c.key"
        :data-key="c.key"
        class="ent-tab"
        :class="{ active: active === c.key }"
        role="tab"
        :aria-selected="active === c.key"
        :aria-controls="`ent-panel-${c.key}`"
        :id="`ent-tab-${c.key}`"
        @click="active = c.key"
      >
        <span class="ent-tab-emoji">{{ c.emoji }}</span>
        <span class="ent-tab-name">{{ t(c) }}</span>
        <span class="ent-tab-count">{{ DATA[c.key].count || 0 }}</span>
      </button>
    </nav>

    <!-- 每类面板 -->
    <template v-for="c in catList" :key="c.key">
    <section
      v-if="active === c.key"
      :id="`ent-panel-${c.key}`"
      :data-cat="c.key"
      role="tabpanel"
      :aria-labelledby="`ent-tab-${c.key}`"
      class="ent-panel"
    >
      <div class="ent-controls">
        <input
          v-model.trim="filters[c.key].q"
          class="ent-search"
          type="search"
          :placeholder="isEn ? 'Search title / creator / region…' : '搜索片名 / 创作者 / 地区…'"
          :aria-label="isEn ? 'Search' : '搜索'"
          @input="resetPage(c.key)"
        />

        <div class="ent-chip-row" :aria-label="isEn ? 'Genres' : '流派'">
          <button
            v-for="(g, gk) in meta.genres[c.key]"
            :key="gk"
            class="ent-chip"
            :class="{ on: filters[c.key].genres.includes(gk) }"
            :aria-pressed="filters[c.key].genres.includes(gk)"
            @click="toggleGenre(c.key, gk)"
          >
            {{ g.emoji }} {{ isEn ? g.en : g.zh }}
          </button>
        </div>

        <div class="ent-chip-row" :aria-label="isEn ? 'Decades' : '年代'">
          <button
            class="ent-chip"
            :class="{ on: filters[c.key].decade === null }"
            :aria-pressed="filters[c.key].decade === null"
            @click="setDecade(c.key, null)"
          >
            {{ isEn ? 'All years' : '全部年代' }}
          </button>
          <button
            v-for="d in decadeKeys(c.key)"
            :key="d"
            class="ent-chip"
            :class="{ on: filters[c.key].decade === d }"
            :aria-pressed="filters[c.key].decade === d"
            @click="setDecade(c.key, d)"
          >
            {{ d }}s
          </button>
        </div>

        <div class="ent-sort-row">
          <span class="ent-label">{{ isEn ? 'Sort' : '排序' }}</span>
          <button
            class="ent-sort-btn"
            :class="{ on: filters[c.key].sort === 'yearDesc' }"
            :aria-pressed="filters[c.key].sort === 'yearDesc'"
            @click="setSort(c.key, 'yearDesc')"
            >{{ isEn ? 'Newest' : '年份 ↓' }}</button
          >
          <button
            class="ent-sort-btn"
            :class="{ on: filters[c.key].sort === 'yearAsc' }"
            :aria-pressed="filters[c.key].sort === 'yearAsc'"
            @click="setSort(c.key, 'yearAsc')"
            >{{ isEn ? 'Oldest' : '年份 ↑' }}</button
          >
          <button
            class="ent-sort-btn"
            :class="{ on: filters[c.key].sort === 'rating' }"
            :aria-pressed="filters[c.key].sort === 'rating'"
            @click="setSort(c.key, 'rating')"
            >{{ isEn ? 'Rating' : '评分 ↓' }}</button
          >
          <button
            class="ent-sort-btn"
            :class="{ on: filters[c.key].onlyAwards }"
            :aria-pressed="filters[c.key].onlyAwards"
            @click="toggleAwards(c.key)"
            >🏆 {{ isEn ? 'Awards only' : '仅看获奖' }}</button
          >
          <span class="ent-result-count">{{ filtered(c.key).length }} {{ isEn ? 'results' : '条' }}</span>
        </div>
      </div>

      <div class="ent-grid">
        <EntertainmentCard
          v-for="it in visible(c.key)"
          :key="it.id"
          :item="it"
          :cat="c.key"
          :is-en="isEn"
        />
      </div>

      <button v-if="hasMore(c.key)" class="ent-more" @click="more(c.key)">
        {{ isEn ? 'Show more' : '加载更多' }}（{{ remaining(c.key) }}）
      </button>
      <p v-else-if="filtered(c.key).length === 0" class="ent-empty">
        {{ isEn ? '😢 Nothing matches — try clearing a filter.' : '😢 没有匹配的作品——试试清空筛选。' }}
      </p>
    </section>
    </template>
  </div>
</template>
