<script setup>
// 纯展示卡片：只收 props，不做任何内部响应式状态。
// 600+ 卡片随筛选重渲时叶子组件成本最低。
import meta from '../data/entertainment/meta.json'

const props = defineProps({
  item: { type: Object, required: true },
  cat: { type: String, required: true },
  isEn: { type: Boolean, default: false },
})

const catMeta = meta.categories[props.cat]

function g(key) {
  return meta.genres[props.cat][key] || { zh: key, en: key, emoji: '•' }
}
function a(key) {
  return meta.awards[props.cat][key] || { zh: key, en: key, emoji: '🏅' }
}
function regionMeta() {
  return meta.regions[props.item.region] || null
}
// 评分：按评分源顺序取存在的 1–2 个
function ratings() {
  const out = []
  for (const s of catMeta.ratingSources) {
    const v = props.item.rating?.[s.key]
    if (typeof v === 'number') out.push({ label: s.key === 'douban' ? (props.isEn ? s.en : s.label) : s.en, value: v })
    if (out.length === 2) break
  }
  return out
}
function awardTitle(key) {
  const m = a(key)
  return `${m.emoji} ${props.isEn ? m.en : m.zh}`
}
</script>

<template>
  <article class="ent-card">
    <header class="ent-card-top">
      <span class="ent-year-badge">{{ item.year }}</span>
      <span class="ent-region" :title="regionMeta()?.en">{{ regionMeta()?.flag || '🌐' }} {{ item.region }}</span>
      <span v-if="item.awards.length" class="ent-awards">
        <span
          v-for="aw in item.awards.slice(0, 3)"
          :key="aw"
          class="ent-award"
          :title="awardTitle(aw)"
          >{{ a(aw).emoji }}</span
        >
        <span
          v-if="item.awards.length > 3"
          class="ent-award-more"
          :title="item.awards.map(awardTitle).join(' · ')"
          >+{{ item.awards.length - 3 }}</span
        >
        <span class="ent-ribbon" aria-hidden="true">🏆</span>
      </span>
    </header>

    <h4 class="ent-card-title">{{ isEn ? item.en : item.title }}</h4>
    <p class="ent-card-sub">{{ isEn ? item.title : item.en }}</p>
    <p class="ent-card-creator">{{ item.creator }}</p>

    <div class="ent-card-genres">
      <span v-for="ge in item.genres" :key="ge" class="ent-genre-chip"
        >{{ g(ge).emoji }} {{ isEn ? g(ge).en : g(ge).zh }}</span
      >
    </div>

    <div v-if="ratings().length" class="ent-card-rating">
      <span v-for="r in ratings()" :key="r.label" class="ent-rating"
        >{{ r.label }} <b>{{ r.value }}</b></span
      >
    </div>

    <p class="ent-card-note">{{ isEn ? item.noteEn : item.note }}</p>
  </article>
</template>
