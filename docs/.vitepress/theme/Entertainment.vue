<script setup>
import { computed } from 'vue'
import { useData } from 'vitepress'

const { page } = useData()
const isEn = computed(() => page.value.relativePath.startsWith('en/'))

// 四个品味维度：电影 / 音乐 / 游戏 / 摄影
// 每条 = 一份私人推荐，附一句为什么打动我。
const zh = [
  {
    cat: '电影',
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="16" rx="2"/><path d="M3 9h18M3 15h18M8 4v16M16 4v16"/></svg>',
    items: [
      { title: '银翼杀手 2049', meta: 'Denis Villeneuve · 2017', note: '关于「什么是人」的视觉诗——也关乎工程师对造物的责任。' },
      { title: '降临', meta: 'Denis Villeneuve · 2016', note: '非线性语言改写时间感知，语言学与决定论的温柔碰撞。' },
      { title: '盗梦空间', meta: 'Christopher Nolan · 2010', note: '嵌套状态机与递归出口，是工程师最熟悉的浪漫。' },
      { title: '楚门的世界', meta: 'Peter Weir · 1998', note: '当整个世界都是被设计的系统，逃离才是第一个真实测试。' },
      { title: '千与千寻', meta: '宫崎骏 · 2001', note: '名字是身份的键——失去名字就失去对自身的引用。' },
    ],
  },
  {
    cat: '音乐',
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/></svg>',
    items: [
      { title: '坂本龍一 · async', meta: '2017', note: '不对称节拍与沉默，写代码到深夜时的背景音。' },
      { title: '久石让 · Summer', meta: '北野武《菊次郎的夏天》', note: '极简动机的反复变奏，证明好旋律胜过一切编排。' },
      { title: 'Pink Floyd · The Dark Side of the Moon', meta: '1973', note: '一张关于时间、金钱与精神崩塌的概念专辑，工程级精密。' },
      { title: 'Joep Beving · Henosis', meta: '2019', note: '新古典极简主义，长上下文思考时的稳定锚点。' },
    ],
  },
  {
    cat: '游戏',
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="20" height="12" rx="4"/><path d="M7 12h4M9 10v4M16 12h.01M18 14h.01"/></svg>',
    items: [
      { title: 'Outer Wilds', meta: 'Mobius · 2019', note: '用知识而非数值推进的探索——信息即唯一的升级。' },
      { title: 'Portal 2', meta: 'Valve · 2011', note: '空间推理与物理规则的优雅题集，关卡设计教科书。' },
      { title: '极乐迪斯科', meta: 'ZA/UM · 2019', note: '把人格拆成技能互相博弈，RPG 系统即哲学系统。' },
      { title: '塞尔达：旷野之息', meta: '任天堂 · 2017', note: '涌现式玩法设计——规则少而组合无穷。' },
    ],
  },
  {
    cat: '摄影',
    icon: '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7h3l2-2h8l2 2h3v13H3z"/><circle cx="12" cy="13" r="4"/></svg>',
    items: [
      { title: '街头 · 决定性瞬间', meta: 'Cartier-Bresson 体系', note: '在流动场景里等待几何与时机同时闭合的那一帧。' },
      { title: '黑白 · 影调结构', meta: 'Ansel Adams · Zone System', note: '把动态范围当作分区曝光的工程问题来解。' },
      { title: '极简 · 负空间', meta: '杉本博司 · 海景', note: '长曝光把时间压缩成一张静态的灰阶。' },
      { title: '夜景 · 长曝光', meta: '城市与星空', note: '信噪比与三脚架的耐心的回报。' },
    ],
  },
]

const en = [
  {
    cat: 'Film',
    icon: zh[0].icon,
    items: [
      { title: 'Blade Runner 2049', meta: 'Denis Villeneuve · 2017', note: 'A visual poem on what makes us human — and the engineer’s duty to what we build.' },
      { title: 'Arrival', meta: 'Denis Villeneuve · 2016', note: 'Nonlinear language rewiring the perception of time; linguistics meets determinism, gently.' },
      { title: 'Inception', meta: 'Christopher Nolan · 2010', note: 'Nested state machines and recursive exits — an engineer’s kind of romance.' },
      { title: 'The Truman Show', meta: 'Peter Weir · 1998', note: 'When the whole world is a designed system, escape is the first real test.' },
      { title: 'Spirited Away', meta: 'Hayao Miyazaki · 2001', note: 'A name is the key to identity — lose it and you lose the reference to yourself.' },
    ],
  },
  {
    cat: 'Music',
    icon: zh[1].icon,
    items: [
      { title: 'Ryuichi Sakamoto · async', meta: '2017', note: 'Asymmetric beats and silence — the late-night soundtrack for writing code.' },
      { title: 'Joe Hisaishi · Summer', meta: 'Kikujiro', note: 'A minimalist motif in endless variation — proof a good melody beats any arrangement.' },
      { title: 'Pink Floyd · The Dark Side of the Moon', meta: '1973', note: 'A concept album on time, money and breakdown — engineered with precision.' },
      { title: 'Joep Beving · Henosis', meta: '2019', note: 'Neo-classical minimalism — a steady anchor for long-context thinking.' },
    ],
  },
  {
    cat: 'Games',
    icon: zh[2].icon,
    items: [
      { title: 'Outer Wilds', meta: 'Mobius · 2019', note: 'Progress through knowledge, not stats — information is the only upgrade.' },
      { title: 'Portal 2', meta: 'Valve · 2011', note: 'Spatial reasoning and physics as an elegant problem set — a level-design textbook.' },
      { title: 'Disco Elysium', meta: 'ZA/UM · 2019', note: 'Personality split into skills that bargain with each other — the RPG system is the philosophy.' },
      { title: 'The Legend of Zelda: Breath of the Wild', meta: 'Nintendo · 2017', note: 'Emergent gameplay — few rules, infinite combinations.' },
    ],
  },
  {
    cat: 'Photography',
    icon: zh[3].icon,
    items: [
      { title: 'Street · The Decisive Moment', meta: 'Cartier-Bresson', note: 'Waiting in a flowing scene for geometry and timing to close into a single frame.' },
      { title: 'Black & White · Tonal Structure', meta: 'Ansel Adams · Zone System', note: 'Treating dynamic range as a zoned-exposure engineering problem.' },
      { title: 'Minimalism · Negative Space', meta: 'Hiroshi Sugimoto · Seascapes', note: 'Long exposure compressing time into a single static grayscale.' },
      { title: 'Night · Long Exposure', meta: 'City & Stars', note: 'The reward of signal-to-noise ratio and a tripod’s patience.' },
    ],
  },
]

const groups = computed(() => (isEn.value ? en : zh))
</script>

<template>
  <div class="ent">
    <section v-for="g in groups" :key="g.cat" class="ent-cat">
      <h3 class="ent-cat-head">
        <span class="ent-cat-icon" v-html="g.icon"></span>
        <span class="ent-cat-name">{{ g.cat }}</span>
      </h3>
      <ul class="ent-items">
        <li v-for="it in g.items" :key="it.title" class="ent-item">
          <div class="ent-item-head">
            <span class="ent-title">{{ it.title }}</span>
            <span class="ent-meta">{{ it.meta }}</span>
          </div>
          <p class="ent-note">{{ it.note }}</p>
        </li>
      </ul>
    </section>
  </div>
</template>

<style scoped>
.ent { margin: 1.5rem 0 2rem; }
.ent-cat { margin-bottom: 2rem; }
.ent-cat-head {
  display: flex;
  align-items: center;
  gap: 0.5em;
  margin: 0 0 0.8rem;
  padding-bottom: 0.4rem;
  border-bottom: 2px solid var(--tuf-ink);
  font-size: 1.2rem;
  font-weight: 600;
  color: var(--tuf-ink);
}
.ent-cat-icon {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 1.3em;
  height: 1.3em;
  opacity: 0.8;
}
.ent-cat-icon :deep(svg) { width: 1.15em; height: 1.15em; }
.ent-items { list-style: none; margin: 0; padding: 0; }
.ent-item {
  padding: 0.7rem 0;
  border-bottom: 1px solid var(--tuf-rule);
}
.ent-item:last-child { border-bottom: none; }
.ent-item-head {
  display: flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.6rem;
}
.ent-title {
  font-size: 1.02rem;
  font-weight: 600;
  color: var(--tuf-ink);
}
.ent-meta {
  font-family: var(--vp-font-family-mono);
  font-size: 0.78rem;
  color: var(--tuf-faint);
}
.ent-note {
  margin: 0.35rem 0 0;
  font-size: 0.92rem;
  line-height: 1.7;
  color: var(--tuf-muted);
  max-width: 100%;
}
</style>
