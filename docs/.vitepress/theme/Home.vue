<script setup>
import { ref, reactive, computed, onMounted } from 'vue'
import data from '../data/progress.json'

/* ---------- 数据 ---------- */
const cats = Object.values(data)
const topics = cats.reduce((s, c) => s + c.total, 0)
const done = cats.reduce((s, c) => s + c.done, 0)
const pct = computed(() => (topics ? Math.round((done / topics) * 100) : 0))
const asciiBar = computed(() => {
  const n = Math.round(pct.value / 5)
  return '█'.repeat(n) + '░'.repeat(20 - n)
})

const focus = [
  ['llm-inference/', 'PD 分离 · MTP · 量化推理 · vLLM / llama.cpp'],
  ['heterogeneous-compute/', 'A100 / V100 / 昇腾 910B3 · 梯度算力方案'],
  ['platform-eng/', '大模型开放平台 · API 智能路由 · 全栈'],
  ['applied-ai/', 'RAG · Agent · 业务系统落地'],
]

const stack = [
  'Python', 'PyTorch', 'vLLM', 'SGLang', 'llama.cpp', 'TEI',
  'CUDA', 'Ascend-CANN', 'Docker', 'K8s', 'Vue', 'FastAPI',
]

const tiers = [
  ['01', '基础科学', '/posts/foundations/math/', '数理生化 + 天文地学认知心理逻辑科哲经济'],
  ['02', '进阶数理', '/posts/intermediate/advanced-math/', '高数概率线代，直到实变泛函拓扑'],
  ['03', '计算机基础', '/posts/cs/data-structures/', 'CS 核心课全集：从数据结构到分布式'],
  ['04', '高阶专题', '/posts/advanced/llm-principles/', 'ML/DL/RL 到大模型原理、部署、微调'],
]

/* ---------- 窗口布局与拖拽 ---------- */
const defaults = {
  whoami: { x: 40, y: 28 },
  focus: { x: 520, y: 28 },
  clock: { x: 520, y: 330 },
  stack: { x: 40, y: 340 },
  tiers: { x: 40, y: 540 },
  progress: { x: 520, y: 470 },
}
const pos = reactive(JSON.parse(JSON.stringify(defaults)))
const z = reactive(Object.fromEntries(Object.keys(defaults).map((k) => [k, 1])))
let top = 1

const STORAGE_KEY = 'home-terminal-layout'

function onDragStart(e, key) {
  if (e.pointerType === 'touch') return // 触屏走纵向堆叠布局
  const el = e.currentTarget.closest('.win')
  const parent = el.parentElement.getBoundingClientRect()
  const rect = el.getBoundingClientRect()
  const offX = e.clientX - rect.left
  const offY = e.clientY - rect.top
  z[key] = ++top

  const move = (ev) => {
    pos[key].x = Math.min(Math.max(ev.clientX - parent.left - offX, 0), parent.width - 80)
    pos[key].y = Math.min(Math.max(ev.clientY - parent.top - offY, 0), parent.height - 40)
  }
  const up = () => {
    window.removeEventListener('pointermove', move)
    window.removeEventListener('pointerup', up)
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(pos))
    } catch {}
  }
  window.addEventListener('pointermove', move)
  window.addEventListener('pointerup', up)
}

function resetLayout() {
  Object.assign(pos, JSON.parse(JSON.stringify(defaults)))
  try {
    localStorage.removeItem(STORAGE_KEY)
  } catch {}
}

/* ---------- 时钟 ---------- */
const now = ref('')
onMounted(() => {
  try {
    const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) || 'null')
    if (saved) for (const k of Object.keys(defaults)) if (saved[k]) Object.assign(pos[k], saved[k])
  } catch {}
  const tick = () => {
    const d = new Date()
    const p = (n) => String(n).padStart(2, '0')
    now.value = `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
  }
  tick()
  setInterval(tick, 1000)
})
</script>

<template>
  <div class="term-home">
    <div class="desktop">
      <!-- whoami -->
      <div
        class="win win-whoami"
        :style="{ left: pos.whoami.x + 'px', top: pos.whoami.y + 'px', zIndex: z.whoami }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'whoami')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">whoami — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ whoami</p>
          <h1 class="me">徐鸿铎</h1>
          <p class="role">大模型架构工程师 @ 中国工商银行总行</p>
          <p class="cmd">$ cat motto.txt</p>
          <p class="motto">
            白天做大模型推理架构与异构算力调度，<br />
            晚上写一个从高中数理到大模型的完整知识体系。
          </p>
          <p class="cmd">$ ls ./nav</p>
          <div class="nav-row">
            <a href="/posts/" class="nav-link">博文总览</a>
            <a href="/projects/" class="nav-link">项目</a>
            <a href="/about/" class="nav-link">关于我</a>
            <a href="https://github.com/xhongduo-tech" class="nav-link">GitHub</a>
          </div>
        </div>
      </div>

      <!-- focus -->
      <div
        class="win win-focus"
        :style="{ left: pos.focus.x + 'px', top: pos.focus.y + 'px', zIndex: z.focus }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'focus')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">focus — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ ls -l ~/focus</p>
          <div v-for="f in focus" :key="f[0]" class="ls-row">
            <span class="ls-name">{{ f[0] }}</span>
            <span class="ls-desc">{{ f[1] }}</span>
          </div>
        </div>
      </div>

      <!-- clock -->
      <div
        class="win win-clock"
        :style="{ left: pos.clock.x + 'px', top: pos.clock.y + 'px', zIndex: z.clock }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'clock')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">date — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ date</p>
          <p class="clock">{{ now || 'loading…' }}</p>
        </div>
      </div>

      <!-- stack -->
      <div
        class="win win-stack"
        :style="{ left: pos.stack.x + 'px', top: pos.stack.y + 'px', zIndex: z.stack }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'stack')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">stack — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ cat ~/stack.txt</p>
          <div class="tags">
            <span v-for="s in stack" :key="s" class="tag">{{ s }}</span>
          </div>
        </div>
      </div>

      <!-- tiers -->
      <div
        class="win win-tiers"
        :style="{ left: pos.tiers.x + 'px', top: pos.tiers.y + 'px', zIndex: z.tiers }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'tiers')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">tiers — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ open ~/writing-system <span class="comment"># 60 学科 · {{ topics.toLocaleString() }} 选题</span></p>
          <a v-for="t in tiers" :key="t[0]" :href="t[2]" class="tier-row">
            <span class="tier-no">{{ t[0] }}</span>
            <span class="tier-name">{{ t[1] }}</span>
            <span class="tier-desc">{{ t[3] }}</span>
          </a>
        </div>
      </div>

      <!-- progress -->
      <div
        class="win win-progress"
        :style="{ left: pos.progress.x + 'px', top: pos.progress.y + 'px', zIndex: z.progress }"
      >
        <div class="win-bar" @pointerdown="onDragStart($event, 'progress')">
          <span class="dot r"></span><span class="dot y"></span><span class="dot g"></span>
          <span class="win-title">progress — zsh</span>
        </div>
        <div class="win-body">
          <p class="cmd">$ ./progress --all</p>
          <p class="prog-line">
            <span class="prog-bar">{{ asciiBar }}</span> {{ pct }}%
          </p>
          <p class="prog-stat">{{ done }} / {{ topics.toLocaleString() }} 篇 · {{ cats.length }} 个学科</p>
          <p class="cmd">$ <a href="/posts/" class="term-link">open /posts</a><span class="cursor">▊</span></p>
        </div>
      </div>

      <button class="reset" title="复位布局" @click="resetLayout">reset</button>
    </div>
    <p class="hint">拖动窗口标题栏可以自由摆放，布局会自动保存</p>
  </div>
</template>

<style scoped>
.term-home {
  max-width: 1080px;
  margin: 0 auto;
  padding: 24px 20px 8px;
}

/* 桌面 */
.desktop {
  position: relative;
  height: 760px;
  border-radius: 12px;
  border: 1px solid #21262d;
  background:
    radial-gradient(ellipse at 20% 0%, rgba(63, 185, 80, 0.06), transparent 55%),
    linear-gradient(rgba(88, 166, 255, 0.025) 1px, transparent 1px),
    linear-gradient(90deg, rgba(88, 166, 255, 0.025) 1px, transparent 1px),
    #0d1117;
  background-size: auto, 32px 32px, 32px 32px, auto;
  overflow: hidden;
}

/* 窗口 */
.win {
  position: absolute;
  border: 1px solid #30363d;
  border-radius: 8px;
  background: rgba(13, 17, 23, 0.92);
  box-shadow: 0 12px 32px rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(4px);
  font-family: var(--vp-font-family-mono);
  user-select: none;
}
.win-whoami { width: 440px; }
.win-focus { width: 460px; }
.win-clock { width: 460px; }
.win-stack { width: 440px; }
.win-tiers { width: 440px; }
.win-progress { width: 460px; }

.win-bar {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 12px;
  background: #161b22;
  border-bottom: 1px solid #30363d;
  border-radius: 8px 8px 0 0;
  cursor: grab;
  touch-action: none;
}
.win-bar:active {
  cursor: grabbing;
}
.dot {
  width: 11px;
  height: 11px;
  border-radius: 50%;
}
.dot.r { background: #ff5f57; }
.dot.y { background: #febc2e; }
.dot.g { background: #28c840; }
.win-title {
  margin-left: 8px;
  font-size: 12px;
  color: #7d8590;
}
.win-body {
  padding: 14px 16px 16px;
  font-size: 13px;
  line-height: 1.7;
  color: #adbac7;
}

.cmd {
  color: #3fb950;
  margin: 10px 0 6px;
}
.cmd:first-child {
  margin-top: 0;
}
.comment {
  color: #484f58;
}
.term-link {
  color: #58a6ff;
  text-decoration: none;
}
.term-link:hover {
  text-decoration: underline;
}
.cursor {
  color: #3fb950;
  animation: blink 1.1s steps(1) infinite;
  margin-left: 4px;
}
@keyframes blink {
  50% { opacity: 0; }
}

/* whoami */
.me {
  font-size: 30px;
  font-weight: 800;
  color: #e6edf3;
  letter-spacing: 0.02em;
  margin: 2px 0 4px;
}
.role {
  color: #58a6ff;
  margin: 0;
}
.motto {
  color: #adbac7;
  margin: 0;
  font-family: var(--vp-font-family-base);
  line-height: 1.8;
}
.nav-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.nav-link {
  font-size: 12.5px;
  padding: 4px 12px;
  border: 1px solid #30363d;
  border-radius: 6px;
  color: #3fb950;
  text-decoration: none !important;
  transition: border-color 0.2s, background 0.2s;
}
.nav-link:hover {
  border-color: #3fb950;
  background: rgba(63, 185, 80, 0.1);
}

/* focus ls */
.ls-row {
  display: flex;
  gap: 12px;
  white-space: nowrap;
  overflow: hidden;
}
.ls-name {
  color: #58a6ff;
  flex-shrink: 0;
}
.ls-desc {
  color: #7d8590;
  overflow: hidden;
  text-overflow: ellipsis;
  font-family: var(--vp-font-family-base);
  font-size: 12.5px;
}

/* clock */
.clock {
  font-size: 26px;
  font-weight: 700;
  color: #e6edf3;
  font-variant-numeric: tabular-nums;
  margin: 4px 0 2px;
}

/* stack */
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.tag {
  font-size: 12px;
  padding: 3px 10px;
  border: 1px solid #30363d;
  border-radius: 999px;
  color: #adbac7;
  transition: border-color 0.2s, color 0.2s;
}
.tag:hover {
  border-color: #3fb950;
  color: #3fb950;
}

/* tiers */
.tier-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  padding: 5px 0;
  text-decoration: none !important;
  border-bottom: 1px dashed #21262d;
}
.tier-row:last-child {
  border-bottom: none;
}
.tier-no {
  color: #3fb950;
}
.tier-name {
  color: #e6edf3;
  font-weight: 700;
  font-family: var(--vp-font-family-base);
}
.tier-row:hover .tier-name {
  color: #58a6ff;
}
.tier-desc {
  color: #7d8590;
  font-size: 12px;
  font-family: var(--vp-font-family-base);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* progress */
.prog-line {
  margin: 4px 0;
  color: #e6edf3;
}
.prog-bar {
  color: #3fb950;
  letter-spacing: 1px;
}
.prog-stat {
  color: #7d8590;
  margin: 0 0 4px;
}

/* reset & hint */
.reset {
  position: absolute;
  right: 12px;
  bottom: 10px;
  font-family: var(--vp-font-family-mono);
  font-size: 11px;
  color: #484f58;
  background: none;
  border: 1px solid #21262d;
  border-radius: 5px;
  padding: 3px 10px;
  cursor: pointer;
}
.reset:hover {
  color: #3fb950;
  border-color: #3fb950;
}
.hint {
  text-align: center;
  font-size: 12px;
  color: var(--vp-c-text-3);
  margin: 10px 0 0;
}

/* 移动端：堆叠为纵向列表，禁用拖拽 */
@media (max-width: 900px) {
  .desktop {
    height: auto;
    display: flex;
    flex-direction: column;
    gap: 12px;
    padding: 12px;
    overflow: visible;
  }
  .win {
    position: static !important;
    width: 100%;
  }
  .win-bar {
    cursor: default;
  }
  .reset {
    display: none;
  }
}
</style>
