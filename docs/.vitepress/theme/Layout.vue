<script setup>
import { withBase, useRoute, useData } from 'vitepress'
import { computed, onMounted, watch, nextTick, ref } from 'vue'
import { enhancePage } from './enhance'

const THEME_KEY = 'theme-preference'
const route = useRoute()
const { page } = useData()
const theme = ref('light')

const nav = [
  { href: '/', label: '首页', match: (path) => path === '/' },
  { href: '/llm/', label: '大模型', match: (path) => path.startsWith('/llm/') },
  { href: '/quant/', label: '量化', match: (path) => path.startsWith('/quant/') },
]

const byline = computed(() => {
  const fm = page.value.frontmatter || {}
  if (!fm.date) return ''
  const section = fm.section === 'quant' ? '量化' : fm.section === 'llm' ? '大模型' : ''
  const date = String(fm.date).slice(0, 10)
  return [section, date].filter(Boolean).join(' · ')
})

function storedTheme() {
  try {
    return sessionStorage.getItem(THEME_KEY)
  } catch {
    return null
  }
}

function systemTheme() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

function applyTheme(next) {
  theme.value = next
  document.documentElement.setAttribute('data-theme', next)
}

function toggleTheme() {
  const next = theme.value === 'dark' ? 'light' : 'dark'
  try {
    sessionStorage.setItem(THEME_KEY, next)
  } catch {}
  applyTheme(next)
}

onMounted(() => {
  applyTheme(storedTheme() || systemTheme())
  enhancePage()
})
watch(
  () => route.path,
  () => nextTick(() => enhancePage()),
)
</script>

<template>
  <div>
    <header class="site-header">
      <p class="site-title">LLM & Quant</p>
      <nav class="site-nav">
        <span class="nav-links">
          <a
            v-for="item in nav"
            :key="item.href"
            :class="{ active: item.match(route.path) }"
            :href="withBase(item.href)"
            >{{ item.label }}</a
          >
        </span>
        <span class="nav-tools">
          <a
            class="nav-icon-btn"
            href="https://github.com/xhongduo-tech/blog"
            target="_blank"
            rel="noopener"
            aria-label="GitHub"
          >
            <svg viewBox="0 0 16 16" width="1.15em" height="1.15em" fill="currentColor" aria-hidden="true">
              <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z" />
            </svg>
          </a>
          <button
            class="theme-toggle-btn"
            type="button"
            :aria-label="theme === 'dark' ? '切换到浅色模式' : '切换到深色模式'"
            @click="toggleTheme"
          >
            <svg
              v-if="theme === 'dark'"
              xmlns="http://www.w3.org/2000/svg"
              width="1em"
              height="1em"
              fill="currentColor"
              viewBox="0 0 256 256"
              aria-hidden="true"
            >
              <path d="M120,40V16a8,8,0,0,1,16,0V40a8,8,0,0,1-16,0Zm8,24a64,64,0,1,0,64,64A64.07,64.07,0,0,0,128,64ZM58.34,69.66A8,8,0,0,0,69.66,58.34l-16-16A8,8,0,0,0,42.34,53.66Zm0,116.68-16,16a8,8,0,0,0,11.32,11.32l16-16a8,8,0,0,0-11.32-11.32ZM192,72a8,8,0,0,0,5.66-2.34l16-16a8,8,0,0,0-11.32-11.32l-16,16A8,8,0,0,0,192,72Zm5.66,114.34a8,8,0,0,0-11.32,11.32l16,16a8,8,0,0,0-11.32-11.32ZM48,128a8,8,0,0,0-8-8H16a8,8,0,0,0,0,16H40A8,8,0,0,0,48,128Zm80,80a8,8,0,0,0-8,8v24a8,8,0,0,1,16,0V216a8,8,0,0,0-8-8Zm112-88H216a8,8,0,0,0,0,16h24a8,8,0,0,0,0-16Z" />
            </svg>
            <svg
              v-else
              xmlns="http://www.w3.org/2000/svg"
              width="1em"
              height="1em"
              fill="currentColor"
              viewBox="0 0 256 256"
              aria-hidden="true"
            >
              <path d="M235.54,150.21a104.84,104.84,0,0,1-37,52.91A104,104,0,0,1,32,120,103.09,103.09,0,0,1,52.88,57.48a104.84,104.84,0,0,1,52.91-37,8,8,0,0,1,10,10,88.08,88.08,0,0,0,109.8,109.8,8,8,0,0,1,10,10Z" />
            </svg>
          </button>
        </span>
      </nav>
    </header>

    <article class="tuf-article">
      <section>
        <p v-if="byline" class="article-byline">{{ byline }}</p>
        <Content />
      </section>
    </article>

    <footer class="site-footer">LLM & Quant · 徐鸿铎 · 2026</footer>
  </div>
</template>
