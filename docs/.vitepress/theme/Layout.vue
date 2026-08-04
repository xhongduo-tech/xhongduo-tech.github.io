<script setup>
import { withBase, useRoute, useData, useRouter } from 'vitepress'
import { computed, onMounted, watch, nextTick, ref } from 'vue'
import { initEnhancements, enhancePage } from './enhance'

const route = useRoute()
const { page } = useData()
const router = useRouter()

const pageClass = computed(() => page.value.frontmatter?.pageClass || '')

// 语言偏好：博文/项目目前没有英文翻译，relativePath 不会带 en/ 前缀。
// 语言按钮是「原地切换」——在当前页面直接切换 chrome（问候语、导航、
// 页脚）的语言，不跳转到别的页面；偏好用 sessionStorage 记住（与主题
// 切换同一套持久化方式）。只有首页内容有中/英两份翻译：在首页时切语言
// 会跳转到对应语言的首页（/ 或 /en/），其余页面原地切换，正文保持原样。
const LANG_KEY = 'lang-preference'
function getStoredLang() {
  try {
    return sessionStorage.getItem(LANG_KEY)
  } catch {
    return null
  }
}
function setStoredLang(lang) {
  try {
    sessionStorage.setItem(LANG_KEY, lang)
  } catch {}
}

const langPreference = ref(getStoredLang())

const isEn = computed(() => {
  if (page.value.relativePath.startsWith('en/')) return true
  return langPreference.value === 'en'
})

function switchLang() {
  const next = isEn.value ? 'zh' : 'en'
  langPreference.value = next
  setStoredLang(next)
  // 首页才有本地化内容（/ 与 /en/ 是两份翻译），切语言时跳转到对应首页；
  // 其余页面没有英文翻译，原地切换 chrome 即可，不让用户被拽走。
  const rel = page.value.relativePath
  if (rel === 'index.md' || rel === 'en/index.md') {
    router.go(withBase(next === 'en' ? '/en/' : '/'))
  }
}

const t = computed(() =>
  isEn.value
    ? {
        greeting: 'Hi, this is "From Limits to LLMs" — Xu Hongduo’s knowledge base',
        home: 'Home',
        homeLink: '/en/',
        posts: 'Posts',
        projects: 'Projects',
        lang: '中文',
        footer: 'From Limits to LLMs · Xu Hongduo · Powered by VitePress ·',
        source: 'Source',
      }
    : {
        greeting: '你好，这里是「从极限到大模型」—— 徐鸿铎的个人知识库',
        home: '首页',
        homeLink: '/',
        posts: '博文',
        projects: '项目',
        lang: 'EN',
        footer: '从极限到大模型 · 徐鸿铎 · Powered by VitePress ·',
        source: '源码',
      },
)

onMounted(() => {
  initEnhancements()
  enhancePage()
})
watch(
  () => route.path,
  () => nextTick(() => enhancePage()),
)

// 真正落在 en/ 目录下的页面才代表内容本身是英文，同步偏好；
// 未翻译页面保留用户之前的选择，不强制覆盖。immediate: true 确保首次
// 加载 /en/ 页面时也能同步（而不是只在“路由发生变化”时才生效）。
watch(
  () => page.value.relativePath,
  (relativePath) => {
    if (relativePath.startsWith('en/')) {
      langPreference.value = 'en'
      setStoredLang('en')
    }
  },
  { immediate: true },
)
</script>

<template>
  <div :class="pageClass">
    <header class="site-header">
      <p class="site-greeting">{{ t.greeting }}</p>
      <nav class="site-nav">
        <span class="nav-links">
          <a :href="withBase(t.homeLink)">{{ t.home }}</a>
          <a :href="withBase('/posts/')">{{ t.posts }}</a>
          <a :href="withBase('/projects/')">{{ t.projects }}</a>
        </span>
        <span class="nav-tools">
          <button
            class="nav-icon-btn lang-btn"
            type="button"
            :aria-label="t.lang"
            @click="switchLang"
            >{{ t.lang }}</button
          >
          <a
            class="nav-icon-btn"
            href="https://github.com/xhongduo-tech/blog"
            target="_blank"
            rel="noopener"
            aria-label="GitHub"
          >
            <svg viewBox="0 0 16 16" width="1.15em" height="1.15em" fill="currentColor" aria-hidden="true">
              <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/>
            </svg>
          </a>
          <button id="theme-toggle" class="theme-toggle-btn" type="button" aria-label="切换主题"></button>
        </span>
      </nav>
    </header>

    <article class="tuf-article">
      <section>
        <Content />
      </section>
    </article>

    <footer class="site-footer">
      {{ t.footer }}
      <a href="https://github.com/xhongduo-tech/blog" target="_blank" rel="noopener">{{ t.source }}</a>
    </footer>
  </div>
</template>
