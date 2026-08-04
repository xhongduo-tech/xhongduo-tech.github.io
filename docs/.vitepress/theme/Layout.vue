<script setup>
import { withBase, useRoute, useData } from 'vitepress'
import { computed, onMounted, watch, nextTick, ref } from 'vue'
import { initEnhancements, enhancePage } from './enhance'

const route = useRoute()
const { page } = useData()

const pageClass = computed(() => page.value.frontmatter?.pageClass || '')

// 语言偏好：博文/项目目前没有英文翻译，relativePath 不会带 en/ 前缀。
// 只用 relativePath 判断语言会导致「选了英文再点博文/项目，整个页面
// chrome（问候语、导航、页脚）又弹回中文」。这里用 sessionStorage 记住
// 用户上一次的显式选择（与主题切换同一套持久化方式），未翻译页面沿用
// 该偏好，只有正文内容本身仍是中文——比一整页突然切回中文更符合预期。
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
        langLink: '/',
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
        langLink: '/en/',
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
          <a
            class="nav-icon-btn lang-btn"
            :href="withBase(t.langLink)"
            :aria-label="t.lang"
            @click="switchLang"
            >{{ t.lang }}</a
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
