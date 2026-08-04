<script setup>
import { withBase, useRoute, useData } from 'vitepress'
import { computed, onMounted, watch, nextTick } from 'vue'
import { initEnhancements, enhancePage } from './enhance'

const route = useRoute()
const { page } = useData()

const isEn = computed(() => page.value.relativePath.startsWith('en/'))
const t = computed(() =>
  isEn.value
    ? {
        greeting: 'Hi, this is "From Limits to LLMs" — Xu Hongduo’s knowledge base',
        home: 'Home',
        posts: 'Posts',
        projects: 'Projects',
        lang: '中文',
        langLink: '/',
      }
    : {
        greeting: '你好，这里是「从极限到大模型」—— 徐鸿铎的个人知识库',
        home: '首页',
        posts: '博文',
        projects: '项目',
        lang: 'EN',
        langLink: '/en/',
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
</script>

<template>
  <div>
    <header class="site-header">{{ t.greeting }}</header>
    <header class="site-header">
      <nav class="site-nav">
        <a :href="withBase('/')">{{ t.home }}</a>
        <a :href="withBase('/posts/')">{{ t.posts }}</a>
        <a :href="withBase('/projects/')">{{ t.projects }}</a>
        <span class="nav-tools">
          <a class="nav-icon-btn lang-btn" :href="withBase(t.langLink)" :aria-label="t.lang">{{ t.lang }}</a>
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
      从极限到大模型 · 徐鸿铎 · Powered by VitePress ·
      <a href="https://github.com/xhongduo-tech/blog" target="_blank" rel="noopener">源码</a>
    </footer>
  </div>
</template>
