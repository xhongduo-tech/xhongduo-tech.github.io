<script setup>
import { withBase, useRoute, useData, useRouter } from 'vitepress'
import { computed, onMounted, watch, nextTick } from 'vue'
import { initEnhancements, enhancePage } from './enhance'

const route = useRoute()
const { page } = useData()
const router = useRouter()

const pageClass = computed(() => page.value.frontmatter?.pageClass || '')

// 语言由内容驱动：每个页面都有中/英两个版本（/posts/foo 与 /en/posts/foo
// 是一对镜像），页面落在 en/ 目录下即为英文。语言按钮跳到当前页面的
// 另一语言版本，不再依赖存储的偏好。
const isEn = computed(() => page.value.relativePath.startsWith('en/'))

// 相对路径 -> 站点路径：'posts/foo/index.md' -> '/posts/foo/'，'en/index.md' -> '/en/'
function urlFromRelativePath(rel) {
  const noExt = rel.replace(/\.md$/, '')
  const dir = noExt.replace(/(^|\/)index$/, '$1')
  return '/' + dir
}

function switchLang() {
  const rel = page.value.relativePath
  const isEnPage = rel.startsWith('en/')
  const targetRel = isEnPage ? rel.slice(3) : 'en/' + rel
  router.go(withBase(urlFromRelativePath(targetRel)))
}

const t = computed(() =>
  isEn.value
    ? {
        greeting: 'Hi, this is "From Limits to LLMs" — Xu Hongduo’s knowledge base',
        home: 'Home',
        homeLink: '/en/',
        posts: 'Posts',
        postsLink: '/en/posts/',
        knowledge: 'Knowledge Tree',
        knowledgeLink: '/en/knowledge-tree/',
        projects: 'Projects',
        projectsLink: '/en/projects/',
        lang: '中文',
        footer: 'From Limits to LLMs · Xu Hongduo · Powered by VitePress ·',
        source: 'Source',
      }
    : {
        greeting: '你好，这里是「从极限到大模型」—— 徐鸿铎的个人知识库',
        home: '首页',
        homeLink: '/',
        posts: '博文',
        postsLink: '/posts/',
        knowledge: '知识树',
        knowledgeLink: '/knowledge-tree/',
        projects: '项目',
        projectsLink: '/projects/',
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
</script>

<template>
  <div :class="pageClass">
    <header class="site-header">
      <p class="site-greeting">{{ t.greeting }}</p>
      <nav class="site-nav">
        <span class="nav-links">
          <a :class="{ active: route.path === t.homeLink }" :href="withBase(t.homeLink)">{{ t.home }}</a>
          <a :class="{ active: route.path.startsWith(t.postsLink) }" :href="withBase(t.postsLink)">{{ t.posts }}</a>
          <a :class="{ active: route.path.startsWith(t.knowledgeLink) }" :href="withBase(t.knowledgeLink)">{{ t.knowledge }}</a>
          <a :class="{ active: route.path.startsWith(t.projectsLink) }" :href="withBase(t.projectsLink)">{{ t.projects }}</a>
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
