<script setup>
import { withBase, useRoute } from 'vitepress'
import { onMounted, watch, nextTick } from 'vue'
import { initEnhancements, enhancePage } from './enhance'

const route = useRoute()

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
    <header class="site-header">你好，这里是「从极限到大模型」—— 徐鸿铎的个人知识库</header>
    <header class="site-header">
      <nav class="site-nav">
        <a :href="withBase('/')">首页</a>
        <a :href="withBase('/posts/')">博文</a>
        <a :href="withBase('/projects/')">项目</a>
        <a :href="withBase('/about/')">关于我</a>
        <a href="https://github.com/xhongduo-tech/blog" target="_blank" rel="noopener">GitHub</a>
        <button id="theme-toggle" class="theme-toggle-btn" type="button" aria-label="切换主题"></button>
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
