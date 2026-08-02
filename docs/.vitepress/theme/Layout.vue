<script setup>
import { useData, withBase } from 'vitepress'
import { ref, computed, onMounted } from 'vue'

const { page } = useData()

const dark = ref(false)
onMounted(() => {
  dark.value = document.documentElement.classList.contains('dark')
})
function toggleTheme() {
  dark.value = !dark.value
  document.documentElement.classList.toggle('dark', dark.value)
  try {
    localStorage.setItem('tuf-theme', dark.value ? 'dark' : 'light')
  } catch {}
}

const lastUpdated = computed(() => {
  const ts = page.value.lastUpdated
  if (!ts) return ''
  const d = new Date(ts)
  const p = (n) => String(n).padStart(2, '0')
  return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}`
})

const navs = [
  { text: '博文', link: '/posts/' },
  { text: '项目', link: '/projects/' },
  { text: '关于我', link: '/about/' },
]
</script>

<template>
  <div class="tuf-layout">
    <header class="site-head">
      <a class="site-title" :href="withBase('/')">从极限到大模型</a>
      <nav class="site-nav">
        <a
          v-for="n in navs"
          :key="n.link"
          :href="withBase(n.link)"
          :class="{ active: page.relativePath.startsWith(n.link.slice(1, -1) + '/') }"
        >
          {{ n.text }}
        </a>
        <a href="https://github.com/xhongduo-tech" target="_blank" rel="noopener">GitHub</a>
        <button class="theme-btn" :title="dark ? '切换到亮色' : '切换到暗色'" @click="toggleTheme">
          {{ dark ? '☀ 亮色' : '☾ 暗色' }}
        </button>
      </nav>
    </header>

    <main class="tuf-main">
      <article class="tuf-doc">
        <Content />
      </article>
      <p v-if="lastUpdated" class="last-updated">最后更新于 {{ lastUpdated }}</p>
    </main>

    <footer class="site-foot">
      从极限到大模型 · 徐鸿铎 · 本站由 VitePress 构建，源码托管于
      <a href="https://github.com/xhongduo-tech/blog">GitHub</a>
    </footer>
  </div>
</template>
