<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const { frontmatter, page, site, lang } = useData()
const english = computed(() => lang.value === 'en-US')
const relative = computed(() => page.value.relativePath.replace(/^en\//, ''))
const categories: Record<string, { title: string; titleEn: string; url: string; urlEn: string }> = {
  articles: { title: '已发布文章', titleEn: 'Articles', url: '/articles/', urlEn: '/en/articles/' },
  projects: { title: '项目报告', titleEn: 'Projects', url: '/projects/', urlEn: '/en/projects/' },
}
const category = computed(() => categories[relative.value.split('/')[0]])
const isCategoryPage = computed(() => category.value && relative.value === `${category.value.url.slice(1)}index.md`)
const kindLabels: Record<string, string> = { about: '关于', article: '文章', project: '项目' }
const kindLabelsEn: Record<string, string> = { about: 'About', article: 'Article', project: 'Project' }
const kindLabel = computed(() => (english.value ? kindLabelsEn : kindLabels)[frontmatter.value.kind] ?? (english.value ? 'Note' : '笔记'))
const statusLabel = computed(() => frontmatter.value.status === 'published' && frontmatter.value.kind === 'article' ? (english.value ? 'Published' : '已发布') : '')
const date = computed(() => {
  const value = frontmatter.value.updated || frontmatter.value.date
  return value instanceof Date ? value.toISOString().slice(0, 10) : value ? String(value).slice(0, 10) : ''
})
</script>

<template>
  <div class="page-context">
    <nav class="breadcrumbs" :aria-label="english ? 'Breadcrumb' : '当前位置'">
      <a :href="withBase(english ? '/en/' : '/')">{{ site.title }}</a>
      <template v-if="category">
        <span class="breadcrumb-separator" aria-hidden="true">/</span>
        <span v-if="isCategoryPage" aria-current="page">{{ english ? category.titleEn : category.title }}</span>
        <a v-else :href="withBase(english ? category.urlEn : category.url)">{{ english ? category.titleEn : category.title }}</a>
      </template>
    </nav>
    <div class="page-meta">
      <span>{{ kindLabel }}</span>
      <span v-if="statusLabel" class="status-badge">{{ statusLabel }}</span>
      <time v-if="date" :datetime="date">{{ frontmatter.updated ? (english ? 'Updated ' : '更新于 ') : '' }}{{ date }}</time>
    </div>
  </div>
</template>
