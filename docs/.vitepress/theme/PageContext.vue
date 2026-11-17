<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'

const { frontmatter, page, site } = useData()
const categories: Record<string, { title: string; url: string }> = {
  foundations: { title: '模型基础', url: '/foundations/' },
  inference: { title: '推理与系统', url: '/inference/' },
  training: { title: '训练与后训练', url: '/training/' },
  evaluation: { title: '模型评测', url: '/evaluation/' },
  projects: { title: '项目实践', url: '/projects/' },
  writing: { title: '文章与写作', url: '/writing/' },
  articles: { title: '已发布文章', url: '/articles/' },
}
const category = computed(() => categories[page.value.relativePath.split('/')[0]])
const kindLabels: Record<string, string> = {
  topic: '专题提纲', project: '实践项目', overview: '专题地图', about: '关于', article: '文章',
}
const kindLabel = computed(() => kindLabels[frontmatter.value.kind] ?? '笔记')
const statusLabel = computed(() => {
  const status = frontmatter.value.status
  if (status === 'planned') return frontmatter.value.kind === 'project' ? '待实践' : '待研究'
  if (status === 'published') return frontmatter.value.kind === 'article' ? '已发布' : ''
  return status ? String(status) : ''
})
const date = computed(() => {
  const value = frontmatter.value.updated || frontmatter.value.date
  return value instanceof Date ? value.toISOString().slice(0, 10) : value ? String(value).slice(0, 10) : ''
})
</script>

<template>
  <div class="page-context">
    <nav class="breadcrumbs" aria-label="当前位置">
      <a :href="withBase('/')">{{ site.title }}</a>
      <template v-if="category">
        <span class="breadcrumb-separator" aria-hidden="true">/</span>
        <a :href="withBase(category.url)">{{ category.title }}</a>
      </template>
    </nav>
    <div class="page-meta">
      <span>{{ kindLabel }}</span>
      <span v-if="statusLabel" class="status-badge">{{ statusLabel }}</span>
      <time v-if="date" :datetime="date">{{ frontmatter.updated ? '更新于' : '' }} {{ date }}</time>
    </div>
  </div>
</template>
