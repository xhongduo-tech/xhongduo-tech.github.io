<script setup lang="ts">
import { withBase } from 'vitepress'
import { data as articles } from '../data/articles.data'

withDefaults(defineProps<{ limit?: number }>(), { limit: 0 })
</script>

<template>
  <div v-if="articles.length" class="article-index">
    <a
      v-for="article in limit ? articles.slice(0, limit) : articles"
      :key="article.url"
      :href="withBase(article.url)"
      class="article-entry"
    >
      <div class="article-entry-meta">
        <time v-if="article.date" :datetime="article.date">{{ article.date }}</time>
        <span v-if="article.topic">{{ article.topic }}</span>
      </div>
      <h3>{{ article.title }} <span aria-hidden="true">↗</span></h3>
      <p v-if="article.description">{{ article.description }}</p>
    </a>
  </div>
  <div v-else class="articles-empty">
    <span class="empty-mark" aria-hidden="true">—</span>
    <div>
      <p class="empty-title">正式文章，等待第一份完整记录。</p>
      <p>当前已发布文章为 0 篇。专题提纲与项目计划分别保留在研究地图和项目页。</p>
    </div>
  </div>
</template>
