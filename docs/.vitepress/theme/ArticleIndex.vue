<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as articles } from '../data/articles.data'

const { lang } = useData()
const english = computed(() => lang.value === 'en-US')

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
    <p>{{ english ? 'No published articles yet. Reviewed learning, source-reading and experimental notes will appear here.' : '还没有正式文章；经过核对的学习、源码与实验记录会发布在这里。' }}</p>
  </div>
</template>
