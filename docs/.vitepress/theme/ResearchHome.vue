<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import ActivityHeatmap from './ActivityHeatmap.vue'

const { lang } = useData()
const copy = computed(() => lang.value === 'en-US' ? {
  title: 'LLM Field Notes',
  activity: 'Activity log',
  tags: ['Model fundamentals', 'Inference systems', 'Training & alignment', 'Evaluation & experiments'],
  links: [
    { title: 'Articles', url: '/en/articles/' },
    { title: 'Projects', url: '/en/projects/' },
    { title: 'About', url: '/en/about' },
  ],
} : {
  title: '大模型研习录',
  activity: '更新记录',
  tags: ['模型原理', '推理系统', '训练与后训练', '评测与实验'],
  links: [
    { title: '文章', url: '/articles/' },
    { title: '项目', url: '/projects/' },
    { title: '关于我', url: '/about' },
  ],
})
</script>

<template>
  <div class="learning-home">
    <header class="learning-header">
      <p class="learning-kicker" lang="en">LLM FIELD NOTES</p>
      <h1>{{ copy.title }}</h1>
      <ul class="learning-tags" :aria-label="lang === 'en-US' ? 'Topics' : '学习主题'">
        <li v-for="(tag, index) in copy.tags" :key="tag">
          <svg v-if="index === 0" viewBox="0 0 20 20" aria-hidden="true"><circle cx="5" cy="5" r="2" /><circle cx="15" cy="5" r="2" /><circle cx="10" cy="15" r="2" /><path d="m6.7 6.3 2.4 6.5m4.2-6.5-2.4 6.5M7 5h6" /></svg>
          <svg v-else-if="index === 1" viewBox="0 0 20 20" aria-hidden="true"><path d="M11.5 2 5 11h5l-1.5 7L15 8h-5l1.5-6Z" /></svg>
          <svg v-else-if="index === 2" viewBox="0 0 20 20" aria-hidden="true"><path d="m10 2 7 4-7 4-7-4 7-4Zm-7 8 7 4 7-4M3 14l7 4 7-4" /></svg>
          <svg v-else viewBox="0 0 20 20" aria-hidden="true"><path d="M3 16V9m5 7V4m5 12v-5m4 5V7M2 18h16" /></svg>
          <span>{{ tag }}</span>
        </li>
      </ul>
    </header>

    <main class="home-activity">
      <nav class="home-page-nav" :aria-label="lang === 'en-US' ? 'Browse pages' : '浏览页面'">
        <p lang="en">INDEX</p>
        <a v-for="(link, index) in copy.links" :key="link.url" :href="withBase(link.url)">
          <span class="home-nav-number">0{{ index + 1 }}</span>
          <span>{{ link.title }}</span>
          <span class="home-nav-arrow" aria-hidden="true">↗</span>
        </a>
      </nav>
      <section class="activity-archive" aria-labelledby="activity-title">
        <ActivityHeatmap :title="copy.activity" />
      </section>
    </main>
  </div>
</template>
