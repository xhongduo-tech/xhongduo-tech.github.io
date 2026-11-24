<script setup lang="ts">
import { computed, ref } from 'vue'
import { useData } from 'vitepress'
import ArticleIndex from './ArticleIndex.vue'
import ArticleViewSwitch from './ArticleViewSwitch.vue'
import DomainTopology from './DomainTopology.vue'
import TodayTodo from './TodayTodo.vue'

const { lang } = useData()
const english = computed(() => lang.value === 'en-US')
const view = ref<'articles' | 'topology'>('articles')
</script>

<template>
  <div class="articles-page">
    <header class="articles-header">
      <div class="articles-header-copy">
        <p class="articles-overline" lang="en">PUBLISHED NOTES</p>
        <h1>{{ english ? 'Articles' : '文章' }}</h1>
        <p>{{ english ? 'Read reviewed articles or explore the public LLM topic graph. Unfinished research notes remain private.' : '阅读经过核对的正式文章，或探索公开的大模型主题依赖。未完成的研究提纲仍留在私有工作区。' }}</p>
      </div>
      <ArticleViewSwitch :value="view" :english="english" @change="view = $event" />
    </header>
    <div class="articles-layout">
      <div class="articles-main">
        <div id="article-view-panel" class="article-view-panel" :aria-label="view === 'articles' ? (english ? 'Published articles' : '已发布博文') : (english ? 'Detailed topic topology' : '详细主题拓扑')">
          <section v-show="view === 'topology'" class="article-topology-section" :aria-label="english ? 'Detailed topic topology' : '详细主题拓扑'">
            <DomainTopology />
          </section>
          <section v-show="view === 'articles'" class="article-list-section" :aria-labelledby="english ? 'reading-by-date' : 'article-list-heading'">
            <h2 :id="english ? 'reading-by-date' : 'article-list-heading'">{{ english ? 'Reading by date' : '按时间阅读' }}</h2>
            <ArticleIndex />
          </section>
        </div>
      </div>
      <aside class="articles-aside" :aria-label="english ? 'Today’s plan' : '今日计划'">
        <TodayTodo />
      </aside>
    </div>
  </div>
</template>
