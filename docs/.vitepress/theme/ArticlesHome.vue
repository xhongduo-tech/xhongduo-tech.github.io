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
      <p class="articles-overline" lang="en">PUBLISHED NOTES</p>
      <h1>{{ english ? 'Articles' : '文章' }}</h1>
      <p>{{ english ? 'Read reviewed articles or explore the public LLM topic graph. Unfinished research notes remain private.' : '阅读经过核对的正式文章，或探索公开的大模型主题依赖。未完成的研究提纲仍留在私有工作区。' }}</p>
    </header>
    <div class="articles-layout">
      <div class="articles-main">
        <ArticleViewSwitch class="article-view-mobile" :value="view" :english="english" @change="view = $event" />
        <div id="article-view-panel" class="article-view-panel" :aria-label="view === 'articles' ? (english ? 'Published articles' : '已发布博文') : (english ? 'Detailed topic topology' : '详细主题拓扑')">
          <section v-show="view === 'topology'" class="article-topology-section" aria-labelledby="topic-topology-heading">
            <h2 id="topic-topology-heading">{{ english ? 'Topic topology' : '主题拓扑' }}</h2>
            <p>{{ english ? 'A detailed public index of LLM topics and prerequisites. Publication status is separate from domain acceptance.' : '展开具体大模型主题与前置依赖。文章发布状态与领域人工验收分别记录。' }}</p>
            <DomainTopology />
          </section>
          <section v-show="view === 'articles'" class="article-list-section" :aria-labelledby="english ? 'reading-by-date' : 'article-list-heading'">
            <h2 :id="english ? 'reading-by-date' : 'article-list-heading'">{{ english ? 'Reading by date' : '按时间阅读' }}</h2>
            <ArticleIndex />
          </section>
        </div>
      </div>
      <aside class="articles-aside" :aria-label="english ? 'Article tools and today’s plan' : '文章工具与今日计划'">
        <ArticleViewSwitch class="article-view-desktop" :value="view" :english="english" @change="view = $event" />
        <TodayTodo />
      </aside>
    </div>
  </div>
</template>
