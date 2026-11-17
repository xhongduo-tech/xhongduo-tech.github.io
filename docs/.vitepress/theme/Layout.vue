<script setup lang="ts">
import { computed } from 'vue'
import { useData } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import ResearchHome from './ResearchHome.vue'
import PageContext from './PageContext.vue'

const { frontmatter } = useData()
const isHome = computed(() => frontmatter.value.layout === 'home')
</script>

<template>
  <DefaultTheme.Layout>
    <template #nav-bar-title-after>
      <span class="brand-caption" aria-hidden="true">RESEARCH NOTES</span>
    </template>
    <template #home-hero-before>
      <ResearchHome v-if="isHome" />
    </template>
    <template #doc-before>
      <PageContext />
    </template>
    <template #aside-outline-after>
      <div class="reading-note">
        <span class="reading-note-label">阅读说明</span>
        <p>提纲与项目按实际进展更新，已发布文章另行收录。</p>
      </div>
    </template>
  </DefaultTheme.Layout>
</template>
