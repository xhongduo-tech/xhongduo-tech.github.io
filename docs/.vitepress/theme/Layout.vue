<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import ResearchHome from './ResearchHome.vue'
import PageContext from './PageContext.vue'

const { frontmatter, page } = useData()
const isHome = computed(() => frontmatter.value.layout === 'home')
const isRoadmap = computed(() => page.value.relativePath === 'roadmap/index.md')
</script>

<template>
  <DefaultTheme.Layout>
    <template #home-hero-before>
      <ResearchHome v-if="isHome" />
    </template>
    <template #doc-before>
      <PageContext />
    </template>
    <template #sidebar-nav-before>
      <a
        class="topic-map-link"
        :href="withBase('/roadmap/')"
        :aria-current="isRoadmap ? 'page' : undefined"
      >
        <span class="vpi-align-left" aria-hidden="true" />
        完整知识地图
      </a>
    </template>
  </DefaultTheme.Layout>
</template>
