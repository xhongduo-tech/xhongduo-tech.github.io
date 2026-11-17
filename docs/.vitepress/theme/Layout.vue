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
    <template #home-hero-before>
      <ResearchHome v-if="isHome" />
    </template>
    <template #doc-before>
      <PageContext />
    </template>
  </DefaultTheme.Layout>
</template>
