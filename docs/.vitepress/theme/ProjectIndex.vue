<script setup lang="ts">
import { computed } from 'vue'
import { useData, withBase } from 'vitepress'
import { data as projects } from '../data/projects.data'

const { lang } = useData()
const english = computed(() => lang.value === 'en-US')
</script>

<template>
  <div v-if="projects.length" class="article-index project-index">
    <a v-for="project in projects" :key="project.url" :href="withBase(project.url)" class="article-entry">
      <div class="article-entry-meta"><time v-if="project.date" :datetime="project.date">{{ project.date }}</time><span>{{ english ? 'Project report' : '项目报告' }}</span></div>
      <h3>{{ project.title }} <span aria-hidden="true">↗</span></h3>
      <p>{{ project.description }}</p>
    </a>
  </div>
  <div v-else class="projects-empty">
    <span class="projects-empty-mark" aria-hidden="true">{ }</span>
    <div>
      <h3>{{ english ? 'No public project reports yet' : '还没有公开项目报告' }}</h3>
      <p>{{ english ? 'A project appears here only after it has a reproducible setup, baseline, raw observations and a record of limitations.' : '只有形成可复现环境、基线、原始观察和边界记录后，项目才会发布在这里。' }}</p>
    </div>
  </div>
</template>
