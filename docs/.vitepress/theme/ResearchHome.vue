<script setup lang="ts">
import { withBase } from 'vitepress'
import { sections } from '../data/site'
import { data as articles } from '../data/articles.data'
import ArticleIndex from './ArticleIndex.vue'
</script>

<template>
  <div class="research-home">
    <header class="home-intro">
      <h1>一些思考，一些实践</h1>
      <p>在阅读、代码与实验中积累理解。</p>
    </header>

    <section class="home-section" aria-labelledby="writing-title">
      <div class="section-heading">
        <h2 id="writing-title">文章 <span class="article-count">{{ articles.length }}</span></h2>
        <a class="section-link" :href="withBase('/articles/')">全部文章 <span aria-hidden="true">→</span></a>
      </div>
      <ArticleIndex :limit="5" />
    </section>

    <section id="research-map" class="home-section" aria-labelledby="map-title">
      <div class="section-heading">
        <h2 id="map-title">研究方向</h2>
        <a class="section-link" :href="withBase('/projects/')">实验项目 <span aria-hidden="true">→</span></a>
      </div>
      <ul class="research-map">
        <li v-for="section in sections" :key="section.link">
          <a :href="withBase(section.link)" class="map-entry">
            <h3>{{ section.title }}</h3>
            <p>{{ section.description }}</p>
            <span class="map-arrow" aria-hidden="true">→</span>
          </a>
        </li>
      </ul>
    </section>
  </div>
</template>
