<script setup lang="ts">
import { withBase } from 'vitepress'
import { sections } from '../data/site'
import { data as articles } from '../data/articles.data'
import ArticleIndex from './ArticleIndex.vue'
import ResearchIcon from './ResearchIcon.vue'
</script>

<template>
  <div class="research-home">
    <header class="home-intro">
      <p class="hero-eyebrow" lang="en"><span aria-hidden="true" />LLM RESEARCH NOTES</p>
      <h1 lang="en"><span class="hero-word">Token<span class="hero-separator">&nbsp;·</span></span> <span class="hero-word">Attention<span class="hero-separator">&nbsp;·</span></span> <span class="hero-word">Inference</span></h1>
      <p class="home-description">记录模型、训练与推理系统的学习和实践。</p>
    </header>

    <section class="home-section" aria-labelledby="writing-title">
      <div class="section-heading">
        <div><span class="section-eyebrow" lang="en">Articles</span><h2 id="writing-title">文章 <span class="article-count">{{ articles.length }}</span></h2></div>
        <a class="section-link" :href="withBase('/articles/')">全部文章 <span aria-hidden="true">→</span></a>
      </div>
      <ArticleIndex :limit="5" />
    </section>

    <section id="research-map" class="home-section" aria-labelledby="map-title">
      <div class="section-heading">
        <div><span class="section-eyebrow" lang="en">Research areas</span><h2 id="map-title">研究方向</h2></div>
        <a class="section-link" :href="withBase('/projects/')">实验项目 <span aria-hidden="true">→</span></a>
      </div>
      <ul class="research-map">
        <li v-for="section in sections" :key="section.link">
          <a :href="withBase(section.link)" class="map-entry">
            <div class="map-label">
              <ResearchIcon :name="section.icon" />
              <div><span class="map-english" lang="en">{{ section.english }}</span><h3>{{ section.title }}</h3></div>
            </div>
            <p>{{ section.description }}</p>
            <span class="map-arrow" aria-hidden="true">→</span>
          </a>
        </li>
      </ul>
    </section>
  </div>
</template>
