<script setup lang="ts">
import { withBase } from 'vitepress'
import { sections, focusTopics, projects } from '../data/site'
import { data as articles } from '../data/articles.data'
import ArticleIndex from './ArticleIndex.vue'
</script>

<template>
  <div class="research-home">
    <section class="home-intro" aria-labelledby="intro-title">
      <div class="intro-main">
        <p class="eyebrow"><span class="eyebrow-line" />LLM RESEARCH NOTES</p>
        <h1 id="intro-title">从模型机制，<br>走向工程实践<span class="title-period">。</span></h1>
        <p class="intro-description">我是徐鸿铎。这里记录大模型基础、推理、训练与评测的学习过程，让理解逐步落到代码、实验和可复查的笔记中。</p>
        <div class="intro-actions">
          <a class="text-action primary-action" href="#research-map">探索研究地图 <span aria-hidden="true">↗</span></a>
          <a class="text-action" :href="withBase('/about')">关于这些笔记 <span aria-hidden="true">→</span></a>
        </div>
      </div>
      <aside class="intro-coordinate" aria-label="笔记的组织方式">
        <span class="coordinate-caption">A WORKING NOTEBOOK</span>
        <p class="coordinate-title">原理 · 系统 · 实验</p>
        <div class="coordinate-line"><span>01</span><p>理解机制<small>模型为什么这样工作</small></p></div>
        <div class="coordinate-line"><span>02</span><p>连接实现<small>机制怎样落到代码</small></p></div>
        <div class="coordinate-line"><span>03</span><p>留下证据<small>用比较与记录检验理解</small></p></div>
        <p class="coordinate-note">持续整理，按实际进展更新。</p>
      </aside>
    </section>

    <section id="research-map" class="home-section map-section" aria-labelledby="map-title">
      <div class="section-heading">
        <div><p class="section-kicker">01 / RESEARCH MAP</p><h2 id="map-title">专题地图</h2></div>
        <p class="section-description">沿着问题建立联系，<br>从基础出发，也回到基础。</p>
      </div>
      <div class="research-map">
        <a v-for="section in sections" :key="section.link" :href="withBase(section.link)" class="map-entry">
          <span class="map-number">{{ section.number }}</span>
          <div><span class="map-english">{{ section.english }}</span><h3>{{ section.title }}</h3><p>{{ section.description }}</p></div>
          <span class="map-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>

    <section class="home-section focus-section" aria-labelledby="focus-title">
      <div class="section-heading">
        <div><p class="section-kicker">TOPICS TO EXPLORE</p><h2 id="focus-title">先把问题写清楚</h2></div>
        <p class="section-description">以下是专题提纲，<br>尚待阅读、复现与实践。</p>
      </div>
      <div class="focus-list">
        <a v-for="topic in focusTopics" :key="topic.link" :href="withBase(topic.link)" class="focus-entry">
          <div class="focus-name"><span>{{ topic.english }}</span><h3>{{ topic.title }}</h3></div>
          <p>{{ topic.detail }}</p>
          <span class="status-badge">专题提纲 · 待研究</span>
          <span class="focus-arrow" aria-hidden="true">↗</span>
        </a>
      </div>
    </section>

    <section class="home-section projects-section" aria-labelledby="projects-title">
      <div class="section-heading">
        <div><p class="section-kicker">02 / LAB NOTES</p><h2 id="projects-title">在项目中检验理解</h2></div>
        <a class="text-action" :href="withBase('/projects/')">项目总览 <span aria-hidden="true">→</span></a>
      </div>
      <div class="project-grid">
        <a v-for="project in projects" :key="project.link" :href="withBase(project.link)" class="project-entry">
          <div class="project-topline"><span class="project-number">{{ project.number }}</span><span class="status-badge">待实践</span></div>
          <span class="project-english">{{ project.english }}</span>
          <h3>{{ project.title }}</h3>
          <p>{{ project.detail }}</p>
          <span class="project-link">查看任务与验收 <span aria-hidden="true">→</span></span>
        </a>
      </div>
    </section>

    <section class="home-section writing-section" aria-labelledby="writing-title">
      <div class="section-heading">
        <div><p class="section-kicker">03 / WRITING</p><h2 id="writing-title">文章 <span class="article-count">{{ articles.length }}</span></h2></div>
        <a class="text-action" :href="withBase('/writing/')">文章与写作规范 <span aria-hidden="true">→</span></a>
      </div>
      <ArticleIndex :limit="4" />
    </section>

    <div class="home-colophon"><span>徐鸿铎</span><p>在理解、实现与验证之间，慢慢建立自己的判断。</p><a :href="withBase('/about')">关于 <span aria-hidden="true">↗</span></a></div>
  </div>
</template>
