<script setup>
import { withBase } from 'vitepress'
</script>

<template>
  <div>
    <header class="site-header">
      <nav class="site-nav">
        <a :href="withBase('/')">首页</a>
      </nav>
    </header>
    <article class="tuf-article">
      <section>
        <h1>404</h1>
        <p>这一页不存在。回到 <a :href="withBase('/')">首页</a> 或 <a :href="withBase('/posts/')">博文总览</a>。</p>
      </section>
    </article>
  </div>
</template>
