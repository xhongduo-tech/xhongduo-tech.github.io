<script setup>
import { withBase } from 'vitepress'
</script>

<template>
  <div class="tuf-layout">
    <main class="tuf-main">
      <article class="tuf-doc">
        <h1>404</h1>
        <p>这一页不存在。回到 <a :href="withBase('/')">首页</a> 或 <a :href="withBase('/posts/')">博文总览</a>。</p>
      </article>
    </main>
  </div>
</template>
