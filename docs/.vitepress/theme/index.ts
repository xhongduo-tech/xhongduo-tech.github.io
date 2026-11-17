import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import Layout from './Layout.vue'
import ArticleIndex from './ArticleIndex.vue'
import './style.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('ArticleIndex', ArticleIndex)
  },
} satisfies Theme
