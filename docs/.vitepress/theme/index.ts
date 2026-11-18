import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import Layout from './Layout.vue'
import ArticleIndex from './ArticleIndex.vue'
import { LazyMermaid } from './mermaid'
import './style.css'
import './reading-media.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('ArticleIndex', ArticleIndex)
    app.component('Mermaid', LazyMermaid)
  },
} satisfies Theme
