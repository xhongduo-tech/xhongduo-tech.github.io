import type { Theme } from 'vitepress'
import DefaultTheme from 'vitepress/theme-without-fonts'
import Layout from './Layout.vue'
import ArticleIndex from './ArticleIndex.vue'
import ArticlesHome from './ArticlesHome.vue'
import DomainTopology from './DomainTopology.vue'
import ProjectIndex from './ProjectIndex.vue'
import { LazyMermaid } from './mermaid'
import './style.css'
import './reading-media.css'

export default {
  extends: DefaultTheme,
  Layout,
  enhanceApp({ app }) {
    app.component('ArticleIndex', ArticleIndex)
    app.component('ArticlesHome', ArticlesHome)
    app.component('DomainTopology', DomainTopology)
    app.component('ProjectIndex', ProjectIndex)
    app.component('Mermaid', LazyMermaid)
  },
} satisfies Theme
