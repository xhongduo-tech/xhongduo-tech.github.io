import Layout from './Layout.vue'
import NotFound from './NotFound.vue'
import PostList from './PostList.vue'
import KnowledgeTree from './KnowledgeTree.vue'
import LessonNav from './LessonNav.vue'
import SectionMap from './SectionMap.vue'
import LearnPath from './LearnPath.vue'
import DirectionCards from './DirectionCards.vue'
import './tufte-base.css'
import './site.css'

export default {
  Layout,
  NotFound,
  enhanceApp({ app }) {
    app.component('PostList', PostList)
    app.component('KnowledgeTree', KnowledgeTree)
    app.component('LessonNav', LessonNav)
    app.component('SectionMap', SectionMap)
    app.component('LearnPath', LearnPath)
    app.component('DirectionCards', DirectionCards)
  },
}
