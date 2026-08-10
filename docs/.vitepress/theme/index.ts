import Layout from './Layout.vue'
import NotFound from './NotFound.vue'
import HomeStats from './HomeStats.vue'
import ProgressGrid from './ProgressGrid.vue'
import ProgressOverview from './ProgressOverview.vue'
import ProjectList from './ProjectList.vue'
import KnowledgeTree from './KnowledgeTree.vue'
import Entertainment from './Entertainment.vue'
import './fonts.css'
import './tufte-base.css'
import './tufted.css'
import './theme.css'
import './custom.css'

export default {
  Layout,
  NotFound,
  enhanceApp({ app }) {
    app.component('HomeStats', HomeStats)
    app.component('ProgressGrid', ProgressGrid)
    app.component('ProgressOverview', ProgressOverview)
    app.component('ProjectList', ProjectList)
    app.component('KnowledgeTree', KnowledgeTree)
    app.component('Entertainment', Entertainment)
  },
}
