import Layout from './Layout.vue'
import NotFound from './NotFound.vue'
import HomeStats from './HomeStats.vue'
import ProgressGrid from './ProgressGrid.vue'
import ProgressOverview from './ProgressOverview.vue'
import ProjectList from './ProjectList.vue'
import './custom.css'

export default {
  Layout,
  NotFound,
  enhanceApp({ app }) {
    app.component('HomeStats', HomeStats)
    app.component('ProgressGrid', ProgressGrid)
    app.component('ProgressOverview', ProgressOverview)
    app.component('ProjectList', ProjectList)
  },
}
