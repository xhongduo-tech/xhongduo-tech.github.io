import DefaultTheme from 'vitepress/theme'
import Home from './Home.vue'
import HomeStats from './HomeStats.vue'
import ProgressGrid from './ProgressGrid.vue'
import ProgressOverview from './ProgressOverview.vue'
import ProjectList from './ProjectList.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('Home', Home)
    app.component('HomeStats', HomeStats)
    app.component('ProgressGrid', ProgressGrid)
    app.component('ProgressOverview', ProgressOverview)
    app.component('ProjectList', ProjectList)
  },
}
