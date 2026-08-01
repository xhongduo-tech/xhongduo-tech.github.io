import DefaultTheme from 'vitepress/theme'
import HomeStats from './HomeStats.vue'
import ProgressGrid from './ProgressGrid.vue'
import ProgressOverview from './ProgressOverview.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('HomeStats', HomeStats)
    app.component('ProgressGrid', ProgressGrid)
    app.component('ProgressOverview', ProgressOverview)
  },
}
