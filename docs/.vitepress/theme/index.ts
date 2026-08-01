import DefaultTheme from 'vitepress/theme'
import ProgressGrid from './ProgressGrid.vue'
import ProgressOverview from './ProgressOverview.vue'
import './custom.css'

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ProgressGrid', ProgressGrid)
    app.component('ProgressOverview', ProgressOverview)
  },
}
