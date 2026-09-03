import Layout from './Layout.vue'
import NotFound from './NotFound.vue'
import PostList from './PostList.vue'
import './tufte-base.css'
import './site.css'

export default {
  Layout,
  NotFound,
  enhanceApp({ app }) {
    app.component('PostList', PostList)
  },
}
