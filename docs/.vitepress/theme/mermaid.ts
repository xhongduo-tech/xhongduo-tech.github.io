import { defineAsyncComponent } from 'vue'

// Loading starts only when a page renders a Mermaid fence. Keeping the original
// component preserves its configuration, dark-mode updates and SVG rendering.
export const LazyMermaid = defineAsyncComponent(() =>
  import('vitepress-plugin-mermaid/Mermaid.vue'),
)
