import { defineAsyncComponent, defineComponent, h } from 'vue'

// Loading starts only when a page renders a Mermaid fence. Keeping the original
// component preserves its configuration, dark-mode updates and SVG rendering.
export const LazyMermaid = defineAsyncComponent(async () => {
  const { default: Mermaid } = await import('vitepress-plugin-mermaid/Mermaid.vue')

  return defineComponent({
    name: 'ReadingMermaid',
    inheritAttrs: false,
    setup(_, { attrs }) {
      // The upstream component has one div root. These attributes fall through
      // to that existing scroll container; no extra wrapper or DOM observer.
      return () => h(Mermaid, {
        ...attrs,
        tabindex: attrs.tabindex ?? 0,
        role: attrs.role ?? 'region',
        'aria-label': attrs['aria-label'] ?? '图表，较宽时可用左右方向键滚动',
      })
    },
  })
})
