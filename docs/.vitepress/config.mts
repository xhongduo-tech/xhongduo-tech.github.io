import { defineConfig } from 'vitepress'
import { withLazyMermaid } from './mermaid-lazy'

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repository && !repository.endsWith('.github.io') ? `/${repository}/` : '/'
const siteName = "xuhongduo's blog"
const siteDescription = '记录大模型原理、训练、推理与评测的学习过程，用源码和实验验证理解。'

const config = withLazyMermaid(
  defineConfig({
    base,
    lang: 'zh-CN',
    title: siteName,
    titleTemplate: `:title · ${siteName}`,
    description: siteDescription,
    cleanUrls: true,
    lastUpdated: false,
    appearance: true,
    metaChunk: true,
    mermaid: {
      fontFamily: 'var(--vp-font-family-base)',
      theme: 'neutral',
      securityLevel: 'strict',
      flowchart: { nodeSpacing: 24, rankSpacing: 36, padding: 11 },
    },
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
      ['meta', { name: 'theme-color', content: '#faf9f5' }],
      ['meta', { name: 'author', content: '徐鸿铎' }],
    ],
    markdown: {
      math: true,
      lineNumbers: false,
      theme: { light: 'github-dark', dark: 'github-dark' },
      config(md) {
        md.renderer.rules.table_open = () =>
          '<div class="reading-table" role="group" aria-label="表格" tabindex="0"><table>\n'
        md.renderer.rules.table_close = () => '</table></div>\n'

        const renderFence = md.renderer.rules.fence
        if (renderFence) {
          md.renderer.rules.fence = (...args) =>
            renderFence(...args).replace(/<pre(?=\s|>)(?![^>]*\btabindex=)/, '<pre tabindex="0"')
        }
      },
    },
    themeConfig: {
      siteTitle: siteName,
      nav: [
        { text: '文章', link: '/articles/', activeMatch: '^/articles(/|$)' },
        { text: '项目', link: '/projects/', activeMatch: '^/projects(/|$)' },
        { text: '关于我', link: '/about' },
      ],
      sidebar: false,
      outline: { level: [2, 3], label: '本页目录' },
      docFooter: { prev: '上一篇', next: '下一篇' },
      sidebarMenuLabel: '浏览文章',
      returnToTopLabel: '返回顶部',
      darkModeSwitchLabel: '外观',
      lightModeSwitchTitle: '切换到浅色模式',
      darkModeSwitchTitle: '切换到深色模式',
      skipToContentLabel: '跳到正文',
      notFound: {
        title: '这页笔记还没有找到',
        quote: '可以返回首页，或在文章页查看已经发布的内容。',
        linkLabel: '返回首页',
        linkText: '返回首页',
      },
      search: {
        provider: 'local',
        options: {
          translations: {
            button: { buttonText: '搜索笔记', buttonAriaLabel: '搜索笔记' },
            modal: {
              noResultsText: '没有找到相关笔记',
              resetButtonTitle: '清除搜索',
              backButtonTitle: '返回',
              displayDetails: '显示摘要',
              footer: { selectText: '打开', navigateText: '切换', closeText: '关闭' },
            },
          },
          miniSearch: {
            options: {
              tokenize: (text: string) => {
                const normalized = text.toLowerCase()
                const latin = normalized.match(/[a-z0-9]+/g) ?? []
                const chinese = normalized.match(/[\p{Script=Han}]+/gu) ?? []
                const terms = [...latin]
                for (const run of chinese) {
                  const chars = Array.from(run)
                  terms.push(...chars)
                  for (let index = 0; index < chars.length - 1; index++) {
                    terms.push(chars[index] + chars[index + 1])
                  }
                }
                return [...new Set(terms)]
              },
            },
            searchOptions: { prefix: true, fuzzy: false, combineWith: 'AND', boost: { title: 4, titles: 2, text: 1 } },
          },
        },
      },
      footer: {
        copyright: siteName,
      },
    },
    locales: {
      root: { label: '中文', lang: 'zh-CN' },
      en: {
        label: 'English',
        lang: 'en-US',
        themeConfig: {
          nav: [
            { text: 'Articles', link: '/en/articles/', activeMatch: '^/en/articles(/|$)' },
            { text: 'Projects', link: '/en/projects/', activeMatch: '^/en/projects(/|$)' },
            { text: 'About', link: '/en/about' },
          ],
          outline: { level: [2, 3], label: 'On this page' },
          docFooter: { prev: 'Previous', next: 'Next' },
          sidebarMenuLabel: 'Browse',
          returnToTopLabel: 'Back to top',
          darkModeSwitchLabel: 'Appearance',
          lightModeSwitchTitle: 'Switch to light mode',
          darkModeSwitchTitle: 'Switch to dark mode',
          skipToContentLabel: 'Skip to content',
          notFound: {
            title: 'Page not found',
            quote: 'Return home or browse the published articles.',
            linkLabel: 'Go home',
            linkText: 'Go home',
          },
          search: {
            provider: 'local',
            options: {
              translations: {
                button: { buttonText: 'Search', buttonAriaLabel: 'Search notes' },
                modal: {
                  noResultsText: 'No results for',
                  resetButtonTitle: 'Clear search',
                  backButtonTitle: 'Close search',
                  displayDetails: 'Show details',
                  footer: { selectText: 'Open', navigateText: 'Navigate', closeText: 'Close' },
                },
              },
            },
          },
        },
      },
    },
  }),
)

// Prebundle Mermaid's UMD dependencies for the development server.
// The plugin also includes an optional debug package that this site does not use.
config.vite ??= {}
config.vite.optimizeDeps ??= {}
config.vite.optimizeDeps.include = [...new Set([
  ...(config.vite.optimizeDeps.include ?? []).filter((dependency) => dependency !== 'debug'),
  'fastdom',
  'fastdom/extensions/fastdom-promised.js',
])]

export default config
