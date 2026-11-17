import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repository && !repository.endsWith('.github.io') ? `/${repository}/` : '/'

const sidebar = [
  {
    text: '从这里开始',
    items: [
      { text: '研究地图', link: '/' },
      { text: '关于这些笔记', link: '/about' },
    ],
  },
  {
    text: '模型基础',
    items: [{ text: '基础知识与阅读路径', link: '/foundations/' }],
  },
  {
    text: '推理与系统',
    items: [
      { text: '推理专题地图', link: '/inference/' },
      { text: '投机采样', link: '/inference/speculative-decoding' },
      { text: '多 Token 预测 MTP', link: '/inference/mtp' },
      { text: 'DSpark', link: '/inference/dspark' },
      { text: 'DeepInfer', link: '/inference/deepinfer' },
    ],
  },
  {
    text: '训练与评测',
    items: [
      { text: '训练与后训练', link: '/training/' },
      { text: '模型评测', link: '/evaluation/' },
    ],
  },
  {
    text: '项目实践',
    items: [
      { text: '项目总览', link: '/projects/' },
      { text: '推理算法与服务实验', link: '/projects/serving-lab' },
      { text: '数据与监督微调', link: '/projects/finetuning-lab' },
      { text: '偏好优化与后训练', link: '/projects/post-training-lab' },
    ],
  },
  {
    text: '文章与写作',
    items: [
      { text: '已发表博文', link: '/articles/' },
      { text: '文章与写作规范', link: '/writing/' },
      { text: '文章模板', link: '/writing/article-template' },
    ],
  },
]

const config = withMermaid(
  defineConfig({
    base,
    lang: 'zh-CN',
    title: '徐鸿铎',
    titleTemplate: ':title · LLM Research Notes',
    description: '从模型机制到工程实践，记录大模型基础、推理、训练与评测的学习过程。',
    cleanUrls: true,
    lastUpdated: false,
    appearance: true,
    metaChunk: true,
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
      ['meta', { name: 'theme-color', content: '#faf9f5' }],
      ['meta', { name: 'author', content: '徐鸿铎' }],
    ],
    markdown: {
      math: true,
      lineNumbers: false,
      theme: { light: 'github-light', dark: 'github-dark' },
    },
    themeConfig: {
      siteTitle: '徐鸿铎',
      nav: [
        { text: '文章', link: '/articles/' },
        {
          text: '研究',
          items: [
            { text: '模型基础', link: '/foundations/' },
            { text: '推理与系统', link: '/inference/' },
            { text: '训练与后训练', link: '/training/' },
            { text: '评测与实验', link: '/evaluation/' },
          ],
        },
        { text: '项目', link: '/projects/' },
        { text: '关于', link: '/about' },
      ],
      sidebar,
      outline: { level: [2, 3], label: '本页目录' },
      docFooter: { prev: '上一节', next: '下一节' },
      sidebarMenuLabel: '浏览专题',
      returnToTopLabel: '返回顶部',
      darkModeSwitchLabel: '外观',
      lightModeSwitchTitle: '切换到浅色模式',
      darkModeSwitchTitle: '切换到深色模式',
      skipToContentLabel: '跳到正文',
      notFound: {
        title: '这页笔记还没有找到',
        quote: '可以返回研究地图，或使用搜索寻找相关主题。',
        linkLabel: '返回研究地图',
        linkText: '返回研究地图',
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
        copyright: '徐鸿铎 · LLM Research Notes',
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
