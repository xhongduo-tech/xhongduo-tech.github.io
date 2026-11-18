import { defineConfig } from 'vitepress'
import { withLazyMermaid } from './mermaid-lazy'

const repository = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repository && !repository.endsWith('.github.io') ? `/${repository}/` : '/'
const siteName = "xuhongduo's blog"
const siteDescription = '记录模型、训练与推理系统的学习和实践。'

const foundations = {
  text: '模型基础',
  items: [
    { text: '基础知识与阅读路径', link: '/foundations/' },
    { text: '编程、数学与训练机制', link: '/foundations/programming-math' },
    { text: 'Transformer与架构演进', link: '/foundations/transformer-architectures' },
  ],
}
const inference = {
  text: '推理与系统',
  items: [
    { text: '推理专题地图', link: '/inference/' },
    { text: 'KV缓存与Attention执行', link: '/inference/kv-cache' },
    { text: '量化与低精度', link: '/inference/quantization' },
    { text: '投机采样', link: '/inference/speculative-decoding' },
    { text: '多 Token 预测 MTP', link: '/inference/mtp' },
    { text: 'DSpark', link: '/inference/dspark' },
    { text: 'DeepInfer', link: '/inference/deepinfer' },
    { text: '服务、调度与分布式推理', link: '/inference/serving-systems' },
  ],
}
const training = {
  text: '训练与后训练',
  items: [
    { text: '训练专题地图', link: '/training/' },
    { text: '数据与预训练', link: '/training/data-pretraining' },
    { text: '分布式训练与恢复', link: '/training/distributed-training' },
    { text: 'SFT与参数高效适配', link: '/training/sft' },
    { text: '偏好优化与DPO', link: '/training/preference-optimization' },
    { text: 'RL、推理能力与蒸馏', link: '/training/rl-training' },
  ],
}
const evaluation = {
  text: '评测与实验',
  items: [
    { text: '评测与实验方法', link: '/evaluation/' },
    { text: '质量与服务实验', link: '/evaluation/quality-systems' },
    { text: '鲁棒性、安全与解释', link: '/evaluation/robustness-safety' },
  ],
}
const applications = {
  text: '应用与多模态',
  items: [
    { text: '应用主题地图', link: '/applications/' },
    { text: 'RAG与Agent', link: '/applications/retrieval-agents' },
    { text: '多模态机制与成本', link: '/applications/multimodal' },
  ],
}
const projects = {
  text: '项目实践',
  items: [
    { text: '项目总览', link: '/projects/' },
    { text: '推理算法与服务实验', link: '/projects/serving-lab' },
    { text: '数据与监督微调', link: '/projects/finetuning-lab' },
    { text: '偏好优化与后训练', link: '/projects/post-training-lab' },
  ],
}
const writing = {
  text: '文章与写作',
  items: [
    { text: '已发表博文', link: '/articles/' },
    { text: '文章与写作规范', link: '/writing/' },
    { text: '文章模板', link: '/writing/article-template' },
  ],
}

const researchAreas = [foundations, inference, training, evaluation, applications]
const sidebar = {
  '/roadmap/': [{
    text: '研究方向',
    items: researchAreas.map(({ text, items }) => ({ text, link: items[0].link })),
  }],
  '/foundations/': [foundations],
  '/inference/': [inference],
  '/training/': [training],
  '/evaluation/': [evaluation],
  '/applications/': [applications],
  '/projects/': [projects],
  '/articles/': [writing],
  '/writing/': [writing],
}

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
    head: [
      ['link', { rel: 'icon', type: 'image/svg+xml', href: `${base}favicon.svg` }],
      ['meta', { name: 'theme-color', content: '#faf9f5' }],
      ['meta', { name: 'author', content: '徐鸿铎' }],
    ],
    markdown: {
      math: true,
      lineNumbers: false,
      theme: { light: 'github-light', dark: 'github-dark' },
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
        { text: '文章', link: '/articles/', activeMatch: '^/(articles|writing)(/|$)' },
        {
          text: '研究',
          activeMatch: '^/(roadmap|foundations|inference|training|evaluation|applications)(/|$)',
          items: [
            { text: '完整知识地图', link: '/roadmap/' },
            { text: '模型基础', link: '/foundations/' },
            { text: '推理与系统', link: '/inference/' },
            { text: '训练与后训练', link: '/training/' },
            { text: '评测与实验', link: '/evaluation/' },
            { text: '应用与多模态', link: '/applications/' },
          ],
        },
        { text: '项目', link: '/projects/', activeMatch: '^/projects(/|$)' },
        { text: '关于我', link: '/about' },
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
        quote: '可以返回首页，或使用搜索寻找相关主题。',
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
