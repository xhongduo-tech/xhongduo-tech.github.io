import { defineConfig } from 'vitepress'

// GitHub Pages base 路径：
// - 普通项目仓库（如 user/blog）-> '/blog/'
// - user.github.io 仓库或本地开发 -> '/'
const repo = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repo && !repo.endsWith('.github.io') ? `/${repo}/` : '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '我的博客', // TODO: 改成你的站点名
  description: '数学 · 物理 · 计算机 · 大模型', // TODO: 改成你的简介
  cleanUrls: true,
  lastUpdated: true,

  markdown: {
    math: true, // 内置数学公式支持，直接写 $...$ / $$...$$
    lineNumbers: true,
  },

  themeConfig: {
    nav: [
      { text: '首页', link: '/' },
      { text: '博文', link: '/posts/' },
      { text: '项目', link: '/projects/' },
      { text: '关于我', link: '/about/' },
    ],

    sidebar: {
      '/posts/': [
        { text: '博文总览', link: '/posts/' },
        { text: '如何发布博文', link: '/posts/how-to-publish' },
        {
          text: '基础科学',
          collapsed: false,
          items: [
            { text: '基础数学', link: '/posts/foundations/math/' },
            { text: '基础物理', link: '/posts/foundations/physics/' },
            { text: '化学', link: '/posts/foundations/chemistry/' },
            { text: '生物', link: '/posts/foundations/biology/' },
          ],
        },
        {
          text: '进阶数理',
          collapsed: false,
          items: [
            { text: '高等数学', link: '/posts/intermediate/advanced-math/' },
            { text: '概率论与数理统计', link: '/posts/intermediate/probability/' },
            { text: '线性代数', link: '/posts/intermediate/linear-algebra/' },
            { text: '高等物理', link: '/posts/intermediate/advanced-physics/' },
          ],
        },
        {
          text: '计算机基础',
          collapsed: false,
          items: [
            { text: '数据结构', link: '/posts/cs/data-structures/' },
            { text: '计算机组成原理', link: '/posts/cs/computer-organization/' },
            { text: '操作系统', link: '/posts/cs/os/' },
            { text: '数据库', link: '/posts/cs/database/' },
          ],
        },
        {
          text: '高阶专题',
          collapsed: false,
          items: [
            { text: '机器学习', link: '/posts/advanced/machine-learning/' },
            { text: '深度学习', link: '/posts/advanced/deep-learning/' },
            { text: '大模型原理', link: '/posts/advanced/llm-principles/' },
            { text: '大模型部署', link: '/posts/advanced/llm-deployment/' },
            { text: '大模型微调', link: '/posts/advanced/llm-finetuning/' },
            { text: '本体论', link: '/posts/advanced/ontology/' },
          ],
        },
      ],
    },

    outline: { level: [2, 3], label: '本页目录' },
    lastUpdated: { text: '最后更新' },
    docFooter: { prev: '上一篇', next: '下一篇' },
    search: { provider: 'local' },
    socialLinks: [
      { icon: 'github', link: 'https://github.com/xhongduo-tech' },
    ],
    footer: {
      message: 'Simple is professional.',
      copyright: 'Copyright © 2026',
    },
  },
})
