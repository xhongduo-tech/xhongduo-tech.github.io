import { defineConfig } from 'vitepress'

// GitHub Pages base 路径：
// - 普通项目仓库（如 user/blog）-> '/blog/'
// - user.github.io 仓库或本地开发 -> '/'
const repo = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repo && !repo.endsWith('.github.io') ? `/${repo}/` : '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '从极限到大模型',
  description: '徐鸿铎的个人知识库：从数理基础到大模型的系统化写作计划',
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
            { text: '天文学', link: '/posts/foundations/astronomy/' },
            { text: '地球科学', link: '/posts/foundations/earth-science/' },
            { text: '认知科学', link: '/posts/foundations/cognitive-science/' },
            { text: '心理学基础', link: '/posts/foundations/psychology/' },
            { text: '逻辑学', link: '/posts/foundations/logic/' },
            { text: '科学史与科学哲学', link: '/posts/foundations/philosophy-of-science/' },
            { text: '经济学基础', link: '/posts/foundations/economics/' },
          ],
        },
        {
          text: '进阶数理',
          collapsed: false,
          items: [
            { text: '高等数学', link: '/posts/intermediate/advanced-math/' },
            { text: '数学分析', link: '/posts/intermediate/mathematical-analysis/' },
            { text: '概率论与数理统计', link: '/posts/intermediate/probability/' },
            { text: '随机过程', link: '/posts/intermediate/stochastic-processes/' },
            { text: '线性代数', link: '/posts/intermediate/linear-algebra/' },
            { text: '离散数学', link: '/posts/intermediate/discrete-math/' },
            { text: '复变函数与积分变换', link: '/posts/intermediate/complex-analysis/' },
            { text: '实变函数与测度论', link: '/posts/intermediate/real-analysis/' },
            { text: '泛函分析', link: '/posts/intermediate/functional-analysis/' },
            { text: '抽象代数', link: '/posts/intermediate/abstract-algebra/' },
            { text: '拓扑学', link: '/posts/intermediate/topology/' },
            { text: '微分几何', link: '/posts/intermediate/differential-geometry/' },
            { text: '偏微分方程', link: '/posts/intermediate/pde/' },
            { text: '数值分析', link: '/posts/intermediate/numerical-analysis/' },
            { text: '最优化理论', link: '/posts/intermediate/optimization/' },
            { text: '信息论', link: '/posts/intermediate/information-theory/' },
            { text: '高等物理', link: '/posts/intermediate/advanced-physics/' },
          ],
        },
        {
          text: '计算机基础',
          collapsed: false,
          items: [
            { text: '数据结构', link: '/posts/cs/data-structures/' },
            { text: '算法设计与分析', link: '/posts/cs/algorithms/' },
            { text: '程序设计语言', link: '/posts/cs/programming-languages/' },
            { text: '数字逻辑', link: '/posts/cs/digital-logic/' },
            { text: '计算机组成原理', link: '/posts/cs/computer-organization/' },
            { text: '计算机体系结构', link: '/posts/cs/computer-architecture/' },
            { text: '操作系统', link: '/posts/cs/os/' },
            { text: '计算机网络', link: '/posts/cs/computer-networks/' },
            { text: '数据库', link: '/posts/cs/database/' },
            { text: '分布式系统', link: '/posts/cs/distributed-systems/' },
            { text: '编译原理', link: '/posts/cs/compilers/' },
            { text: '软件工程', link: '/posts/cs/software-engineering/' },
            { text: '计算机图形学', link: '/posts/cs/computer-graphics/' },
            { text: '密码学与信息安全', link: '/posts/cs/cryptography-security/' },
          ],
        },
        {
          text: '高阶专题',
          collapsed: false,
          items: [
            { text: '机器学习', link: '/posts/advanced/machine-learning/' },
            { text: '深度学习', link: '/posts/advanced/deep-learning/' },
            { text: '强化学习', link: '/posts/advanced/reinforcement-learning/' },
            { text: '计算机视觉', link: '/posts/advanced/computer-vision/' },
            { text: '自然语言处理', link: '/posts/advanced/nlp/' },
            { text: '语音技术', link: '/posts/advanced/speech/' },
            { text: '信息检索', link: '/posts/advanced/information-retrieval/' },
            { text: '推荐系统', link: '/posts/advanced/recommender-systems/' },
            { text: '大模型原理', link: '/posts/advanced/llm-principles/' },
            { text: '大模型部署', link: '/posts/advanced/llm-deployment/' },
            { text: '大模型微调', link: '/posts/advanced/llm-finetuning/' },
            { text: 'AI 基础设施', link: '/posts/advanced/ai-infra/' },
            { text: 'AI 安全与对齐', link: '/posts/advanced/ai-safety/' },
            { text: '具身智能', link: '/posts/advanced/embodied-ai/' },
            { text: '自动驾驶', link: '/posts/advanced/autonomous-driving/' },
            { text: 'AI for Science', link: '/posts/advanced/ai4science/' },
            { text: '量子计算', link: '/posts/advanced/quantum-computing/' },
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
      message: '从极限到大模型 · From Limits to LLMs',
      copyright: 'Copyright © 2026 徐鸿铎',
    },
  },
})
