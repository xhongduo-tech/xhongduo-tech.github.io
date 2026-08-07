import { defineConfig } from 'vitepress'
import taskLists from 'markdown-it-task-lists'

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
  appearance: false, // 暗色模式由自定义主题管理

  locales: {
    root: { lang: 'zh-CN', label: '中文' },
    en: { lang: 'en', label: 'English' },
  },

  head: [
    // 首屏前确定亮/暗主题（sessionStorage，跟随系统，语义与 Tufted-Blog 一致）
    [
      'script',
      {},
      `(function(){var t=null;try{t=sessionStorage.getItem('theme-preference')}catch(e){}if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}document.documentElement.setAttribute('data-theme',t)})()`,
    ],
    // 客户端 MathJax 配置：排版 \(...\) / \[...\]（含 mhchem 化学式）
    [
      'script',
      { type: 'text/javascript', id: 'MathJax-config' },
      `window.MathJax={tex:{inlineMath:[['\\\\\\\\(','\\\\\\\\)']],displayMath:[['\\\\[','\\\\]']]},svg:{fontCache:'global'},options:{skipHtmlTags:['script','noscript','style','textarea','pre','code']}};`,
    ],
    [
      'script',
      { type: 'text/javascript', id: 'MathJax-script', src: 'https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js', defer: true },
      '',
    ],
  ],

  markdown: {
    math: true, // MathJax（含 mhchem），直接写 $...$ / $$...$$ / \ce{...}
    lineNumbers: false, // 行号由主题 JS 生成（与模板一致）
    theme: 'min-light', // 颜色由主题 CSS 统一为素色（与模板一致）
    config: (md) => {
      md.use(taskLists) // 让 - [ ] / - [x] 渲染为复选框
      // 客户端 MathJax：mathjax3 只负责 tokenize 保护 $...$，渲染输出 \(...\) / \[...\]
      // 由浏览器端 MathJax 排版。避免服务端内联 SVG（10257 页导致构建 OOM / 页面臃肿）。
      const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      md.renderer.rules.math_inline = (tokens, idx) => '\\(' + esc(tokens[idx].content) + '\\)'
      md.renderer.rules.math_block = (tokens, idx) => '\\[' + esc(tokens[idx].content) + '\\]'
    },
  },
})
