import { defineConfig } from 'vitepress'

const repo = process.env.GITHUB_REPOSITORY?.split('/')[1]
const base = repo && !repo.endsWith('.github.io') ? `/${repo}/` : '/'

export default defineConfig({
  base,
  lang: 'zh-CN',
  title: '从极限到大模型',
  description: '徐鸿铎 · 大模型架构',
  cleanUrls: true,
  lastUpdated: true,
  appearance: false,
  metaChunk: true,

  head: [
    [
      'script',
      {},
      `(function(){var t=null;try{t=sessionStorage.getItem('theme-preference')}catch(e){}if(!t){t=window.matchMedia('(prefers-color-scheme: dark)').matches?'dark':'light'}document.documentElement.setAttribute('data-theme',t)})()`,
    ],
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
    math: true,
    lineNumbers: false,
    theme: 'min-light',
    config: (md) => {
      const esc = (s: string) =>
        s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\{/g, '&#123;').replace(/\}/g, '&#125;')
      md.renderer.rules.math_inline = (tokens, idx) => '\\(' + esc(tokens[idx].content) + '\\)'
      md.renderer.rules.math_block = (tokens, idx) => '\\[' + esc(tokens[idx].content) + '\\]'
    },
  },
})
