const { createMarkdownRenderer } = require('vitepress')
const { readFileSync } = require('node:fs')
const { execSync } = require('node:child_process')
;(async () => {
  // 重新生成可疑文件列表（同 scantag2 逻辑）
  const files = execSync("find docs/posts -name '*.md' ! -name 'index.md'", {encoding:'utf8'}).trim().split('\n')
  const suspicious = new Set()
  for (const f of files) {
    const lines = readFileSync(f, 'utf8').split('\n')
    lines.forEach((ln) => {
      if (ln.trim().startsWith('$$')) return
      if (/\$[^$]*<\/?[a-z]+>[^$]*\$/.test(ln)) suspicious.add(f)
    })
  }
  const md = await createMarkdownRenderer(process.cwd(), { math: true }, '/')
  let real = []
  for (const f of [...suspicious]) {
    const body = readFileSync(f, 'utf8').replace(/^---[\s\S]*?---\n/, '')
    try { md.render(body) } catch (e) { real.push(f) }
  }
  console.log('真实失败文件数:', real.length)
  real.forEach(f => console.log(' -', f))
})()
