const { createMarkdownRenderer } = require('vitepress')
const { readFileSync } = require('node:fs')
const { execSync } = require('node:child_process')
;(async () => {
  const files = execSync("find docs/posts -name '*.md' ! -name 'index.md' | sort", { encoding: 'utf8' }).trim().split('\n')
  const md = await createMarkdownRenderer(process.cwd(), { math: true }, '/')
  const probe = readFileSync('docs/posts/foundations/math/set-concept.md', 'utf8').replace(/^---[\s\S]*?---\n/, '')
  for (const f of files) {
    const body = readFileSync(f, 'utf8').replace(/^---[\s\S]*?---\n/, '')
    try {
      md.render(body)
    } catch (e) {
      console.log('渲染本身失败:', f, '|', e.message)
      return
    }
    // probe: render set-concept (healthy math) — if this now fails, the previous file corrupted state
    try {
      md.render(probe)
    } catch (e) {
      console.log('污染源(渲染后探针失败):', f, '|', e.message)
      return
    }
  }
  console.log('无污染源（全部通过）')
})()
