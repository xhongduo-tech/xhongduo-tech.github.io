const { createMarkdownRenderer } = require('vitepress')
const { readFileSync } = require('node:fs')
const { execSync } = require('node:child_process')
;(async () => {
  const files = execSync("find docs/posts -name '*.md' ! -name 'index.md' | sort", { encoding: 'utf8' }).trim().split('\n')
  const md = await createMarkdownRenderer(process.cwd(), { math: true }, '/')
  let failed = null
  for (const f of files) {
    const body = readFileSync(f, 'utf8').replace(/^---[\s\S]*?---\n/, '')
    try {
      md.render(body)
    } catch (e) {
      failed = f
      console.log('失败于:', f, '|', e.message)
      break
    }
  }
  if (!failed) console.log('全部通过（单实例按序渲染无失败）')
})()
