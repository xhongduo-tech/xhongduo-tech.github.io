const { createMarkdownRenderer } = require('vitepress')
const { readFileSync } = require('node:fs')
const { execSync } = require('node:child_process')

function findUnbalanced(html) {
  const VOID = new Set(['br', 'img', 'hr', 'input', 'meta', 'link', 'area', 'base', 'col', 'embed', 'source', 'track', 'wbr'])
  const stack = []
  const re = /<\/?([a-zA-Z][a-zA-Z0-9-]*)((?:"[^"]*"|[^"'>])*?)(\/?)>/g
  let m
  const problems = []
  while ((m = re.exec(html))) {
    const [full, name, , self] = m
    if (full.startsWith('</')) {
      const idx = stack.lastIndexOf(name)
      if (idx >= 0) {
        for (let k = stack.length - 1; k > idx; k--) problems.push(`未闭合 ${stack[k]} 却先闭合 ${name}`)
        stack.length = idx
      } else {
        problems.push(`孤立闭合标签 ${name}`)
      }
    } else if (!self && !VOID.has(name)) {
      stack.push(name)
    }
  }
  if (stack.length) problems.push(`最终未闭合: ${stack.join('>')}`)
  return problems
}

;(async () => {
  const files = execSync("find docs/posts -name '*.md' ! -name 'index.md'", { encoding: 'utf8' }).trim().split('\n')
  const md = await createMarkdownRenderer(process.cwd(), { math: true }, '/')
  const broken = []
  for (const f of files) {
    const body = readFileSync(f, 'utf8').replace(/^---[\s\S]*?---\n/, '')
    try {
      const html = md.render(body)
      const probs = findUnbalanced(html)
      if (probs.length) broken.push(`${f}  [${probs[0]}]`)
    } catch (e) {
      broken.push(`${f}  [render异常: ${e.message}]`)
    }
  }
  console.log('HTML 失衡文件数:', broken.length)
  broken.slice(0, 40).forEach(b => console.log(' -', b))
})()
