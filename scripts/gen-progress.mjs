// 解析 docs/posts/*/*/index.md 中的章（###/####）与选题（- [ ] / - [x]），
// 生成 docs/.vitepress/data/progress.json 供 ProgressGrid / ProgressOverview 使用。
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'

const root = 'docs/posts'
const data = {}

for (const tier of readdirSync(root, { withFileTypes: true })) {
  if (!tier.isDirectory()) continue
  for (const cat of readdirSync(join(root, tier.name), { withFileTypes: true })) {
    if (!cat.isDirectory()) continue
    const file = join(root, tier.name, cat.name, 'index.md')
    let md
    try {
      md = readFileSync(file, 'utf8')
    } catch {
      continue
    }
    const name = (md.match(/^#\s+(.+)/m) || [])[1]?.trim() || cat.name
    const chapters = []
    let cur = null
    for (const line of md.split('\n')) {
      const h = line.match(/^#{3,4}\s+(.+)/)
      if (h) {
        cur = { title: h[1].trim(), items: [] }
        chapters.push(cur)
        continue
      }
      const it = line.match(/^- \[( |x)\]\s+(.+)/)
      if (it) {
        if (!cur) {
          cur = { title: '', items: [] }
          chapters.push(cur)
        }
        cur.items.push({ done: it[1] === 'x', title: it[2].trim() })
      }
    }
    const total = chapters.reduce((s, c) => s + c.items.length, 0)
    const done = chapters.reduce((s, c) => s + c.items.filter((i) => i.done).length, 0)
    data[`${tier.name}/${cat.name}`] = { name, chapters, total, done }
  }
}

mkdirSync('docs/.vitepress/data', { recursive: true })
writeFileSync('docs/.vitepress/data/progress.json', JSON.stringify(data))
console.log(`progress.json: ${Object.keys(data).length} categories`)
