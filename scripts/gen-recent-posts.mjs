// 生成「最近研究」数据 → docs/.vitepress/data/recent-posts.json
//
// 扫描技术专题下的全部文章，读取 frontmatter 的 date 字段，
// 按日期倒序取最近 N 篇（默认 12），供 /posts/ 首页「最近研究」区渲染。
// 仅收录技术专题（博文），非技术知识树条目不进入研究动态。

import { readFileSync, readdirSync, writeFileSync, existsSync } from 'node:fs'

const ROOT = new URL('../docs/', import.meta.url).pathname
const techTopics = JSON.parse(
  readFileSync(new URL('../docs/.vitepress/data/tech-topics.json', import.meta.url), 'utf-8'),
)
const DOMAIN_NAMES = {
  'math-physics': '数理基础',
  cs: '计算机科学',
  ai: 'AI 与大模型',
  engineering: '工程技术',
}
const LIMIT = Number(process.argv[2] || 12)

function frontmatterOf(file) {
  const head = readFileSync(file, 'utf-8').slice(0, 600)
  const m = head.match(/^---\n([\s\S]*?)\n---/)
  if (!m) return {}
  const fm = {}
  for (const line of m[1].split('\n')) {
    const kv = line.match(/^([A-Za-z_-]+):\s*(.+)$/)
    if (kv) fm[kv[1]] = kv[2].trim().replace(/^["']|["']$/g, '')
  }
  return fm
}

const posts = []
for (const topic of techTopics.tech) {
  const dir = `${ROOT}posts/${topic}`
  if (!existsSync(dir)) continue
  const domain = techTopics.domains
    ? Object.entries(techTopics.domains).find(([, v]) => v.includes(topic))?.[0]
    : 'other'
  for (const f of readdirSync(dir)) {
    if (!f.endsWith('.md') || f === 'index.md') continue
    const slug = f.replace(/\.md$/, '')
    const fm = frontmatterOf(`${dir}/${f}`)
    const date = fm.date || ''
    if (!date) continue
    posts.push({
      path: `/posts/${topic}/${slug}`,
      title: fm.title || slug,
      category: topic,
      date,
      domain: DOMAIN_NAMES[domain] || '',
    })
  }
}

// 按日期倒序，同一天保持稳定顺序
posts.sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0))

// 每个专题只保留最新一篇，保证研究动态的跨专题多样性
const seen = new Set()
const diverse = []
for (const p of posts) {
  if (seen.has(p.category)) continue
  seen.add(p.category)
  diverse.push(p)
}

const out = { generated: 'by scripts/gen-recent-posts.mjs', limit: LIMIT, posts: diverse.slice(0, LIMIT) }
writeFileSync(
  new URL('../docs/.vitepress/data/recent-posts.json', import.meta.url),
  JSON.stringify(out, null, 2) + '\n',
)
console.log(`recent-posts.json 生成：技术文章 ${posts.length} 篇，取最近 ${Math.min(LIMIT, posts.length)} 篇`)
for (const p of out.posts.slice(0, 6)) console.log(`  ${p.date}  ${p.title}  [${p.domain}]`)
