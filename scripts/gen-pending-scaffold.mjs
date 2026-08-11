// 为 knowledge-tree-detail.mjs 中的 329 个待建专题生成脚手架 index.md。
// 数据源：docs/.vitepress/data/knowledge-tree-detail.mjs
// 运行：node scripts/gen-pending-scaffold.mjs [--force]（--force 覆盖已存在的 index.md）
import { writeFileSync, mkdirSync, existsSync, readFileSync } from 'node:fs'

const FORCE = process.argv.includes('--force')

// 从 detail.mjs 文本中解析全部专题（避免 import mjs 的语法差异）
function parseDetail(file) {
  const content = readFileSync(file, 'utf8')
  const topics = []
  const keyRe = /^  '([^']+)': \{/gm
  let m
  while ((m = keyRe.exec(content))) {
    const key = m[1]
    const blockStart = content.indexOf('{', m.index) + 1
    // 找到对应块的结束（逐字符括号匹配）
    let depth = 1
    let i = blockStart
    while (depth > 0 && i < content.length) {
      if (content[i] === '{') depth++
      else if (content[i] === '}') depth--
      i++
    }
    const block = content.slice(blockStart, i - 1)
    const title = (block.match(/title: "([^"]+)"/) || [])[1] || key.split('/').pop()
    // 行内字符串（允许 \" 转义）：books 行缩进 10 空格，chapters 行缩进 6 空格
    const books = [...block.matchAll(/^ {10}"((?:[^"\\]|\\.)*)",?$/gm)].map((x) => x[1].replace(/\\"/g, '"'))
    const chapters = [...block.matchAll(/^ {6}"((?:[^"\\]|\\.)*)",?$/gm)].map((x) => x[1].replace(/\\"/g, '"'))
    topics.push({ key, title, books, chapters })
  }
  return topics
}

// 章节标题 → 条目文字：去掉括号书籍标注中的书名号部分可保留，仅清理尾随标注
function itemText(ch) {
  return ch
}

const TIER_META = {
  foundations: { name: '基础科学', num: '第一级' },
  intermediate: { name: '进阶数理', num: '第二级' },
  cs: { name: '计算机基础', num: '第三级' },
  advanced: { name: '高阶 AI 专题', num: '第四级' },
  life: { name: '生命与健康科学', num: '第五级' },
  engineering: { name: '工程技术', num: '第六级' },
  humanities: { name: '人文与艺术', num: '第七级' },
  social: { name: '社会科学', num: '第八级' },
  frontier: { name: '交叉与前沿', num: '第九级' },
}

// 分篇：每 ~8 章一组，命名「第N篇」
function groupChapters(chapters) {
  const groups = []
  const SIZE = 8
  for (let i = 0; i < chapters.length; i += SIZE) {
    const slice = chapters.slice(i, i + SIZE)
    groups.push({ title: `第${groups.length + 1}篇`, items: slice })
  }
  return groups
}

// key 拆分：第一段是 tier，其余段保持为嵌套路径（如 advanced/llm-principles/rwkv → tier=advanced, sub=llm-principles/rwkv）
function splitKey(key) {
  const parts = key.split('/')
  return { tier: parts[0], sub: parts.slice(1).join('/') }
}

function scaffoldMd(topic) {
  const { tier, sub } = splitKey(topic.key)
  const key = sub
  const meta = TIER_META[tier] || { name: tier, num: '' }
  const booksStr =
    topic.books.length > 0
      ? topic.books.map((b) => `- ${b}`).join('\n')
      : '- 该学科权威教材体系'

  const lines = []
  lines.push('---')
  lines.push('pageClass: plain-doc')
  lines.push('---')
  lines.push('')
  lines.push(`# ${topic.title}`)
  lines.push('')
  lines.push(`对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。`)
  lines.push('')
  lines.push(`## 对标教材`)
  lines.push('')
  lines.push(booksStr)
  lines.push('')
  lines.push(`## 主题规划`)
  lines.push('')
  lines.push(`<ProgressGrid cat="${tier}/${key}" />`)
  lines.push('')
  for (const g of groupChapters(topic.chapters)) {
    lines.push(`### ${g.title}`)
    lines.push('')
    for (const ch of g.items) {
      lines.push(`- [ ] ${itemText(ch)}`)
    }
    lines.push('')
  }
  return lines.join('\n')
}

// main
const detailFile = 'docs/.vitepress/data/knowledge-tree-detail.mjs'
const topics = parseDetail(detailFile)
console.log(`解析到 ${topics.length} 个待建专题`)

let count = 0
for (const topic of topics) {
  const { tier, sub } = splitKey(topic.key)
  const dir = `docs/posts/${tier}/${sub}`
  const file = `${dir}/index.md`
  mkdirSync(dir, { recursive: true })
  if (!FORCE && existsSync(file)) {
    console.log(`跳过已存在：${file}`)
    continue
  }
  writeFileSync(file, scaffoldMd(topic))
  count++
}
console.log(`已生成 ${count} 个脚手架 index.md`)

// 汇总统计
const totalCh = topics.reduce((s, t) => s + t.chapters.length, 0)
console.log(`待建专题 ${topics.length} 个 · 章节 ${totalCh} 条`)
