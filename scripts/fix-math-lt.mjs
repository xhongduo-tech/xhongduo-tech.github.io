// 修复新专题文章中的数学区间裸 `<` → `\lt`
// 只在数学定界符内替换（$...$ / $$...$$ / \(...\) / \[...\]），不动正文/HTML 标签。
// 用法：node scripts/fix-math-lt.mjs <topic-json> 或 node scripts/fix-math-lt.mjs all

import { readFileSync, writeFileSync, readdirSync, existsSync } from 'node:fs'
import path from 'node:path'

const ROOT = new URL('../docs/posts/', import.meta.url).pathname

function collectTopics() {
  const files = [process.argv[2]]
  if (process.argv[2] === 'all') {
    // 扫描所有 posts 目录
    const all = []
    for (const tier of readdirSync(ROOT)) {
      const tierDir = path.join(ROOT, tier)
      if (!existsSync(tierDir)) continue
      for (const key of readdirSync(tierDir)) {
        if (existsSync(path.join(tierDir, key, 'index.md'))) all.push(`${tier}/${key}`)
      }
    }
    return all
  }
  return JSON.parse(readFileSync(process.argv[2], 'utf-8')).map((t) => `${t.tier}/${t.key}`)
}

// 在数学区间内把裸 `<`（后非字母、/、!）替换为 `\lt`
function fixMath(content) {
  const out = []
  let i = 0
  const n = content.length
  while (i < n) {
    // 找数学起点
    let start = -1, close = -1, type = ''
    const dbl = content.indexOf('$$', i)
    const inl = content.indexOf('$', i)
    const paren = content.indexOf('\\(', i)
    const brack = content.indexOf('\\[', i)
    // 取最近的起点
    const candidates = []
    if (dbl !== -1) candidates.push([dbl, '$$', '$$'])
    if (inl !== -1) candidates.push([inl, '$', '$'])
    if (paren !== -1) candidates.push([paren, '\\(', '\\)'])
    if (brack !== -1) candidates.push([brack, '\\[', '\\]'])
    if (!candidates.length) break
    candidates.sort((a, b) => a[0] - b[0])
    ;[start, , close] = candidates[0]
    // 避免 $$ 被 $ 匹配（若 $ 起点紧跟 $，跳过）
    if (close === '$' && content[start + 1] === '$') {
      // 这是 $$ 已被 dbl 处理，跳过 inl
      candidates.shift()
      if (!candidates.length) { out.push(content.slice(i)); break }
      ;[start, , close] = candidates[0]
    }
    // 输出起点前的文本
    out.push(content.slice(i, start))
    // 找闭合
    const end = content.indexOf(close, start + close.length)
    if (end === -1) { out.push(content.slice(start)); break }
    let math = content.slice(start + close.length, end)
    // 替换数学内裸 <
    math = math.replace(/<(?![a-zA-Z/!])/g, '\\lt ')
    out.push(close + math + close)
    i = end + close.length
  }
  return out.join('')
}

let total = 0, fixedFiles = 0
for (const topic of collectTopics()) {
  const dir = path.join(ROOT, topic)
  if (!existsSync(dir)) continue
  for (const f of readdirSync(dir)) {
    if (!f.endsWith('.md') || f === 'index.md') continue
    const fp = path.join(dir, f)
    const orig = readFileSync(fp, 'utf-8')
    const fixed = fixMath(orig)
    if (fixed !== orig) {
      writeFileSync(fp, fixed)
      fixedFiles++
      const before = (orig.match(/<(?![a-zA-Z/!])/g) || []).length
      const after = (fixed.match(/<(?![a-zA-Z/!])/g) || []).length
      total += before - after
    }
  }
}
console.log(`修复 ${fixedFiles} 个文件，消除 ${total} 处裸 <`)
