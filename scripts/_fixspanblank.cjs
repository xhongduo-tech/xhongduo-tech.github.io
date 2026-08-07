const { readFileSync, writeFileSync } = require('node:fs')
const targets = [
  'docs/posts/advanced/ontology/occams-razor.md',
  'docs/posts/foundations/psychology/schools-structuralism-functionalism-gestalt.md',
  'docs/posts/foundations/psychology/schools-behaviorism-psychoanalysis.md',
  'docs/posts/foundations/psychology/schools-humanism-cognitive.md',
]
for (const f of targets) {
  const lines = readFileSync(f, 'utf8').split('\n')
  // 计算每行 span 开闭计数，维护深度；删除 span 打开但未闭合时遇到的空行
  const out = []
  let depth = 0
  for (const ln of lines) {
    const opens = (ln.match(/<span/g) || []).length
    const closes = (ln.match(/<\/span>/g) || []).length
    // 空行且 span 打开中 -> 删除（合并段落）
    if (ln.trim() === '' && depth > 0) {
      // skip blank line inside span
      continue
    }
    out.push(ln)
    depth = Math.max(0, depth + opens - closes)
  }
  writeFileSync(f, out.join('\n'))
  console.log('已处理:', f)
}
