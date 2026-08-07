const { readFileSync, writeFileSync } = require('node:fs')
const { execSync } = require('node:child_process')
const files = execSync("find docs/posts -name '*.md' ! -name 'index.md'", {encoding:'utf8'}).trim().split('\n')
let hits = []
for (const f of files) {
  const lines = readFileSync(f, 'utf8').split('\n')
  for (let i = 0; i < lines.length; i++) {
    // span 开在 i 行
    if (/<span class="(marginnote|sidenote)">/.test(lines[i])) {
      // 向后找 </span>（同文件，可能跨行）
      for (let j = i + 1; j < lines.length && j < i + 60; j++) {
        if (lines[j].includes('</span>')) {
          // 检查 i..j 之间是否有空行
          const between = lines.slice(i + 1, j)
          if (between.some(l => l.trim() === '')) {
            hits.push(`${f}:${i + 1}->${j + 1}`)
          }
          break
        }
      }
    }
  }
}
console.log('跨空行 span 数:', hits.length)
hits.slice(0,25).forEach(h => console.log(' -', h))
