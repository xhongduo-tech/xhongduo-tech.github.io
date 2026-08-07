const { readFileSync, writeFileSync } = require('node:fs')
const file = process.argv[2]
const src = readFileSync(file, 'utf8')
const start = src.indexOf('export const detailed = ')
const objSrc = src.slice(start + 'export const detailed = '.length)
const detailed = eval('(' + objSrc + ')')
let gen = readFileSync('scripts/gen-new-curriculum.mjs', 'utf8')
let replaced = 0, failed = []
for (const [key, chapters] of Object.entries(detailed)) {
  const [tier, cat] = key.split('/')
  const re = new RegExp(`^  \\['${tier}', '${cat}'[\\s\\S]*?\\n  \\]\\],\\n`, 'm')
  const found = gen.match(re)
  if (!found) { failed.push(key); continue }
  const nm = found[0].match(/, '([^']+)', '([^']*)', \[/)
  const name = nm ? nm[1] : cat
  const bench = nm ? nm[2] : ''
  const newEntry = `  ['${tier}', '${cat}', '${name}', '${bench}', [\n` +
    chapters.map(([t, its]) => `    ['${t}', [${its.map(i => `'${i}'`).join(', ')}]],`).join('\n') + `\n  ]],\n`
  gen = gen.replace(found[0], newEntry)
  replaced++
}
writeFileSync('scripts/gen-new-curriculum.mjs', gen)
console.log('已应用:', replaced)
if (failed.length) console.log('失败:', failed.join(','))
