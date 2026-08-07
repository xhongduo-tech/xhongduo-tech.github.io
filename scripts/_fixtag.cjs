const { readFileSync, writeFileSync } = require('node:fs')
const targets = [
  'docs/posts/advanced/nlp/ngram-markov-estimation.md',
  'docs/posts/advanced/nlp/perplexity-lm-evaluation.md',
]
let total = 0
for (const f of targets) {
  const lines = readFileSync(f, 'utf8').split('\n')
  let changed = false
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i]
    if (ln.trim().startsWith('$$')) continue
    let out = '', inMath = false
    for (let j = 0; j < ln.length; j++) {
      const c = ln[j]
      if (c === '$') { inMath = !inMath; out += c; continue }
      if (inMath && c === '<' && ln.slice(j, j + 3) === '<s>') { out += '〈s〉'; j += 2; total++; changed = true; continue }
      if (inMath && c === '<' && ln.slice(j, j + 4) === '</s>') { out += '〈/s〉'; j += 3; total++; changed = true; continue }
      out += c
    }
    if (out !== ln) lines[i] = out
  }
  if (changed) writeFileSync(f, lines.join('\n'))
}
console.log('替换总数:', total)
