// 把 /tmp/kt-audit/patches/*.json 的补丁合并进 docs/.vitepress/data/knowledge-tree-detail.mjs。
// 每个补丁：{ 'tier/key': [完整修正后章节数组, ...] }，直接替换该专题的 chapters。
// 用「从后往前」替换避免索引偏移。同时展平嵌套论文专题 key。
import { readFileSync, writeFileSync, readdirSync } from 'node:fs'

const DETAIL = 'docs/.vitepress/data/knowledge-tree-detail.mjs'
const PATCH_DIR = '/tmp/kt-audit/patches'

// 嵌套 key → 展平 key 映射（gen-progress 只扫两层目录，嵌套专题需展平为独立专题）
const FLATTEN = {
  'advanced/reinforcement-learning/decision-transformer': 'advanced/decision-transformer-paper',
  'advanced/llm-principles/rwkv': 'advanced/rwkv',
  'advanced/llm-principles/gshard': 'advanced/gshard',
  'advanced/computer-vision/dit-sora': 'advanced/dit-sora',
}

// 解析 detail.mjs 中每个专题块的位置
function locateBlocks(content) {
  const blocks = []
  const keyRe = /^  '([^']+)': \{/gm
  let m
  while ((m = keyRe.exec(content))) {
    const key = m[1]
    const keyStart = m.index
    const keyLen = m[0].length
    const blockStart = content.indexOf('{', keyStart) + 1
    let depth = 1
    let i = blockStart
    while (depth > 0 && i < content.length) {
      if (content[i] === '{') depth++
      else if (content[i] === '}') depth--
      i++
    }
    const blockEnd = i - 1 // 指向 '}' 位置
    blocks.push({ key, keyStart, keyLen, blockStart, blockEnd })
  }
  return blocks
}

// 在 block 文本内替换 chapters 数组
function replaceChapters(blockText, newChapters) {
  // 定位 chapters: [ 到该数组的结束 ]（该 ] 后跟换行和 }, 即 `],`）
  const marker = 'chapters: ['
  const chIdx = blockText.indexOf(marker)
  if (chIdx < 0) return null
  // marker 的 '[' 即数组开始，depth 初始为 1
  let depth = 1
  let arrEnd = -1
  let i = chIdx + marker.length
  for (; i < blockText.length; i++) {
    const ch = blockText[i]
    if (ch === '[') depth++
    else if (ch === ']') {
      depth--
      if (depth === 0) {
        arrEnd = i
        break
      }
    }
  }
  if (arrEnd < 0) return null
  const arrStart = chIdx + marker.length // 指向 '['
  // 标准格式：每项缩进 6 空格、逗号在项末尾（最后一项无逗号）
  // 数组以 ']' 结尾（原始 '],' 中的逗号需移除，因为新数组最后一项无逗号）
  const items = newChapters.map((c) => `      "${c.replace(/"/g, '\\"')}"`)
  const newArr = '\n' + items.join(',\n') + '\n    '
  // 替换 [ 到 ] 的内容；] 后的 ',' 保留但需确认格式
  const after = blockText.slice(arrEnd + 1)
  return blockText.slice(0, arrStart) + newArr + ']' + after
}

const content = readFileSync(DETAIL, 'utf8')
const blocks = locateBlocks(content)
const byKey = {}
for (const b of blocks) byKey[b.key] = b

// 读取全部补丁
const patchFiles = readdirSync(PATCH_DIR).filter((f) => f.endsWith('.json'))
const patches = {}
for (const f of patchFiles) {
  const data = JSON.parse(readFileSync(`${PATCH_DIR}/${f}`, 'utf8'))
  for (const [k, v] of Object.entries(data)) patches[k] = v
  console.log(`读取补丁 ${f}: ${Object.keys(data).length} 个专题`)
}

// 应用补丁：从后往前替换，避免偏移
let replaced = 0
let missing = []
const ops = []
for (const [patchKey, chapters] of Object.entries(patches)) {
  if (!byKey[patchKey]) {
    missing.push(patchKey)
    continue
  }
  ops.push({ key: patchKey, block: byKey[patchKey], chapters })
}
// 按 blockStart 从大到小排序
ops.sort((a, b) => b.block.blockStart - a.block.blockStart)

let result = content
for (const op of ops) {
  const { block, chapters } = op
  const blockText = result.slice(block.blockStart, block.blockEnd)
  const newBlock = replaceChapters(blockText, chapters)
  if (newBlock === null) {
    console.log(`⚠️ ${op.key} 替换失败`)
    continue
  }
  result = result.slice(0, block.blockStart) + newBlock + result.slice(block.blockEnd)
  replaced++
}

console.log(`\n已替换 ${replaced} 个专题的 chapters`)
if (missing.length) console.log(`⚠️ 补丁中未在 detail 找到的 key: ${missing.join(', ')}`)

// 展平嵌套 key：把 detail 中嵌套 key 改成展平 key（从后往前）
for (const [nested, flat] of Object.entries(FLATTEN)) {
  const idx = result.lastIndexOf(`  '${nested}': {`)
  if (idx >= 0) {
    result = result.slice(0, idx) + `  '${flat}': {` + result.slice(idx + `  '${nested}': {`.length)
    console.log(`展平 key: ${nested} → ${flat}`)
  } else {
    console.log(`⚠️ 嵌套 key 未找到（可能补丁已改名）: ${nested}`)
  }
}

writeFileSync(DETAIL, result)
console.log(`\n✅ 已写回 ${DETAIL}`)

// 校验
const finalContent = readFileSync(DETAIL, 'utf8')
const finalBlocks = locateBlocks(finalContent)
const totalCh = finalBlocks.reduce((s, b) => {
  const blockText = finalContent.slice(b.blockStart, b.blockEnd)
  return s + [...blockText.matchAll(/^      "((?:[^"\\]|\\.)*)",?$/gm)].length
}, 0)
console.log(`detail 现有专题: ${finalBlocks.length} · 章节总数: ${totalCh}`)
