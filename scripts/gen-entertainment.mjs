// 娱乐大典数据生成器：
// 读取 docs/.vitepress/data/entertainment/raw/<cat>-*.json（每文件为一个 JSON 数组），
// 逐条校验 -> 去重 -> 按年份排序 -> 统计 -> 写出：
//   docs/.vitepress/data/entertainment/<cat>.json （含 { cat, count, stats, items }）
//   docs/.vitepress/data/entertainment/meta.json （标签映射 + 分类元数据，组件直接导入）
// 同时渲染 .claude/entertainment-charter.md 给内容 Agent 作为宪章。
//
// 规则：
//   - 空目录不是错误：0 条也写合法空文件并告警，保证首日构建不红。
//   - schema 错误（必填缺、键非法、数值越界、重复）全部收集后 exit 1。
//   - 未知 region 仅告警（UI 回退显示原始代码）。
import { readdirSync, readFileSync, writeFileSync, mkdirSync } from 'node:fs'
import { join } from 'node:path'
import {
  CATEGORIES,
  GENRES,
  AWARDS,
  REGIONS,
  RATING_KEYS,
  REQUIRED_FIELDS,
  MAX_GENRES,
} from './ent-schema.mjs'

const RAW_DIR = 'docs/.vitepress/data/entertainment/raw'
const OUT_DIR = 'docs/.vitepress/data/entertainment'
const CHARTER_PATH = '.claude/entertainment-charter.md'

const errors = [] // { msg }
const warnings = []
const seenEn = new Map() // normalized en -> { cat, file, idx }

function err(msg) {
  errors.push({ msg })
}
function warn(msg) {
  warnings.push(msg)
}

// 规范化英文标题用于去重：小写、去重音符号与标点、折叠空白
function normEn(s) {
  return s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
}

function slugify(s) {
  return s
    .toLowerCase()
    .normalize('NFD')
    .replace(/[̀-ͯ]/g, '')
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80)
}

function isInt(x) {
  return Number.isInteger(x)
}

function validateEntry(cat, file, idx, e) {
  const where = `${file}[${idx}]`
  if (typeof e !== 'object' || e === null || Array.isArray(e)) {
    err(`${where} 必须是对象`)
    return null
  }
  for (const f of REQUIRED_FIELDS) {
    if (typeof e[f] !== 'string' || !e[f].trim()) {
      err(`${where} 缺少必填字符串字段 ${f}`)
      return null
    }
  }
  const [yMin, yMax] = CATEGORIES[cat].yearRange
  if (!isInt(e.year) || e.year < yMin || e.year > yMax) {
    err(`${where} year=${e.year} 不在 ${cat} 范围 [${yMin}, ${yMax}]`)
    return null
  }
  if (!Array.isArray(e.genres) || e.genres.length === 0 || e.genres.length > MAX_GENRES) {
    err(`${where} genres 必须为非空数组且 ≤${MAX_GENRES}`)
    return null
  }
  for (const g of e.genres) {
    if (!GENRES[cat][g]) {
      err(`${where} 未知流派键 ${g}（合法: ${Object.keys(GENRES[cat]).join(', ')}）`)
      return null
    }
  }
  e.genres = [...new Set(e.genres)] // 去重流派
  if (!Array.isArray(e.awards)) {
    err(`${where} awards 必须为数组`)
    return null
  }
  const aw = []
  for (const a of e.awards) {
    if (!AWARDS[cat][a]) {
      err(`${where} 未知奖项键 ${a}（合法: ${Object.keys(AWARDS[cat]).join(', ')}）`)
      return null
    }
    if (!aw.includes(a)) aw.push(a)
  }
  e.awards = aw
  if (e.region === undefined || e.region === null || e.region === '') {
    err(`${where} 缺少 region`)
    return null
  }
  if (typeof e.region !== 'string' || !REGIONS[e.region]) {
    warn(`[${cat}] ${where} 未知地区键 ${e.region}，UI 回退显示原始代码`)
  }
  if (e.rating !== undefined && e.rating !== null) {
    if (typeof e.rating !== 'object' || Array.isArray(e.rating)) {
      err(`${where} rating 必须为对象`)
      return null
    }
    for (const [k, v] of Object.entries(e.rating)) {
      const max = RATING_KEYS[cat][k]
      if (max === undefined) {
        err(`${where} rating 含未知键 ${k}（合法: ${Object.keys(RATING_KEYS[cat]).join(', ') || '无（本类无评分）'}）`)
        return null
      }
      if (typeof v !== 'number' || !Number.isFinite(v) || v < 0 || v > max) {
        err(`${where} rating.${k}=${v} 超出刻度 [0, ${max}]`)
        return null
      }
    }
  }
  if (e.id === undefined || e.id === null || e.id === '') {
    e.id = slugify(e.en)
  }
  if (typeof e.id !== 'string' || !e.id.trim()) {
    err(`${where} id 必须为非空字符串`)
    return null
  }
  // 去重
  const key = normEn(e.en)
  if (seenEn.has(key)) {
    const prev = seenEn.get(key)
    err(`${where} 与 ${prev.file}[${prev.idx}]（en: "${prev.en}"）重复；若为翻拍/重制请用可区分的 en 标题`)
    return null
  }
  seenEn.set(key, { cat, file, idx, en: e.en })
  return e
}

function mergeCat(cat) {
  const files = readdirSync(RAW_DIR, { withFileTypes: true })
    .filter((f) => f.isFile() && f.name.startsWith(`${cat}-`) && f.name.endsWith('.json'))
    .map((f) => f.name)
    .sort()
  if (files.length === 0) {
    warn(`[${cat}] 无 raw 文件，写入空数据集`)
    return []
  }
  const items = []
  for (const file of files) {
    let raw
    try {
      raw = readFileSync(join(RAW_DIR, file), 'utf8')
    } catch {
      err(`[${cat}] 无法读取 ${file}`)
      continue
    }
    let arr
    try {
      arr = JSON.parse(raw)
    } catch (e2) {
      err(`[${cat}] ${file} 不是合法 JSON: ${e2.message}`)
      continue
    }
    if (!Array.isArray(arr)) {
      err(`[${cat}] ${file} 必须是 JSON 数组`)
      continue
    }
    arr.forEach((e, idx) => {
      const v = validateEntry(cat, file, idx, e)
      if (v) items.push(v)
    })
  }
  // 排序：year 升序，同年在按 en 字母序
  items.sort((a, b) => a.year - b.year || a.en.localeCompare(b.en))
  return items
}

function computeStats(cat, items) {
  const genres = {}
  const decades = {}
  const regions = {}
  let totalAwards = 0
  for (const it of items) {
    for (const g of it.genres) genres[g] = (genres[g] || 0) + 1
    const d = Math.floor(it.year / 10) * 10
    decades[d] = (decades[d] || 0) + 1
    if (REGIONS[it.region]) regions[it.region] = (regions[it.region] || 0) + 1
    totalAwards += it.awards.length
  }
  const years = items.map((i) => i.year)
  return {
    totalAwards,
    genres,
    decades,
    regions,
    yearRange: years.length ? [Math.min(...years), Math.max(...years)] : null,
  }
}

function buildMeta() {
  return {
    schemaVersion: 1,
    categories: Object.fromEntries(
      Object.entries(CATEGORIES).map(([cat, m]) => [
        cat,
        {
          zh: m.zh,
          en: m.en,
          emoji: m.emoji,
          color: m.color,
          soft: m.soft,
          ink: m.ink,
          ratingSources: m.ratingSources,
        },
      ]),
    ),
    genres: GENRES,
    awards: AWARDS,
    regions: REGIONS,
  }
}

// ---------- 宪章渲染 ----------
function renderCharter() {
  const lines = []
  lines.push('# 娱乐大典 · 内容宪章（Agent 必读）')
  lines.push('')
  lines.push('这是为娱乐页「综合大典版」生成内容时给每位 Agent 的规范。')
  lines.push('')
  lines.push('## 条目 Schema（四类一致）')
  lines.push('')
  lines.push('```json')
  lines.push(JSON.stringify(
    {
      id: 'blade-runner-2049',
      title: '银翼杀手 2049',
      en: 'Blade Runner 2049',
      year: 2017,
      genres: ['scifi', 'noir'],
      creator: 'Denis Villeneuve',
      region: 'US',
      rating: { douban: 8.3, imdb: 8.0, metacritic: 81 },
      awards: ['oscar'],
      note: '关于「什么是人」的视觉诗。',
      noteEn: 'A visual poem on what makes us human.',
    },
    null,
    2,
  ))
  lines.push('```')
  lines.push('')
  lines.push('- 必填：`title`（中文名）、`en`（英文名，去重键）、`year`（整数，范围见下）、`genres`（1–3 个本类流派键）、`creator`（导演/工作室/艺术家/摄影师）、`region`（ISO 代码）、`note`（一句中文打动理由）、`noteEn`（一句英文理由）。')
  lines.push('- 可空：`rating`（对象，键在本类评分源内、值在刻度内）、`awards`（本类奖项键数组）。`id` 缺省由 en 派生，可省略。')
  lines.push('- 质量红线：**禁止编造**。年份/创作者/奖项必须真实可核实；存疑事实最多用 2 次 WebSearch 核实。`note`/`noteEn` 要具体、非套话（说清它为何好/独特，而非「经典之作」）。')
  lines.push('')
  for (const cat of Object.keys(CATEGORIES)) {
    const m = CATEGORIES[cat]
    lines.push(`## ${cat}（${m.zh}）· year ${m.yearRange[0]}–${m.yearRange[1]}`)
    lines.push('')
    lines.push(`- 流派键：${Object.entries(GENRES[cat]).map(([k, v]) => `${k}(${v.emoji}${v.zh}/${v.en})`).join(' ')}`)
    lines.push(`- 奖项键：${Object.entries(AWARDS[cat]).map(([k, v]) => `${k}(${v.zh})`).join(' ')}`)
    lines.push(`- 评分源：${m.ratingSources.length ? m.ratingSources.map((s) => `${s.key}(0–${s.max})`).join(' ') : '无（本类无评分）'}`)
    lines.push('')
  }
  lines.push('## 地区键（ISO）')
  lines.push('')
  lines.push(Object.keys(REGIONS).join(' '))
  lines.push('')
  lines.push('## 红线')
  lines.push('')
  lines.push('- 只写你自己那份 raw JSON 文件，输出必须是合法 JSON **数组**（2 空格缩进）。')
  lines.push('- 不跑 git、不碰其他文件/其他分类的 raw 文件、不运行合并脚本、不开子 Agent。')
  lines.push('- 同名作品（翻拍/重制/重制版）用可区分的 en 标题，避免去重冲突。')
  lines.push('- 完成后返回简短报告：写入条数、各流派计数、你做的消歧决定。')
  return lines.join('\n') + '\n'
}

// ---------- 主流程 ----------
mkdirSync(RAW_DIR, { recursive: true })
mkdirSync(OUT_DIR, { recursive: true })

const results = {}
let grandTotal = 0
let grandAwards = 0
for (const cat of Object.keys(CATEGORIES)) {
  const items = mergeCat(cat)
  const stats = computeStats(cat, items)
  results[cat] = { cat, count: items.length, stats, items }
  grandTotal += items.length
  grandAwards += stats.totalAwards
  writeFileSync(
    join(OUT_DIR, `${cat}.json`),
    JSON.stringify({ cat, count: items.length, stats, items }, null, 2) + '\n',
  )
}

const meta = buildMeta()
writeFileSync(join(OUT_DIR, 'meta.json'), JSON.stringify(meta, null, 2) + '\n')
writeFileSync(CHARTER_PATH, renderCharter())

// ---------- 输出汇总 ----------
console.log('== 娱乐大典数据生成 ==')
for (const cat of Object.keys(CATEGORIES)) {
  const r = results[cat]
  const perGenre = Object.entries(r.stats.genres)
    .sort((a, b) => b[1] - a[1])
    .map(([g, n]) => `${g}:${n}`)
    .join(' ')
  console.log(`- ${cat}: ${r.count} 条 | 奖项 ${r.stats.totalAwards} | 年代 ${r.stats.yearRange ? r.stats.yearRange.join('–') : '-'}`)
  if (perGenre) console.log(`    ${perGenre}`)
}
console.log(`合计: ${grandTotal} 条 / 大奖 ${grandAwards}`)
console.log(`警告 ${warnings.length} 条`)
for (const w of warnings) console.log(`  ⚠ ${w}`)

if (errors.length) {
  console.error(`\n=== ${errors.length} 个 schema 错误 ===`)
  for (const e of errors) console.error(`  ✗ ${e.msg}`)
  process.exit(1)
}
console.log('✓ 全部通过')
