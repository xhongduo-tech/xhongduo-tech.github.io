import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import matter from 'gray-matter'

const root = path.resolve('docs')
const errors = []
const pages = []
const kinds = new Set(['overview', 'topic', 'project', 'article', 'about'])
async function walk(dir) {
  for (const item of await readdir(dir, { withFileTypes: true })) {
    if (item.name.startsWith('.') || item.name === 'public') continue
    const target = path.join(dir, item.name)
    if (item.isDirectory()) await walk(target)
    else if (item.name.endsWith('.md')) {
      const { data, content } = matter(await readFile(target, 'utf8'))
      const rel = path.relative(root, target).replaceAll(path.sep, '/')
      const route = '/' + rel.replace(/(^|\/)index\.md$/, '$1').replace(/\.md$/, '')
      pages.push({ target, rel, route, data, content })
    }
  }
}
await walk(root)
const normalize = (value) => value.replace(/\.html$/, '').replace(/\/$/, '') || '/'
const routes = new Set(pages.map((page) => normalize(page.route)))
for (const page of pages) {
  const { rel, data, content } = page
  const fail = (message) => errors.push(`${rel}: ${message}`)
  if (!data.title || typeof data.title !== 'string') fail('缺少title')
  if (!data.description || typeof data.description !== 'string') fail('缺少description')
  if (!kinds.has(data.kind)) fail('kind必须说明内容角色')
  if (!data.status) fail('缺少真实状态status')
  if (data.kind === 'article') {
    if (data.status !== 'published') fail('未发表草稿应放在drafts，而不是公开docs')
    const articleDate = data.date instanceof Date ? data.date.toISOString().slice(0, 10) : String(data.date)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(articleDate)) fail('文章需有YYYY-MM-DD日期')
    if (!['foundations', 'inference', 'training', 'evaluation'].includes(data.area)) fail('文章需有有效area')
    if (!['mechanism', 'source', 'simulation', 'hardware'].includes(data.evidence)) fail('文章需说明证据层次')
    if (!/https:\/\//.test(content)) fail('文章需有直接支持主张的资料链接')
    for (const section of ['问题', '机制', '实验', '结论', '参考']) {
      if (!new RegExp(`^## .*${section}`, 'm').test(content)) fail(`缺少${section}段落`)
    }
    if (/\bTODO\b|待补充|TBD/.test(content)) fail('文章仍含未处理的写作占位')
    if (data.evidence === 'source' && !/\/(blob|tree)\/[a-f0-9]{40}\//.test(content)) fail('源码文章需固定commit入口')
    if (data.evidence === 'hardware' && (!data.model || !data.hardware || !data.results)) fail('硬件实测需有model、hardware和results')
  }
  const prose = content.replace(/```[\s\S]*?```/g, '')
  for (const match of prose.matchAll(/\[[^\]]*\]\(([^)]+)\)/g)) {
    const target = match[1].split('#')[0]
    if (!target || /^(https?:|mailto:)/.test(target)) continue
    let resolved
    if (target.startsWith('/')) resolved = normalize(target)
    else resolved = normalize(path.posix.normalize(path.posix.join(path.posix.dirname('/' + rel), target)).replace(/\.md$/, ''))
    if (!routes.has(resolved)) fail(`内部链接不存在: ${target}`)
  }
}
for (const legacy of ['cs', 'quant', 'econ', 'litho', 'llm', '_test', 'path']) {
  if (pages.some((page) => page.rel.startsWith(legacy + '/'))) errors.push(`旧内容目录意外恢复: ${legacy}`)
}
if (errors.length) {
  process.stderr.write(errors.join('\n') + '\n')
  process.exitCode = 1
} else {
  const articles = pages.filter((page) => page.data.kind === 'article' && page.data.status === 'published')
  process.stdout.write(`内容检查通过：${pages.length}页，${articles.length}篇正式博文；元数据、内部链接与旧内容清理正常。\n`)
}
