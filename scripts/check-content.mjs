import { readdir, readFile } from 'node:fs/promises'
import path from 'node:path'
import matter from 'gray-matter'
import { createArticleTopology } from '../docs/.vitepress/data/article-topology-utils.mjs'
import { domains, domainEdges, domainProgress } from '../docs/.vitepress/data/domain-progress.mjs'
import { topicNodes } from '../docs/.vitepress/data/topic-topology.mjs'

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
const topicIds = new Set(topicNodes.map(node => node.id))
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
    if (!Array.isArray(data.prerequisites) || data.prerequisites.some(value => typeof value !== 'string' || !value.startsWith('/'))) {
      fail('prerequisites必须是站内文章路径组成的字符串数组')
    }
    if (!['foundations', 'inference', 'training', 'evaluation', 'applications'].includes(data.area)) fail('文章需有有效area')
    if (!['mechanism', 'source', 'simulation', 'hardware'].includes(data.evidence)) fail('文章需说明证据层次')
    if (data.topicId && !topicIds.has(data.topicId)) fail(`文章关联了未知主题: ${data.topicId}`)
    if (!/https:\/\//.test(content)) fail('文章需有直接支持主张的资料链接')
    for (const section of (data.contentLang === 'en' ? ['Problem', 'Mechanism', 'Experiment', 'Conclusion', 'Reference'] : ['问题', '机制', '实验', '结论', '参考'])) {
      if (!new RegExp(`^## .*${section}`, 'm').test(content)) fail(`缺少${section}段落`)
    }
    if (/\bTODO\b|待补充|TBD/.test(content)) fail('文章仍含未处理的写作占位')
    if (data.evidence === 'source' && !/\/(blob|tree)\/[a-f0-9]{40}\//.test(content)) fail('源码文章需固定commit入口')
    if (data.evidence === 'hardware' && (!data.model || !data.hardware || !data.results)) fail('硬件实测需有model、hardware和results')
  }
  if (data.kind === 'project') {
    if (data.status !== 'published') fail('未完成的私有项目任务不应放在公开docs')
    const projectDate = data.date instanceof Date ? data.date.toISOString().slice(0, 10) : String(data.date)
    if (!/^\d{4}-\d{2}-\d{2}$/.test(projectDate)) fail('公开项目需有YYYY-MM-DD日期')
    if (!data.results) fail('公开项目需有真实results记录')
  }
  if (data.translations !== undefined) {
    if (!data.translations || typeof data.translations !== 'object' || Array.isArray(data.translations)) fail('translations必须记录其他语言页面路径')
    else for (const target of Object.values(data.translations)) {
      if (typeof target !== 'string' || !routes.has(normalize(target))) fail(`翻译页面不存在: ${target}`)
    }
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
const publishedArticles = pages.filter((page) => page.data.kind === 'article' && page.data.status === 'published')
try {
  createArticleTopology(publishedArticles.map((page) => ({
    title: page.data.title,
    url: normalize(page.route),
    date: page.data.date,
    topic: '',
    prerequisites: Array.isArray(page.data.prerequisites)
      ? page.data.prerequisites.map((value) => typeof value === 'string' ? normalize(value) : String(value))
      : page.data.prerequisites,
  })))
} catch (error) {
  errors.push(`文章拓扑无效: ${error.message}`)
}
const domainIds = new Set(domains.map(domain => domain.id))
if (domainIds.size !== domains.length) errors.push('领域拓扑存在重复节点')
for (const [source, target] of domainEdges) {
  if (!domainIds.has(source) || !domainIds.has(target)) errors.push(`领域拓扑边引用未知节点: ${source} -> ${target}`)
}
const publicEvidence = new Map(pages
  .filter(page => ['article', 'project'].includes(page.data.kind) && page.data.status === 'published')
  .map(page => [normalize(page.route), page]))
for (const domain of domains) {
  const progress = domainProgress[domain.id]
  if (!progress || typeof progress.completed !== 'boolean') {
    errors.push(`${domain.id}: 缺少真实的领域完成状态`)
  } else if (progress.completed) {
    if (!progress.evidence || typeof progress.evidence !== 'string') errors.push(`${domain.id}: 点亮领域必须关联公开验收证据`)
    else if (!publicEvidence.has(normalize(progress.evidence))) errors.push(`${domain.id}: 领域验收证据必须是已发布文章或项目`)
  }
}
for (const legacy of ['cs', 'quant', 'econ', 'litho', 'llm', '_test', 'path']) {
  if (pages.some((page) => page.rel.startsWith(legacy + '/'))) errors.push(`旧内容目录意外恢复: ${legacy}`)
}
if (errors.length) {
  process.stderr.write(errors.join('\n') + '\n')
  process.exitCode = 1
} else {
  const articles = publishedArticles
  process.stdout.write(`内容检查通过：${pages.length}页，${articles.length}篇正式博文；元数据、内部链接、文章依赖与旧内容清理正常。\n`)
}
