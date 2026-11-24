import { createContentLoader } from 'vitepress'

export interface Article {
  title: string
  description: string
  url: string
  date: string
  topic: string
  topicId: string
  prerequisites: string[]
}

declare const data: Article[]
export { data }

function articleDate(value: unknown): string {
  if (!value) return ''
  const date = value instanceof Date ? value : new Date(String(value))
  return Number.isNaN(date.valueOf()) ? '' : date.toISOString().slice(0, 10)
}

function normalizeRoute(value: unknown): string {
  const route = String(value ?? '').split('#')[0].split('?')[0]
    .replace(/\.(?:md|html)$/, '').replace(/\/$/, '')
  return route || '/'
}

export default createContentLoader('articles/**/*.md', {
  transform(pages) {
    return pages
      .filter(({ frontmatter }) => frontmatter.kind === 'article' && frontmatter.status === 'published')
      .map(({ url, frontmatter }) => ({
        title: String(frontmatter.title ?? '未命名文章'),
        description: String(frontmatter.description ?? ''),
        url: normalizeRoute(url),
        date: articleDate(frontmatter.date),
        topic: String(frontmatter.topic ?? ''),
        topicId: String(frontmatter.topicId ?? ''),
        prerequisites: Array.isArray(frontmatter.prerequisites)
          ? frontmatter.prerequisites.map(normalizeRoute)
          : [],
      }))
      .sort((left, right) => right.date.localeCompare(left.date) || left.url.localeCompare(right.url))
  },
})
