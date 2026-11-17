import { createContentLoader } from 'vitepress'

export interface Article {
  title: string
  description: string
  url: string
  date: string
  topic: string
}

declare const data: Article[]
export { data }

function articleDate(value: unknown): string {
  if (!value) return ''
  const date = value instanceof Date ? value : new Date(String(value))
  return Number.isNaN(date.valueOf()) ? '' : date.toISOString().slice(0, 10)
}

export default createContentLoader('**/*.md', {
  transform(pages) {
    return pages
      .filter(({ frontmatter }) => frontmatter.kind === 'article' && frontmatter.status === 'published')
      .map(({ url, frontmatter }) => ({
        title: String(frontmatter.title ?? '未命名文章'),
        description: String(frontmatter.description ?? ''),
        url,
        date: articleDate(frontmatter.date),
        topic: String(frontmatter.topic ?? ''),
      }))
      .sort((left, right) => right.date.localeCompare(left.date))
  },
})
