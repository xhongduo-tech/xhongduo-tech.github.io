import { createContentLoader } from 'vitepress'

export interface Project {
  title: string
  description: string
  url: string
  date: string
}

declare const data: Project[]
export { data }

export default createContentLoader('projects/**/*.md', {
  transform(pages) {
    return pages
      .filter(({ frontmatter }) => frontmatter.kind === 'project' && frontmatter.status === 'published')
      .map(({ url, frontmatter }) => {
        const value = frontmatter.date
        const date = value instanceof Date ? value : new Date(String(value || ''))
        return {
          title: String(frontmatter.title),
          description: String(frontmatter.description || ''),
          url,
          date: Number.isNaN(date.valueOf()) ? '' : date.toISOString().slice(0, 10),
        }
      })
      .sort((left, right) => right.date.localeCompare(left.date))
  },
})
