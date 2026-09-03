import { createContentLoader } from 'vitepress'

export interface Post {
  title: string
  url: string
  date: string
  section: 'llm' | 'quant'
}

function formatDate(raw: unknown): string {
  if (!raw) return ''
  const d = new Date(String(raw))
  if (Number.isNaN(+d)) return String(raw)
  return d.toISOString().slice(0, 10)
}

export default createContentLoader(['llm/*.md', 'quant/*.md'], {
  transform(raw): Post[] {
    return raw
      .filter((page) => {
        const segs = page.url.replace(/\/$/, '').split('/').filter(Boolean)
        return segs.length >= 2
      })
      .map((page) => {
        const segs = page.url.replace(/\/$/, '').split('/').filter(Boolean)
        const section = segs[0] === 'quant' ? 'quant' : 'llm'
        return {
          title: String(page.frontmatter.title || segs[segs.length - 1]),
          url: page.url,
          date: formatDate(page.frontmatter.date),
          section,
        }
      })
      .sort((a, b) => {
        if (a.date !== b.date) return a.date < b.date ? 1 : -1
        return a.title.localeCompare(b.title, 'zh')
      })
  },
})
