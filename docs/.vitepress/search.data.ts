import { createContentLoader } from 'vitepress'
import { isSectionId, type SectionId } from './data/sections'
import { getLesson } from './data/curriculum'

export interface SearchEntry {
  title: string
  url: string
  section: SectionId
  /** 课程 · 单元 · 课序 的连写，用于匹配与展示 */
  path: string
  indexInCourse: number
  courseSize: number
  appendix: boolean
}

export default createContentLoader(['llm/*.md', 'quant/*.md', 'econ/*.md', 'litho/*.md', 'cs/*.md'], {
  transform(raw): SearchEntry[] {
    const rows = raw
      .map((page) => {
        const segs = page.url.replace(/\/$/, '').split('/').filter(Boolean)
        if (segs.length < 2) return null
        const section: SectionId = isSectionId(segs[0]) ? segs[0] : 'llm'
        const slug = segs[segs.length - 1]
        const lesson = getLesson(section, slug)
        const path = lesson
          ? [lesson.course, lesson.unit, lesson.sequence].filter(Boolean).join(' · ')
          : ''
        return {
          title: String(page.frontmatter.title || slug),
          url: page.url,
          section,
          path,
          indexInCourse: lesson ? lesson.indexInCourse : 0,
          courseSize: lesson ? lesson.courseSize : 0,
          appendix: lesson ? lesson.appendix : false,
        }
      })
      .filter((r): r is SearchEntry => r !== null)
    // 栏内按课程内课序排，便于结果自上而下就是阅读顺序
    const order = new Map<string, number>()
    for (const r of rows) {
      const k = r.section + ':' + r.url
      order.set(k, order.size)
    }
    return rows.sort((a, b) => {
      if (a.section !== b.section) {
        const secOrder = ['llm', 'quant', 'econ', 'litho', 'cs']
        return secOrder.indexOf(a.section) - secOrder.indexOf(b.section)
      }
      return a.indexInCourse - b.indexInCourse
    })
  },
})
