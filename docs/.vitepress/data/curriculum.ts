import { trees } from './trees'
import type { Node } from './trees/schema'
import { SECTION_IDS, type SectionId } from './sections'

export interface LessonRef {
  slug: string
  title: string
}

export interface Lesson {
  slug: string
  title: string
  section: SectionId
  appendix: boolean
  course: string
  unit: string
  sequence: string
  indexInCourse: number
  courseSize: number
  indexInUnit: number
  unitSize: number
  indexInSequence: number
  sequenceSize: number
  prev: LessonRef | null
  next: LessonRef | null
  prereq: LessonRef | null
}

interface LeafHit {
  slug: string
  title: string
  appendix: boolean
  course: string
  unit: string
  sequence: string
}

const lessons = new Map<string, Lesson>()
const orderBySection = Object.fromEntries(SECTION_IDS.map((id) => [id, [] as string[]])) as Record<
  SectionId,
  string[]
>

function key(section: string, slug: string) {
  return `${section}:${slug}`
}

function collect(
  nodes: Node[],
  ctx: { course: string; unit: string; sequence: string; appendix: boolean },
  out: LeafHit[],
) {
  for (const node of nodes) {
    const appendix = Boolean(node.appendix || ctx.appendix)
    const next = { ...ctx, appendix }
    if (node.kind === 'branch') next.course = node.name
    if (node.kind === 'mainline') next.unit = node.name
    if (node.kind === 'group') next.sequence = node.name
    if (node.kind === 'leaf' && node.slug) {
      out.push({ slug: node.slug, title: node.name, ...next })
    }
    if (node.children) collect(node.children, next, out)
  }
}

function countBy(leaves: LeafHit[], pick: (leaf: LeafHit) => string) {
  const sizes = new Map<string, number>()
  for (const leaf of leaves) {
    const k = pick(leaf)
    sizes.set(k, (sizes.get(k) || 0) + 1)
  }
  const seen = new Map<string, number>()
  return (leaf: LeafHit) => {
    const k = pick(leaf)
    const n = (seen.get(k) || 0) + 1
    seen.set(k, n)
    return { index: n, size: sizes.get(k) || 1 }
  }
}

function refOf(leaf: LeafHit | undefined): LessonRef | null {
  return leaf ? { slug: leaf.slug, title: leaf.title } : null
}

function registerSection(section: SectionId) {
  const children = trees[section].children
  const core: LeafHit[][] = []
  const appendix: LeafHit[][] = []

  for (const branch of children) {
    const leaves: LeafHit[] = []
    collect([branch], { course: branch.name, unit: '', sequence: '', appendix: Boolean(branch.appendix) }, leaves)
    if (!leaves.length) continue
    if (branch.appendix) appendix.push(leaves)
    else core.push(leaves)
  }

  const ordered: LeafHit[] = []

  const register = (leaves: LeafHit[], spine: LeafHit[]) => {
    const inCourse = countBy(leaves, (l) => l.course)
    const inUnit = countBy(leaves, (l) => `${l.course}\0${l.unit}`)
    const inSeq = countBy(leaves, (l) => `${l.course}\0${l.unit}\0${l.sequence}`)
    const offset = spine.length

    for (let j = 0; j < leaves.length; j++) {
      const leaf = leaves[j]
      const coursePos = inCourse(leaf)
      const unitPos = inUnit(leaf)
      const seqPos = inSeq(leaf)
      const prior = j > 0 ? leaves[j - 1] : spine[offset - 1]
      const lesson: Lesson = {
        slug: leaf.slug,
        title: leaf.title,
        section,
        appendix: leaf.appendix,
        course: leaf.course,
        unit: leaf.unit,
        sequence: leaf.sequence,
        indexInCourse: coursePos.index,
        courseSize: coursePos.size,
        indexInUnit: unitPos.index,
        unitSize: unitPos.size,
        indexInSequence: seqPos.index,
        sequenceSize: seqPos.size,
        prev: refOf(prior),
        next: refOf(leaves[j + 1]),
        prereq: refOf(prior),
      }
      const k = key(section, leaf.slug)
      if (!lessons.has(k)) lessons.set(k, lesson)
    }
    spine.push(...leaves)
    ordered.push(...leaves)
  }

  const coreSpine: LeafHit[] = []
  for (const leaves of core) register(leaves, coreSpine)
  for (let i = 0; i < coreSpine.length - 1; i++) {
    const cur = lessons.get(key(section, coreSpine[i].slug))
    const nxt = coreSpine[i + 1]
    if (cur) cur.next = refOf(nxt)
  }

  for (const leaves of appendix) {
    const spine: LeafHit[] = []
    register(leaves, spine)
  }

  orderBySection[section] = ordered.map((l) => l.slug)
}

for (const id of SECTION_IDS) registerSection(id)

export function getLesson(section: string, slug: string): Lesson | null {
  return lessons.get(key(section, slug)) ?? null
}

export function curriculumOrder(section: SectionId): string[] {
  return orderBySection[section]
}
