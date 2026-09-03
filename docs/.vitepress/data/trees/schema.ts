export type Kind = 'branch' | 'mainline' | 'group' | 'leaf'

export interface Node {
  kind: Kind
  name: string
  slug?: string
  children?: Node[]
  /** 同时挂在其他主线/分支上的显示名 */
  alsoIn?: string[]
}

/** '标题|slug'；纯英文则自动生成 slug */
export function leaf(spec: string): Node {
  const [name, explicit] = spec.split('|').map((s) => s.trim())
  const slug =
    explicit ||
    name
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '')
  return { kind: 'leaf', name, slug: slug || name }
}

export function group(kind: Kind, name: string, children: Node[]): Node {
  return { kind, name, children }
}

export const B = (name: string, children: Node[]) => group('branch', name, children)
export const M = (name: string, children: Node[]) => group('mainline', name, children)
export const G = (name: string, children: Node[]) => group('group', name, children)
export const L = leaf

export function leafCount(nodes: Node[]): number {
  let n = 0
  for (const node of nodes) {
    if (node.kind === 'leaf') n++
    else if (node.children) n += leafCount(node.children)
  }
  return n
}

const KIND_BY_DEPTH: Kind[] = ['branch', 'mainline', 'group']

/** 嵌套数组大纲：字符串为叶子，[名称, 子节点] 为分组 */
export type Outline = string | readonly [string, readonly Outline[]]

export function fromOutline(items: readonly Outline[], depth = 0): Node[] {
  return items.map((item) => {
    if (typeof item === 'string') return leaf(item)
    const [name, kids] = item
    const kind = KIND_BY_DEPTH[Math.min(depth, KIND_BY_DEPTH.length - 1)]
    return { kind, name, children: fromOutline(kids, depth + 1) }
  })
}
