// 由 VitePress 构建时用 import.meta.glob 生成博文索引（路径 + 标题 + 分类），
// 供篇末导航（上一篇/下一篇）使用。用 ?raw 只取 markdown 原文（轻量字符串），
// 避免把每篇博文的编译产物（含 MathJax SVG）打进客户端 bundle。
// 标题优先取 frontmatter.title，否则取首个 # 标题；路径转为不含 base 的路由 path。
const mdFiles = import.meta.glob('../../../docs/posts/**/*.md', {
  eager: true,
  query: '?raw',
  import: 'default',
}) as Record<string, string>

export interface PostEntry {
  path: string
  title: string
  category: string // tier/category，用于同分类上下篇分组
}

function extractTitle(src: string): string {
  const fm = src.match(/^---\r?\n([\s\S]*?)\r?\n---\r?\n/)
  if (fm) {
    const fmTitle = fm[1].match(/^title:\s*(.+)$/m)
    if (fmTitle) return fmTitle[1].trim().replace(/^["']|["']$/g, '')
  }
  const body = fm ? src.slice(fm[0].length) : src
  const h1 = body.match(/^#\s+(.+)/m)
  return h1 ? h1[1].trim() : ''
}

const posts: PostEntry[] = []

for (const [filePath, src] of Object.entries(mdFiles)) {
  // filePath: ../../../docs/posts/foundations/math/set-concept.md
  const rel = filePath.replace(/^\.\.\/\.\.\/\.\.\/docs\//, '')
  if (!rel.startsWith('posts/')) continue
  const route = '/' + rel.replace(/\.md$/, '').replace(/(^|\/)index$/, '$1')
  const title = extractTitle(src) || rel.split('/').pop()?.replace(/\.md$/, '') || rel
  // 分类：posts/<tier>/<cat> -> <tier>/<cat>（index.md 也归属该分类）
  const parts = rel.split('/')
  const category = parts.length >= 4 ? `${parts[1]}/${parts[2]}` : ''
  posts.push({ path: route, title, category })
}

// 按路径排序，保证同级顺序稳定
posts.sort((a, b) => a.path.localeCompare(b.path))

export default posts
