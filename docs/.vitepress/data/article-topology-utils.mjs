const NODE_WIDTH = 272
const NODE_HEIGHT = 96
const COLUMN_GAP = 104
const ROW_GAP = 24
const PADDING = 24

function displayWidth(character) {
  return character.codePointAt(0) <= 0xff ? 0.58 : 1
}

function fitText(value, maximum = 22) {
  const characters = Array.from(String(value))
  let width = 0
  for (let index = 0; index < characters.length; index++) {
    width += displayWidth(characters[index])
    if (width > maximum) return characters.slice(0, Math.max(1, index)).join('') + '…'
  }
  return characters.join('')
}

export function wrapNodeTitle(value) {
  const lines = ['']
  let width = 0
  for (const character of Array.from(String(value))) {
    const nextWidth = width + displayWidth(character)
    if (lines[lines.length - 1] && nextWidth > 15) {
      if (lines.length === 2) {
        lines[1] = fitText(lines[1], 14)
        return lines
      }
      lines.push(character)
      width = displayWidth(character)
    } else {
      lines[lines.length - 1] += character
      width = nextWidth
    }
  }
  return lines
}

export function createArticleTopology(articles) {
  if (!Array.isArray(articles)) throw new TypeError('Articles must be an array')
  if (!articles.length) return { nodes: [], edges: [], width: 0, height: 0 }

  const byUrl = new Map()
  for (const article of articles) {
    if (!article?.url || byUrl.has(article.url)) throw new Error(`Duplicate or missing article URL: ${article?.url}`)
    if (!Array.isArray(article.prerequisites)) throw new Error(`Prerequisites must be an array: ${article.url}`)
    byUrl.set(article.url, article)
  }
  for (const article of articles) {
    if (new Set(article.prerequisites).size !== article.prerequisites.length) {
      throw new Error(`Duplicate prerequisite: ${article.url}`)
    }
    for (const prerequisite of article.prerequisites) {
      if (prerequisite === article.url) throw new Error(`Article cannot depend on itself: ${article.url}`)
      if (!byUrl.has(prerequisite)) throw new Error(`Missing published prerequisite: ${article.url} -> ${prerequisite}`)
    }
  }

  const depths = new Map()
  const visiting = new Set()
  function depth(article) {
    if (depths.has(article.url)) return depths.get(article.url)
    if (visiting.has(article.url)) throw new Error(`Article dependency cycle includes: ${article.url}`)
    visiting.add(article.url)
    const value = article.prerequisites.length
      ? Math.max(...article.prerequisites.map(url => depth(byUrl.get(url)))) + 1
      : 0
    visiting.delete(article.url)
    depths.set(article.url, value)
    return value
  }

  const layers = []
  for (const article of [...articles].sort((a, b) => String(a.date).localeCompare(String(b.date)) || a.url.localeCompare(b.url))) {
    const level = depth(article)
    layers[level] ??= []
    layers[level].push(article)
  }
  const rowCount = Math.max(...layers.map(layer => layer.length))
  const width = PADDING * 2 + layers.length * NODE_WIDTH + (layers.length - 1) * COLUMN_GAP
  const height = PADDING * 2 + rowCount * NODE_HEIGHT + (rowCount - 1) * ROW_GAP
  const nodes = layers.flatMap((layer, level) => layer.map((article, row) => {
    const layerHeight = layer.length * NODE_HEIGHT + (layer.length - 1) * ROW_GAP
    return {
      ...article,
      x: PADDING + level * (NODE_WIDTH + COLUMN_GAP),
      y: PADDING + (height - PADDING * 2 - layerHeight) / 2 + row * (NODE_HEIGHT + ROW_GAP),
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
      lines: wrapNodeTitle(article.title),
      caption: fitText([article.topic, article.date].filter(Boolean).join(' · '), 20),
    }
  }))
  const positions = new Map(nodes.map(node => [node.url, node]))
  const edges = nodes.flatMap(node => node.prerequisites.map(prerequisite => {
    const source = positions.get(prerequisite)
    const startX = source.x + NODE_WIDTH
    const endX = node.x
    const bend = Math.max(36, (endX - startX) / 2)
    return {
      id: `${prerequisite}->${node.url}`,
      source: prerequisite,
      target: node.url,
      path: `M ${startX} ${source.y + NODE_HEIGHT / 2} C ${startX + bend} ${source.y + NODE_HEIGHT / 2}, ${endX - bend} ${node.y + NODE_HEIGHT / 2}, ${endX} ${node.y + NODE_HEIGHT / 2}`,
    }
  }))
  return { nodes, edges, width, height }
}
