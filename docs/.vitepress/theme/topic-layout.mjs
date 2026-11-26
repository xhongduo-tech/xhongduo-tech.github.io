import {
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
} from '../data/topic-topology.mjs'

const ROW_PITCH = 84
const GROUP_HEADER = 42
const GROUP_BOTTOM = 18
const GROUP_GAP = 22
const NODE_GAP = 28
const LANE_PADDING = 14

export function buildFocusedDomainLayout(nodes, groups, domain, columns = 1) {
  const columnCount = columns === 2 ? 2 : 1
  const positions = {}
  const layoutGroups = []
  let y = 110

  for (const group of groups.filter(candidate => candidate.domain === domain)) {
    const groupNodes = nodes.filter(node => node.group === group.id)
    const rows = Math.ceil(groupNodes.length / columnCount)
    const width = LANE_PADDING * 2 + columnCount * TOPIC_NODE_WIDTH + (columnCount - 1) * NODE_GAP
    const height = GROUP_HEADER + (rows - 1) * ROW_PITCH + TOPIC_NODE_HEIGHT + GROUP_BOTTOM
    const x = 48
    layoutGroups.push({ ...group, x, y, width, height })
    groupNodes.forEach((node, index) => {
      positions[node.id] = {
        x: x + LANE_PADDING + (index % columnCount) * (TOPIC_NODE_WIDTH + NODE_GAP),
        y: y + GROUP_HEADER + Math.floor(index / columnCount) * ROW_PITCH,
      }
    })
    y += height + GROUP_GAP
  }

  const firstGroup = layoutGroups[0]
  const lastGroup = layoutGroups[layoutGroups.length - 1]
  const layoutDomain = firstGroup ? {
    x: firstGroup.x - 12,
    y: 74,
    width: firstGroup.width + 24,
    height: lastGroup.y + lastGroup.height + 22 - 74,
  } : null
  return { id: domain, positions, groups: layoutGroups, domain: layoutDomain, columns: columnCount }
}
