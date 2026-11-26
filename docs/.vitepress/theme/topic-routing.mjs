const CORNER_RADIUS = 10
const NODE_PADDING = 3

function normalizePoints(points) {
  const unique = points.filter((point, index) => index === 0 || point.x !== points[index - 1].x || point.y !== points[index - 1].y)
  return unique.filter((point, index) => {
    if (index === 0 || index === unique.length - 1) return true
    const previous = unique[index - 1]
    const next = unique[index + 1]
    return !((previous.x === point.x && point.x === next.x) || (previous.y === point.y && point.y === next.y))
  })
}

function segmentIntersectsNode(a, b, node, width, height) {
  const left = node.x - NODE_PADDING
  const right = node.x + width + NODE_PADDING
  const top = node.y - NODE_PADDING
  const bottom = node.y + height + NODE_PADDING
  if (a.x === b.x) {
    return a.x > left && a.x < right && Math.max(Math.min(a.y, b.y), top) < Math.min(Math.max(a.y, b.y), bottom)
  }
  if (a.y === b.y) {
    return a.y > top && a.y < bottom && Math.max(Math.min(a.x, b.x), left) < Math.min(Math.max(a.x, b.x), right)
  }
  return true
}

export function topicRouteCollisionCount(points, obstacles, width, height) {
  let collisions = 0
  for (let index = 1; index < points.length; index += 1) {
    for (const obstacle of obstacles) {
      if (segmentIntersectsNode(points[index - 1], points[index], obstacle, width, height)) collisions += 1
    }
  }
  return collisions
}

function routeLength(points) {
  let length = 0
  for (let index = 1; index < points.length; index += 1) {
    length += Math.abs(points[index].x - points[index - 1].x) + Math.abs(points[index].y - points[index - 1].y)
  }
  return length
}

function roundedPath(points) {
  const number = value => Number(value.toFixed(2))
  let path = `M ${number(points[0].x)} ${number(points[0].y)}`
  for (let index = 1; index < points.length - 1; index += 1) {
    const previous = points[index - 1]
    const current = points[index]
    const next = points[index + 1]
    const incoming = Math.abs(current.x - previous.x) + Math.abs(current.y - previous.y)
    const outgoing = Math.abs(next.x - current.x) + Math.abs(next.y - current.y)
    const radius = Math.min(CORNER_RADIUS, incoming / 2, outgoing / 2)
    const before = {
      x: current.x - Math.sign(current.x - previous.x) * radius,
      y: current.y - Math.sign(current.y - previous.y) * radius,
    }
    const after = {
      x: current.x + Math.sign(next.x - current.x) * radius,
      y: current.y + Math.sign(next.y - current.y) * radius,
    }
    path += ` L ${number(before.x)} ${number(before.y)} Q ${number(current.x)} ${number(current.y)} ${number(after.x)} ${number(after.y)}`
  }
  const last = points[points.length - 1]
  return `${path} L ${number(last.x)} ${number(last.y)}`
}

export function routeTopicEdge(source, target, visibleNodes, width, height) {
  const horizontal = Math.abs(target.x - source.x) > 40
  const forward = target.x > source.x
  const down = target.y > source.y
  const start = horizontal
    ? { x: source.x + (forward ? width : 0), y: source.y + height / 2 }
    : { x: source.x + width / 2, y: source.y + (down ? height : 0) }
  const end = horizontal
    ? { x: target.x + (forward ? 0 : width), y: target.y + height / 2 }
    : { x: target.x + width / 2, y: target.y + (down ? 0 : height) }
  const direction = horizontal ? (forward ? 1 : -1) : (down ? 1 : -1)
  const candidates = []

  if (start.x === end.x || start.y === end.y) candidates.push([start, end])
  if (horizontal) {
    const middle = (start.x + end.x) / 2
    candidates.push([start, { x: middle, y: start.y }, { x: middle, y: end.y }, end])
  } else {
    const middle = (start.y + end.y) / 2
    candidates.push([start, { x: start.x, y: middle }, { x: end.x, y: middle }, end])
  }

  const stubs = [8, 14, 24, 36]
  if (horizontal) {
    const channels = [
      source.y - 8, source.y + height + 8,
      target.y - 8, target.y + height + 8,
      source.y - 10, source.y + height + 10,
      target.y - 10, target.y + height + 10,
      source.y - 18, source.y + height + 18,
      target.y - 18, target.y + height + 18,
      source.y - 28, source.y + height + 28,
      target.y - 28, target.y + height + 28,
    ]
    for (const stub of stubs) {
      const sourceLane = start.x + direction * stub
      const targetLane = end.x - direction * stub
      for (const channel of channels) {
        candidates.push([
          start,
          { x: sourceLane, y: start.y },
          { x: sourceLane, y: channel },
          { x: targetLane, y: channel },
          { x: targetLane, y: end.y },
          end,
        ])
      }
    }
  } else {
    const channels = [
      source.x - 8, source.x + width + 8,
      target.x - 8, target.x + width + 8,
      source.x - 10, source.x + width + 10,
      target.x - 10, target.x + width + 10,
      source.x - 18, source.x + width + 18,
      target.x - 18, target.x + width + 18,
      source.x - 28, source.x + width + 28,
      target.x - 28, target.x + width + 28,
    ]
    for (const stub of stubs) {
      const sourceLane = start.y + direction * stub
      const targetLane = end.y - direction * stub
      for (const channel of channels) {
        candidates.push([
          start,
          { x: start.x, y: sourceLane },
          { x: channel, y: sourceLane },
          { x: channel, y: targetLane },
          { x: end.x, y: targetLane },
          end,
        ])
      }
    }
  }

  const obstacles = visibleNodes.filter(node => node.id !== source.id && node.id !== target.id)
  const routes = candidates.map(points => normalizePoints(points)).filter(points => points.length > 1)
  const scored = routes.map((points, index) => ({
    points,
    collisions: topicRouteCollisionCount(points, obstacles, width, height),
    score: routeLength(points) + (points.length - 2) * 14 + index * 0.001,
  }))
  scored.sort((a, b) => a.collisions - b.collisions || a.score - b.score)
  const points = scored[0].points
  return { path: roundedPath(points), points }
}
