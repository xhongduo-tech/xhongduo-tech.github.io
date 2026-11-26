import test from 'node:test'
import assert from 'node:assert/strict'
import {
  topicNodes,
  topicEdges,
  topicGroups,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
} from '../docs/.vitepress/data/topic-topology.mjs'
import {
  routeTopicEdge,
  topicRouteCollisionCount,
} from '../docs/.vitepress/theme/topic-routing.mjs'
import { buildFocusedDomainLayout } from '../docs/.vitepress/theme/topic-layout.mjs'

const WIDTH = TOPIC_NODE_WIDTH
const HEIGHT = TOPIC_NODE_HEIGHT

function assertRouteAvoidsNodes(source, target, nodes) {
  const route = routeTopicEdge(source, target, nodes, WIDTH, HEIGHT)
  const obstacles = nodes.filter(node => node.id !== source.id && node.id !== target.id)
  assert.equal(topicRouteCollisionCount(route.points, obstacles, WIDTH, HEIGHT), 0, `Route crosses an unrelated node: ${source.id} -> ${target.id}`)
  assert.doesNotMatch(route.path, /NaN|Infinity/)
  return route
}

test('a vertical dependency routes around intermediate topic cards', () => {
  const source = { id: 'source', x: 60, y: 100 }
  const blocker = { id: 'blocker', x: 60, y: 184 }
  const target = { id: 'target', x: 60, y: 268 }
  const route = assertRouteAvoidsNodes(source, target, [source, blocker, target])
  assert.ok(route.points.some(point => point.x !== source.x + WIDTH / 2), 'The route should use a side channel')
})

test('a horizontal dependency cannot appear to connect through an unrelated card', () => {
  const source = { id: 'serving', x: 60, y: 268 }
  const guardrail = { id: 'guardrail', x: 660, y: 100 }
  const target = { id: 'benchmark', x: 960, y: 100 }
  assertRouteAvoidsNodes(source, target, [source, guardrail, target])
})

test('every responsive domain route avoids all unrelated visible topic cards', () => {
  const domains = new Set(topicNodes.map(node => node.domain))
  let checked = 0
  for (const domain of domains) {
    for (const columns of [1, 2]) {
      const layout = buildFocusedDomainLayout(topicNodes, topicGroups, domain, columns)
      const visible = topicNodes.filter(node => node.domain === domain).map(node => ({ id: node.id, ...layout.positions[node.id] }))
      const visibleIds = new Set(visible.map(node => node.id))
      const byId = new Map(visible.map(node => [node.id, node]))
      for (const [sourceId, targetId] of topicEdges) {
        if (!visibleIds.has(sourceId) || !visibleIds.has(targetId)) continue
        assertRouteAvoidsNodes(byId.get(sourceId), byId.get(targetId), visible)
        checked += 1
      }
    }
  }
  assert.ok(checked > 0)
})

test('routes retain the expected source and target ports after normalization', () => {
  const source = { id: 'source', x: 60, y: 100 }
  const target = { id: 'target', x: 60, y: 436 }
  const blocker = { id: 'blocker', x: 60, y: 268 }
  const route = assertRouteAvoidsNodes(source, target, [source, blocker, target])
  assert.deepEqual(route.points[0], { x: source.x + WIDTH / 2, y: source.y + HEIGHT })
  assert.deepEqual(route.points.at(-1), { x: target.x + WIDTH / 2, y: target.y })
})
