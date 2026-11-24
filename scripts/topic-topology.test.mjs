import test from 'node:test'
import assert from 'node:assert/strict'
import {
  topicNodes,
  topicEdges,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
  TOPIC_WORLD_WIDTH,
  TOPIC_WORLD_HEIGHT,
} from '../docs/.vitepress/data/topic-topology.mjs'

const DOMAIN_X = {
  foundations: 60,
  training: 360,
  inference: 660,
  evaluation: 960,
  applications: 1260,
  projects: 1560,
}

test('topic nodes form a public bilingual index with unique stable IDs', () => {
  assert.ok(topicNodes.length >= 58 && topicNodes.length <= 66)
  assert.deepEqual(new Set(topicNodes.map(node => node.domain)), new Set(Object.keys(DOMAIN_X)))

  const ids = new Set()
  for (const node of topicNodes) {
    assert.deepEqual(Object.keys(node).sort(), ['domain', 'id', 'title', 'titleEn', 'x', 'y'].sort())
    assert.match(node.id, /^[a-z0-9]+(?:-[a-z0-9]+)*$/)
    assert.ok(!ids.has(node.id), `Duplicate topic ID: ${node.id}`)
    ids.add(node.id)
    assert.ok(Object.hasOwn(DOMAIN_X, node.domain), `Invalid domain: ${node.domain}`)
    assert.ok(node.title.trim().length > 0)
    assert.ok(node.titleEn.trim().length > 0)
    assert.ok(node.titleEn.length <= 28, `English title is too long: ${node.titleEn}`)
  }
})

test('topic nodes use the six fixed columns and fit the world bounds', () => {
  assert.equal(TOPIC_NODE_WIDTH, 240)
  assert.equal(TOPIC_NODE_HEIGHT, 68)
  assert.equal(TOPIC_WORLD_WIDTH, 1850)

  for (const [domain, x] of Object.entries(DOMAIN_X)) {
    const nodes = topicNodes.filter(node => node.domain === domain)
    assert.ok(nodes.length > 0, `Missing domain: ${domain}`)
    nodes.forEach((node, index) => {
      assert.equal(node.x, x, `Unexpected column for ${node.id}`)
      assert.equal(node.y, 110 + index * 88, `Unexpected row for ${node.id}`)
    })
  }

  for (const node of topicNodes) {
    assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y))
    assert.ok(node.x >= 0 && node.y >= 0)
    assert.ok(node.x + TOPIC_NODE_WIDTH <= TOPIC_WORLD_WIDTH, `Node exceeds world width: ${node.id}`)
    assert.ok(node.y + TOPIC_NODE_HEIGHT <= TOPIC_WORLD_HEIGHT, `Node exceeds world height: ${node.id}`)
  }
  const expectedHeight = Math.max(...topicNodes.map(node => node.y)) + TOPIC_NODE_HEIGHT + 100
  assert.equal(TOPIC_WORLD_HEIGHT, expectedHeight)
})

test('topic edges are unique, non-reflexive and reference existing nodes', () => {
  assert.ok(topicEdges.length >= 70 && topicEdges.length <= 100)
  const ids = new Set(topicNodes.map(node => node.id))
  const edges = new Set()
  for (const edge of topicEdges) {
    assert.ok(Array.isArray(edge) && edge.length === 2)
    const [source, target] = edge
    assert.ok(ids.has(source), `Missing edge source: ${source}`)
    assert.ok(ids.has(target), `Missing edge target: ${target}`)
    assert.notEqual(source, target, `Self-edge: ${source}`)
    const key = `${source}\0${target}`
    assert.ok(!edges.has(key), `Duplicate edge: ${source} -> ${target}`)
    edges.add(key)
  }
})

test('topic prerequisites are acyclic', () => {
  const adjacency = new Map(topicNodes.map(node => [node.id, []]))
  const indegree = new Map(topicNodes.map(node => [node.id, 0]))
  for (const [source, target] of topicEdges) {
    adjacency.get(source).push(target)
    indegree.set(target, indegree.get(target) + 1)
  }

  const queue = [...indegree].filter(([, degree]) => degree === 0).map(([id]) => id)
  let visited = 0
  for (let index = 0; index < queue.length; index += 1) {
    const source = queue[index]
    visited += 1
    for (const target of adjacency.get(source)) {
      indegree.set(target, indegree.get(target) - 1)
      if (indegree.get(target) === 0) queue.push(target)
    }
  }
  assert.equal(visited, topicNodes.length, 'Topic dependency graph contains a cycle')
})
