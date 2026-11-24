import test from 'node:test'
import assert from 'node:assert/strict'
import { createArticleTopology, wrapNodeTitle } from '../docs/.vitepress/data/article-topology-utils.mjs'

const article = (url, prerequisites = []) => ({
  url, title: `文章 ${url}`, description: '', date: '2026-11-23', topic: '推理', prerequisites,
})

test('an empty public index produces no invented nodes', () => {
  assert.deepEqual(createArticleTopology([]), { nodes: [], edges: [], width: 0, height: 0 })
})

test('a chain is laid out from prerequisite to dependent article', () => {
  const graph = createArticleTopology([
    article('/c', ['/b']), article('/a'), article('/b', ['/a']),
  ])
  assert.equal(graph.nodes.length, 3)
  assert.equal(graph.edges.length, 2)
  const positions = Object.fromEntries(graph.nodes.map(node => [node.url, node]))
  assert.ok(positions['/a'].x < positions['/b'].x)
  assert.ok(positions['/b'].x < positions['/c'].x)
  assert.deepEqual(graph.edges.map(edge => [edge.source, edge.target]), [
    ['/a', '/b'], ['/b', '/c'],
  ])
  assert.ok(graph.edges.every(edge => edge.path.startsWith('M ')))
})

test('a diamond keeps shared prerequisites and centers independent nodes', () => {
  const graph = createArticleTopology([
    article('/d', ['/b', '/c']), article('/c', ['/a']),
    article('/a'), article('/b', ['/a']),
  ])
  const positions = Object.fromEntries(graph.nodes.map(node => [node.url, node]))
  assert.equal(graph.edges.length, 4)
  assert.ok(positions['/b'].x > positions['/a'].x)
  assert.equal(positions['/b'].x, positions['/c'].x)
  assert.notEqual(positions['/b'].y, positions['/c'].y)
  assert.ok(positions['/d'].x > positions['/b'].x)
})

test('missing, self, duplicate and cyclic prerequisites are rejected', () => {
  assert.throws(() => createArticleTopology([article('/a', ['/missing'])]), /Missing published prerequisite/)
  assert.throws(() => createArticleTopology([article('/a', ['/a'])]), /cannot depend on itself/)
  assert.throws(() => createArticleTopology([article('/a', ['/a', '/a'])]), /Duplicate prerequisite/)
  assert.throws(() => createArticleTopology([
    article('/a', ['/b']), article('/b', ['/a']),
  ]), /cycle/)
})

test('long titles remain bounded to two SVG lines', () => {
  const lines = wrapNodeTitle('这是一篇用于验证很长文章标题在知识拓扑节点中不会溢出的技术文章，并继续增加文字以确认超出两行后的截断行为')
  assert.equal(lines.length, 2)
  assert.ok(lines[1].endsWith('…'))
  assert.ok(lines.every(line => line.length > 0))
})
