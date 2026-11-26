import test from 'node:test'
import assert from 'node:assert/strict'
import {
  topicNodes,
  topicGroups,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
} from '../docs/.vitepress/data/topic-topology.mjs'
import { buildFocusedDomainLayout } from '../docs/.vitepress/theme/topic-layout.mjs'

const domains = [...new Set(topicNodes.map(node => node.domain))]

test('focused layouts keep every topic inside its specialization and preserve reading order', () => {
  for (const domain of domains) {
    for (const columns of [1, 2]) {
      const layout = buildFocusedDomainLayout(topicNodes, topicGroups, domain, columns)
      const domainNodes = topicNodes.filter(node => node.domain === domain)
      assert.equal(Object.keys(layout.positions).length, domainNodes.length)
      assert.equal(layout.groups.length, topicGroups.filter(group => group.domain === domain).length)
      for (const group of layout.groups) {
        const groupNodes = domainNodes.filter(node => node.group === group.id)
        groupNodes.forEach((node, index) => {
          const position = layout.positions[node.id]
          const expectedX = group.x + 14 + (index % columns) * (TOPIC_NODE_WIDTH + 28)
          const expectedY = group.y + 42 + Math.floor(index / columns) * 84
          assert.deepEqual(position, { x: expectedX, y: expectedY })
          assert.ok(position.x >= group.x && position.x + TOPIC_NODE_WIDTH <= group.x + group.width)
          assert.ok(position.y >= group.y && position.y + TOPIC_NODE_HEIGHT <= group.y + group.height)
        })
      }
    }
  }
})

test('desktop two-column groups are compact without overlapping nodes or group lanes', () => {
  for (const domain of domains) {
    const single = buildFocusedDomainLayout(topicNodes, topicGroups, domain, 1)
    const double = buildFocusedDomainLayout(topicNodes, topicGroups, domain, 2)
    assert.ok(double.domain.height <= single.domain.height)
    assert.ok(double.domain.width > single.domain.width)
    for (const group of double.groups) {
      const nodes = topicNodes.filter(node => node.group === group.id).map(node => ({ id: node.id, ...double.positions[node.id] }))
      for (let left = 0; left < nodes.length; left += 1) {
        for (let right = left + 1; right < nodes.length; right += 1) {
          const a = nodes[left]
          const b = nodes[right]
          const separated = a.x + TOPIC_NODE_WIDTH <= b.x || b.x + TOPIC_NODE_WIDTH <= a.x || a.y + TOPIC_NODE_HEIGHT <= b.y || b.y + TOPIC_NODE_HEIGHT <= a.y
          assert.ok(separated, `Overlapping nodes: ${a.id} and ${b.id}`)
        }
      }
    }
    for (let index = 1; index < double.groups.length; index += 1) {
      assert.ok(double.groups[index - 1].y + double.groups[index - 1].height <= double.groups[index].y)
    }
  }
})
