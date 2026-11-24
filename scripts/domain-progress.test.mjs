import test from 'node:test'
import assert from 'node:assert/strict'
import { domains, domainEdges, domainProgress } from '../docs/.vitepress/data/domain-progress.mjs'

test('public domain topology has six unique, bounded and connected domains', () => {
  assert.equal(domains.length, 6)
  const ids = new Set(domains.map(domain => domain.id))
  assert.equal(ids.size, domains.length)
  for (const domain of domains) {
    assert.ok(domain.x >= 0 && domain.x + 190 <= 1024)
    assert.ok(domain.y >= 0 && domain.y + 88 <= 324)
    assert.ok(domain.url.startsWith('/') && domain.urlEn.startsWith('/en/'))
    assert.equal(typeof domainProgress[domain.id]?.completed, 'boolean')
  }
  assert.equal(domainEdges.length, 6)
  for (const [source, target] of domainEdges) {
    assert.ok(ids.has(source) && ids.has(target))
    assert.notEqual(source, target)
  }
})

test('a lit domain always names the public evidence it relies on', () => {
  for (const domain of domains) {
    const progress = domainProgress[domain.id]
    if (progress.completed) assert.ok(progress.evidence.startsWith('/'))
  }
})
