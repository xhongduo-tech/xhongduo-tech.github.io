import test from 'node:test'
import assert from 'node:assert/strict'
import { mkdtemp, mkdir, writeFile, readFile, rm } from 'node:fs/promises'
import { execFileSync, spawnSync } from 'node:child_process'
import { tmpdir } from 'node:os'
import path from 'node:path'

const repo = process.cwd()
const checker = path.join(repo, 'scripts/check-content.mjs')
const creator = path.join(repo, 'scripts/new-note.mjs')
const article = (status = 'published', extra = '') => `---
title: 可核对的生成实验
description: 固定输入和基线的机制检查
kind: article
status: ${status}
area: inference
evidence: mechanism
date: 2026-11-16
${extra}---
# 可核对的生成实验
## 问题与先修
一个明确问题。
## 机制
说明机制。
## 实验
说明实验条件与边界。
## 结论
给出范围。
## 参考
[原始资料](https://arxiv.org/abs/2211.17192)
`
async function fixture(run) {
  const dir = await mkdtemp(path.join(tmpdir(), 'blog-authoring-'))
  try { await mkdir(path.join(dir, 'docs'), { recursive: true }); await run(dir) }
  finally { await rm(dir, { recursive: true, force: true }) }
}
function check(dir) { return spawnSync(process.execPath, [checker], { cwd: dir, encoding: 'utf8' }) }

test('published article accepts an ISO date parsed by YAML', async () => {
  await fixture(async (dir) => {
    await writeFile(path.join(dir, 'docs/article.md'), article())
    const result = check(dir)
    assert.equal(result.status, 0, result.stderr)
    assert.match(result.stdout, /1篇正式博文/)
  })
})

test('public draft and broken internal link are rejected', async () => {
  await fixture(async (dir) => {
    await writeFile(path.join(dir, 'docs/article.md'), article('draft') + '\n[missing](/does-not-exist)\n')
    const result = check(dir)
    assert.equal(result.status, 1)
    assert.match(result.stderr, /未发表草稿/)
    assert.match(result.stderr, /内部链接不存在/)
  })
})

test('hardware claims need model, hardware and raw-result metadata', async () => {
  await fixture(async (dir) => {
    await writeFile(path.join(dir, 'docs/article.md'), article().replace('evidence: mechanism', 'evidence: hardware'))
    const result = check(dir)
    assert.equal(result.status, 1)
    assert.match(result.stderr, /硬件实测/)
  })
})

test('new-note writes a private draft, refuses overwrites and path traversal', async () => {
  await fixture(async (dir) => {
    execFileSync(process.execPath, [creator, 'retrieval-check', '--area', 'applications', '--title', '检索证据的一个问题'], { cwd: dir })
    const content = await readFile(path.join(dir, 'drafts/retrieval-check.md'), 'utf8')
    assert.match(content, /status: draft/)
    assert.match(content, /area: applications/)
    assert.match(content, /检索证据的一个问题/)
    assert.notEqual(spawnSync(process.execPath, [creator, 'retrieval-check'], { cwd: dir }).status, 0)
    assert.notEqual(spawnSync(process.execPath, [creator, '../escape'], { cwd: dir }).status, 0)
  })
})
