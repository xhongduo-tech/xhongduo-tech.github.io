import { mkdir, writeFile, access } from 'node:fs/promises'
import path from 'node:path'

const [slug, ...args] = process.argv.slice(2)
function option(name, fallback) {
  const index = args.indexOf(name)
  return index === -1 ? fallback : args[index + 1]
}
const area = option('--area', 'inference')
const title = option('--title', slug)
if (!slug || !/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug) || !['foundations', 'inference', 'training', 'evaluation', 'applications'].includes(area) || !title) {
  process.stderr.write('用法：npm run note:new -- slug --area inference --title "文章问题"\n')
  process.exit(1)
}
const target = path.resolve('drafts', slug + '.md')
await mkdir(path.dirname(target), { recursive: true })
try {
  await access(target)
  process.stderr.write('草稿已存在，不覆盖。\n')
  process.exit(1)
} catch (error) {
  if (error.code !== 'ENOENT') throw error
}
const date = new Intl.DateTimeFormat('en-CA', { timeZone: process.env.BLOG_TIMEZONE || 'America/Los_Angeles' }).format(new Date())
const body = `---
title: ${JSON.stringify(title)}
description: 待补充文章范围
kind: article
status: draft
area: ${area}
evidence: mechanism
date: ${date}
prerequisites: []
---

# ${title}

## 问题与先修

待补充原方案、具体缺口、问题边界与依赖。

## 机制与正确性

待补充关键推导或状态不变量、符号/形状、假设及反例。

## 最小参考实现

待补充手算/reference、实现范围、边界与误差检查。

## 源码路径

待补充固定commit、数据/状态调用链，以及源码如何对应前述推导或不变量。

## 实验与结果

实验待执行。先记录假设、预期与否证条件，再记录基线、变量、消融/反例、环境和原始结果。

## 结论与局限

待补充逐项证据、失效边界、条件变化，以及本人修改与排错记录。

## 参考资料

待补充第一方资料。
`
await writeFile(target, body, { encoding: 'utf8', flag: 'wx' })
process.stdout.write(`已创建未公开草稿：${target}\n`)
