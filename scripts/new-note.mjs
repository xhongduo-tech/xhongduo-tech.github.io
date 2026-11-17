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
---

# ${title}

## 问题与先修

待补充。

## 机制与正确性

待补充。

## 源码路径

待补充仓库、commit和调用链。

## 实验与结果

实验待执行。记录基线、变量、环境、配置与原始结果。

## 结论与局限

待补充。

## 参考资料

待补充第一方资料。
`
await writeFile(target, body, { encoding: 'utf8', flag: 'wx' })
process.stdout.write(`已创建未公开草稿：${target}\n`)
