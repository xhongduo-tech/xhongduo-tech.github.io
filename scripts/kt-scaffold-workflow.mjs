// 知识树缺口专题脚手架生成工作流
// 每个缺口专题一个代理：创建目录 + 生成 index.md（对标教材章节规划，`- [ ]` 未勾选条目）
// 输入：args = [{ name, tier, key, textbooks, rationale }]（或 { topics: [...] }）
// 输出：每个专题的 { path, created, chapters } 报告

export const meta = {
  name: 'kt-scaffold',
  description: '为知识树缺口专题生成脚手架 index.md（对标教材章节规划）',
  phases: [{ title: '脚手架', detail: '每专题一个代理创建目录并写 index.md' }],
}

const topics = Array.isArray(args) ? args : args.topics || []

function promptFor(t) {
  const dir = `/Users/xuhongduo/Projects/blog/docs/posts/${t.tier}/${t.key}`
  return `你是知识树专题脚手架专家，为新增专题【${t.name}】创建脚手架。

背景：${t.rationale || '该专题是知识树审查发现的缺口。'}

任务：
1. 创建目录：${dir}（mkdir -p）。
2. 在该目录写入 index.md，格式必须完全符合本站既有专题（先读一个范本：/Users/xuhongduo/Projects/blog/docs/posts/foundations/math/index.md 的前 30 行了解格式）：

---
pageClass: plain-doc
---

# ${t.name}

<一句简介：该学科是什么、为什么值得学，1-2 句话>

## 对标教材

- <教材1>
- <教材2>
- <教材3（如有）>

## 主题规划

<ProgressGrid cat="${t.tier}/${t.key}" />

### 第1篇

- 未勾选条目：章节1标题（教材名 第N章）
- 未勾选条目：章节2标题（教材名 第N章）

### 第2篇
- 未勾选条目：……

3. 章节规划要求：
   - 依据对标教材的目录结构（${(t.textbooks || []).join('；')}），按学科内在逻辑组织成 2-5 个「第N篇」分组
   - 每篇 4-8 条，全文 12-30 条「未勾选条目」——即 markdown 复选框：短横线 + 空格 + 左方括号 + 空格 + 右方括号 + 空格 + 标题
   - 每条 = 一个可写的独立主题，标题简洁，括号内标注对标教材与章节（如「配位键与晶体场理论（Cotton 第20章）」）
   - 从基础到进阶自然递进
4. 只写脚手架（index.md），不写文章正文。
5. 完成后用 ls 确认文件存在。

输出 JSON：{ path: '${t.tier}/${t.key}', created: true/false, chapters: 一个数字 }`
}

const RESULT_SCHEMA = {
  type: 'object',
  required: ['path', 'created', 'chapters'],
  properties: {
    path: { type: 'string' },
    created: { type: 'boolean' },
    chapters: { type: 'integer' },
  },
}

phase('脚手架')
const results = await parallel(
  topics.map((t) => () =>
    agent(promptFor(t), { label: `scaffold:${t.key}`, phase: '脚手架', schema: RESULT_SCHEMA, agentType: 'general-purpose' }),
  ),
)

const ok = results.filter(Boolean)
log(`脚手架完成：${ok.length}/${topics.length}`)
return { done: ok }
