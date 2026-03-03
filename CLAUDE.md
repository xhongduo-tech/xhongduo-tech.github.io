# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 架构概览

纯静态博客，托管于 GitHub Pages，无构建步骤、无框架、无 npm。

```
index.html      # 首页：文章列表 + 分类/标签/搜索
post.html       # 文章页：通用模板，通过 ?slug= 参数加载对应 .md
about.html      # 关于页
css/style.css   # 全部样式（主题变量、排版、组件）
js/app.js       # 全部逻辑（路由、Markdown 渲染、反馈系统）
posts/
  posts.json    # 文章索引（唯一的"数据库"）
  *.md          # 文章内容（无 frontmatter，H2 起写正文）
  _template.md  # 格式参考，不会出现在列表中
```

**数据流**：`posts.json` → `app.js` 读取索引 → 按 slug 加载对应 `.md` → 用 marked.js 渲染 → KaTeX 处理数学公式 → highlight.js 处理代码高亮。三个库均从 CDN 加载，无本地依赖。

## 发布新文章

1. 在 `posts/` 新建 `slug-name.md`（文件名即 slug，用小写字母和连字符）
2. 内容从 `## ` 开始，H1 由系统自动渲染为标题
3. 在 `posts/posts.json` 数组**开头**插入条目：

```json
{
  "title":   "文章标题",
  "slug":    "slug-name",
  "date":    "YYYY-MM-DD",
  "author":  "claude",
  "tags":    ["分类标签（首个 tag 决定所属分类）", "其他标签"],
  "summary": "摘要，显示在列表页"
}
```

4. `git add posts/slug-name.md posts/posts.json && git commit -m "post: slug-name" && git push origin main`

## posts.json 关键字段

- **`tags[0]`**：首个标签决定文章所属分类（`底层原理` / `模型解析` / `智能体` / `工程实践`），分类顺序在 `app.js` 的 `CATEGORY_ORDER` 中定义
- **`author`**：`"claude"` = 仅显示 Claude Code 署名；`"both"` = 显示 Claude Code & xuhongduo（经用户核验后手动改为 `"both"`）
- **`slug`**：必须与 `.md` 文件名（去掉 `.md`）完全一致，否则 404

## 写作风格规范

所有文章统一采用**技术直白**风格（参见 `/tmp/style-guide.md`，每次对话初始时可能不存在，需重新生成）：

- 开篇直接给出定义或结论，不用"你一定遇到过"等铺垫
- 先结论后展开机制，用 `---` 控制章节节奏
- 禁止：口语化比喻替代定义、读者心理模拟、感叹号夸张、空洞过渡句
- 用表格替代冗长列举，精确术语替代生活类比

## 修改注意事项

- **下线文章**：从 `posts/posts.json` 删除条目（`.md` 文件可保留）
- **删除文章**：还需一并删除对应 `.md` 文件
- **新增分类**：需同步修改 `app.js` 中的 `CATEGORY_ORDER` 数组
- **样式变量**：CSS 主题色、字体、间距均定义在 `css/style.css` 顶部的 `:root` / `[data-theme="dark"]` 中
- **数学公式**：使用 `$...$`（行内）和 `$$...$$`（块级），由 KaTeX 渲染
