# CLAUDE.md

This file is kept for compatibility and provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## 架构概览

纯静态博客，托管于 GitHub Pages，无构建步骤、无框架、无 npm。

```
index.html                      # 首页：文章列表 + 分类/标签/搜索
post.html                       # 旧 ?slug= 跳转 shim（自动跳转到 posts/<slug>.html）
about.html                      # 关于页（简历 + 自我介绍）
projects.html                   # 项目页（手工维护）
css/style.css                   # 全部样式（主题变量、排版、组件）
js/site.js                      # 共享逻辑（主题、导航、代码复制、页脚年份）
js/index.js                     # 首页文章列表与筛选
js/post.js                      # 文章页阅读进度、KaTeX / highlight.js 初始化
posts/
  posts.json                    # 文章索引（唯一的"数据库"）
  *.html                        # 每篇文章一个独立 HTML 文件
archive/
  knowledge-tree.html           # 历史页面归档
  llm-basics.html               # 历史页面归档
```

**数据流**：每篇文章是独立 HTML；`posts.json` 只保存元数据供首页索引与渲染。文章页仍可使用 KaTeX（`$...$` / `$$...$$`）与 highlight.js，依赖从 CDN 加载，无本地构建。

## 发布新文章

1. 在 `posts/` 新建 `slug-name.html`（文件名即 slug，用小写字母和连字符），可复制 `posts/what-is-nvidia-mps.html` 作为模板
2. 在 `posts/posts.json` 数组**开头**插入条目：

```json
{
  "title":   "文章标题",
  "slug":    "slug-name",
  "date":    "YYYY-MM-DD",
  "author":  "both",
  "tags":    ["分类标签（首个 tag 决定所属分类）", "其他标签"],
  "summary": "摘要，显示在列表页",
  "url":     "posts/slug-name.html"
}
```

3. `git add posts/slug-name.html posts/posts.json && git commit -m "post: slug-name" && git push origin main`

## posts.json 关键字段

- **`tags[0]`**：首个标签决定文章所属分类，分类顺序在 `js/index.js` 的 `CATEGORY_ORDER` 中定义
- **`url`**：从站点根目录出发的文章 HTML 路径
- **`slug`**：必须与 `.html` 文件名（去掉 `.html`）完全一致，否则旧链接跳转会失败
- **`author`**：统一使用 `"both"`

## 写作风格规范

所有文章统一采用**技术直白**风格：

- 开篇直接给出定义或结论，不用"你一定遇到过"等铺垫
- 先结论后展开机制，用 `---` / `<hr>` 控制章节节奏
- 禁止：口语化比喻替代定义、读者心理模拟、感叹号夸张、空洞过渡句
- 用表格替代冗长列举，精确术语替代生活类比

## 修改注意事项

- **下线文章**：从 `posts/posts.json` 删除条目（`.html` 文件可保留）
- **删除文章**：还需一并删除对应 `.html` 文件
- **新增分类**：需同步修改 `js/index.js` 中的 `CATEGORY_ORDER` 数组
- **样式变量**：CSS 主题色、字体、间距均定义在 `css/style.css` 顶部的 `:root` / `[data-theme="dark"]` 中
- **数学公式**：使用 `$...$`（行内）和 `$$...$$`（块级），由 KaTeX auto-render 处理
- **项目页**：`projects.html` 为手工维护页面，不再自动展示历史项目
- **归档页面**：`knowledge-tree.html` 与 `llm-basics.html` 已移至 `archive/`，不参与主导航
