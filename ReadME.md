# xhd.log

技术随笔，记录代码、架构与思考碎片。欢迎在知识海洋中遨游。

## 架构

纯静态博客，托管于 GitHub Pages，无构建步骤、无框架、无 npm。

```
index.html      # 首页：文章列表 + 分类/标签/搜索
post.html       # 文章页：通过 ?slug= 加载对应 .md
about.html      # 关于页
css/style.css   # 全部样式（主题变量、排版、组件）
js/app.js       # 全部逻辑（路由、Markdown 渲染、反馈系统）
posts/
  posts.json    # 文章索引
  *.md          # 文章内容（从 ## 起写正文）
```

**数据流**：`posts.json` → `app.js` 读取索引 → 按 slug 加载 `.md` → marked.js 渲染 → KaTeX 数学公式 → highlight.js 代码高亮。依赖均从 CDN 加载。

## 本地预览

```bash
# 任意静态服务器均可，例如：
python3 -m http.server 8000
# 或
npx serve .
```

访问 `http://localhost:8000`。

## 发布新文章

1. 在 `posts/` 新建 `slug-name.md`（文件名即 slug）
2. 内容从 `## ` 开始，H1 由系统自动渲染
3. 在 `posts/posts.json` 数组**开头**插入条目：

```json
{
  "title":   "文章标题",
  "slug":    "slug-name",
  "date":    "YYYY-MM-DD",
  "author":  "both",
  "tags":    ["分类标签", "其他标签"],
  "summary": "摘要"
}
```

4. `git add posts/slug-name.md posts/posts.json && git commit -m "post: slug-name" && git push`

## 分类

首个 tag 决定文章分类：`底层原理` / `模型解析` / `智能体` / `工程实践`。顺序在 `app.js` 的 `CATEGORY_ORDER` 中定义。

## License

MIT
