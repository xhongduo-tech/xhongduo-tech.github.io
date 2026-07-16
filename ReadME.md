# xhd.log

徐鸿铎的技术笔记，记录大模型推理、Agent 工程、CUDA/GPU 与系统实践。

## 架构

纯静态博客，托管于 GitHub Pages，无构建步骤、无框架、无 npm。

```
index.html                      # 首页：文章列表 + 分类/标签/搜索
post.html                       # 旧 ?slug= 跳转 shim
about.html                      # 关于页（简历 + 自我介绍）
projects.html                   # 项目页（手工维护）
css/style.css                   # 全部样式（主题变量、排版、组件）
js/site.js                      # 共享逻辑（主题、导航、代码复制）
js/index.js                     # 首页文章列表与筛选
js/post.js                      # 文章页阅读进度、KaTeX / highlight.js
posts/
  posts.json                    # 文章索引（元数据）
  *.html                        # 每篇文章一个独立 HTML 文件
archive/
  knowledge-tree.html           # 历史页面归档
  llm-basics.html               # 历史页面归档
```

**数据流**：每篇文章是独立 HTML；`posts.json` 只保存元数据供首页索引与渲染。文章页仍可使用 KaTeX（`$...$` / `$$...$$`）与 highlight.js，依赖从 CDN 加载。

## 本地预览

```bash
python3 -m http.server 8000
# 或
npx serve .
```

访问 `http://localhost:8000`。

## 发布新文章

1. 在 `posts/` 新建 `slug-name.html`，使用现有文章（如 `what-is-nvidia-mps.html`）作为模板。
2. 在 `posts/posts.json` 数组**开头**插入条目：

```json
{
  "title":   "文章标题",
  "slug":    "slug-name",
  "date":    "YYYY-MM-DD",
  "author":  "both",
  "tags":    ["分类标签", "其他标签"],
  "summary": "摘要",
  "url":     "posts/slug-name.html"
}
```

3. `git add posts/slug-name.html posts/posts.json && git commit -m "post: slug-name" && git push`

## 分类

首个 tag 决定文章分类。当前分类顺序在 `js/index.js` 的 `CATEGORY_ORDER` 中定义：

`LLM 推理` / `Agents` / `CUDA / GPU` / `模型部署` / `系统工程` / `训练与微调` / `论文 / 前沿` / `工程实践`

## License

MIT
