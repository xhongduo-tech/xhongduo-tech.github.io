# 发布新文章 — 操作手册

## 文件结构

```
personal_blog/
├── index.html
├── post.html                 # 旧 ?slug= 跳转 shim
├── about.html
├── projects.html
├── css/style.css
├── js/site.js
├── js/index.js
├── js/post.js
└── posts/
    ├── posts.json            # 文章索引（每次发布都要更新）
    └── *.html                # 每篇文章一个独立 HTML 文件
```

---

## 发布流程（3 步）

### 第 1 步：写 HTML 文章

在 `posts/` 目录新建 HTML 文件，文件名即 slug，例如：

```text
posts/deep-dive-into-rust-lifetimes.html
```

可直接复制 `posts/what-is-nvidia-mps.html` 作为模板，然后替换标题、元信息、正文内容。

要求：

- 文件名只使用小写字母、数字、连字符
- 文章头部使用 `post-header` 写入标题、日期、分类、标签
- 正文包裹在 `class="prose"` 的容器中
- 代码块使用 `.code-wrapper` + `pre code.language-xxx`（可省略语言类）
- 行内/块级数学公式仍可用 `$...$` / `$$...$$`，由 `js/post.js` 调用 KaTeX auto-render 处理

### 第 2 步：注册到 `posts/posts.json`

在数组**开头**插入新条目：

```json
{
  "title": "深入理解 Rust 生命周期",
  "slug": "deep-dive-into-rust-lifetimes",
  "date": "2026-03-07",
  "author": "both",
  "tags": ["工程实践", "Rust", "系统编程"],
  "summary": "一句话描述文章核心内容，显示在首页列表。",
  "url": "posts/deep-dive-into-rust-lifetimes.html"
}
```

字段约束：

- `slug` 必须与 HTML 文件名完全一致（去掉 `.html`）
- `url` 为从站点根目录出发的文章路径
- `tags[0]` 决定文章分类，分类顺序由 `js/index.js` 的 `CATEGORY_ORDER` 控制
- `date` 使用 `YYYY-MM-DD`

### 第 3 步：提交并推送

```bash
git add posts/your-new-post.html posts/posts.json
git commit -m "post: your-new-post"
git push origin main
```

---

## 下线与删除

- 下线文章：仅从 `posts/posts.json` 删除条目
- 删除文章：同时删除对应 `posts/<slug>.html`
