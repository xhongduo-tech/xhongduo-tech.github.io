# 发布新文章 — 操作手册

## 文件结构

```
myblog_codex/
├── index.html
├── post.html
├── about.html
├── css/style.css
├── js/app.js
└── posts/
    ├── posts.json      # 文章索引（每次发布都要更新）
    ├── _template.md    # 格式参考（不会出现在列表）
    └── *.md            # 文章正文（无 frontmatter）
```

---

## 发布流程（3 步）

### 第 1 步：写文章

在 `posts/` 目录新建 Markdown 文件，文件名即 slug，例如：

```text
posts/deep-dive-into-rust-lifetimes.md
```

要求：

- 文件名只使用小写字母、数字、连字符
- 不写 frontmatter
- 正文从 `##` 开始（H1 由系统根据 `posts.json.title` 自动渲染）

### 第 2 步：注册到 `posts/posts.json`

在数组**开头**插入新条目：

```json
{
  "title": "深入理解 Rust 生命周期",
  "slug": "deep-dive-into-rust-lifetimes",
  "date": "2026-03-07",
  "author": "both",
  "tags": ["工程实践", "Rust", "系统编程"],
  "summary": "一句话描述文章核心内容，显示在首页列表。"
}
```

字段约束：

- `slug` 必须与 Markdown 文件名完全一致（去掉 `.md`）
- `author` 固定为 `"both"`（显示 `Codex & xuhongduo`）
- `tags[0]` 决定文章分类，分类顺序由 `js/app.js` 的 `CATEGORY_ORDER` 控制
- `date` 使用 `YYYY-MM-DD`

### 第 3 步：提交并推送

```bash
git add posts/your-new-post.md posts/posts.json
git commit -m "post: your-new-post"
git push origin main
```

---

## 下线与删除

- 下线文章：仅从 `posts/posts.json` 删除条目
- 删除文章：同时删除对应 `posts/<slug>.md`
