# 发布新文章 — 操作手册

## 文件结构

```
myblogs/
├── index.html          # 首页（文章列表）
├── post.html           # 文章阅读页（通用，无需修改）
├── about.html          # 关于页
├── css/style.css       # 样式
├── js/app.js           # 逻辑
└── posts/
    ├── posts.json      # ← 文章索引（每次发布都要更新这个）
    ├── _template.md    # 格式参考模板（不会出现在列表）
    ├── my-first-post.md
    └── another-post.md
```

---

## 发布流程（3 步）

### 第 1 步：写文章

在 `posts/` 目录下新建 Markdown 文件，文件名即为 slug：

```
posts/deep-dive-into-rust-lifetimes.md
```

- 文件名只用小写字母、数字、连字符
- 无需写 YAML frontmatter，标题/日期/标签都在 posts.json 里管理
- 内容从 H2 (`##`) 开始写，H1 由系统自动渲染为文章标题
- 格式参考 `posts/_template.md`

### 第 2 步：注册到索引

打开 `posts/posts.json`，在数组**开头**插入新条目：

```json
[
  {
    "title":   "深入理解 Rust 生命周期",
    "slug":    "deep-dive-into-rust-lifetimes",
    "date":    "2026-03-01",
    "tags":    ["Rust", "内存管理", "系统编程"],
    "summary": "一句话描述文章核心内容，显示在首页列表中。"
  },
  ...已有文章...
]
```

字段说明：

| 字段      | 类型     | 说明                              |
| --------- | -------- | --------------------------------- |
| `title`   | string   | 文章完整标题                      |
| `slug`    | string   | 必须与 `.md` 文件名（去掉扩展名）完全一致 |
| `date`    | string   | 格式 `YYYY-MM-DD`，用于排序和展示 |
| `tags`    | string[] | 标签数组，用于分类过滤            |
| `summary` | string   | 一两句摘要，展示在列表页          |

### 第 3 步：推送

```bash
git add posts/your-new-post.md posts/posts.json
git commit -m "post: your-new-post"
git push origin main
```

GitHub Pages 会在 1-2 分钟内自动更新。

---

## 支持的 Markdown 功能

| 功能          | 语法示例                         |
| ------------- | -------------------------------- |
| 标题          | `## H2` `### H3` `#### H4`      |
| 粗体 / 斜体   | `**粗体**` `*斜体*`              |
| 行内代码      | `` `code` ``                     |
| 代码块        | ` ```rust ... ``` `（支持高亮）  |
| 链接          | `[文字](URL)`                    |
| 图片          | `![alt](URL)`                    |
| 引用块        | `> 引用内容`                     |
| 有序列表      | `1. 2. 3.`                       |
| 无序列表      | `- item`                         |
| 表格          | `\| col \| col \|`               |
| 水平线        | `---`                            |

### 支持的代码高亮语言（常用）

`rust` `go` `python` `typescript` `javascript` `bash` `sh`
`c` `cpp` `java` `sql` `yaml` `toml` `json` `dockerfile`
`html` `css` `markdown` `plaintext`

---

## 删除 / 下线文章

只需从 `posts/posts.json` 中删除对应条目即可（`.md` 文件可保留）。
