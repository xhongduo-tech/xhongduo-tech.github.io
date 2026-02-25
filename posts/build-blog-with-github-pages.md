# 用 GitHub Pages 搭建免费的个人博客

这篇文章介绍如何将这套博客系统部署到 GitHub Pages，整个过程不需要服务器，完全免费。

## 目录结构

```
my-blog/
├── index.html          # 首页（文章列表）
├── post.html           # 文章阅读页
├── about.html          # 关于页
├── css/style.css       # 全局样式
├── js/app.js           # 核心逻辑
└── posts/
    ├── posts.json      # 文章索引 ← 每次新增文章都要更新这里
    ├── my-first-post.md
    └── another-post.md
```

## 部署步骤

### 第一步：创建 GitHub 仓库

在 GitHub 上新建一个仓库，命名为 `<你的用户名>.github.io`，例如：

```
xuhongduo.github.io
```

### 第二步：推送代码

```bash
git init
git add .
git commit -m "初始化博客"
git branch -M main
git remote add origin https://github.com/<用户名>/<用户名>.github.io.git
git push -u origin main
```

### 第三步：开启 GitHub Pages

进入仓库 **Settings → Pages**，将 Source 设为 `Deploy from a branch`，Branch 选 `main`，点击 Save。

几分钟后访问 `https://<用户名>.github.io` 即可看到博客。

## 如何发布新文章

1. 在 `posts/` 目录下新建 Markdown 文件，文件名即为 slug，例如 `my-new-post.md`
2. 在 `posts/posts.json` 中添加一条记录：

```json
{
  "title": "文章标题",
  "slug": "my-new-post",
  "date": "2026-03-01",
  "tags": ["标签1", "标签2"],
  "summary": "文章简介，显示在列表页"
}
```

3. `git add . && git commit -m "新增文章" && git push`

推送后 GitHub Pages 会自动更新，通常不到 1 分钟即可访问。

## 功能一览

| 功能 | 说明 |
| ---- | ---- |
| Markdown 渲染 | 支持全部 GFM 语法 |
| 代码高亮 | 自动识别语言 |
| 深色模式 | 一键切换，记忆偏好 |
| 全文搜索 | 按标题、摘要、标签实时过滤 |
| 标签分类 | 点击标签筛选文章 |
| 上下篇导航 | 文章底部自动生成 |
| 响应式布局 | 适配手机、平板、桌面 |

## 小贴士

> 文章的 `slug` 要与文件名保持一致（不含 `.md` 后缀），否则会加载失败。

代码块示例：

```python
def hello(name: str) -> str:
    return f"Hello, {name}!"

print(hello("World"))
```

---

就这么简单！享受写作吧。
