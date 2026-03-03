# 文章标题（此处不需要写，标题来自 posts.json 的 title 字段）

## 引言

这是一篇示例文章，展示博客支持的所有 Markdown 格式。写作时直接使用 H2 作为顶级章节标题——H1 由系统自动渲染为文章标题。

---

## 代码块

支持语法高亮，在三个反引号后指定语言名称：

```rust
fn main() {
    let msg = "hello, world";
    println!("{}", msg);
}
```

```python
def binary_search(arr: list, target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) >> 1
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1
```

```bash
# 构建并推送
git add posts/my-new-post.md posts/posts.json
git commit -m "post: add my-new-post"
git push origin main
```

```sql
SELECT u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2025-01-01'
GROUP BY u.id
ORDER BY order_count DESC
LIMIT 10;
```

行内代码：`O(log n)` 时间复杂度，`O(1)` 额外空间。

---

## 表格

| 算法        | 时间复杂度    | 空间复杂度 | 稳定性 |
| ----------- | ------------- | ---------- | ------ |
| Quick Sort  | O(n log n)    | O(log n)   | 不稳定 |
| Merge Sort  | O(n log n)    | O(n)       | 稳定   |
| Heap Sort   | O(n log n)    | O(1)       | 不稳定 |
| Tim Sort    | O(n log n)    | O(n)       | 稳定   |

---

## 列表

**有序列表：**

1. 分析问题，确定瓶颈
2. 建立 benchmark，量化当前性能
3. 做最小化修改，重新 benchmark
4. 反复迭代，直到满足目标

**无序列表：**

- 不要过早优化
- 先测量，再改
- 保持代码可读性
- 写注释说明 *为什么*，而不是 *做了什么*

---

## 引用

> Programs must be written for people to read, and only incidentally for machines to execute.
>
> — Harold Abelson, *SICP*

---

## 标题层级

### H3 子章节

H3 用于二级细分，颜色偏灰，视觉权重低于 H2。

#### H4 小节

H4 使用小号大写字母样式，用于更细粒度的分组。

---

## 强调

**粗体**用于强调关键术语，*斜体*用于引用或次要说明，~~删除线~~用于已废弃的做法。

---

## 链接

[GitHub](https://github.com) · [Rust 官方文档](https://doc.rust-lang.org)

---

## 水平分割线

三个或更多破折号即可生成分割线（如上所示）。

---

*本文件是格式模板，不会出现在文章列表中（文件名以 `_` 开头仅作约定，实际由 posts.json 控制哪些文章可见）。*
