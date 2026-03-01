## Git 的核心模型

Git 是一个**内容寻址文件系统（Content-Addressable Filesystem）**，上层封装了版本控制的操作接口。

区别于 CVS/SVN 的差量存储（delta storage）模型，Git 对每次提交保存完整的**快照（snapshot）**。内容相同的文件在整个仓库历史中只存储一份，空间效率并不差。

Git 仓库的核心是四类不可变对象，全部存储在 `.git/objects/`：

| 对象类型 | 存储内容 | 指向关系 |
|---------|---------|---------|
| **blob** | 单个文件的内容（不含文件名） | 无 |
| **tree** | 目录结构：文件名 + 权限 + 指向 blob/tree 的引用 | blob、tree |
| **commit** | 作者、时间戳、日志信息、指向根 tree 的引用 | tree、parent commit |
| **tag** | 带注释的标签（annotated tag） | 任意对象 |

每个对象由其**内容的 SHA-1 哈希**（40 位十六进制）唯一标识。对象一旦写入就不可修改——修改内容意味着产生新哈希、创建新对象。

---

## 对象图：一次提交的结构

执行 `git commit` 后，Git 在对象图中新增的节点如下：

```
commit a3f9c2e
├── tree 7d8b4a1          ← 项目根目录快照
│   ├── blob c4e2a8f  README.md
│   ├── blob 9f1b3d6  main.go
│   └── tree 3a7c5e2  src/
│       └── blob 1e9d4b8  parser.go
└── parent 8b2f1c4        ← 上一次提交
```

检查任意对象的原始内容：

```bash
git cat-file -p a3f9c2e   # 查看 commit 对象
git cat-file -p 7d8b4a1   # 查看 tree 对象
git cat-file -t a3f9c2e   # 查看对象类型
```

---

## refs 与 HEAD

**分支（branch）本质是一个可移动的指针**，内容是一个 commit 哈希，存储在 `.git/refs/heads/<branch-name>`。

```bash
cat .git/refs/heads/main
# 输出：a3f9c2e4b5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0
```

**HEAD** 是当前工作位置的指针，存储在 `.git/HEAD`：

- 正常状态：指向一个 ref（`ref: refs/heads/main`）
- 分离 HEAD（detached HEAD）：直接指向一个 commit 哈希

```bash
cat .git/HEAD
# 附着状态：ref: refs/heads/main
# 分离状态：a3f9c2e4b5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0
```

`git commit` 的本质：创建新 commit 对象，然后把当前 branch 的指针移动到新 commit。

---

## 暂存区（Index）

**暂存区（Index / Staging Area）**是工作目录与下一次提交之间的中间层，存储在 `.git/index`。

它记录的是"下一次提交将包含哪些文件的哪个版本"。三个区域的关系：

```
工作目录 → (git add) → 暂存区 → (git commit) → 对象库
```

`git add` 做了两件事：
1. 将文件内容写入对象库，生成 blob 对象
2. 更新 `.git/index`，记录文件名到 blob 哈希的映射

`git commit` 做了三件事：
1. 根据 index 构建 tree 对象（递归处理目录结构）
2. 创建 commit 对象，指向根 tree 和 parent commit
3. 更新当前 branch 指针

---

## 分支操作的机制

### merge

`git merge` 将两个分支的历史合并，存在两种路径：

**Fast-forward**：目标分支是当前分支的直接后继，只移动指针，不产生新 commit：

```
main: A → B → C
               ↑ feature
```

执行 `git merge feature` 后，`main` 直接移动到 C。

**Three-way merge**：两个分支有分叉历史，Git 找到公共祖先（merge base），将三方差异合并，产生一个新的 merge commit：

```
      C ← D  (feature)
     /
A ← B
     \
      E ← F  (main)
```

merge 后 main 的历史中保留了分支轨迹。

### rebase

`git rebase` 将一个分支的 commits 在另一个基点上**重放（replay）**，产生哈希不同但内容等价的新 commit：

```bash
git checkout feature
git rebase main
```

原 commit C、D 被重放为 C'、D'（新的哈希），feature 的 parent 从 B 变为 F。

> rebase 改写了 commit 哈希。已推送到远端的分支不应 rebase，否则会与他人的本地历史冲突。

---

## 常用操作速查

### 撤销与回退

| 操作 | 命令 | 影响范围 |
|-----|------|---------|
| 撤销暂存（保留工作目录改动） | `git restore --staged <file>` | 仅 index |
| 丢弃工作目录改动 | `git restore <file>` | 仅工作目录 |
| 回退到某次提交（保留改动为未暂存） | `git reset <commit>` | index + HEAD |
| 回退到某次提交（丢弃所有改动） | `git reset --hard <commit>` | index + 工作目录 + HEAD |
| 生成一个逆向 commit（不修改历史） | `git revert <commit>` | 对象库 |

### 历史查看

```bash
git log --oneline --graph --all      # 可视化分支历史
git log -p -- path/to/file           # 某文件的变更历史
git diff HEAD~3 HEAD                 # 最近 3 次提交的变更
git blame -L 10,20 main.go          # 某文件指定行的作者信息
git bisect start / good / bad        # 二分查找引入 bug 的 commit
```

### 储藏（Stash）

```bash
git stash push -m "描述"             # 储藏当前改动
git stash list                       # 列出所有储藏
git stash pop                        # 恢复最近一次储藏并删除
git stash apply stash@{2}            # 恢复指定储藏，不删除
```

### 引用修订语法

| 表达式 | 含义 |
|-------|------|
| `HEAD~3` | HEAD 往上 3 个 parent（一级） |
| `HEAD^2` | merge commit 的第 2 个 parent |
| `main@{yesterday}` | main 在昨天的位置（来自 reflog） |
| `v1.0..main` | v1.0 之后、main 之前的所有 commit |

---

## 工作流设计

常见的分支工作流：

| 工作流 | 适用场景 | 长期分支 |
|-------|---------|---------|
| **Trunk-Based Development** | 持续部署、小团队 | 仅 main |
| **GitHub Flow** | Web 应用、频繁发布 | main + feature branches |
| **Git Flow** | 有明确版本周期的项目 | main + develop + release + hotfix |

Trunk-Based Development 要求 feature flags 配合，适合 CI/CD 完善的团队；Git Flow 适合版本号明确、需要维护多个线上版本的软件。

---

## 参考资料

- [Pro Git（免费在线书）](https://git-scm.com/book/en/v2) — 第 10 章 Git Internals 对对象模型有完整说明
- [Git Object Model](https://git-scm.com/book/en/v2/Git-Internals-Git-Objects)
- [A Successful Git Branching Model](https://nvie.com/posts/a-successful-git-branching-model/) — Git Flow 原始论文
