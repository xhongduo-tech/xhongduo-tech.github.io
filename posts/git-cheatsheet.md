# Git 常用命令速查手册

整理了日常开发最常用的 Git 操作，方便随时查阅。

## 仓库初始化

```bash
git init                         # 初始化本地仓库
git clone <url>                  # 克隆远程仓库
git clone <url> --depth 1        # 浅克隆（只拉最新一次提交）
```

## 暂存 & 提交

```bash
git status                       # 查看工作区状态
git add <file>                   # 暂存指定文件
git add .                        # 暂存所有变更
git commit -m "描述"             # 提交
git commit --amend               # 修改最后一次提交（未推送时用）
```

## 分支管理

```bash
git branch                       # 查看本地分支
git branch -a                    # 查看所有分支（含远程）
git branch <name>                # 创建分支
git checkout <name>              # 切换分支
git checkout -b <name>           # 创建并切换
git switch -c <name>             # 同上（新语法）
git branch -d <name>             # 删除已合并分支
git branch -D <name>             # 强制删除分支
```

## 合并 & 变基

```bash
git merge <branch>               # 合并分支（保留历史）
git rebase <branch>              # 变基（线性历史）
git rebase -i HEAD~3             # 交互式变基（整理最近 3 次提交）
git cherry-pick <commit>         # 摘取指定提交
```

## 撤销 & 回滚

```bash
git restore <file>               # 丢弃工作区修改
git restore --staged <file>      # 取消暂存
git reset HEAD~1                 # 撤销最后一次提交（保留修改）
git reset --hard HEAD~1          # 撤销并丢弃修改（危险！）
git revert <commit>              # 生成反向提交（安全回滚）
```

## Stash 暂存区

```bash
git stash                        # 保存当前工作进度
git stash list                   # 查看所有 stash
git stash pop                    # 恢复最新 stash 并删除
git stash apply stash@{1}        # 恢复指定 stash
git stash drop stash@{0}         # 删除指定 stash
```

## 远程操作

```bash
git remote -v                    # 查看远程地址
git remote add origin <url>      # 添加远程
git fetch origin                 # 拉取但不合并
git pull origin main             # 拉取并合并
git push origin <branch>         # 推送分支
git push --force-with-lease      # 安全强推（推荐替代 --force）
```

## 日志 & 查看

```bash
git log --oneline --graph        # 紧凑图形化日志
git log -p <file>                # 查看文件的历史变更
git diff                         # 查看未暂存的修改
git diff --staged                # 查看已暂存的修改
git blame <file>                 # 查看每行的最后修改者
git show <commit>                # 查看某次提交详情
```

## 标签

```bash
git tag                          # 列出标签
git tag v1.0.0                   # 创建轻量标签
git tag -a v1.0.0 -m "Release"   # 创建附注标签
git push origin --tags           # 推送所有标签
```

## 配置

```bash
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global core.editor vim
git config --list                # 查看所有配置
```

## 常见场景

### 误删了分支，找回提交

```bash
git reflog                       # 找到对应的 commit hash
git checkout -b recovered <hash>
```

### 清理已合并的本地分支

```bash
git branch --merged | grep -v '\*\|main\|master' | xargs git branch -d
```

### 只拉某个文件夹（稀疏检出）

```bash
git clone --no-checkout <url>
cd repo
git sparse-checkout set src/
git checkout main
```

---

记住：**永远不要对公共分支做 `reset --hard` 或 `force push`**，这会破坏团队协作。
