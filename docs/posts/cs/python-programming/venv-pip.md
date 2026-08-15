---
title: 虚拟环境与包管理：venv 与 pip
date: 2026-08-07
---

# 虚拟环境与包管理：venv 与 pip

<div class="epigraph">
<p>每个项目都该有自己的「图书馆」——这就是虚拟环境存在的意义。</p>
<footer>—— 本专题编者按</footer>
</div>

<div class="article-byline">
<p>第三级 · Python 编程入门与进阶 ｜ 官方 Python 教程 第12章 ｜ 2026-08-07</p>
</div>

## 为什么从虚拟环境开始

「装个库就完事了」——很多初学者这么想，直到两个项目撞车：项目 A 要 `numpy 1.24`，项目 B 要 `numpy 1.20`，装了一个另一个就坏。这就是**依赖地狱**。官方 Python 教程第 12 章给出的解药是**虚拟环境（virtual environment）** 与包管理工具 **pip**。

本节学三件事：为什么需要隔离环境、venv 怎么用、pip 怎么装包与复现依赖。这是从「写脚本」走向「做工程」的标志性一步。

## 1 依赖地狱：为什么要隔离环境

**依赖（dependency）**：程序依赖的第三方包。`pip install` 默认装进全局的 `site-packages`——于是所有项目共享同一份包，版本冲突在所难免。

**虚拟环境（venv）**：为单个项目创建**独立**的 Python 环境，拥有自己的 `site-packages` 与 `python` 命令，包互不干扰。

**重点：venv 解决的是「隔离」，pip 解决的是「安装」。** 两者常被混淆。venv 创造一方小天地，pip 在小天地里装东西——装在哪里，取决于你激活了哪个环境。全局安装就像所有住户共用一个大仓库，虚拟环境则是每户一间独立的储物间。<span class="marginnote">Python 3.3 前虚拟环境要靠第三方工具 `virtualenv`，3.3 起标准库内建 `venv` 模块——本专题按官方教程第 12 章，使用内置 `venv`。新版 Python 还会默认引导 `pip` 一起安装。</span>

## 2 venv：创建、激活与退出

创建与激活是四个命令：

```bash
python -m venv .venv          # 创建虚拟环境目录 .venv
source .venv/bin/activate     # macOS / Linux：激活
.venv\Scripts\activate        # Windows：激活
deactivate                    # 退出虚拟环境
```

激活后，终端提示符前会出现 `(.venv)` 前缀，`python` 与 `pip` 都指向环境内的版本：

```bash
(.venv) $ which python        # /path/to/project/.venv/bin/python
(.venv) $ pip list            # 环境内的包列表（默认很少）
```

**重点：`.venv` 是一个普通目录，不是系统魔法。** 它包含一份「私有」的解释器、`site-packages` 与激活脚本；激活只是把环境内路径放到 `PATH` 最前面。删掉 `.venv` 就能彻底清空这个环境——所以它从不进版本控制，`.gitignore` 里应包含它。<span class="marginnote">「不提交 `.venv`」是工程铁律：依赖列表（`requirements.txt`）进版本库，环境本体不进。别人克隆项目后，用 `pip install -r requirements.txt` 重建一个一模一样的环境——「代码 + 依赖清单」才是完整的项目。</span>

## 3 pip：安装、冻结与复现

**pip（Pip Installs Packages）**：Python 官方包安装器，从 **PyPI**（Python Package Index，Python 的「应用商店」）下载并安装包。

```bash
pip install requests          # 安装最新版
pip install "requests==2.31.0"   # 指定精确版本
pip install "numpy>=1.20,<2"     # 版本范围
pip uninstall requests        # 卸载
pip show requests             # 查看包信息
```

安装后，代码里 `import requests` 即可使用——注意 import 名与包名常不同（`pip install pillow` 后 `import PIL`）。

**依赖复现**是工程的核心场景：把当前环境的包快照成清单，供他人重建：

```bash
pip freeze > requirements.txt     # 导出所有包与版本
pip install -r requirements.txt   # 按清单重建环境
```

**重点：`requirements.txt` 是依赖的「快递单」。** 它一行一个「包==版本」，锁定精确版本，保证「在我机器上能跑」变成「在任何机器上都能跑」。CI、部署、团队协作都靠它。一份典型的清单长这样：

```text
requests==2.31.0
numpy==1.26.0
pandas==2.1.0
```

版本锁定让「换台机器、重建环境」变成完全可复现的操作：先 `git clone` 拉代码，再 `python -m venv .venv && pip install -r requirements.txt`，一个与原作者一致的环境就绪。这正是「可复现性」的起点——在机器学习、数据科学的工程里，它和「结果可复现」同等重要。

## 5 依赖清单最佳实践

工程上围绕 `requirements.txt` 有两条成熟惯例：

**第一，区分「直接依赖」与「全部依赖」。** `pip freeze` 导出的是环境中**所有**包（含传递依赖，即依赖的依赖）；而项目真正需要写进清单的，是你 `import` 的**直接依赖**。实践上常维护两个文件：`requirements.in`（手写的直接依赖）与 `requirements.txt`（导出的完整锁版）。<span class="marginnote">只锁 `requirements.txt` 有隐患：传递依赖升级可能悄悄破坏环境。高级方案是用 `pip-tools`、`poetry` 这类工具，把「我要什么」与「实际装了什么」分离管理——本专题介绍 pip 原语，工具只是把同样的原语封装得更好。</span>

**第二，环境名与目录约定。** 用 `.venv` 作为虚拟环境目录名已是社区默认，且必须写进 `.gitignore`：

```text
.venv/
__pycache__/
```

**重点：`.venv`、缓存、编译产物都不该进版本控制。** 进版本库的只有代码与依赖清单——别人克隆后 `pip install -r requirements.txt` 即可重建。把环境目录误提交，轻则仓库臃肿，重则在不同机器上冲突。

## 4 核心对比表：venv 与 pip 的分工

| 维度 | venv | pip |
| --- | --- | --- |
| 职责 | 创建隔离环境 | 安装/管理包 |
| 作用对象 | 解释器与 site-packages | 第三方包 |
| 类比 | 独立的储物间 | 往储物间搬东西的搬运工 |
| 常用命令 | `python -m venv`、`activate` | `install`、`freeze`、`uninstall` |
| 与项目关系 | 每个项目一个 | 作用于当前激活的环境 |

**核心观察：先建环境、再装依赖。** 正确顺序永远是 `python -m venv .venv` → `source .venv/bin/activate` → `pip install ...`。忘激活就 `pip install`，会把包装进全局——这是「装到了但别处看不见/版本冲突」的源头。

**辨析｜易错点：** `pip install` 装在哪里，取决于**当前激活的环境**，而不是「项目所在目录」。同一个项目，激活不同环境装的东西互不相通。此外，`conda`（Anaconda 自带）是另一套环境 + 包管理方案，它管的不只是 Python 包（还有 C 库等），选择哪个由团队约定。<span class="marginnote">`pip install` 通常需要联网访问 PyPI；在内网或无网络环境可用 `pip download` 离线下载、`--index-url` 指定镜像源。国内常用清华、阿里云镜像加速——把镜像地址写进 `~/.pip/pip.conf` 即可长期生效。</span>

## 6 小结

- **依赖地狱**源于全局共享包导致的版本冲突，虚拟环境用**隔离**解决它。
- `python -m venv .venv` 创建环境，`source .venv/bin/activate` 激活，`deactivate` 退出。
- `.venv` 是普通目录，不进版本控制；激活只是把环境内路径置于 `PATH` 之首。
- **pip** 从 PyPI 安装包；`pip freeze > requirements.txt` 导出、`pip install -r requirements.txt` 重建。
- 顺序铁律：先建环境、再激活、后安装；忘激活会把包装进全局。
- 受限网络：用 `--index-url` 指定镜像源、`pip download` 离线打包，可应对内网安装。

在下一节，我们将为整个专题收尾——测试代码：unittest 与 pytest，让程序「可验证、可回归」。
