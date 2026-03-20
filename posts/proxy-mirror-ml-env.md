## 核心结论

训练环境里的“代理与加速”不是单点配置，而是一条可控链路。最稳妥的做法是同时管住 4 个入口：`pip/conda/uv` 的包索引、Hugging Face 的访问端点与缓存目录、`HTTP_PROXY/HTTPS_PROXY` 代理变量、局域网内的 Nexus 私有仓库或代理仓库。

这条链路可以写成一个简单公式：

$$
可控链路 = 包索引镜像 + 代理变量 + 本地缓存 + 内网仓库
$$

如果只改其中一个点，结果通常是不稳定的。比如你把 `pip` 切到内网镜像，但容器里没有 `HTTP_PROXY`，那么 `pip` 也许能装包，`huggingface_hub` 却依然会超时；或者你本机能访问镜像，容器里却因为没有继承代理变量而失败。

对新手最有用的一条命令是：

```bash
pip config set global.index-url https://nexus.lan/repository/pypi-group/simple/
```

它的含义很直接：把 `pip install` 默认访问的 Python 包索引，从公网 `pypi.org` 改成公司内网的 Nexus Group 地址。之后同一台机器上的大多数 `pip install` 都会先走内网。

工具 | 首选配置 | 命令行覆盖 | 额外索引/补充
--- | --- | --- | ---
`pip` | `pip config` 或 `pip.conf` / `pip.ini` | `-i` / `--index-url` | `--extra-index-url`
`conda` | `.condarc` | `-c` / `--channel`，必要时配 `--override-channels` | 多个 `channels`
`uv` | `uv.toml` 或 `pyproject.toml` 中 `[tool.uv]` / `[[tool.uv.index]]` | `--index` / `--default-index`，兼容 `--index-url` / `--extra-index-url` | 多个 `[[tool.uv.index]]`
`huggingface_hub` | 环境变量，如 `HF_HOME`、`HF_HUB_CACHE` | 运行命令前 `export` | 离线模式 `HF_HUB_OFFLINE=1`

---

## 问题定义与边界

这里说的“训练环境”，指的是跑模型训练、数据处理、实验复现的一类机器：物理机、云主机、Kubernetes Pod、Docker 容器都算。它们的共同问题是：经常要下载很多 Python 包和模型文件，但网络条件往往受限。

“镜像”是指把上游仓库内容转发或缓存到离你更近、受你控制的地址。  
“代理”是指把请求先发给一个中间服务器，由它代你访问目标站点。  
“缓存”是指把已经下载过的文件保留在本地，下次直接复用。

边界要先定清楚。本文只讨论这几类出站访问：

1. Python 包安装：`pip`、`conda`、`uv`
2. Hugging Face Hub 模型/数据下载
3. 容器与宿主机之间的代理继承差异
4. 局域网内通过 Nexus 统一暴露 PyPI 入口

不展开的内容有两类：

1. 操作系统级全局透明代理
2. CUDA、apt/yum、Docker 镜像仓库的加速配置

一个最常见的新手场景是：容器里执行 `pip install torch`，默认会去 `https://pypi.org/simple`。如果公司网络策略只允许内网出口或者只开放代理，而你没配镜像和代理，这条请求就会直接失败。此时最小修复不是“反复重试”，而是显式指定：

```bash
pip install -i https://nexus.lan/repository/pypi-group/simple/ torch
```

这就是把“默认公网路径”改成“内网可控路径”。

玩具例子可以这样理解。假设你要拿一本教材：

- 直接去城外总书库，路远，可能封路
- 先去学校图书馆镜像点，快很多
- 图书馆没有时，再走备用路径

训练环境也是同样逻辑，只不过“教材”变成 wheel、conda package、模型权重文件。

---

## 核心机制与推导

先看包索引的优先级。核心规则可以概括为：

$$
最终索引 = CLI\ 显式参数 \gt 项目级/用户级配置 \gt 工具默认值
$$

对 `pip` 来说，`pip config set global.index-url ...` 会写入配置文件；但如果你在命令行又写了 `pip install -i ...`，那么命令行优先。对 `conda` 来说，`.condarc` 中的 `channels`、`default_channels`、`channel_alias` 决定默认通道；`-c` 或 `--override-channels` 可以临时改写。对 `uv` 来说，推荐配置是 `[[tool.uv.index]]` 或 `uv.toml`；命令行的 `--index`、`--default-index` 优先级更高，同时也兼容 `--index-url` 与 `--extra-index-url`。

这里有一个容易误解的点：`pip` 的 `--extra-index-url` 不是“严格第二优先级兜底”，而是“额外搜索位置”。从 pip 官方说明看，它会综合多个位置寻找候选包，再选出匹配版本。工程上因此不能把它当成绝对安全的私有包隔离机制，否则会遇到依赖混淆风险。

下面用一个玩具例子说明“覆盖链”：

- 用户配置：`pip.conf` 里写内网 Nexus
- 当前项目没单独配置
- 命令行执行 `pip install -i https://pypi.org/simple requests`

最终访问公网，因为命令行显式参数覆盖了用户配置。

`uv` 的机制也类似。比如执行：

```bash
uv add requests --index-url https://nexus.lan/repository/pypi-group/simple/
```

如果命令行已给出索引地址，那么它会优先使用这个地址，而不是退回用户级配置。这和“命令行覆盖配置文件”是同一条原则。

Hugging Face 这边要分成两个层面看：

1. 访问哪个 Hub 或镜像端点
2. 下载后的缓存落到哪里

`HF_HOME` 是 Hugging Face 本地数据根目录，白话讲就是“整套缓存和令牌放在哪个文件夹”。  
`HF_HUB_CACHE` 是仓库缓存目录，白话讲就是“模型、数据集这些下载文件具体放在哪”。  
`HF_HUB_OFFLINE=1` 表示离线模式，白话讲就是“只准读本地缓存，不准再发网络请求”。

而 `HF_ENDPOINT` 这个变量在很多镜像实践中仍然常见，含义是“把 Hub 基地址改到一个兼容镜像或私有网关”。要注意，具体是否生效与库版本、镜像实现是否兼容有关；工程上它常被用于内部镜像站或第三方 HF 镜像服务。

真实工程例子通常是这样的：一台 8 卡训练机上跑多个容器，模型目录挂载到 `/mnt/cache/huggingface`。如果不统一 `HF_HOME` 和 `HF_HUB_CACHE`，每个容器都会重新下载一遍同样的模型。假设一个模型 20GB，4 个容器重复拉取就是 80GB 外网流量和多份磁盘占用。统一缓存后，下载次数趋近于 1，后续多数任务变成读本地磁盘。

---

## 代码实现

先给一个最小可运行的 Python 例子，用来模拟“谁覆盖谁”的规则：

```python
def resolve_index(cli=None, local=None, fallback=None):
    """
    返回最终生效的索引地址。
    优先级：CLI > local config > fallback
    """
    if cli:
        return cli
    if local:
        return local
    return fallback

nexus = "https://nexus.lan/repository/pypi-group/simple/"
pypi = "https://pypi.org/simple"

assert resolve_index(cli=nexus, local=pypi, fallback="default") == nexus
assert resolve_index(cli=None, local=nexus, fallback=pypi) == nexus
assert resolve_index(cli=None, local=None, fallback=pypi) == pypi
```

这个例子虽然简单，但它对应的是你每天都在碰到的真实行为：命令行参数优先，其次是本地配置，最后才是工具默认值。

下面是 `pip`、`conda`、`uv`、Hugging Face、Docker 的一组最小配置片段。

```bash
# 1) pip：把默认索引切到局域网 Nexus Group
pip config set global.index-url https://nexus.lan/repository/pypi-group/simple/

# 临时覆盖：本次安装用另一个索引
pip install -i https://nexus.lan/repository/pypi-group/simple/ numpy

# 有主索引也有补充索引的情况
pip install \
  --index-url https://nexus.lan/repository/pypi-group/simple/ \
  --extra-index-url https://pypi.org/simple \
  requests
```

```yaml
# 2) ~/.condarc
channel_alias: https://nexus.lan/repository/conda
default_channels:
  - https://nexus.lan/repository/conda/main
  - https://nexus.lan/repository/conda/r
custom_channels:
  conda-forge: https://nexus.lan/repository/conda
channels:
  - conda-forge
  - defaults
show_channel_urls: true
```

```toml
# 3) ~/.config/uv/uv.toml
index-url = "https://nexus.lan/repository/pypi-group/simple/"

# 项目级更推荐写在 pyproject.toml
[[tool.uv.index]]
name = "nexus"
url = "https://nexus.lan/repository/pypi-group/simple/"
default = true
```

```bash
# 4) Hugging Face 常见环境变量
export HF_ENDPOINT=https://hf-mirror.lan
export HF_HOME=/mnt/cache/huggingface
export HF_HUB_CACHE=/mnt/cache/huggingface/hub

# 完全离线时启用
export HF_HUB_OFFLINE=1
```

```python
# 5) Python 里强制只读本地缓存
from huggingface_hub import hf_hub_download

path = hf_hub_download(
    repo_id="bert-base-uncased",
    filename="config.json",
    local_files_only=True,
)
print(path)
```

```bash
# 6) Docker 容器显式传代理与缓存挂载
docker run --rm \
  -e HTTP_PROXY=http://proxy.lan:3128 \
  -e HTTPS_PROXY=http://proxy.lan:3128 \
  -e NO_PROXY=localhost,127.0.0.1,.lan \
  -e HF_ENDPOINT=https://hf-mirror.lan \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_CACHE=/cache/huggingface/hub \
  -v /mnt/cache/huggingface:/cache/huggingface \
  python:3.11-slim \
  bash -lc "pip install -i https://nexus.lan/repository/pypi-group/simple/ requests"
```

如果你希望“所有终端、所有容器、所有 CI 任务”都统一走一条入口，那么最有效的做法是局域网部署一个 Nexus PyPI Group：

- `pypi-proxy`：代理官方 PyPI
- `pypi-hosted`：放公司内部私有包
- `pypi-group`：把两者合并成一个对外地址

这样客户端只需要记住一个 URL：`https://nexus.lan/repository/pypi-group/simple/`。新终端、新服务器、新容器都复用同一配置。

---

## 工程权衡与常见坑

训练环境最怕“看起来配了，其实链路没闭合”。下面这些坑最常见：

问题 | 现象 | 规避方法
--- | --- | ---
`uv` 没显式指定索引 | `uv add` 或 `uv tool install` 仍访问公网 | 对关键命令直接带 `--index` / `--default-index` 或在项目级 `[[tool.uv.index]]` 固定
容器不继承主机代理 | 主机能装包，容器里超时 | 用 `docker run -e HTTP_PROXY=... -e HTTPS_PROXY=...`，或在 `~/.docker/config.json` 配 `proxies`
只配镜像，不配 HF 缓存 | 模型每次重下，占带宽占磁盘 | 统一 `HF_HOME`、`HF_HUB_CACHE`，并把缓存目录挂载到持久卷
把 `extra-index-url` 当安全兜底 | 私有包与公网同名时可能选错来源 | 私有包优先放内部仓库；敏感场景避免依赖公网补充索引
Nexus 用 HTTP 非 HTTPS | pip 需要额外信任配置，容易报证书错误 | 尽量启用 HTTPS；确实只能 HTTP 时再评估 `trusted-host`
只在本机写配置 | CI、容器、远程节点行为不一致 | 把索引、代理、缓存路径做成启动模板或镜像基线

其中有两个坑尤其值得单独说明。

第一，`Docker` 分“客户端代理”和“daemon 代理”。前者影响你启动容器时给容器注入什么代理环境变量，后者影响 `dockerd` 自己去拉镜像时如何联网。这两件事不是一回事。你可能已经让 `docker pull` 正常了，但容器里的 `pip install` 仍然失败，因为容器内根本没有 `HTTP_PROXY`。

第二，Hugging Face 的“离线”要做完整。只设置 `local_files_only=True` 还不够稳，因为有些代码路径仍可能尝试访问元数据；设置 `HF_HUB_OFFLINE=1` 才是更强的总开关。如果缓存中没有文件，它会直接报错，这反而是你想要的行为，因为它暴露了“离线资产没有准备完整”这个事实。

---

## 替代方案与适用边界

不是每个团队都能马上搭好 Nexus，也不是每台机器都能接入统一代理。所以要给出几种可退化方案。

方案 | 适用场景 | 限制
--- | --- | ---
命令行 `--index-url` | 临时排障、单次安装 | 不可复用，容易漏配
`--extra-index-url` 补充索引 | 主镜像不全，偶尔缺包 | 有依赖混淆风险，不适合高安全场景
项目级 `pyproject.toml` / `uv.toml` | 单个项目希望自带固定镜像 | 跨项目不统一
统一 Nexus Group | 团队、实验室、内网集群 | 需要维护仓库服务
HF 本地缓存 + 离线模式 | 完全离线训练、重复跑实验 | 要提前同步好模型文件
共享网络文件系统缓存 | 多机复用同一批模型 | 需要处理权限与并发读写

新手最容易上手的替代方案，是“主镜像 + 额外公网补充索引”：

```bash
pip install \
  --index-url https://nexus.lan/repository/pypi-group/simple/ \
  --extra-index-url https://pypi.org/simple \
  requests
```

它适合什么场景？适合你已经有内网镜像，但镜像还不完整，偶尔有个冷门包没同步。它不适合什么场景？不适合强合规、强隔离、严格禁止公网的训练环境。

如果你的目标是完全离线训练，正确做法不是“运行时碰碰运气”，而是提前同步资产：

1. 在联网机下载 wheel、conda 包、模型权重
2. 推送到 Nexus、共享盘或离线介质
3. 在线下机设置 `HF_HUB_OFFLINE=1`
4. 代码里配 `local_files_only=True`

这类方案的优点是可复现、可审计；缺点是前期资产准备更重。

---

## 参考资料

- pip `pip config` 文档：说明 `global.index-url`、用户级/站点级配置的写入方式。用途：先读它，建立 pip 配置优先级概念。https://pip.pypa.io/en/stable/cli/pip_config/
- pip `pip install` 文档：说明 `--index-url`、`--extra-index-url`、`--no-index` 的行为与风险。用途：理解临时覆盖与额外索引。https://pip.pypa.io/en/stable/cli/pip_install/
- conda `.condarc` 配置文档：说明 `channels`、`default_channels`、`channel_alias`。用途：重写 conda 默认通道。https://docs.conda.io/projects/conda/en/stable/user-guide/configuration/settings.html
- uv 配置文件文档：说明用户级、项目级 `uv.toml` / `pyproject.toml` 的读取方式。用途：明确 uv 的配置落点。https://docs.astral.sh/uv/concepts/configuration-files/
- uv 包索引文档：说明 `[[tool.uv.index]]`、`--index`、`--default-index`、兼容 `--index-url` 的行为。用途：理解 uv 的索引优先级。https://docs.astral.sh/uv/concepts/indexes/
- Hugging Face Hub 环境变量文档：说明 `HF_HOME`、`HF_HUB_CACHE`、`HF_HUB_OFFLINE`。用途：控制缓存与离线模式。https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- Docker CLI 代理文档：说明 `~/.docker/config.json` 的 `proxies` 与容器环境变量注入。用途：处理“主机能联网、容器不能联网”的问题。https://docs.docker.com/engine/cli/proxy/
- Docker daemon 代理文档：说明 `dockerd` 访问镜像仓库时的代理配置。用途：区分 daemon 和容器的两条链路。https://docs.docker.com/engine/daemon/proxy/
- Sonatype Nexus PyPI 文档：说明 PyPI 的 proxy、hosted、group 仓库模型。用途：搭建统一内网入口。https://help.sonatype.com/en/pypi-repositories.html
- Sonatype Repository Types 文档：说明 Group 仓库以及成员顺序。用途：理解为什么一个组地址能聚合多个源。https://help.sonatype.com/en/repository-types.html
