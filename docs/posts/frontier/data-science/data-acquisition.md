---
title: 数据获取（爬虫 / API）
date: 2026-08-07
---

# 数据获取（爬虫 / API）

<div class="epigraph">
<p>垃圾进，垃圾出。</p>
<footer>—— 计算机界俗语（Garbage in, garbage out）</footer>
</div>

<div class="article-byline">
<p>第九级 · 数据科学 ｜ 《数据科学导论》 第3章 ｜ 2026-08-07</p>
</div>

## 为什么从数据获取开始

流程的第二步是数据获取。你可能有再漂亮的统计方法、再强大的模型，**只要数据源不靠谱，整条流水线就建立在流沙上**。数据获取要解决两个问题：到哪里拿数据？怎么把数据「干净地」搬进来？<span class="marginnote">数据获取与第三级《数据库》里的数据导入、第五级《信息检索》里的抓取，在技术上同源；数据科学这里更强调「为分析服务的获取」，即每拿一批数据都要带着它的元数据与质量判断。</span>

## 1 数据来源的三条主要通路

按获取方式，数据来源大致分三类：

- **结构化数据接口（API）**：平台主动提供的数据出口。微博、GitHub、天气、金融行情都提供 RESTful API，返回 JSON 或 XML。<span class="marginnote">API 的全称是 Application Programming Interface。对数据科学家来说，它的意义是「一份数据可以按约定被程序化、可重复地取到」，且通常带鉴权、限流与文档——比爬虫干净得多。调用前先读文档，确认 `rate limit`（速率限制）与字段语义。</span>
- **网页抓取（爬虫，Web Scraping）**：平台没有 API，或 API 数据不全时，直接从 HTML 页面抽取信息。核心工具链是「请求 → 解析 → 抽取」。
- **文件与数据库导出**：CSV、Excel、Parquet 等静态文件，或从数据库直接 `SELECT`。这是企业内部最常见的来源，往往要配合数据仓库访问权限。

**重点：获取方式的选择顺序应该是「API 优先，爬虫兜底」。** API 有稳定的 schema、明确的使用条款与限流策略；爬虫则要面对页面结构变化、反爬机制与合规风险，维护成本高出一个数量级。能用 API 就不写爬虫，是数据工程界的默认纪律。

## 2 爬虫的核心链路

一个最小可用的爬虫由四步构成：

1. **请求（Request）**：用 HTTP 客户端（Python 的 `requests`）向目标 URL 发起请求，拿到 HTML 文本。
2. **解析（Parse）**：用解析库（`BeautifulSoup`、`lxml`）把 HTML 字符串变成可查询的树结构。
3. **抽取（Extract）**：用 CSS 选择器或 XPath 定位目标元素，取出文本与属性值。
4. **存储（Store）**：把抽取结果规整成结构化的表格（DataFrame、CSV、数据库表）。

下面是一个「抽取标题列表」的骨架示意：

```python
import requests
from bs4 import BeautifulSoup

resp = requests.get("https://example.com/news", timeout=10)
soup = BeautifulSoup(resp.text, "html.parser")

titles = [h.get_text(strip=True) for h in soup.select("h2.title")]
print(len(titles), titles[:3])
```

**辨析｜易错点：** 爬虫的「解析」环节最容易被新手搞错：**页面里的数据不一定都在 HTML 里**。现代网站大量使用 JavaScript 动态渲染，`requests` 拿到的 HTML 可能只是个空壳，真实数据藏在额外的 XHR 请求（API 调用）返回的 JSON 里。这种情况下要么改用无头浏览器（`Playwright`、`Selenium`），要么直接找那个 XHR 的 URL——后者往往更高效。

## 3 API 调用的通用形态

大多数数据 API 遵循 REST 风格：用 HTTP 动词表达操作，用 URL 表达资源，用 JSON 表达数据。一次典型的调用长这样：

```python
import requests

params = {
    "query": "人工智能",
    "page": 1,
    "size": 20,
}
headers = {"Authorization": "Bearer YOUR_TOKEN"}
resp = requests.get(
    "https://api.example.com/v1/search",
    params=params,
    headers=headers,
    timeout=15,
)
data = resp.json()          # 解析响应
records = data["items"]     # 抽取记录列表
```

四个关键点要牢记：

- **鉴权（Authentication）**：通常用 API Key 或 OAuth Token，放在请求头里。密钥必须放在环境变量或密钥管理服务里，绝不硬编码进代码仓库。<span class="marginnote">把 API Key 提交进 git 是数据科学项目里最高频的安全事故之一。即使仓库是私有的，也强烈建议用环境变量 + `.gitignore` 规避。数据隐私与安全的细节见第24篇《数据隐私与安全》。</span>
- **分页（Pagination）**：一次请求通常只返回一页，需要按文档翻页累积。
- **限流（Rate Limiting）**：服务方限制单位时间请求数，超限会被返回 `429 Too Many Requests`。应对策略是控制频率、加退避重试。
- **错误处理**：网络抖动、参数非法、配额用尽都会返回非 200 状态码，代码里要按状态码分类处理。

## 4 数据获取的合规与伦理底线

数据获取不是纯技术问题。**「能不能拿到」和「该不该拿」是两件事。** 三条底线必须守住：

1. **遵守条款与法律**：遵守目标网站的服务条款、robots.txt 协议，以及所在司法辖区的数据保护法律（如欧盟 GDPR、中国《个人信息保护法》）。
2. **尊重隐私与授权**：不采集可识别个人的敏感信息，除非有明确法律依据与授权。爬取公开网页 ≠ 可以自由使用这些数据。
3. **控制抓取负载**：控制请求频率，避免对目标服务器造成过载——这也是技术上的自我保护，过快的爬虫会被封禁。

**辨析｜易错点：** 很多人以为「公开网页上的数据就可以随便抓」。公开可访问（publicly accessible）与授权使用（authorized use）是两回事。**技术可行性与法律、伦理合法性必须分开判断**，数据获取环节的合规判断，将直接影响后面第30篇《数据伦理》所讨论的整条价值链。

## 5 获取之后：立即记录元数据

拿到数据不等于拿到可用的数据。专业的做法是**在获取环节就记录元数据（metadata）**：数据的来源 URL、获取时间、字段定义、更新频率、已知限制。<span class="marginnote">元数据是「关于数据的数据」。它让后来者（包括六个月后的你自己）能回答：这份数据从哪来、可信吗、能复现吗？没有元数据的数据科学，等于没有工程图纸的建筑。</span>这份元数据清单会在下一节《数据清洗》直接派上用场——因为你必须先知道字段「应该是什么」，才能判断哪些值「是错的」。

## 6 一个完整的数据获取小案例

把前面三条通路组装成一个 10 分钟能跑通的小案例：**用公开 API 构建一个「每日天气 + 空气质量」的数据集**。

**第一步，找 API 并读文档**。免费天气 API（如 Open-Meteo、OpenWeatherMap）都提供 REST 接口，返回 JSON。先确认鉴权方式（多数免费档需要 API Key）、请求格式与字段语义。

**第二步，写一个稳健的拉取函数**。要点是：参数化城市列表、限流（每次请求间隔至少 1 秒）、错误处理（网络失败重试 3 次，超时设置 15 秒）：

```python
import requests, time

def fetch_weather(city, lat, lon, api_key):
    url = "https://api.example.com/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    for attempt in range(3):
        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException:
            time.sleep(2 ** attempt)      # 指数退避
    return None
```

**第三步，规整并记录元数据**。把 JSON 摊平成一张表（城市、日期、最高温、最低温、降水量、空气质量），并记录「来源 URL、拉取时间、字段单位」——这就是第3篇的元数据纪律在起作用。

**第四步，增量更新**。每天定时拉当天数据追加进表，用日期做主键去重（第20篇《数据流水线》的增量思想在这里的雏形）。

**辨析｜易错点：** 这个案例最常踩的坑是「把 `raise_for_status()` 省略」——API 返回 429（限流）或 500（服务端错误）时，`resp.json()` 会解出一个错误对象而不是数据，你的表会悄悄混进垃圾行。**任何拉取函数都必须检查 HTTP 状态码**，这是数据获取的底线。

## 7 小结

- 数据来源三条通路：**API、爬虫、文件/数据库导出**；选用顺序是 **API 优先，爬虫兜底**。
- 爬虫核心链路：**请求 → 解析 → 抽取 → 存储**；警惕 JS 动态渲染导致「HTML 空壳」。
- API 调用四要素：**鉴权、分页、限流、错误处理**；密钥永不入库。
- 数据获取有**合规与伦理底线**：条款、法律、隐私、负载控制。
- 获取后立即记录**元数据**，为清洗与后续分析提供依据。

在下一节，我们处理最脏也最耗时的环节：拿到手的数据往往缺值、重复、格式混乱——怎么把它们整理成可分析的形态，这就是**数据清洗**。
