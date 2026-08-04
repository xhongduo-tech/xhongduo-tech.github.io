# 从极限到大模型

<div class="epigraph">
<p>如无必要，勿增实体。</p>
<footer>—— 奥卡姆的威廉<span class="marginnote">奥卡姆剃刀是这个网站的方法论底色：无论是白天做架构取舍，还是夜晚写作取材，标准都是同一条——先做减法，能不加的实体、能不写的段落，就不加、不写。</span></footer>
</div>

我是**徐鸿铎**<span class="marginnote">联系方式：<br>GitHub <a href="https://github.com/xhongduo-tech">@xhongduo-tech</a><br>CSDN <a href="https://blog.csdn.net/weixin_43098506" target="_blank" rel="noopener noreferrer">@weixin_43098506</a><br>Email x.hongduo@hotmail.com</span>，大模型架构工程师，就职于国企数据中心。
白天的工作是让大模型在有限的算力上跑得更快——推理架构、异构集群、量化与调度；
夜晚与周末，我在写这个网站<span class="marginnote">本站由 VitePress 构建，排版致敬 <a href="https://edwardtufte.github.io/tufte-css/" target="_blank" rel="noopener noreferrer">Tufte 风格</a>：衬线字体、类纸张背景、边注与正文并排。<br>源码在 <a href="https://github.com/xhongduo-tech/blog" target="_blank" rel="noopener noreferrer">GitHub</a> 开源。</span>：一个从高中数理出发，经大学数学物理与计算机科学，
最终抵达 AI 与大模型前沿的完整知识体系。

## 专注领域

**大模型推理架构** —— 显存带宽与请求延迟是两个互相拉扯的约束，压榨算力就是在两者间找平衡：PD 分离<span class="marginnote">把 Prefill（预填充）与 Decode（解码）拆到不同实例上分别扩缩容——前者算力密集，后者显存与延迟敏感。</span>、MTP 多 Token 预测<span class="marginnote">一步预测多个后续 token，用多余算力换取更少的解码轮数。</span>、动态优先级调度、量化推理（AWQ / GGUF / w8a8），在生产环境落地 vLLM、llama.cpp、TEI 等推理引擎。

**异构算力调度** —— 让业务方感知不到底层差异是核心目标：统筹 NVIDIA A100 / V100 与华为昇腾 910B3<span class="marginnote">昇腾 910B3 与 NVIDIA 系显卡的算子库、编译链路完全不同，混合调度的核心是在两套生态之上抹平差异，让上层任务无感知。</span> 混合集群，自研梯度算力部署方案，支撑近 50 个大模型重点场景。

**大模型平台工程** —— 从 0 到 1 独立设计并实现大模型开放平台：14+ 模型在线管理，兼容 OpenAI / Anthropic 接口以降低业务方接入成本，KV Cache 感知的 API 智能路由<span class="marginnote">KV Cache 感知路由：把请求优先转发到已缓存对应上下文 KV 的实例，命中时可跳过重复的前缀计算，是多实例部署里常见的推理加速手段。</span>。

**AI 应用落地** —— 提示词工程 + RAG<span class="marginnote">RAG（检索增强生成）：先从知识库中检索相关片段，再交给大模型生成回答，用来缓解大模型的知识时效性与幻觉问题。</span> 驱动的业务系统，且各自有明确的验收门槛：需求项检查要求逐条可追溯，履历合规校验容错空间极小，检索精排与 OCR 全链路要接住上游数据的脏乱。

## 工程方法论

- **先测量，再决策**：架构选型从硬件约束倒推——显存带宽、请求延迟、GPU 利用率里哪个是真正瓶颈，决定了该上 PD 分离还是量化，该垂直扩容还是水平调度。
- **复用优先，自研聚焦**：vLLM、RagFlow、Dify 等开源组件承担通用能力；自研精力放在业务强相关、开源方案覆盖不到的部分——路由策略、梯度调度、合规校验规则。
- **先跑通，再规模化**：新系统先以最小可用方案验证业务价值，跑通后再补齐监控、灰度<span class="marginnote">灰度发布：先在小比例流量上验证新版本，观察指标正常后再逐步放量，是控制上线风险的常见手段。</span>、批量处理等工程化能力，不为还不存在的规模提前投入。

## 写作体系

本站的内容按四级递进组织。这不是博客的目录，是一份长期学习计划——
每个学科对标经典教材，逐章逐节写作。<span class="marginnote">全部 60 个学科的知识地图与实时写作进度见 [博文总览](/posts/)。</span>

**第一级 · [基础科学](/posts/foundations/math/)**<span class="marginnote">第一级共 11 个学科，对标高中到大学低年级的教材体系，全部清单见 [知识地图](/posts/)。</span> —— 数学、物理、化学、生物，及天文、地学、认知、心理、逻辑、科哲、经济学：一切的地基。

**第二级 · [进阶数理](/posts/intermediate/advanced-math/)**<span class="marginnote">第二级共 17 个学科：从微积分、概率、线代一路到实变、泛函、拓扑与微分几何。</span> —— 高等数学、概率统计、线性代数、离散数学，直到实变、泛函、拓扑与微分几何。

**第三级 · [计算机基础](/posts/cs/data-structures/)**<span class="marginnote">第三级共 14 门 CS 核心课，对标考研 408 科目与 CMU 经典课程体系。</span> —— 数据结构、组成原理、操作系统、网络、数据库、编译原理、分布式系统：CS 核心课全集。

**第四级 · [高阶专题](/posts/advanced/llm-principles/)**<span class="marginnote">第四级共 18 个学科：从机器学习、深度学习到 LLM 原理、微调、部署与 AI 基础设施，直抵前沿。</span> —— 机器学习、深度学习、强化学习，大模型原理、微调、部署与基础设施，直至 AI 安全与量子计算。

<HomeStats />

## 工作经历

**国企数据中心 — 大模型架构工程师**<span class="marginnote">入职初期从事金融领域大数据分析，后主动转岗大模型方向。</span>（2023.09 至今）

- **算力统筹**：管理 A100、V100、昇腾 910B3 混合集群，自研梯度算力部署方案<span class="marginnote">"梯度"指按任务优先级与算力需求分级投放资源：高优任务独占 A100，长尾任务共享 V100 与昇腾。</span>，支撑全中心近 50 个大模型重点场景
- **推理架构**：为提升 GPU 利用率，落地容器化部署、动态优先级调度、PD 分离、MTP、NVIDIA MPS<span class="marginnote">NVIDIA MPS（Multi-Process Service）：让多个进程共享同一张 GPU 的计算资源，避免多任务抢占带来的上下文切换开销。</span>、TEI 向量推理引擎
- **模型部署**：牵头技术选型与部署 Qwen、Gemma、DeepSeek、GLM 等系列开源大模型<span class="marginnote">Qwen、Gemma、DeepSeek、GLM 等系列覆盖对话、代码、文档、检索精排与 OCR 全链路，按场景选型部署。</span>，覆盖对话、代码、文档、检索精排、OCR 全链路
- **平台建设**：牵头大模型 API 统一接入平台，搭建内网 PyPI 平台，引入 RagFlow、Dify<span class="marginnote">RagFlow 是面向 RAG 的开源知识库引擎，Dify 是 LLMOps 平台，两者配合支撑检索增强与 Agent 业务。</span> 支撑 RAG 与 Agent 需求
- **业务落地**：BRDM<span class="marginnote">BRDM：用 Qwen2-72B 对需求文档做条目抽取与逐条校验，替代人工评审中的重复劳动。</span> 需求项智能检查系统（Qwen2-72B）、员工履历检查系统（提示词 + RAG）

**中国气象局华云集团 — 全栈工程师**<span class="marginnote">校招实习期间的第一段全栈开发经历，服务于气象数据可视化场景，是从学生阶段走向工程实践的起点。</span>（2021.09 – 2022.08）

- 基于风速风向数据实现风羽图<span class="marginnote">风羽图（wind barb）：气象学中表示风向与风速的图形语言，由指向杆与羽片组成，每根长羽代表 10 节。</span>二维 / 三维实时绘制；参与卫星火情监测预测系统与 FY3E / FY4B 卫星数据库管理

## 教育与论文

- **香港理工大学**<span class="marginnote">香港理工大学电子计算学系开设，课程覆盖机器学习、大数据系统与云计算，是从计算机本科走向大模型方向的关键过渡。</span> · 人工智能与大数据，硕士（2022 – 2023）
- **北方工业大学**<span class="marginnote">本科阶段打下数据结构、操作系统、计算机网络等 CS 核心课基础——这也是本站「第三级 · 计算机基础」模块的选题依据。</span> · 计算机科学与技术，学士（2017 – 2021）
- 《基于改进的 HRNet<span class="marginnote">HRNet（高分辨率网络）：全程保持高分辨率特征、反复与低分辨率特征融合，是姿态估计与关键点检测的经典主干。</span> 的手部穴位检测》，**IJCNN 2022**<span class="marginnote">IJCNN（国际神经网络联合会议）由 IEEE 计算智能学会与国际神经网络学会联合主办，是机器学习与神经网络方向的重要国际会议之一。</span>，第二作者（中科院计算所实习期间）

## 荣誉

- 北京市优秀毕业生（2021）<span class="marginnote">北京市优秀毕业生是市教委面向应届生的综合荣誉，覆盖学业成绩与科研、实践表现。</span>；连续三年一等奖学金（2018 – 2020）
- 蓝桥杯<span class="marginnote">蓝桥杯侧重算法与程序设计能力，是国内规模较大的软件类学科竞赛之一。</span>编程竞赛多次获奖，VEX<span class="marginnote">VEX 是涵盖机械设计、编程与团队协作的机器人竞赛体系，训练的是与蓝桥杯互补的工程能力。</span> 机器人竞赛获奖

## 技术栈

- **大模型**：vLLM / SGLang / llama.cpp / TEI<span class="marginnote">vLLM 以 PagedAttention 管理 KV Cache 见长，SGLang 面向复杂采样与结构化输出，llama.cpp 是 CPU 与边缘端轻量推理的首选。</span> · 推理优化（PD 分离 / 量化 / MTP）· RAG / Agent
- **AI 协同开发**：Claude Code / Codex<span class="marginnote">Claude Code 是 Anthropic 出品的终端原生智能体编程工具，Codex 是 OpenAI 的同类产品——两者都能自主读写代码库、执行多步骤任务，而不只是行内补全。</span> 深度使用，日常承担编码与架构决策；GLM-5.2、Kimi K3<span class="marginnote">GLM 是智谱 AI 的旗舰模型系列，Kimi 是月之暗面的旗舰模型系列——日常用于与 OpenAI / Anthropic 模型对比评测与长上下文场景验证。</span> 等前沿模型高频调用，token 消耗积累超千亿<span class="marginnote">统计口径为云端 API 调用的输入与输出 token 总和，主要来自日常编码、代码审查与技术写作。</span>
- **工程**：Python / PyTorch / Docker / Kubernetes<span class="marginnote">Kubernetes 承载推理服务的编排与扩缩容。</span> / Vue<span class="marginnote">Vue 负责平台前端。</span> / FastAPI<span class="marginnote">FastAPI 是 Python 生态的主流异步 Web 框架。</span>
- **算力**：NVIDIA CUDA / 华为昇腾 CANN<span class="marginnote">CANN（昇腾 AI 计算架构）是华为对标 CUDA 的异构计算平台，提供算子库、图编译与运行时。</span>

## 从这里开始

- [博文总览](/posts/)<span class="marginnote">60 个学科、5832 个选题的完整知识地图，写作进度实时更新；本站源码在 GitHub 开源，欢迎 star 与 issue。</span> —— 60 个学科的完整知识地图与写作进度
- [项目](/projects/)<span class="marginnote">8 个项目，横跨 2021–2026：从卫星可视化的全栈实习，到大模型推理架构与开放平台的独立开发，部分运行于内网环境。</span> —— 我做过的东西：推理架构、开放平台、业务系统与论文
- [样式演示](/posts/style-demo)<span class="marginnote">多级标题、行内与块级公式、化学方程式、代码高亮、表格与边注——写博文前不妨先看一眼这份排版参考。</span> —— 本站支持的排版能力（公式、化学方程式、边注）
