# From Limits to LLMs

<div class="epigraph">
<p>Entities should not be multiplied beyond necessity.</p>
<footer>—— William of Ockham<span class="marginnote">Occam's razor is the methodological undertone of this site: whether I'm making architecture trade-offs by day or deciding what to write by night, the standard is the same — subtract first, and don't add an entity, or a paragraph, you don't need.</span></footer>
</div>

I'm **Xu Hongduo**<span class="marginnote">Contact:<br>GitHub <a href="https://github.com/xhongduo-tech">@xhongduo-tech</a><br>Email x.hongduo@hotmail.com</span>, an LLM Infrastructure Engineer at a state-owned enterprise data center.
By day, I make large models run faster on limited hardware — inference architecture,
heterogeneous clusters, quantization and scheduling. By night, I write this site<span class="marginnote">Built with VitePress; typography in homage to <a href="https://edwardtufte.github.io/tufte-css/" target="_blank" rel="noopener noreferrer">Tufte style</a> — serif faces, paper-like background, margin notes set beside the text.<br>Source on <a href="https://github.com/xhongduo-tech/blog" target="_blank" rel="noopener noreferrer">GitHub</a>.</span>:
a complete knowledge system that starts from high-school math and physics,
passes through university mathematics and computer science,
and arrives at the frontier of AI and large language models.

## Focus Areas

**LLM Inference Architecture** — PD disaggregation<span class="marginnote">Splitting Prefill and Decode onto separate instances that scale independently — the former is compute-bound, the latter is memory- and latency-sensitive.</span>, multi-token prediction (MTP)<span class="marginnote">Predicting several future tokens per step, trading extra compute for fewer decode rounds.</span>, dynamic priority scheduling, quantized inference (AWQ / GGUF / w8a8), running vLLM, llama.cpp and TEI in production.

**Heterogeneous Compute Scheduling** — managing mixed NVIDIA A100 / V100 and Huawei Ascend 910B3<span class="marginnote">Ascend 910B3 and NVIDIA GPUs differ end-to-end — kernel libraries, compiler toolchains, everything. Scheduling across both means papering over that gap so upper layers stay agnostic.</span> clusters; a self-designed tiered deployment scheme supporting ~50 key LLM scenarios.

**LLM Platform Engineering** — independently built a full-stack open platform: 14+ models online, OpenAI / Anthropic compatible APIs, KV-Cache-aware intelligent routing<span class="marginnote">KV-cache-aware routing: forwarding requests to whichever instance already holds the matching context's KV cache, skipping redundant prefix computation on a hit — a common optimization in multi-instance serving.</span>.

**Applied AI** — prompt engineering + RAG<span class="marginnote">RAG (Retrieval-Augmented Generation): retrieve relevant passages first, then hand them to the model for generation — used to offset an LLM's stale knowledge and hallucination.</span> business systems: requirement review, resume compliance checking, retrieval & re-ranking, OCR pipelines.

## The Writing System

Content here is organized in four ascending tiers. This is not a blog index —
it is a long-term study plan: every discipline follows a classic textbook,
written chapter by chapter, section by section.<span class="marginnote">The full map of 60 disciplines and live writing progress: see [Posts](/posts/).</span>

**Tier 1 · [Foundations](/posts/foundations/math/)**<span class="marginnote">Tier 1 covers 11 disciplines, mirroring high-school to early-undergraduate textbooks; the full map lives at [Posts](/posts/).</span> — math, physics, chemistry, biology, plus astronomy, earth science, cognitive science, psychology, logic, philosophy of science and economics.

**Tier 2 · [Intermediate Mathematics](/posts/intermediate/advanced-math/)**<span class="marginnote">Tier 2 covers 17 disciplines — the backbone of undergraduate mathematics, from calculus to real analysis, functional analysis, topology and differential geometry.</span> — calculus, probability, linear algebra, discrete math, up to real analysis, functional analysis, topology and differential geometry.

**Tier 3 · [Computer Science](/posts/cs/data-structures/)**<span class="marginnote">Tier 3 covers 14 CS core courses, mirroring the classic undergraduate curriculum.</span> — data structures, computer organization, operating systems, networks, databases, compilers, distributed systems: the complete CS core.

**Tier 4 · [Advanced Topics](/posts/advanced/llm-principles/)**<span class="marginnote">Tier 4 covers 18 disciplines — from machine learning and deep learning to LLM principles, fine-tuning, deployment and AI infrastructure, right up to the frontier.</span> — machine learning, deep learning, reinforcement learning, LLM principles, fine-tuning, deployment and infrastructure, up to AI safety and quantum computing.

<HomeStats />

## Experience

**State-owned Enterprise Data Center — LLM Infrastructure Engineer**<span class="marginnote">Started as a data analyst in financial domains; transferred to the LLM track by choice.</span> (2023.09 – present)

- **Compute**: managing A100 / V100 / Ascend 910B3 clusters; tiered deployment scheme<span class="marginnote">"Tiered" means allocating resources by task priority and compute demand: high-priority jobs get dedicated A100s, long-tail jobs share V100s and Ascend.</span> supporting ~50 key scenarios
- **Inference**: containerized deployment, dynamic priority scheduling, PD disaggregation, MTP, NVIDIA MPS<span class="marginnote">NVIDIA MPS (Multi-Process Service): lets multiple processes share a single GPU's compute resources, avoiding the context-switch overhead of preemptive multitasking.</span>, TEI
- **Models**: deployed Qwen, Gemma, DeepSeek, GLM families<span class="marginnote">The Qwen, Gemma, DeepSeek and GLM families cover chat, code, documents, retrieval & re-ranking and OCR — chosen per scenario.</span> covering chat, code, docs, retrieval & OCR
- **Platforms**: unified LLM API gateway, intranet PyPI, RagFlow & Dify<span class="marginnote">RagFlow is an open-source knowledge-base engine for RAG; Dify is an LLMOps platform — together they power retrieval-augmented and agentic workflows.</span> adoption for RAG and Agents
- **Applications**: BRDM<span class="marginnote">BRDM: extracting requirement items from spec documents with Qwen2-72B and validating each one, offloading the repetitive part of manual review.</span> requirement review system (Qwen2-72B), resume compliance checker (prompt + RAG)

**Huayun Group, China Meteorological Administration — Full-stack Engineer**<span class="marginnote">My first full-stack role, taken as a campus-recruit intern, building meteorological visualization tools — the starting point on the way from student to engineer.</span> (2021.09 – 2022.08)

- Real-time 2D/3D wind-barb<span class="marginnote">Wind barbs: the graphic language of meteorology for wind direction and speed — a shaft with barbs, each long barb marking 10 knots.</span> rendering; satellite fire monitoring system; FY3E / FY4B satellite data management

## Education & Publication

- **The Hong Kong Polytechnic University**<span class="marginnote">Run by the Department of Computing; coursework spans machine learning, big-data systems and cloud computing — the pivot from a CS bachelor's to the LLM track.</span> — MSc, Artificial Intelligence & Big Data (2022 – 2023)
- **North China University of Technology**<span class="marginnote">The undergraduate years that laid down data structures, operating systems and computer networks — the same core courses behind this site's "Tier 3 · Computer Science".</span> — BEng, Computer Science (2017 – 2021)
- *Hand Acupoint Detection with an Improved HRNet*<span class="marginnote">HRNet (High-Resolution Net) keeps high-resolution features throughout and repeatedly fuses low-resolution ones — a classic backbone for pose estimation and keypoint detection.</span>, **IJCNN 2022**<span class="marginnote">IJCNN (the International Joint Conference on Neural Networks) is jointly organized by the IEEE Computational Intelligence Society and the International Neural Network Society — a major venue for machine learning and neural networks.</span>, second author (during internship at ICT, CAS)

## Honors

- Beijing Outstanding Graduate (2021)<span class="marginnote">Beijing Outstanding Graduate is a comprehensive honor granted by the municipal education commission, covering academic performance as well as research and practice.</span>; First-class Scholarship for three consecutive years (2018 – 2020)
- Lanqiao Cup<span class="marginnote">Lanqiao Cup emphasizes algorithms and programming — one of the larger software-focused academic contests in China.</span> programming contest, multiple awards; VEX<span class="marginnote">VEX robotics spans mechanical design, embedded programming and team collaboration — a complementary flavor of engineering practice to Lanqiao Cup.</span> robotics, awarded

## Tech Stack

- **LLM**: vLLM / SGLang / llama.cpp / TEI<span class="marginnote">vLLM manages KV cache with PagedAttention; SGLang targets complex sampling and structured output; llama.cpp is the go-to for CPU and edge inference.</span> · inference optimization (PD / quantization / MTP) · RAG / Agent
- **Engineering**: Python / PyTorch / Docker / Kubernetes<span class="marginnote">Kubernetes handles serving orchestration and autoscaling.</span> / Vue<span class="marginnote">Vue powers the platform frontend.</span> / FastAPI<span class="marginnote">FastAPI is the mainstream async web framework in Python.</span>
- **Compute**: NVIDIA CUDA / Huawei Ascend CANN<span class="marginnote">CANN (Ascend AI Computing Architecture) is Huawei's heterogeneous compute platform competing with CUDA — operator libraries, graph compilation and runtime.</span>

## Start Here

- [Posts](/posts/)<span class="marginnote">60 disciplines, 5832 topics, live writing progress; the source of this site is open on GitHub — stars and issues welcome.</span> — the full map of 60 disciplines with live progress
- [Projects](/projects/)<span class="marginnote">8 projects spanning 2021–2026: from a full-stack internship in satellite visualization to independent work on LLM inference architecture and open platforms, some running on intranets only.</span> — inference architecture, open platforms, business systems and publications
- [Style Demo](/posts/style-demo)<span class="marginnote">Heading levels, inline and block math, chemical equations, syntax-highlighted code, tables and margin notes — worth a look before you start writing a post.</span> — typography this site supports (math, chemistry, margin notes)
