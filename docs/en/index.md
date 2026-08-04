# From Limits to LLMs

<div class="epigraph">
<p>Entities should not be multiplied beyond necessity.</p>
<footer>—— William of Ockham</footer>
</div>

I'm **Xu Hongduo**, an LLM Infrastructure Engineer at the R&D Center of ICBC Head Office.
By day, I make large models run faster on limited hardware — inference architecture,
heterogeneous clusters, quantization and scheduling. By night, I write this site:
a complete knowledge system that starts from high-school math and physics,
passes through university mathematics and computer science,
and arrives at the frontier of AI and large language models.<span class="marginnote">Contact:<br>GitHub <a href="https://github.com/xhongduo-tech">@xhongduo-tech</a><br>Email x.hongduo@hotmail.com</span>

## Focus Areas

**LLM Inference Architecture** — PD disaggregation, multi-token prediction (MTP), dynamic priority scheduling, quantized inference (AWQ / GGUF / w8a8), running vLLM, llama.cpp and TEI in production.

**Heterogeneous Compute Scheduling** — managing mixed NVIDIA A100 / V100 and Huawei Ascend 910B3 clusters; a self-designed tiered deployment scheme supporting ~50 key LLM scenarios.

**LLM Platform Engineering** — independently built a full-stack open platform: 14+ models online, OpenAI / Anthropic compatible APIs, KV-Cache-aware intelligent routing.

**Applied AI** — prompt engineering + RAG business systems: requirement review, resume compliance checking, retrieval & re-ranking, OCR pipelines.

## The Writing System

Content here is organized in four ascending tiers. This is not a blog index —
it is a long-term study plan: every discipline follows a classic textbook,
written chapter by chapter, section by section.<span class="marginnote">The full map of 60 disciplines and live writing progress: see [Posts](/posts/).</span>

**Tier 1 · [Foundations](/posts/foundations/math/)** — math, physics, chemistry, biology, plus astronomy, earth science, cognitive science, psychology, logic, philosophy of science and economics.

**Tier 2 · [Intermediate Mathematics](/posts/intermediate/advanced-math/)** — calculus, probability, linear algebra, discrete math, up to real analysis, functional analysis, topology and differential geometry.

**Tier 3 · [Computer Science](/posts/cs/data-structures/)** — data structures, computer organization, operating systems, networks, databases, compilers, distributed systems: the complete CS core.

**Tier 4 · [Advanced Topics](/posts/advanced/llm-principles/)** — machine learning, deep learning, reinforcement learning, LLM principles, fine-tuning, deployment and infrastructure, up to AI safety and quantum computing.

<HomeStats />

## Experience

**ICBC Head Office, R&D Center — LLM Infrastructure Engineer** (2023.09 – present)<span class="marginnote">Started as a data analyst in mortgage and risk domains; transferred to the LLM track by choice.</span>

- **Compute**: managing A100 / V100 / Ascend 910B3 clusters; tiered deployment scheme supporting ~50 key scenarios
- **Inference**: containerized deployment, dynamic priority scheduling, PD disaggregation, MTP, NVIDIA MPS, TEI
- **Models**: deployed Qwen, Gemma, DeepSeek, GLM families covering chat, code, docs, retrieval & OCR
- **Platforms**: unified LLM API gateway, intranet PyPI, RagFlow & Dify adoption for RAG and Agents
- **Applications**: BRDM requirement review system (Qwen2-72B), resume compliance checker (prompt + RAG)

**Huayun Group, China Meteorological Administration — Full-stack Engineer** (2021.09 – 2022.08)

- Real-time 2D/3D wind-barb rendering; satellite fire monitoring system; FY3E / FY4B satellite data management

## Education & Publication

- **The Hong Kong Polytechnic University** — MSc, Artificial Intelligence & Big Data (2022 – 2023)
- **North China University of Technology** — BEng, Computer Science (2017 – 2021)
- *Hand Acupoint Detection with an Improved HRNet*, **IJCNN 2022**, second author (during internship at ICT, CAS)

## Honors

- Beijing Outstanding Graduate (2021); First-class Scholarship for three consecutive years (2018 – 2020)
- Multiple awards in Lanqiao Cup programming contest and VEX robotics

## Tech Stack

- **LLM**: vLLM / SGLang / llama.cpp / TEI · inference optimization (PD / quantization / MTP) · RAG / Agent
- **Engineering**: Python / PyTorch / Docker / Kubernetes / Vue / FastAPI
- **Compute**: NVIDIA CUDA / Huawei Ascend CANN

## Start Here

- [Posts](/posts/) — the full map of 60 disciplines with live progress
- [Projects](/projects/) — inference architecture, open platforms, business systems and publications
- [Style Demo](/posts/style-demo) — typography this site supports (math, chemistry, margin notes)
