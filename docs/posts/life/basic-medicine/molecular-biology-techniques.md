---
title: 分子生物学技术
date: 2026-08-07
---

# 分子生物学技术

<div class="epigraph">
<p>分子生物学的工具箱里，PCR 是复印机、测序是字典、基因编辑是改写笔。</p>
<footer>—— 生化格言化改写（分子生物学核心技术）</footer>
</div>

<div class="article-byline">
<p>第五级 · 基础医学 ｜ 生物化学与分子生物学 分子生物学技术 ｜ 2026-08-07</p>
</div>

## 为什么从分子生物学技术开始

理解基因与蛋白质，离不开研究它们的**分子生物学技术**——PCR 扩增 DNA、测序读基因、印迹杂交测 RNA/蛋白、克隆表达蛋白、基因编辑改基因。这些技术是**分子诊断（病原检测、基因突变）、法医鉴定、基因治疗**的基础。<span class="marginnote">核心技术清单：<strong>PCR</strong>`（扩增 DNA）、<strong>反转录 PCR（RT-PCR）/qPCR</strong>`（测 RNA/定量）、<strong>DNA 测序</strong>（Sanger/二代 NGS）、<strong>Southern/Northern/Western 印迹</strong>（DNA/RNA/蛋白）、<strong>分子克隆</strong>（重组 DNA）、<strong>CRISPR 基因编辑</strong>、<strong>核酸杂交/基因芯片</strong>。理解原理（碱基互补、酶学）即可推断应用。</span>

"PCR 为什么能'复印'一段 DNA""新冠核酸怎么检测（RT-qPCR）""基因编辑（CRISPR）怎么剪 DNA"——分子生物学技术回答这些问题。

## 1 PCR：DNA 的体外扩增

**PCR（聚合酶链式反应，polymerase chain reaction）**在体外指数扩增特定 DNA 片段，需要：

**模板 DNA**、**一对引物**（侧翼）、**dNTP**、**耐热 DNA 聚合酶（Taq）**、Mg²⁺ 与缓冲液。
**三步循环**（每轮 DNA 量翻倍）：
  1. **变性**（95℃）：双链解开。
  2. **退火**（55~60℃）：引物与模板结合。
  3. **延伸**（72℃）：聚合酶从引物 3' 端合成新链。

**PCR 的关键**：引物**特异性**决定扩增片段；Taq 酶**耐热**（不需每轮加酶）；循环 30~40 轮→DNA 扩增约百万倍。

**PCR 的应用**：基因克隆、**病原检测（乙肝、结核、新冠）**、法医鉴定（STR）、突变检测。<span class="marginnote"><strong>PCR 的"复印原理"</strong>：引物决定"复印哪一段"（特异性）、Taq 酶耐热（高温解链不灭活）→自动循环放大——"PCR=DNA 的复印机"。<strong>反转录 PCR（RT-PCR）</strong>：先反转录（RNA→cDNA）再 PCR——<strong>检测 RNA 病毒（新冠）与基因表达</strong>。<strong>实时定量 PCR（qPCR）</strong>：荧光信号随扩增增加→定量起始模板——"测病毒载量/基因表达量"。<strong>新冠核酸检测 = RT-qPCR</strong>（RNA→cDNA→定量扩增）——"分子诊断的全民实践"。<strong>PCR 的临床</strong>：病原核酸（灵敏度高）、肿瘤突变（EGFR 检测指导靶向药）、遗传病（基因诊断）。</span>

## 2 DNA 测序与印迹杂交

**DNA 测序（DNA sequencing）**：

**Sanger 测序（第一代）**：双脱氧核苷酸（ddNTP）终止法——读单个基因片段（金标准，验证用）。
**二代测序（NGS，高通量）**：大规模并行测序——**全基因组/外显子组测序**、肿瘤突变图谱、产前筛查（NIPT）。应用：罕见病诊断、肿瘤精准治疗、病原鉴定。

**印迹杂交（blotting）**：

**Southern 印迹**：检测 DNA（限制酶切→电泳→转膜→探针杂交）——基因缺失/重排。
**Northern 印迹**：检测 RNA（基因表达水平）。
**Western 印迹（蛋白印迹）**：抗体检测蛋白——"蛋白表达与修饰（磷酸化）检测的常用方法"（HIV 确诊、阿尔茨海默 tau 检测等）。

**核酸杂交**：探针（标记的互补核酸）与靶序列配对——**FISH（荧光原位杂交）**定位染色体基因、基因芯片。<span class="marginnote"><strong>测序的"代际"</strong>：Sanger（ddNTP 终止，读单片段，用于验证）→NGS（并行测序，全基因组/外显子，用于发现）→单分子测序（长读长）。<strong>NGS 的临床</strong>：肿瘤<strong>驱动基因检测</strong>（EGFR、ALK、KRAS）指导靶向治疗、遗传病全外显子测序、<strong>NIPT（无创产前筛查）</strong>测胎儿游离 DNA。<strong>Western 印迹</strong>是"蛋白检测金标准"：SDS-PAGE 按大小分离蛋白→转膜→一抗结合→二抗显色——"测蛋白表达（或磷酸化）"。<strong>FISH</strong>用荧光探针定位染色体片段（白血病融合基因、HER2 扩增）。</span>

## 3 基因克隆与重组 DNA 技术

**基因克隆（gene cloning）**：把目的基因插入**载体（质粒）**→转入宿主（大肠杆菌）→扩增/表达。

**重组 DNA 的"工具酶"**：

**限制性内切酶**：识别并切割特定 DNA 序列（"分子剪刀"，产生黏性末端）。
**DNA 连接酶**：连接 DNA 片段（"分子胶水"）。
**载体**：质粒（环形 DNA，含复制起点、多克隆位点、筛选标记如抗生素抗性）。

**应用**：**生产重组蛋白**（胰岛素、生长激素、疫苗抗原、单克隆抗体——"生物制药"）、基因治疗载体、转基因。

**基因表达分析**：RT-qPCR、报告基因（GFP）、原位杂交（RNAscope）、单细胞 RNA 测序（scRNA-seq）。<span class="marginnote"><strong>重组 DNA 的"分子手术"</strong>：限制酶（切）→连接酶（缝）→质粒载体（运）→宿主表达——"基因工程的基础"。<strong>重组胰岛素</strong>（第一个基因工程药物）：人胰岛素基因插入质粒→大肠杆菌/酵母表达→纯化——"糖尿病的生物制药革命"（替代猪胰岛素）。<strong>生物制药</strong>：重组蛋白药（胰岛素、EPO、单抗、疫苗）都是基因工程的产物。<strong>表达载体</strong>可加标签（His-tag）便于纯化、加信号肽便于分泌。<strong>scRNA-seq</strong>（单细胞测序）揭示细胞异质性（肿瘤微环境、发育图谱）。</span>

## 4 基因编辑与基因治疗

**CRISPR-Cas9 基因编辑**：

**Cas9 核酸酶** + **向导 RNA（sgRNA）**（与靶序列互补）→Cas9 在靶点切割 DNA。
修复：**非同源末端连接（NHEJ）**→基因敲除（indel）；**同源重组（HDR）**→基因敲入/修复。

**应用**：**基因敲除小鼠**（研究基因功能）、**疾病模型**、**基因治疗**（镰状细胞贫血、β-地中海贫血的 CRISPR 治疗已获批）、**CAR-T 细胞改造**（基因编辑免疫细胞抗癌）。

**基因治疗（gene therapy）**：把正常基因/编辑工具送入患者细胞——病毒载体（AAV、慢病毒）或脂质纳米颗粒（LNP，mRNA 疫苗技术）。

**mRNA 疫苗（新冠）**：mRNA 包在 LNP 中→进入细胞翻译出抗原→诱导免疫——"分子生物学的公共卫生应用"。<span class="marginnote"><strong>CRISPR 是"基因的改写笔"</strong>：sgRNA 把 Cas9"导航"到靶序列→切割→NHEJ（敲除）/HDR（敲入）——"靶向基因编辑"（2020 诺奖）。<strong>CRISPR 的临床突破</strong>：<strong>β-地中海贫血/镰状细胞贫血</strong>的 CRISPR 治疗（编辑 BCL11A 增强子重新激活胎儿血红蛋白，已获批）；<strong>CAR-T</strong>（嵌合抗原受体 T 细胞）改造患者 T 细胞靶向肿瘤——"个体化细胞基因治疗"。<strong>基因治疗的载体</strong>：AAV（小基因、体内）、慢病毒（体外、整合）、LNP（mRNA，非病毒）。<strong>mRNA 疫苗</strong>（新冠）证明"mRNA 技术"的潜力（翻译抗原→免疫）——"分子生物学从实验室走进疫苗接种"。<strong>基因编辑的伦理与安全</strong>（脱靶、生殖系编辑）是持续议题。</span>

## 5 核心对比表：三大印迹技术

| 技术 | 检测分子 | 探针/抗体 | 应用 |
| --- | --- | --- | --- |
| Southern 印迹 | DNA | DNA 探针 | 基因缺失、重排 |
| Northern 印迹 | RNA | DNA/RNA 探针 | 基因表达（mRNA 水平） |
| Western 印迹 | 蛋白 | 抗体 | 蛋白表达、修饰 |

## 6 小结

- **PCR**（变性-退火-延伸，Taq 酶）体外扩增 DNA；**RT-qPCR** 检测 RNA（新冠核酸）与定量基因表达。
- **测序**：Sanger（验证）→NGS（全基因组/外显子，肿瘤与遗传病诊断）。
- **印迹**：Southern（DNA）、Northern（RNA）、Western（蛋白，抗体检测）。
- **基因克隆**（限制酶+连接酶+质粒）生产重组蛋白（胰岛素）；**CRISPR 基因编辑**（Cas9+sgRNA）用于基因敲除/修复、基因治疗（地中海贫血、CAR-T）；**mRNA 疫苗**是分子生物学的公共卫生应用。

在下一节，我们连接细胞信号与生化通路——**细胞信号转导**：受体与第二信使的生化机制、MAPK 通路、以及"信号转导异常与疾病（肿瘤、糖尿病）"。
