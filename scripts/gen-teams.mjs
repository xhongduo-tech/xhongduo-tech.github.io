// 生成 60 个专题专家小组的定义文件（.claude/agents/<tier>-<topic>.md）。
// 每个文件是某专题的专家域简报，可被 Agent 工具按 subagent_type 复用。
import { writeFileSync, mkdirSync } from 'node:fs'

const TIER = {
  foundations: '第一级 · 基础科学',
  intermediate: '第二级 · 进阶数理',
  cs: '第三级 · 计算机基础',
  advanced: '第四级 · 高阶专题',
}

// [tier, key, 专题名, 对标教材/体系]
const TEAMS = [
  // ---- 第一级 基础科学 ----
  ['foundations', 'astronomy', '天文学', '刘学富《基础天文学》'],
  ['foundations', 'biology', '生物学', '人教版高中生物（五册）'],
  ['foundations', 'chemistry', '化学', '人教版高中化学（必修两册 + 选择性必修三册）'],
  ['foundations', 'cognitive-science', '认知科学', '《认知心理学》经典教材与认知科学导论课程'],
  ['foundations', 'earth-science', '地球科学', '《普通地质学》与地球科学概论课程体系'],
  ['foundations', 'economics', '经济学基础', '曼昆《经济学原理》'],
  ['foundations', 'logic', '逻辑学', '大学逻辑学通识经典教材'],
  ['foundations', 'math', '基础数学', '人教A版高中数学（必修第一/二册 + 选择性必修三册）'],
  ['foundations', 'philosophy-of-science', '科学史与科学哲学', '丹皮尔《科学史》、库恩《科学革命的结构》、罗森堡《科学哲学》'],
  ['foundations', 'physics', '基础物理', '人教版高中物理（必修三册 + 选择性必修三册）'],
  ['foundations', 'psychology', '心理学基础', '彭聃龄《普通心理学》'],

  // ---- 第二级 进阶数理 ----
  ['intermediate', 'abstract-algebra', '抽象代数', '杨子胥《近世代数》、丘维声《抽象代数基础》'],
  ['intermediate', 'advanced-math', '高等数学', '同济《高等数学》（第七/八版，上下册）'],
  ['intermediate', 'advanced-physics', '高等物理', '程守洙《普通物理学》'],
  ['intermediate', 'complex-analysis', '复变函数与积分变换', '西安交通大学版《复变函数与积分变换》'],
  ['intermediate', 'differential-geometry', '微分几何', '陈维桓《微分几何》、Do Carmo《曲线和曲面的微分几何》'],
  ['intermediate', 'discrete-math', '离散数学', 'Kenneth H. Rosen《离散数学及其应用》'],
  ['intermediate', 'functional-analysis', '泛函分析', '程其襄、张恭庆《泛函分析》'],
  ['intermediate', 'information-theory', '信息论', 'Cover & Thomas《Elements of Information Theory》'],
  ['intermediate', 'linear-algebra', '线性代数', '同济《线性代数》、Gilbert Strang《Introduction to Linear Algebra》'],
  ['intermediate', 'mathematical-analysis', '数学分析', '华东师范大学《数学分析》'],
  ['intermediate', 'numerical-analysis', '数值分析', '李庆扬《数值分析》'],
  ['intermediate', 'optimization', '最优化理论', 'Boyd《Convex Optimization》、《最优化方法》'],
  ['intermediate', 'pde', '偏微分方程', '谷超豪、姜礼尚《数学物理方程》'],
  ['intermediate', 'probability', '概率论与数理统计', '盛骤《概率论与数理统计》'],
  ['intermediate', 'real-analysis', '实变函数与测度论', '周民强《实变函数论》、那汤松《实变函数论》'],
  ['intermediate', 'stochastic-processes', '随机过程', '张波《应用随机过程》、Ross《Stochastic Processes》'],
  ['intermediate', 'topology', '拓扑学', '尤承业《基础拓扑学讲义》、Munkres《Topology》'],

  // ---- 第三级 计算机基础 ----
  ['cs', 'algorithms', '算法设计与分析', 'CLRS《算法导论》'],
  ['cs', 'compilers', '编译原理', '龙书《编译原理》'],
  ['cs', 'computer-architecture', '计算机体系结构', 'Hennessy & Patterson《Computer Architecture: A Quantitative Approach》'],
  ['cs', 'computer-graphics', '计算机图形学', 'GAMES101（闫令琪）、虎书《Fundamentals of Computer Graphics》'],
  ['cs', 'computer-networks', '计算机网络', '谢希仁《计算机网络》、Tanenbaum《Computer Networks》'],
  ['cs', 'computer-organization', '计算机组成原理', '唐朔飞《计算机组成原理》、CS:APP《深入理解计算机系统》'],
  ['cs', 'cryptography-security', '密码学与信息安全', 'William Stallings《密码编码学与网络安全》'],
  ['cs', 'data-structures', '数据结构', '严蔚敏《数据结构（C语言版）》'],
  ['cs', 'database', '数据库', '《数据库系统概念》（Silberschatz）、DDIA'],
  ['cs', 'digital-logic', '数字逻辑', '阎石《数字电子技术基础》'],
  ['cs', 'distributed-systems', '分布式系统', 'MIT 6.824、《数据密集型应用系统设计》（DDIA）'],
  ['cs', 'os', '操作系统', '《操作系统概念》（恐龙书）、OSTEP'],
  ['cs', 'programming-languages', '程序设计语言', 'Sebesta《程序设计语言原理》、PLT 课程体系'],
  ['cs', 'software-engineering', '软件工程', 'Pressman《软件工程：实践者的研究方法》、邹欣《构建之法》'],

  // ---- 第四级 高阶专题 ----
  ['advanced', 'ai-infra', 'AI 基础设施', '大规模 AI 训练/推理基础设施技术栈（CUDA/NCCL/并行策略/vLLM 等）'],
  ['advanced', 'ai-safety', 'AI 安全与对齐', '对齐问题/可解释性/鲁棒性/AI 治理经典课程体系'],
  ['advanced', 'ai4science', 'AI for Science', '各领域经典论文、课程与专著体系'],
  ['advanced', 'autonomous-driving', '自动驾驶', '模块化栈 + 端到端新范式技术全景'],
  ['advanced', 'computer-vision', '计算机视觉', 'Szeliski《计算机视觉：算法与应用》、CS231n'],
  ['advanced', 'deep-learning', '深度学习', 'Goodfellow《深度学习》（花书）、李沐《动手学深度学习》'],
  ['advanced', 'embodied-ai', '具身智能', 'Craig《机器人学导论》、Lynch & Park《Modern Robotics》'],
  ['advanced', 'information-retrieval', '信息检索', 'Manning《Introduction to Information Retrieval》'],
  ['advanced', 'llm-deployment', '大模型部署', 'vLLM/SGLang/TensorRT-LLM 等推理引擎与部署工程体系'],
  ['advanced', 'llm-finetuning', '大模型微调', '继续预训练/指令微调/偏好对齐知识树'],
  ['advanced', 'llm-principles', '大模型原理', 'LLM 完整知识树（Tokenizer/架构/预训练/推理/评测）'],
  ['advanced', 'machine-learning', '机器学习', '周志华《机器学习》（西瓜书）'],
  ['advanced', 'nlp', '自然语言处理', '宗成庆《自然语言处理》、Jurafsky《Speech and Language Processing》'],
  ['advanced', 'ontology', '本体论', '形而上学传统 + 知识表示工程（双线）'],
  ['advanced', 'quantum-computing', '量子计算', 'Nielsen & Chuang《量子计算与量子信息》、IBM Qiskit'],
  ['advanced', 'recommender-systems', '推荐系统', '项亮《推荐系统实践》+ 工业界经典论文'],
  ['advanced', 'reinforcement-learning', '强化学习', 'Sutton & Barto《强化学习（第2版）》'],
  ['advanced', 'speech', '语音技术', '《语音信号处理》、《Spoken Language Processing》'],
]

function agentFile([tier, key, name, benchmark]) {
  const id = `${tier}-${key}`
  const tierLabel = TIER[tier]
  return `---
name: ${id}
description: 专题专家：负责「${name}」（${tierLabel}）分类全部博文的撰写。对标 ${benchmark}。写该专题博文时使用本专家。
tools: Bash, Read, Write, Edit, WebFetch, WebSearch, Glob, Grep
---

# ${name} 专家小组

你是「从极限到大模型」博客 ${tierLabel}《${name}》专题的资深专家写作者，负责把该专题对标教材的体系逐节写成高质量博文。

## 领域坐标
- 专题 key：${tier}/${key}
- 对标教材 / 体系：${benchmark}
- 写作约束：全部博文遵循 \`.claude/writing-charter.md\`（编辑章程），**写作前必须通读**

## 本组工作方法（每篇必走）
1. 读 \`.claude/writing-charter.md\`、本专题规划 \`docs/posts/${tier}/${key}/index.md\`、范本 \`docs/posts/foundations/math/set-concept.md\`
2. 基于对标教材的权威知识撰写（这些教材的经典内容是标准知识）；细节拿不准时用 ≤2 次全网搜索（OpenStax/arXiv/MIT OCW/官方文档）核对
3. 按章程产出 Markdown，写入 \`docs/posts/${tier}/${key}/<slug>.md\`
4. 需要时配 ≤1 张手写 SVG 图，存 \`docs/public/images/${key}/\`，文章以 \`/images/${key}/...\` 引用
5. 更新 \`docs/posts/${tier}/${key}/index.md\` 中对应条目为 \`- [x] [标题](./<slug>)\`
6. 向主控返回简短报告：标题、slug、参考来源、是否配图（**不要**改动 \`progress.json\`，主控统一重生成）
`
}

mkdirSync('.claude/agents', { recursive: true })
for (const t of TEAMS) {
  const id = `${t[0]}-${t[1]}`
  writeFileSync(`.claude/agents/${id}.md`, agentFile(t))
}
console.log(`已生成 ${TEAMS.length} 个专家小组定义文件`)
