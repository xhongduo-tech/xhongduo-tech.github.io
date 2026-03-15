/* ===================================================
   主题初始化（已在 HTML <script> 中内联执行，这里补
   充按钮文字更新与切换逻辑）
   =================================================== */
document.addEventListener('DOMContentLoaded', () => {
  // 年份
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // 主题按钮
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    const updateBtn = () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      themeBtn.textContent = isDark ? '[light]' : '[dark]';
    };
    updateBtn();
    themeBtn.addEventListener('click', () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const next = isDark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateBtn();
      syncHljsTheme(next);
    });
  }

  // 初始化 hljs 主题（跟随系统色，若无手动设置）
  const savedTheme = localStorage.getItem('theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  syncHljsTheme(savedTheme);

  // 代码块 copy 按钮（事件委托）
  document.addEventListener('click', e => {
    if (e.target.classList.contains('code-copy')) {
      const btn = e.target;
      const code = btn.closest('.code-wrapper')?.querySelector('code');
      if (!code) return;
      navigator.clipboard.writeText(code.textContent).then(() => {
        btn.textContent = 'copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = 'copy';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {
        btn.textContent = 'failed';
        setTimeout(() => { btn.textContent = 'copy'; }, 2000);
      });
    }
  });

  // 根据当前页面执行对应逻辑
  if (document.getElementById('post-list')) {
    initIndexPage();
  } else if (document.getElementById('post-content')) {
    initPostPage();
  }
});

/* ===== hljs 主题同步 ===== */
function syncHljsTheme(theme) {
  const link = document.getElementById('hljs-theme');
  if (!link) return;
  const base = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/';
  link.href = theme === 'light'
    ? base + 'github.min.css'
    : base + 'atom-one-dark.min.css';
}

/* ===================================================
   首页：文章列表
   =================================================== */
async function initIndexPage() {
  const CATEGORY_ORDER = ['理论基础', '前沿追踪', '模型训练', '模型微调', '模型部署', '智能体', '系统基础', '工程实践'];
  const CATEGORY_CLASS = {
    '理论基础': 'cat-theory',
    '前沿追踪': 'cat-frontier',
    '模型训练': 'cat-training',
    '模型微调': 'cat-finetune',
    '模型部署': 'cat-deploy',
    '智能体':   'cat-agent',
    '系统基础': 'cat-systems',
    '工程实践': 'cat-engineering',
  };

  let posts = [];
  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error('fetch failed');
    posts = await res.json();
  } catch (e) {
    document.getElementById('post-list').innerHTML =
      '<p class="no-result">// 暂无文章 · 请添加 posts/posts.json</p>';
    return;
  }

  // 按日期降序
  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  // 渲染统计数字
  const statsEl = document.getElementById('blog-stats');
  if (statsEl) {
    const cats = new Set(posts.map(p => p.tags?.[0]).filter(Boolean));
    statsEl.innerHTML =
      `<span class="stat-num">${posts.length}</span><span class="stat-label"> posts</span>` +
      `<span class="stat-sep">·</span>` +
      `<span class="stat-num">${cats.size}</span><span class="stat-label"> categories</span>`;
  }

  // State
  let activeCategory = 'all';
  let activeTag      = null;
  let searchQuery    = '';

  // 渲染分类 Tab
  const categoryTabsEl = document.getElementById('category-tabs');
  ['all', ...CATEGORY_ORDER].forEach(cat => {
    const btn = document.createElement('button');
    btn.className   = 'cat-tab' + (cat === 'all' ? ' active' : '');
    btn.dataset.cat = cat;
    btn.textContent = cat === 'all' ? '全部' : cat;
    categoryTabsEl.appendChild(btn);
  });
  categoryTabsEl.addEventListener('click', e => {
    const btn = e.target.closest('.cat-tab');
    if (!btn) return;
    activeCategory = btn.dataset.cat;
    categoryTabsEl.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // 切换分类时重置 tag 筛选
    activeTag = null;
    tagFilter.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
    renderList();
  });

  // 每篇文章取 tags[1]（第一个非分类次级 tag），去重后渲染过滤器
  const allTags = [...new Set(posts.map(p => (p.tags || [])[1]).filter(Boolean))].sort();
  const tagFilter = document.getElementById('tag-filter');
  allTags.forEach(tag => {
    const btn = document.createElement('button');
    btn.className = 'tag';
    btn.textContent = tag;
    btn.addEventListener('click', () => {
      activeTag = activeTag === tag ? null : tag;
      tagFilter.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
      if (activeTag) btn.classList.add('active');
      renderList();
    });
    tagFilter.appendChild(btn);
  });

  // 搜索
  const searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', () => {
    searchQuery = searchInput.value.trim().toLowerCase();
    renderList();
  });

  /**
   * 影响力因子：仅显示 核心主题、分类方向、评分依据。
   * 评分依据必须基于具体 topic（如 Broken Neural Scaling Law: S 形修正），而非宽泛分类。
   * 具体 topic 优先匹配，顺序靠前的条目优先。
   */
  function computeImpactFactor(post) {
    const title   = post.title   || '';
    const summary = post.summary || '';
    const tags    = post.tags    || [];
    const cat     = tags[0]      || '';
    const text    = [title, summary, ...tags].join(' ');

    // 具体 topic 优先：每个条目对应一个可识别的具体研究主题，评分依据针对该 topic
    const TOPIC_MAP = [
      // 具体 topic：Scaling Law 相关
      { keys: ['Broken Neural Scaling Law', 'BNSL', 'S 形修正', '断裂神经扩展'], score: 4, name: 'Broken Neural Scaling Law: S 形修正', desc: '评分 4：Shen 等将单一幂律扩展为带平滑拐点的分段幂律，解释涌现为斜率突变而非跳变；修正传统 scaling law 对下游任务的拟合偏差，是理解规模-能力非单调关系的核心工作。' },
      { keys: ['Inverse Scaling', 'U 形涌现', '逆向 scaling'], score: 4, name: 'Inverse Scaling / U 形涌现', desc: '评分 4：Wei 等发现部分任务随规模先变差再回升；Inverse Scaling Prize 推动任务设计反思，U 形曲线揭示干扰信号与容量竞争，对 benchmark 设计与能力评估有直接影响。' },
      { keys: ['Chinchilla', '计算最优'], score: 5, name: 'Chinchilla 计算最优', desc: '评分 5：Hoffmann 等证明数据与算力应同步扩展，推翻「模型越大越好」的粗放结论；指导 OpenAI/Google 等训练配比，是 scaling 决策的理论基石。' },
      { keys: ['涌现能力', 'emergent', 'Emergent Abilities'], score: 4, name: '涌现能力', desc: '评分 4：规模增大带来的突变式能力跃迁；Wei 等 2022 系统化定义，指导 scaling 决策，是理解 LLM 能力边界的核心概念。' },
      { keys: ['扩展定律', 'Scaling Law', 'scaling law'], score: 5, name: 'Scaling Law', desc: '评分 5：Kaplan/Chinchilla 等奠定规模-数据-算力关系；指导超大规模训练决策，是 AI 投入产出可预测性的理论基石。' },
      // 奠基性范式
      { keys: ['RLHF'], score: 5, name: 'RLHF', desc: '评分 5：InstructGPT 开创人类偏好对齐范式，引用破万；确立 ChatGPT/Claude 等主流模型训练基线，工业界几乎全部闭源模型均采用。' },
      { keys: ['LoRA'], score: 5, name: 'LoRA', desc: '评分 5：Hu et al. 2022 低秩适配，引用超 2 万；参数高效微调事实标准，HuggingFace/PEFT 生态标配。' },
      { keys: ['CoT', '思维链'], score: 5, name: 'Chain-of-Thought', desc: '评分 5：Wei et al. 2022 首次系统化推理提示，引用超 1.5 万；开启 LLM 推理能力显式化研究线，GPT-4/Claude 等均内置，是复杂推理任务的基础方法论。' },
      { keys: ['RAG'], score: 5, name: 'RAG', desc: '评分 5：Lewis et al. 2020 检索增强生成，引用超 1 万；工业界知识库问答、企业搜索的核心架构，解决幻觉与知识截止的事实标准。' },
      { keys: ['Transformer'], score: 5, name: 'Transformer', desc: '评分 5：Vaswani et al. 2017 注意力机制，引用超 10 万；当前所有主流 LLM/VLM 的基础架构。' },
      // 具体 topic：推理与 CoT
      { keys: ['Zero-shot CoT', '零样本思维链'], score: 4, name: 'Zero-shot CoT', desc: '评分 4：Kojima 等用「Let\'s think step by step」触发推理，无需示例；大幅降低 CoT 使用门槛，工业界广泛采用。' },
      // ★★★★ 高影响：>3k 引用或广泛工程采用
      { keys: ['DPO'], score: 4, name: 'DPO', desc: '评分 4：Rafailov et al. 2023 无需奖励模型的对齐，引用超 5000；简化 RLHF 管线、降低训练成本，Zephyr/Llama 等开源模型广泛采用，对齐研究的重要分支。' },
      { keys: ['SFT'], score: 4, name: 'SFT', desc: '评分 4：监督微调是 LLM 指令对齐的基线；几乎覆盖所有指令模型训练流程，工业界标配环节，虽非单篇高引但作为流程基石影响广泛。' },
      { keys: ['ReAct'], score: 4, name: 'ReAct', desc: '评分 4：Yao et al. 2022 推理-行动交替，引用超 3000；Agent 工程代表框架，LangChain/AutoGPT 等均借鉴，是工具调用与推理结合的主流范式。' },
      { keys: ['ToT', 'Tree of Thought'], score: 4, name: 'Tree of Thoughts', desc: '评分 4：Yao et al. 2023 树结构多路径推理，引用超 3000；扩展 CoT 为显式搜索，数学/代码等需验证任务的标准增强方法。' },
      { keys: ['HNSW'], score: 4, name: 'HNSW', desc: '评分 4：Malkov & Yashunin 2018 近似最近邻，Faiss/Weaviate/Milvus 等工业标准；向量检索与 RAG 的底层索引事实选择，工程落地必备。' },
      { keys: ['Embedding', '嵌入模型'], score: 4, name: 'Embedding', desc: '评分 4：语义嵌入是 RAG、检索、多模态对齐的核心组件；OpenAI/Cohere 等提供商用 API，工业界语义搜索基础设施。' },
      { keys: ['知识图谱'], score: 4, name: '知识图谱', desc: '评分 4：结构化知识表示，与向量检索互补；多跳推理、知识问答的经典路径，企业知识管理的重要技术栈。' },
      { keys: ['MoE'], score: 4, name: 'MoE', desc: '评分 4：稀疏激活扩展模型容量而不成比例增加计算；Mixtral/DeepSeek-MoE 等证明可行，是 scaling 效率的重要方向，引用与工业采用双高。' },
      { keys: ['多模态'], score: 4, name: '多模态', desc: '评分 4：图文/音视频统一建模是 GPT-4V/Claude/Gemini 的核心能力；多模态理解与生成是 AI 泛化到真实世界的关键扩展。' },
      { keys: ['长上下文'], score: 4, name: '长上下文', desc: '评分 4：百万 token 上下文是 Claude/Gemini 的差异化能力；长文档理解、代码库分析、多轮对话依赖，工业界刚需方向。' },
      { keys: ['FlashAttention'], score: 4, name: 'FlashAttention', desc: '评分 4：Dao et al. 2022 注意力 IO 优化，引用超 3000；PyTorch 2.0/vLLM 等标配，大模型训练与推理的显存与速度关键优化。' },
      { keys: ['ZeRO'], score: 4, name: 'ZeRO', desc: '评分 4：微软 DeepSpeed ZeRO 系列，分布式训练事实标准；参数/梯度/优化器分片成为大模型训练标配，引用与采用双高。' },
      { keys: ['涌现能力'], score: 4, name: '涌现能力', desc: '评分 4：规模增大带来的突变式能力跃迁；指导模型 scaling 决策，是理解 LLM 能力边界的核心概念，学术争议与工业关注并存。' },
      // ★★★ 重要专项：特定领域广泛引用
      { keys: ['自洽性', 'Self-Consistency'], score: 3, name: 'Self-Consistency', desc: '评分 3：Wang et al. 2022 多路径多数投票，引用约 8000；数学/推理任务的标准解码增强，实现简单、收益稳定，但依赖采样成本。' },
      { keys: ['GoT', 'Graph of Thought'], score: 3, name: 'Graph of Thoughts', desc: '评分 3：Besta et al. 2023 推理拓扑从树扩展为 DAG；支持分支复用与聚合，是 ToT 的拓扑泛化，引用与关注持续增长。' },
      { keys: ['MCTS', '蒙特卡洛树'], score: 3, name: 'MCTS', desc: '评分 3：蒙特卡洛树搜索是博弈与规划经典算法；AlphaGo 奠定地位，LLM+搜索结合的重要方向，数学证明、代码生成等任务有应用。' },
      { keys: ['MemGPT'], score: 3, name: 'MemGPT', desc: '评分 3：Packer et al. 2023 分层上下文管理；扩展 LLM 记忆边界，Agent 长程交互的实用方案，记忆系统研究代表工作。' },
      { keys: ['Function Calling', '工具调用'], score: 3, name: 'Function Calling', desc: '评分 3：OpenAI 工具调用协议成为事实标准；Agent 落地的核心接口，GPT/Claude/Gemini 均支持，工程采用广泛。' },
      { keys: ['数学推理'], score: 3, name: '数学推理', desc: '评分 3：MATH/GSM8K 等 benchmark 驱动；过程奖励、形式化验证等方向活跃，是衡量推理能力的关键任务域。' },
      { keys: ['上下文学习', 'ICL', 'In-Context Learning'], score: 3, name: 'ICL', desc: '评分 3：Brown et al. 2020 首次系统展示 few-shot 适应；无需微调即可任务适应，是 LLM 核心特性，引用与理论分析丰富。' },
      { keys: ['注意力机制'], score: 3, name: '注意力机制', desc: '评分 3：Transformer 核心组件；GQA/MQA、稀疏注意力等变体广泛采用，是模型效率与能力的关键调节点。' },
      { keys: ['位置编码', 'RoPE', 'YaRN'], score: 3, name: '位置编码', desc: '评分 3：RoPE 等相对位置编码是长上下文基础；YaRN/LongRoPE 等扩展方法支撑百万 token，工程实现必备知识。' },
      { keys: ['Constitutional AI', 'RLAIF'], score: 3, name: 'Constitutional AI', desc: '评分 3：Anthropic 宪法式自洽对齐；减少人类标注、提升可控性，Claude 系列采用，对齐研究的重要分支。' },
      { keys: ['PRM', '过程奖励'], score: 3, name: 'PRM', desc: '评分 3：过程奖励模型对推理步骤评分；数学/代码等链式任务的关键增强，OpenAI 等采用，与 ORM 互补。' },
      { keys: ['量化'], score: 3, name: '量化', desc: '评分 3：AWQ/GPTQ 等使大模型边缘部署可行；工业界推理成本控制标配，学术与工程双线活跃。' },
      { keys: ['分布式训练', 'Megatron', '流水线并行'], score: 3, name: '分布式训练', desc: '评分 3：TP/PP/DP 等并行范式支撑千亿级训练；Megatron/DeepSpeed 等框架标配，大模型训练基础设施。' },
      { keys: ['Reflexion', '反思'], score: 3, name: 'Reflexion', desc: '评分 3：语言化反馈实现跨 episode 学习；HotPotQA 等任务显著提升，Agent 自我修正的代表方法。' },
      { keys: ['记忆系统', '反思记忆'], score: 3, name: '记忆系统', desc: '评分 3：Agent 长程交互的存储与检索；Generative Agents 等奠定架构，多 Agent 协作与个人化的重要组件。' },
      // ★★ 专项实现：引用或落地相对局限
      { keys: ['CRDT'], score: 2, name: 'CRDT', desc: '评分 2：无冲突复制数据类型，分布式一致性专项；多 Agent 共享状态等场景有用，但应用面较窄。' },
      { keys: ['PPO'], score: 2, name: 'PPO', desc: '评分 2：Schulman et al. 2017 近端策略优化；RLHF 常用算法，但作为 RL 通用方法，在 LLM 语境下更多是工具而非范式。' },
      { keys: ['GQA', 'MQA'], score: 2, name: 'GQA/MQA', desc: '评分 2：Grouped/Multi-Query Attention 减少 KV 缓存；Llama 2/3 等采用，推理加速的工程优化，非范式级创新。' },
      { keys: ['状态空间模型', 'Mamba', 'RetNet'], score: 2, name: 'SSM', desc: '评分 2：Mamba/RetNet 等线性复杂度序列模型；长序列效率有优势，但尚未成为主流，处于探索期。' },
      { keys: ['稀疏注意力', '线性注意力'], score: 2, name: '稀疏/线性注意力', desc: '评分 2：降低注意力复杂度的方法；Longformer/Sparse Transformer 等有应用，但 FlashAttention 等使标准注意力仍主流。' },
      { keys: ['数据工程', '预训练'], score: 2, name: '数据工程', desc: '评分 2：数据清洗、去重、配比是训练质量基础；工业界重视但学术引用分散，属工程关键环节。' },
      { keys: ['评测', 'Benchmark'], score: 2, name: '评测', desc: '评分 2：MMLU/SWE-bench 等驱动能力迭代；必要基础设施，但单篇 benchmark 影响力有限。' },
      { keys: ['安全', '红队', 'Jailbreak'], score: 2, name: '安全', desc: '评分 2：对齐与对抗是持续研究方向；工业界刚需，但方法分散，尚未形成统一范式。' },
      { keys: ['推理优化', 'Speculative', 'KV Cache'], score: 2, name: '推理优化', desc: '评分 2：投机解码、KV 压缩等提升推理效率；部署必备，属工程优化范畴。' },
      { keys: ['模型部署', 'vLLM', '显存优化'], score: 2, name: '模型部署', desc: '评分 2：服务化与资源优化；vLLM/SGLang 等有影响，但更多是工程实现。' },
      // ★ 工程实践与系统基础
      { keys: ['系统基础', 'Linux', 'Docker', 'Kubernetes', '数据库'], score: 2, name: '系统基础', desc: '评分 2：操作系统、容器、数据库等是 AI 工程底座；必要但不直接贡献 AI 能力突破。' },
    ];

    let topicScore = 0;
    let topicName  = '';
    let topicDesc  = '';
    for (const item of TOPIC_MAP) {
      if (item.keys.some(k => text.includes(k))) {
        if (item.score > topicScore) {
          topicScore = item.score;
          topicName  = item.name;
          topicDesc  = item.desc;
        }
      }
    }

    const catMeta = {
      '前沿追踪': { score: 4, label: '前沿研究领域' },
      '理论基础': { score: 4, label: '基础理论方向' },
      '模型训练': { score: 3, label: '模型训练方向' },
      '模型微调': { score: 3, label: '微调工程方向' },
      '模型部署': { score: 2, label: '部署工程方向' },
      '智能体':   { score: 3, label: 'Agent 应用方向' },
      '系统基础': { score: 2, label: '系统基础方向' },
      '工程实践': { score: 1, label: '工程实践方向' },
    };
    const catInfo = catMeta[cat] || { score: 2, label: '通用方向' };

    let level = topicScore > 0 ? topicScore : catInfo.score;
    const isSynthesis  = /与|融合|结合|集成|混合/.test(title);
    const isComparison = /选型|对比|比较/.test(title);
    if (topicScore === 0 && isSynthesis)    level = Math.min(5, level + 1);
    if (isComparison && tags.length >= 4)   level = Math.min(5, level + 1);
    level = Math.max(1, Math.min(5, level));

    let reasonDesc;
    if (topicName && topicDesc) {
      reasonDesc = topicDesc;
    } else {
      reasonDesc = '该主题暂无独立文献评估，按分类给出参考分。';
    }

    const factors = [
      { label: '核心主题', value: topicName || title },
      { label: '分类方向', value: catInfo.label },
      { label: '评分依据', value: reasonDesc },
    ];

    return { level, factors };
  }

  function postCardHtml(p) {
    const tags = p.tags || [];
    const tagsHtml = tags.map((t, i) => {
      const catCls = i === 0 ? (CATEGORY_CLASS[t] || '') : '';
      return `<span class="tag ${catCls}">${escapeHtml(t)}</span>`;
    }).join('');
    const { level, factors } = computeImpactFactor(p);
    const impactData = escapeHtml(JSON.stringify({ level, factors }));
    const barsHtml = [1, 2, 3, 4, 5]
      .map(i => `<span class="impact-bar${i <= level ? ' active' : ''}"></span>`)
      .join('');
    const impactHtml = `<span class="impact-factor impact-level-${level}" data-impact="${impactData}">${barsHtml}</span>`;
    return `
      <a class="post-card" href="post.html?slug=${encodeURIComponent(p.slug)}">
        <div class="post-card-meta">
          <span class="post-date">${p.date}</span>
          <div class="post-tags">${tagsHtml}</div>
          ${impactHtml}
        </div>
        <div class="post-card-title">${escapeHtml(p.title)}</div>
        ${p.summary ? `<div class="post-card-summary">${escapeHtml(p.summary)}</div>` : ''}
      </a>`;
  }

  function renderList() {
    const filtered = posts.filter(p => {
      const matchCat    = activeCategory === 'all' || (p.tags && p.tags[0] === activeCategory);
      const matchTag    = !activeTag || (p.tags && p.tags.includes(activeTag));
      const q           = searchQuery;
      const matchSearch = !q ||
        p.title.toLowerCase().includes(q) ||
        (p.summary || '').toLowerCase().includes(q) ||
        (p.tags || []).some(t => t.toLowerCase().includes(q));
      return matchCat && matchTag && matchSearch;
    });

    const list = document.getElementById('post-list');
    if (filtered.length === 0) {
      list.innerHTML = '<p class="no-result">// 没有找到相关文章</p>';
      return;
    }

    // 「全部」且无任何筛选时：按分类分组展示
    if (activeCategory === 'all' && !activeTag && !searchQuery) {
      const groups = {};
      CATEGORY_ORDER.forEach(cat => { groups[cat] = []; });
      filtered.forEach(p => {
        const cat = p.tags?.[0];
        if (cat && groups[cat]) groups[cat].push(p);
      });
      list.innerHTML = CATEGORY_ORDER
        .filter(cat => groups[cat].length > 0)
        .map(cat => `
          <div class="post-section">
            <div class="post-section-header">
              <span class="section-label">// ${cat}</span>
              <span class="section-count">${groups[cat].length}</span>
            </div>
            ${groups[cat].map(postCardHtml).join('')}
          </div>`)
        .join('');
    } else {
      // 分类 / 标签 / 搜索筛选时：平铺展示
      list.innerHTML = filtered.map(postCardHtml).join('');
    }
  }

  renderList();
  setupImpactTooltip();
}

/* ===================================================
   影响力因子 Tooltip
   =================================================== */
function setupImpactTooltip() {
  let tip = document.getElementById('impact-tooltip');
  if (!tip) {
    tip = document.createElement('div');
    tip.id = 'impact-tooltip';
    tip.className = 'impact-tooltip';
    document.body.appendChild(tip);
  }

  const listEl = document.getElementById('post-list');
  if (!listEl) return;

  listEl.addEventListener('mouseover', e => {
    const el = e.target.closest('.impact-factor');
    if (!el) return;
    let data;
    try { data = JSON.parse(el.dataset.impact); } catch { return; }

    const { level, factors } = data;
    const rowsHtml = (factors || []).map(f =>
      `<div class="itp-row">
        <span class="itp-key">${escapeHtml(f.label)}</span>
        <span class="itp-val">${escapeHtml(f.value)}</span>
      </div>`
    ).join('');

    tip.innerHTML = `
      <div class="itp-header">
        <span class="itp-title">影响力因子</span>
        <span class="itp-score">${level} / 5</span>
      </div>
      ${rowsHtml}`;

    const rect  = el.getBoundingClientRect();
    const tipW  = 270;
    let left = rect.left + rect.width / 2 - tipW / 2;
    left = Math.max(8, Math.min(left, window.innerWidth - tipW - 8));
    tip.style.left      = left + 'px';
    tip.style.top       = (rect.top - 8) + 'px';
    tip.style.transform = 'translateY(-100%)';
    tip.classList.add('visible');
  });

  listEl.addEventListener('mouseout', e => {
    const el = e.target.closest('.impact-factor');
    if (!el) return;
    if (el.contains(e.relatedTarget)) return;
    tip.classList.remove('visible');
  });
}

/* ===================================================
   文章页：渲染单篇文章
   =================================================== */
async function initPostPage() {
  const params    = new URLSearchParams(location.search);
  const slug      = params.get('slug');
  const contentEl = document.getElementById('post-content');

  if (!slug) {
    contentEl.innerHTML = '<p class="no-result">// 未指定文章</p>';
    return;
  }

  // 加载文章索引
  let posts = [];
  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error();
    posts = await res.json();
    posts.sort((a, b) => new Date(b.date) - new Date(a.date));
  } catch (e) {
    contentEl.innerHTML = '<p class="no-result">// 无法加载文章索引</p>';
    return;
  }

  const idx  = posts.findIndex(p => p.slug === slug);
  const post = posts[idx];
  if (!post) {
    contentEl.innerHTML = '<p class="no-result">// 文章不存在</p>';
    return;
  }

  // 加载 Markdown
  let mdText = '';
  try {
    const res = await fetch(`posts/${post.slug}.md`);
    if (!res.ok) throw new Error('404');
    mdText = await res.text();
  } catch (e) {
    contentEl.innerHTML = '<p class="no-result">// 文章文件未找到</p>';
    return;
  }

  // 页面标题
  document.title = `${post.title} · xhd.log`;

  // 配置 marked（用 plain object renderer，避免 new marked.Renderer() 兼容问题）
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });

    marked.use({
      renderer: {
        // 代码块：添加语言标签 + copy 按钮
        code(text, lang) {
          let highlighted = escapeHtml(text);
          let language    = 'plaintext';

          if (typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
            language    = lang;
            highlighted = hljs.highlight(text, { language }).value;
          }

          const langLabel = language !== 'plaintext'
            ? `<span class="code-lang">${language}</span>`
            : `<span class="code-lang">text</span>`;

          return `
        <div class="code-wrapper">
          <div class="code-header">
            ${langLabel}
            <button class="code-copy">copy</button>
          </div>
          <pre><code class="hljs language-${language}">${highlighted}</code></pre>
        </div>`;
        }
      }
    });
  }

  // 阅读时长（中文按字符，英文按单词）
  const charCount = mdText.replace(/\s/g, '').length;
  const readTime  = Math.max(1, Math.ceil(charCount / 400));

  const tagsHtml = (post.tags || [])
    .map(t => `<span class="tag">${escapeHtml(t)}</span>`)
    .join('');

  // follow 状态（localStorage）
  const followKey       = `follow:${post.slug}`;
  const isFollowing     = localStorage.getItem(followKey) === '1';
  const followBtnClass  = isFollowing ? 'follow-btn following' : 'follow-btn';
  const followBtnText   = isFollowing ? '[ following ]' : '[ + follow ]';

  // 保护数学公式不被 marked 处理，渲染后替换为 KaTeX HTML
  let bodyHtml;
  try {
    if (typeof marked !== 'undefined') {
      const mathStore = [];
      const protectedMd = mdText
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
          mathStore.push({ display: true, math });
          return `<!--KMATH${mathStore.length - 1}-->`;
        })
        .replace(/\$([^$\n]+?)\$/g, (_, math) => {
          mathStore.push({ display: false, math });
          return `<!--KMATH${mathStore.length - 1}-->`;
        });
      // await 兼容 marked 在某些配置下返回 Promise<string> 的情况
      let html = await Promise.resolve(marked.parse(protectedMd));
      if (typeof katex !== 'undefined' && mathStore.length > 0) {
        html = html.replace(/<!--KMATH(\d+)-->/g, (_, i) => {
          const { display, math } = mathStore[Number(i)];
          return katex.renderToString(math, { throwOnError: false, displayMode: display });
        });
      }
      bodyHtml = html;
    } else {
      bodyHtml = `<pre>${escapeHtml(mdText)}</pre>`;
    }
  } catch (e) {
    // 降级：不渲染数学公式，但文章内容正常显示
    try {
      bodyHtml = typeof marked !== 'undefined'
        ? await Promise.resolve(marked.parse(mdText))
        : `<pre>${escapeHtml(mdText)}</pre>`;
    } catch (_) {
      bodyHtml = `<pre>${escapeHtml(mdText)}</pre>`;
    }
  }

  contentEl.innerHTML = `
    <header class="post-header">
      <h1 class="post-title">${escapeHtml(post.title)}</h1>
      <div class="post-info">
        <span class="post-date">${post.date}</span>
        <span class="post-read-time">约 ${readTime} 分钟阅读</span>
        <div class="post-tags">${tagsHtml}</div>
      </div>
      <div class="post-author">
        ${post.author === 'both'
          ? `${CLAUDE_SVG}<span>Claude Sonnet/Opus 4.6</span><span class="author-sep">&amp;</span>${OPENAI_SVG}<span>GPT-5.4</span><span class="author-sep">&amp;</span><span>xuhongduo</span>`
          : `${CLAUDE_SVG}<span>Claude Sonnet/Opus 4.6</span>`}
      </div>
      <div class="post-follow">
        <button class="${followBtnClass}" id="follow-btn">${followBtnText}</button>
        <span class="follow-count" id="follow-count"></span>
      </div>
    </header>
    <div class="post-body">${bodyHtml}</div>
  `;

  // 上一篇 / 下一篇（列表已按日期降序，idx-1 = 更新，idx+1 = 更早）
  const navEl = document.getElementById('post-nav');
  const prev  = posts[idx - 1];
  const next  = posts[idx + 1];

  navEl.innerHTML = [
    prev
      ? `<a class="post-nav-item prev" href="post.html?slug=${encodeURIComponent(prev.slug)}">
           <div class="post-nav-label">← newer</div>
           <div class="post-nav-title">${escapeHtml(prev.title)}</div>
         </a>`
      : '<span></span>',
    next
      ? `<a class="post-nav-item next" href="post.html?slug=${encodeURIComponent(next.slug)}">
           <div class="post-nav-label">older →</div>
           <div class="post-nav-title">${escapeHtml(next.title)}</div>
         </a>`
      : '<span></span>',
  ].join('');

  // 阅读进度条
  const progressBar = document.getElementById('progress-bar');
  if (progressBar) {
    const updateProgress = () => {
      const docH = document.documentElement.scrollHeight - window.innerHeight;
      const pct  = docH > 0 ? Math.min(100, (window.scrollY / docH) * 100) : 0;
      progressBar.style.width = pct + '%';
    };
    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  setupFollow(post);
  setupFeedback(post);
}

/* ===================================================
   关注 (Follow) + 计数
   =================================================== */
async function setupFollow(post) {
  const btn     = document.getElementById('follow-btn');
  const countEl = document.getElementById('follow-count');
  if (!btn) return;

  const key     = `follow:${post.slug}`;
  const apiBase = `https://api.counterapi.dev/v1/xhd-blog/follow-${post.slug}`;

  // 非阻塞加载关注总数
  if (countEl) {
    try {
      const res = await fetch(apiBase);
      if (res.ok) {
        const data = await res.json();
        countEl.textContent = data.count ?? 0;
      } else {
        countEl.textContent = 0;
      }
    } catch (e) {
      countEl.style.display = 'none';
    }
  }

  btn.addEventListener('click', async () => {
    const current = localStorage.getItem(key) === '1';
    if (current) {
      // 取消关注：仅改变本地状态（计数不减）
      localStorage.removeItem(key);
      btn.textContent = '[ + follow ]';
      btn.classList.remove('following');
    } else {
      // 关注：本地保存 + 远端计数 +1
      localStorage.setItem(key, '1');
      btn.textContent = '[ following ]';
      btn.classList.add('following');
      try {
        const res = await fetch(`${apiBase}/up`);
        if (res.ok && countEl) {
          const data = await res.json();
          countEl.textContent = data.count ?? countEl.textContent;
          countEl.style.display = '';
        }
      } catch (e) {
        // 静默失败，按钮状态已正确
      }
    }
  });
}

/* ===================================================
   纠错反馈
   =================================================== */
function setupFeedback(post) {
  const navEl = document.getElementById('post-nav');
  if (!navEl) return;

  // 纠错按钮
  const section = document.createElement('div');
  section.className = 'feedback-section';
  section.innerHTML = `<button class="feedback-btn">[ 纠错反馈 ]</button>`;
  navEl.insertAdjacentElement('afterend', section);

  // Modal
  const modal = document.createElement('div');
  modal.className = 'feedback-modal';
  modal.setAttribute('aria-hidden', 'true');
  modal.innerHTML = `
    <div class="feedback-panel">
      <div class="feedback-title">// 纠错反馈</div>
      <div class="feedback-hint">输入 <kbd>@</kbd> 可搜索文章中的标题与段落</div>
      <div class="mention-dropdown" id="mention-dropdown"></div>
      <textarea class="feedback-textarea" id="feedback-textarea"
        placeholder="描述具体错误，例如：@第二章 公式推导有误，正确结果应为..."></textarea>
      <div class="feedback-actions">
        <button class="feedback-cancel">取消</button>
        <button class="feedback-submit">提交到 GitHub Issues →</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const textarea    = modal.querySelector('#feedback-textarea');
  const dropdown    = modal.querySelector('#mention-dropdown');
  const submitBtn   = modal.querySelector('.feedback-submit');
  const cancelBtn   = modal.querySelector('.feedback-cancel');

  // 收集文章中所有可引用内容（标题 + 段落首句）
  function getSearchItems() {
    const items = [];
    let h2Text = '';
    let h3Text = '';
    document.querySelectorAll('.post-body h2, .post-body h3, .post-body h4, .post-body p, .post-body li')
      .forEach(el => {
        const text = el.textContent.trim();
        if (!text) return;
        if (el.tagName === 'H2') {
          h2Text = text; h3Text = '';
          items.push({ label: text, ref: text, crumb: '' });
        } else if (el.tagName === 'H3') {
          h3Text = text;
          items.push({ label: text, ref: text, crumb: h2Text });
        } else if (el.tagName === 'H4') {
          items.push({ label: text, ref: text, crumb: h3Text || h2Text });
        } else if (el.tagName === 'P' && text.length > 20) {
          // 段落：截取首句（句号/问号/感叹号前）
          const sentence = text.split(/[。？！.?!]/)[0].slice(0, 50);
          if (sentence.length > 10) {
            const crumb = h3Text || h2Text;
            items.push({ label: sentence + '…', ref: sentence, crumb });
          }
        } else if (el.tagName === 'LI' && text.length > 10 && text.length < 80) {
          const crumb = h3Text || h2Text;
          items.push({ label: text.slice(0, 50) + (text.length > 50 ? '…' : ''), ref: text.slice(0, 50), crumb });
        }
      });
    return items;
  }

  // 显示/隐藏 modal
  function openModal() {
    modal.setAttribute('aria-hidden', 'false');
    modal.classList.add('open');
    textarea.focus();
  }
  function closeModal() {
    modal.setAttribute('aria-hidden', 'true');
    modal.classList.remove('open');
    dropdown.innerHTML = '';
    dropdown.style.display = 'none';
  }

  section.querySelector('.feedback-btn').addEventListener('click', openModal);
  cancelBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  // @ 触发下拉
  let mentionStart = -1;
  // 缓存，避免每次输入都重新遍历 DOM
  let _searchCache = null;
  function getCachedItems() {
    if (!_searchCache) _searchCache = getSearchItems();
    return _searchCache;
  }

  textarea.addEventListener('input', () => {
    const val    = textarea.value;
    const cursor = textarea.selectionStart;
    const before = val.slice(0, cursor);
    const atIdx  = before.lastIndexOf('@');

    if (atIdx === -1) { dropdown.style.display = 'none'; mentionStart = -1; return; }

    // 如果 @ 后面已经有空格（说明引用已完成），不弹出
    const afterAt = before.slice(atIdx + 1);
    if (afterAt.includes(' ') && afterAt.length > 10) {
      dropdown.style.display = 'none'; mentionStart = -1; return;
    }

    const query   = afterAt.toLowerCase();
    const matched = getCachedItems().filter(item =>
      item.label.toLowerCase().includes(query) ||
      item.crumb.toLowerCase().includes(query)
    ).slice(0, 8); // 最多展示 8 条

    if (matched.length === 0) { dropdown.style.display = 'none'; mentionStart = -1; return; }

    mentionStart = atIdx;
    dropdown.style.display = 'block';
    dropdown.innerHTML = matched.map(item => `
      <div class="mention-item" data-ref="${escapeHtml(item.ref)}">
        <span class="mention-label">${escapeHtml(item.label)}</span>
        ${item.crumb ? `<span class="mention-crumb">↳ ${escapeHtml(item.crumb)}</span>` : ''}
      </div>`
    ).join('');
  });

  dropdown.addEventListener('click', e => {
    const item = e.target.closest('.mention-item');
    if (!item || mentionStart === -1) return;
    const ref    = item.dataset.ref;
    const val    = textarea.value;
    const cursor = textarea.selectionStart;
    textarea.value = val.slice(0, mentionStart) + '@[' + ref + '] ' + val.slice(cursor);
    textarea.selectionStart = textarea.selectionEnd = mentionStart + ref.length + 4;
    textarea.focus();
    dropdown.style.display = 'none';
    mentionStart = -1;
  });

  // 提交
  submitBtn.addEventListener('click', () => {
    const body = textarea.value.trim();
    if (!body) { textarea.focus(); return; }
    const title  = encodeURIComponent('[纠错] ' + post.title);
    const labels = encodeURIComponent('error-report');
    const bodyEnc = encodeURIComponent(body + '\n\n---\n文章：' + post.slug);
    const url = `https://github.com/xhongduo-tech/xhongduo-tech.github.io/issues/new?title=${title}&labels=${labels}&body=${bodyEnc}`;
    window.open(url, '_blank', 'noopener');
    closeModal();
  });
}

/* ===== 品牌图标 ===== */
const CLAUDE_SVG = `<svg class="author-icon claude-icon" viewBox="0 0 24 24" fill="currentColor" aria-label="Claude"><path d="m4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z"/></svg>`;
const OPENAI_SVG = `<svg class="author-icon openai-icon" viewBox="0 0 24 24" fill="currentColor" aria-label="GPT"><path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zm-9.022 12.61a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/></svg>`;

/* ===== 工具函数 ===== */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
