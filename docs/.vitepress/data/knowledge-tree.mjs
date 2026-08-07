// 全人类知识树 · 12 棵领域树
// 每棵树：基础 → 核心 → 进阶 → 专业 → 前沿，逐级依赖清晰。
// 节点 = 已有专题（path 为 tier/key，指向 /posts/<tier>/<key>/）。
// 跨树重合以「引用」呈现（tag: 'ref'），不重复建主题。
// 增补节点（tag: 'add'）为当前缺失、规划待建的专题。

export const trees = [
  {
    id: 'math',
    name: '数学知识树',
    desc: '从算术几何出发，经微积分与代数，到分析、代数、几何、数论、概率与优化的完整数学大厦。',
    branches: [
      {
        level: '基础',
        nodes: [
          { name: '基础数学', path: 'foundations/math' },
          { name: '初等几何与三角', path: 'foundations/elementary-geometry-trigonometry' },
        ],
      },
      {
        level: '核心',
        nodes: [
          { name: '高等数学（微积分）', path: 'intermediate/advanced-math' },
          { name: '线性代数', path: 'intermediate/linear-algebra' },
          { name: '概率论与数理统计', path: 'intermediate/probability' },
          { name: '离散数学', path: 'intermediate/discrete-math' },
        ],
      },
      {
        level: '进阶',
        nodes: [
          { name: '数学分析', path: 'intermediate/mathematical-analysis' },
          { name: '实变函数与测度论', path: 'intermediate/real-analysis' },
          { name: '复变函数与积分变换', path: 'intermediate/complex-analysis' },
          { name: '抽象代数', path: 'intermediate/abstract-algebra' },
          { name: '拓扑学', path: 'intermediate/topology' },
          { name: '微分几何', path: 'intermediate/differential-geometry' },
          { name: '泛函分析', path: 'intermediate/functional-analysis' },
          { name: '数理逻辑', path: 'intermediate/mathematical-logic' },
          { name: '图论', path: 'intermediate/graph-theory' },
        ],
      },
      {
        level: '专业',
        nodes: [
          { name: '常微分方程', path: 'intermediate/ordinary-differential-equations' },
          { name: '偏微分方程', path: 'intermediate/pde' },
          { name: '数值分析', path: 'intermediate/numerical-analysis' },
          { name: '最优化理论', path: 'intermediate/optimization' },
          { name: '信息论', path: 'intermediate/information-theory' },
          { name: '数论', path: 'intermediate/number-theory' },
          { name: '组合数学', path: 'intermediate/combinatorics' },
          { name: '计算复杂性理论', path: 'intermediate/computational-complexity' },
          { name: '计算几何', path: 'intermediate/computational-geometry' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '随机过程', path: 'intermediate/stochastic-processes' },
          { name: '博弈论', path: 'intermediate/game-theory' },
          { name: '数学物理方法', path: 'intermediate/mathematical-physics-methods' },
          { name: '统计学习理论', path: 'advanced/machine-learning', tag: 'ref' },
        ],
      },
    ],
  },

  {
    id: 'physics',
    name: '物理知识树',
    desc: '从经典力学、电磁学、热学、光学到相对论与量子力学的现代物理图景。',
    branches: [
      {
        level: '基础',
        nodes: [{ name: '基础物理', path: 'foundations/physics' }],
      },
      {
        level: '核心',
        nodes: [
          { name: '高等物理（普通物理）', path: 'intermediate/advanced-physics' },
          { name: '数学方法（引用数学树）', path: 'intermediate/advanced-math', tag: 'ref' },
        ],
      },
      {
        level: '进阶',
        nodes: [
          { name: '理论力学', path: 'intermediate/theoretical-mechanics' },
          { name: '电动力学', path: 'intermediate/electrodynamics' },
          { name: '量子力学', path: 'intermediate/quantum-mechanics' },
          { name: '统计力学与热力学', path: 'intermediate/statistical-mechanics' },
        ],
      },
      {
        level: '专业',
        nodes: [
          { name: '光学工程（引用工程树）', path: 'engineering/optical-engineering', tag: 'ref' },
          { name: '天文学（引用地球空间树）', path: 'foundations/astronomy', tag: 'ref' },
          { name: '相对论', path: 'intermediate/relativity' },
          { name: '粒子物理', path: 'advanced/particle-physics' },
          { name: '凝聚态物理', path: 'advanced/condensed-matter-physics' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '量子计算（引用计算机树）', path: 'advanced/quantum-computing', tag: 'ref' },
          { name: '量子场论', path: 'advanced/quantum-field-theory' },
          { name: '天体物理', path: 'advanced/astrophysics' },
          { name: '宇宙学', path: 'advanced/cosmology' },
        ],
      },
    ],
  },

  {
    id: 'chemistry',
    name: '化学知识树',
    desc: '从元素与反应出发，经四大化学到物质结构与化学反应机理。',
    branches: [
      {
        level: '基础',
        nodes: [{ name: '化学', path: 'foundations/chemistry' }],
      },
      {
        level: '核心',
        nodes: [
          { name: '无机化学', path: 'intermediate/inorganic-chemistry' },
          { name: '有机化学', path: 'intermediate/organic-chemistry' },
          { name: '物理化学', path: 'intermediate/physical-chemistry' },
          { name: '分析化学', path: 'intermediate/analytical-chemistry' },
        ],
      },
      {
        level: '进阶',
        nodes: [
          { name: '结构化学', path: 'intermediate/structural-chemistry' },
          { name: '量子化学', path: 'advanced/quantum-chemistry' },
          { name: '高分子化学', path: 'engineering/materials-science', tag: 'ref' },
          { name: '化学生物学', path: 'advanced/chemical-biology' },
        ],
      },
      {
        level: '专业',
        nodes: [
          { name: '化学工程（引用工程树）', path: 'engineering/chemical-engineering', tag: 'ref' },
          { name: '材料科学与工程（引用工程树）', path: 'engineering/materials-science', tag: 'ref' },
          { name: '药物化学（引用医学树）', path: 'life/pharmacy', tag: 'ref' },
          { name: '环境化学', path: 'engineering/environmental-engineering', tag: 'ref' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '计算化学', path: 'advanced/computational-chemistry' },
          { name: '电化学与储能', path: 'advanced/electrochemistry-energy-storage' },
        ],
      },
    ],
  },

  {
    id: 'life-science',
    name: '生命科学树',
    desc: '从细胞与遗传出发，经生理、神经、生态到生物技术与合成生物。',
    branches: [
      {
        level: '基础',
        nodes: [{ name: '生物', path: 'foundations/biology' }],
      },
      {
        level: '核心',
        nodes: [
          { name: '分子生物学', path: 'intermediate/molecular-biology' },
          { name: '细胞生物学', path: 'intermediate/cell-biology' },
          { name: '遗传学', path: 'intermediate/genetics' },
          { name: '进化论', path: 'intermediate/evolution' },
        ],
      },
      {
        level: '进阶',
        nodes: [
          { name: '神经科学', path: 'life/neuroscience' },
          { name: '生态学', path: 'life/ecology' },
          { name: '认知科学', path: 'foundations/cognitive-science' },
          { name: '生理学', path: 'life/basic-medicine', tag: 'ref' },
          { name: '微生物学与免疫学', path: 'life/basic-medicine', tag: 'ref' },
        ],
      },
      {
        level: '专业',
        nodes: [
          { name: '生物技术与生物工程', path: 'life/biotechnology' },
          { name: '生物信息学', path: 'life/bioinformatics' },
          { name: '合成生物学（引用交叉树）', path: 'frontier/synthetic-biology', tag: 'ref' },
          { name: '医学与健康（引用医学树）', path: 'life/basic-medicine', tag: 'ref' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '基因组学与精准医学', path: 'advanced/genomics-precision-medicine' },
          { name: '计算神经科学', path: 'advanced/computational-neuroscience' },
        ],
      },
    ],
  },

  {
    id: 'earth-space',
    name: '地球与空间科学树',
    desc: '从天文与地质出发，经大气、海洋到环境与遥感测绘。',
    branches: [
      {
        level: '基础',
        nodes: [
          { name: '天文学', path: 'foundations/astronomy' },
          { name: '地球科学', path: 'foundations/earth-science' },
        ],
      },
      {
        level: '核心',
        nodes: [
          { name: '普通地质学', path: 'foundations/earth-science', tag: 'ref' },
          { name: '地理学（自然地理）', path: 'foundations/physical-geography' },
          { name: '大气科学', path: 'foundations/atmospheric-science' },
          { name: '海洋科学', path: 'foundations/oceanography' },
        ],
      },
      {
        level: '专业',
        nodes: [
          { name: '遥感科学与技术', path: 'frontier/remote-sensing' },
          { name: '测绘科学与技术（引用工程树）', path: 'engineering/surveying-mapping', tag: 'ref' },
          { name: '环境科学与工程（引用工程树）', path: 'engineering/environmental-engineering', tag: 'ref' },
          { name: '地质资源与地质工程（引用工程树）', path: 'engineering/geological-engineering', tag: 'ref' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '地球系统科学', path: 'advanced/earth-system-science' },
          { name: '行星科学', path: 'advanced/planetary-science' },
        ],
      },
    ],
  },

  {
    id: 'computer-ai',
    name: '计算机与 AI 知识树',
    desc: '从数学物理基础，经编程、数据结构与系统，到机器学习、深度学习与大模型全链路。',
    branches: [
      {
        level: '基础（引用）',
        nodes: [
          { name: '离散数学（数学树）', path: 'intermediate/discrete-math', tag: 'ref' },
          { name: '概率论与数理统计（数学树）', path: 'intermediate/probability', tag: 'ref' },
          { name: '线性代数（数学树）', path: 'intermediate/linear-algebra', tag: 'ref' },
          { name: '基础物理（物理树）', path: 'foundations/physics', tag: 'ref' },
        ],
      },
      {
        level: '编程与数据',
        nodes: [
          { name: '程序设计语言', path: 'cs/programming-languages' },
          { name: '数据结构', path: 'cs/data-structures' },
          { name: '算法设计与分析', path: 'cs/algorithms' },
        ],
      },
      {
        level: '计算机系统',
        nodes: [
          { name: '数字逻辑', path: 'cs/digital-logic' },
          { name: '计算机组成原理', path: 'cs/computer-organization' },
          { name: '计算机体系结构', path: 'cs/computer-architecture' },
          { name: '操作系统', path: 'cs/os' },
          { name: '计算机网络', path: 'cs/computer-networks' },
          { name: '编译原理', path: 'cs/compilers' },
          { name: '数据库', path: 'cs/database' },
        ],
      },
      {
        level: '系统与软件',
        nodes: [
          { name: '分布式系统', path: 'cs/distributed-systems' },
          { name: '高性能计算', path: 'cs/high-performance-computing' },
          { name: '云计算', path: 'cs/cloud-computing' },
          { name: '计算机图形学', path: 'cs/computer-graphics' },
          { name: '密码学与信息安全', path: 'cs/cryptography-security' },
          { name: '软件工程', path: 'cs/software-engineering' },
          { name: '区块链', path: 'cs/blockchain' },
        ],
      },
      {
        level: '机器学习与深度学习',
        nodes: [
          { name: '机器学习', path: 'advanced/machine-learning' },
          { name: '深度学习', path: 'advanced/deep-learning' },
          { name: '强化学习', path: 'advanced/reinforcement-learning' },
          { name: '图神经网络', path: 'advanced/graph-neural-networks' },
          { name: '生成模型', path: 'advanced/generative-models' },
        ],
      },
      {
        level: 'AI 各模态',
        nodes: [
          { name: '自然语言处理', path: 'advanced/nlp' },
          { name: '计算机视觉', path: 'advanced/computer-vision' },
          { name: '语音技术', path: 'advanced/speech' },
          { name: '多模态学习', path: 'advanced/multimodal-learning' },
          { name: '推荐系统', path: 'advanced/recommender-systems' },
          { name: '信息检索', path: 'advanced/information-retrieval' },
        ],
      },
      {
        level: '大模型',
        nodes: [
          { name: '大模型原理', path: 'advanced/llm-principles' },
          { name: '大模型微调', path: 'advanced/llm-finetuning' },
          { name: '大模型部署', path: 'advanced/llm-deployment' },
          { name: '大模型量化', path: 'advanced/llm-deployment', tag: 'ref' },
          { name: '大模型推理与 KV Cache', path: 'advanced/llm-deployment', tag: 'ref' },
        ],
      },
      {
        level: 'AI 前沿',
        nodes: [
          { name: '具身智能', path: 'advanced/embodied-ai' },
          { name: '自动驾驶', path: 'advanced/autonomous-driving' },
          { name: 'AI 智能体', path: 'advanced/ai-agents' },
          { name: 'AI 基础设施', path: 'advanced/ai-infra' },
          { name: 'AI 安全与对齐', path: 'advanced/ai-safety' },
          { name: 'AI for Science', path: 'advanced/ai4science' },
          { name: '量子计算', path: 'advanced/quantum-computing' },
          { name: '本体论（知识表示）', path: 'advanced/ontology' },
        ],
      },
    ],
  },

  {
    id: 'engineering',
    name: '工程技术树',
    desc: '从数理基础到全部工科主干学科，含设计、制造与系统集成。',
    branches: [
      {
        level: '基础（引用）',
        nodes: [
          { name: '高等数学（数学树）', path: 'intermediate/advanced-math', tag: 'ref' },
          { name: '高等物理（物理树）', path: 'intermediate/advanced-physics', tag: 'ref' },
        ],
      },
      {
        level: '力学与机械',
        nodes: [
          { name: '机械工程', path: 'engineering/mechanical-engineering' },
          { name: '材料科学与工程', path: 'engineering/materials-science' },
          { name: '冶金工程', path: 'engineering/metallurgical-engineering' },
        ],
      },
      {
        level: '电气与电子',
        nodes: [
          { name: '电气工程', path: 'engineering/electrical-engineering' },
          { name: '电子科学与技术', path: 'engineering/electronic-science' },
          { name: '微电子与集成电路', path: 'engineering/microelectronics' },
          { name: '信息与通信工程', path: 'engineering/communications' },
          { name: '控制科学与工程', path: 'engineering/control-engineering' },
          { name: '光学工程', path: 'engineering/optical-engineering' },
          { name: '仪器科学与技术', path: 'engineering/instrumentation' },
        ],
      },
      {
        level: '化工与材料加工',
        nodes: [
          { name: '化学工程', path: 'engineering/chemical-engineering' },
          { name: '轻工技术与工程', path: 'engineering/light-industry' },
          { name: '纺织科学与工程', path: 'engineering/textile-engineering' },
        ],
      },
      {
        level: '土木与建筑',
        nodes: [
          { name: '土木工程', path: 'engineering/civil-engineering' },
          { name: '水利工程', path: 'engineering/water-conservancy' },
          { name: '城乡规划学', path: 'engineering/urban-planning' },
          { name: '风景园林学', path: 'engineering/landscape-architecture' },
        ],
      },
      {
        level: '交通与运输',
        nodes: [
          { name: '交通运输工程', path: 'engineering/transportation-engineering' },
          { name: '船舶与海洋工程', path: 'engineering/naval-architecture' },
          { name: '航空航天', path: 'engineering/aerospace-engineering' },
        ],
      },
      {
        level: '能源与资源',
        nodes: [
          { name: '动力工程及工程热物理', path: 'engineering/energy-power' },
          { name: '核科学与技术', path: 'engineering/nuclear-engineering' },
          { name: '矿业工程', path: 'engineering/mining-engineering' },
          { name: '石油与天然气工程', path: 'engineering/petroleum-gas' },
          { name: '地质资源与地质工程', path: 'engineering/geological-engineering' },
        ],
      },
      {
        level: '环境与安全',
        nodes: [
          { name: '环境科学与工程', path: 'engineering/environmental-engineering' },
          { name: '安全科学与工程', path: 'engineering/safety-engineering' },
          { name: '兵器科学与技术', path: 'engineering/military-engineering' },
        ],
      },
      {
        level: '交叉工程',
        nodes: [
          { name: '生物医学工程', path: 'engineering/biomedical-engineering' },
          { name: '农业工程', path: 'engineering/agricultural-engineering' },
          { name: '林业工程', path: 'engineering/forestry-engineering' },
          { name: '测绘科学与技术', path: 'engineering/surveying-mapping' },
        ],
      },
    ],
  },

  {
    id: 'medicine',
    name: '医学与健康树',
    desc: '从基础医学与人体生理，到临床各科、药学与公共卫生。',
    branches: [
      {
        level: '基础（引用）',
        nodes: [
          { name: '生物学（生命科学树）', path: 'foundations/biology', tag: 'ref' },
          { name: '化学（化学树）', path: 'foundations/chemistry', tag: 'ref' },
        ],
      },
      {
        level: '基础医学',
        nodes: [
          { name: '基础医学（解剖/生理/生化/病理/药理/免疫/微生物）', path: 'life/basic-medicine' },
          { name: '神经科学（生命科学树）', path: 'life/neuroscience', tag: 'ref' },
        ],
      },
      {
        level: '临床医学',
        nodes: [
          { name: '临床医学（诊断/内/外/妇产/儿/神经/精神）', path: 'life/clinical-medicine' },
          { name: '医学技术', path: 'life/medical-technology' },
          { name: '康复医学', path: 'life/rehabilitation-medicine' },
        ],
      },
      {
        level: '药学与口腔',
        nodes: [
          { name: '药学', path: 'life/pharmacy' },
          { name: '口腔医学', path: 'life/stomatology' },
          { name: '护理学', path: 'life/nursing' },
        ],
      },
      {
        level: '中医药',
        nodes: [
          { name: '中医学', path: 'life/traditional-chinese-medicine' },
          { name: '中药学与中药鉴定', path: 'life/traditional-chinese-pharmacy' },
          { name: '中西医结合', path: 'life/integrated-medicine' },
        ],
      },
      {
        level: '公共卫生',
        nodes: [
          { name: '公共卫生与预防医学', path: 'life/public-health' },
          { name: '体育科学（健康）', path: 'life/sports-science' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '生物医学工程（工程树）', path: 'engineering/biomedical-engineering', tag: 'ref' },
          { name: '兽医学（引用农学树）', path: 'life/veterinary', tag: 'ref' },
          { name: '精准医学', path: 'advanced/precision-medicine' },
        ],
      },
    ],
  },

  {
    id: 'agriculture',
    name: '农学树',
    desc: '从植物生理与遗传到作物、畜牧、水产、林草与食品加工。',
    branches: [
      {
        level: '基础（引用）',
        nodes: [
          { name: '植物学与植物生理（生命科学树）', path: 'foundations/biology', tag: 'ref' },
          { name: '土壤学', path: 'life/agriculture', tag: 'ref' },
        ],
      },
      {
        level: '种植',
        nodes: [
          { name: '农学（作物栽培/植保/园艺/土壤）', path: 'life/agriculture' },
          { name: '草学', path: 'life/grassland-science' },
        ],
      },
      {
        level: '养殖',
        nodes: [
          { name: '畜牧学', path: 'life/animal-husbandry' },
          { name: '兽医学', path: 'life/veterinary' },
          { name: '水产学', path: 'life/aquaculture-fisheries' },
        ],
      },
      {
        level: '林与食品',
        nodes: [
          { name: '林学', path: 'life/forestry' },
          { name: '食品科学与工程', path: 'life/food-science' },
        ],
      },
      {
        level: '工程（引用工程树）',
        nodes: [
          { name: '农业工程', path: 'engineering/agricultural-engineering', tag: 'ref' },
          { name: '林业工程', path: 'engineering/forestry-engineering', tag: 'ref' },
        ],
      },
    ],
  },

  {
    id: 'social-science',
    name: '社会科学树',
    desc: '从经济与心理基础，到法学、政治、社会、管理与教育。',
    branches: [
      {
        level: '基础（引用）',
        nodes: [
          { name: '经济学基础', path: 'foundations/economics' },
          { name: '心理学基础', path: 'foundations/psychology' },
          { name: '逻辑学（人文树）', path: 'foundations/logic', tag: 'ref' },
        ],
      },
      {
        level: '经济',
        nodes: [
          { name: '微观经济学', path: 'foundations/economics', tag: 'ref' },
          { name: '宏观经济学', path: 'foundations/economics', tag: 'ref' },
          { name: '博弈论（数学树）', path: 'intermediate/game-theory', tag: 'ref' },
          { name: '金融学', path: 'social/finance' },
          { name: '会计学', path: 'social/business-management', tag: 'ref' },
          { name: '工商管理', path: 'social/business-management' },
        ],
      },
      {
        level: '政治与法律',
        nodes: [
          { name: '政治学', path: 'social/political-science' },
          { name: '法学', path: 'social/law' },
          { name: '国际关系', path: 'social/international-relations' },
          { name: '马克思主义理论', path: 'social/marxist-theory' },
          { name: '军事学', path: 'social/military-science' },
          { name: '公安学', path: 'social/public-security' },
        ],
      },
      {
        level: '社会与人口',
        nodes: [
          { name: '社会学', path: 'social/sociology' },
          { name: '人类学', path: 'social/anthropology' },
          { name: '民族学', path: 'social/ethnology' },
          { name: '人口学', path: 'social/demography' },
          { name: '城市研究', path: 'social/urban-studies' },
        ],
      },
      {
        level: '管理与公共',
        nodes: [
          { name: '管理学', path: 'social/management' },
          { name: '公共管理', path: 'social/public-management' },
          { name: '教育学', path: 'social/education' },
          { name: '新闻传播学', path: 'social/communication' },
        ],
      },
      {
        level: '心理与行为',
        nodes: [
          { name: '心理学应用深化', path: 'social/psychology-deepening' },
          { name: '认知科学（生命树）', path: 'foundations/cognitive-science', tag: 'ref' },
          { name: '行为经济学', path: 'social/behavioral-economics' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '计算社会科学（交叉树）', path: 'frontier/computational-social-science', tag: 'ref' },
          { name: '区域国别学（交叉树）', path: 'frontier/area-studies', tag: 'ref' },
        ],
      },
    ],
  },

  {
    id: 'humanities',
    name: '人文与艺术树',
    desc: '从语言与文学，经历史与哲学，到艺术各门类与宗教文化。',
    branches: [
      {
        level: '基础',
        nodes: [
          { name: '语言学', path: 'humanities/linguistics' },
          { name: '逻辑学', path: 'foundations/logic' },
        ],
      },
      {
        level: '文学与翻译',
        nodes: [
          { name: '中国文学', path: 'humanities/chinese-literature' },
          { name: '外国文学', path: 'humanities/foreign-literature' },
          { name: '文学理论', path: 'humanities/literary-theory' },
          { name: '翻译学', path: 'humanities/translation-studies' },
        ],
      },
      {
        level: '历史与考古',
        nodes: [
          { name: '中国历史', path: 'humanities/chinese-history' },
          { name: '世界历史', path: 'humanities/world-history' },
          { name: '考古学', path: 'humanities/archaeology' },
          { name: '科学史与科学哲学', path: 'foundations/philosophy-of-science' },
        ],
      },
      {
        level: '哲学与思想',
        nodes: [
          { name: '哲学深化', path: 'humanities/philosophy-deepening' },
          { name: '本体论（计算机树）', path: 'advanced/ontology', tag: 'ref' },
          { name: '宗教学', path: 'humanities/religious-studies' },
        ],
      },
      {
        level: '艺术',
        nodes: [
          { name: '艺术史', path: 'humanities/art-history' },
          { name: '音乐', path: 'humanities/music' },
          { name: '舞蹈学', path: 'humanities/dance' },
          { name: '戏剧与影视学', path: 'humanities/drama-film' },
          { name: '设计学', path: 'humanities/design' },
          { name: '建筑', path: 'humanities/architecture-history' },
        ],
      },
      {
        level: '文化与社会',
        nodes: [
          { name: '民俗学', path: 'humanities/folklore' },
          { name: '文化研究', path: 'humanities/cultural-studies' },
          { name: '图书情报与档案管理', path: 'humanities/library-information' },
        ],
      },
      {
        level: '前沿',
        nodes: [
          { name: '数字人文（交叉树）', path: 'frontier/digital-humanities', tag: 'ref' },
          { name: '环境人文', path: 'humanities/environmental-humanities' },
        ],
      },
    ],
  },

  {
    id: 'frontier',
    name: '交叉与前沿树',
    desc: '横跨多学科的复杂性、系统、数据与新兴交叉领域。',
    branches: [
      {
        level: '系统与复杂性',
        nodes: [
          { name: '系统科学', path: 'frontier/systems-science' },
          { name: '复杂性科学', path: 'frontier/complexity-science' },
          { name: '网络科学', path: 'frontier/network-science' },
        ],
      },
      {
        level: '数据与计算',
        nodes: [
          { name: '数据科学', path: 'frontier/data-science' },
          { name: '计算社会科学', path: 'frontier/computational-social-science' },
          { name: '科学计量学', path: 'frontier/scientometrics' },
        ],
      },
      {
        level: '智能交叉',
        nodes: [
          { name: '智能科学与技术', path: 'frontier/intelligent-science' },
          { name: '认知计算', path: 'frontier/cognitive-computing' },
        ],
      },
      {
        level: '生命与环境交叉',
        nodes: [
          { name: '合成生物学', path: 'frontier/synthetic-biology' },
          { name: '遥感科学与技术', path: 'frontier/remote-sensing' },
        ],
      },
      {
        level: '社会与治理',
        nodes: [
          { name: '国家安全学', path: 'frontier/national-security' },
          { name: '区域国别学', path: 'frontier/area-studies' },
          { name: '未来学', path: 'frontier/futurology' },
        ],
      },
      {
        level: '人文交叉',
        nodes: [
          { name: '数字人文', path: 'frontier/digital-humanities' },
          { name: '认知科学（生命树）', path: 'foundations/cognitive-science', tag: 'ref' },
        ],
      },
    ],
  },
]
