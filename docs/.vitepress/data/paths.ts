import type { SectionId } from './sections'

export interface PathStage {
  name: string
  note: string
  /** [栏, slug] 的有序列表；标题与位置由组件从知识树解析 */
  items: [SectionId, string][]
}

export interface LearnPath {
  id: string
  name: string
  goal: string
  stages: PathStage[]
}

/** 方向一：大模型部署与后训练。跨栏选段，树序即读序。 */
export const llmDeployPath: LearnPath = {
  id: 'llm-deploy',
  name: '大模型部署与后训练',
  goal: '从程序与系统底座出发，经深度学习与 Transformer、后训练与强化学习，到推理系统与部署研究前沿。',
  stages: [
    {
      name: '系统底座',
      note: '计算机栏选段：程序怎么变成进程、缓存与网络长什么样。部署算法的所有约束都来自这一层。',
      items: [
        ['cs', 'compile-link-entry'],
        ['cs', 'stack-heap-static'],
        ['cs', 'pointer-alias'],
        ['cs', 'dram-timing'],
        ['cs', 'pipeline-five-stage'],
        ['cs', 'kernel-user'],
        ['cs', 'socket-nonblock'],
        ['cs', 'app-backpressure'],
        ['cs', 'raft'],
      ],
    },
    {
      name: '深度学习与语言模型基础',
      note: '大模型栏前段：张量、优化、反传，到词元与注意力。',
      items: [
        ['llm', 'tensor-shape-broadcast'],
        ['llm', 'matmul-as-layer'],
        ['llm', 'softmax-numerics'],
        ['llm', 'cross-entropy-mle'],
        ['llm', 'sgd-learning-rate'],
        ['llm', 'momentum-adaptive-lr'],
        ['llm', 'chain-rule-graph'],
        ['llm', 'why-residual'],
        ['llm', 'why-layernorm'],
        ['llm', 'token-as-discrete-unit'],
      ],
    },
    {
      name: '注意力与架构',
      note: 'SDPA 到位置编码与 KV 权衡：部署侧所有优化的作用对象。',
      items: [
        ['llm', 'sdpa'],
        ['llm', 'mha'],
        ['llm', 'causal-self-attention'],
        ['llm', 'rope'],
        ['llm', 'gqa'],
        ['llm', 'sliding-window-attention'],
        ['llm', 'mamba'],
      ],
    },
    {
      name: '后训练与强化学习',
      note: 'SFT 之后的世界：偏好、奖励、策略梯度与信任域，最后收在自训练闭环。',
      items: [
        ['llm', 'instruction-data-lineage'],
        ['llm', 'reward-model'],
        ['llm', 'rlhf-pipeline'],
        ['llm', 'mdp-five-tuple'],
        ['llm', 'policy-gradient-theorem'],
        ['llm', 'baselines-advantage'],
        ['llm', 'trust-region-monotone'],
        ['llm', 'ppo-clip-view'],
        ['llm', 'rl-foundations-map'],
        ['llm', 'grpo'],
        ['llm', 'rlvr'],
        ['llm', 'self-improvement-loop'],
        ['llm', 'closure-metrics'],
      ],
    },
    {
      name: '推理与部署系统',
      note: '算账先行：prefill/decode 两阶段、KV 字节、投机解码与量化，再看通信与集群。',
      items: [
        ['llm', 'prefill-compute'],
        ['llm', 'kv-cache-size-math'],
        ['llm', 'paged-attention'],
        ['llm', 'speculative-decoding'],
        ['llm', 'gptq'],
        ['llm', 'pretrain-comm'],
      ],
    },
    {
      name: '评测与前沿',
      note: '评测协议的坑，再加附录里的推理系统与硬件 2026 对照。',
      items: [
        ['llm', 'perplexity-eval-pitfalls'],
        ['llm', 'mt-bench'],
      ],
    },
  ],
}

/** 方向二：量化投资研究。金融栏做理论底座，量化栏走到研究终点。 */
export const quantResearchPath: LearnPath = {
  id: 'quant-research',
  name: '量化投资研究',
  goal: '经济学与计量打底，从订单簿读到因子与衍生品，用回测纪律约束自己，最后走完「从问题到组合」的全流程。',
  stages: [
    {
      name: '理论与计量底座',
      note: '金融栏选段：优化的语言、均衡的直觉、识别的纪律。',
      items: [
        ['econ', 'euclidean-open-set'],
        ['econ', 'lagrange-multiplier'],
        ['econ', 'preference-choice'],
        ['econ', 'welfare-theorems'],
        ['econ', 'subgame-perfect'],
        ['econ', 'potential-outcomes'],
        ['econ', 'ols-gauss-markov'],
        ['econ', 'iv-weak-instruments'],
        ['econ', 'difference-in-differences'],
        ['econ', 'clustered-se'],
      ],
    },
    {
      name: '随机分析与市场结构',
      note: '量化栏开局：路径与积分，然后是订单簿的世界。',
      items: [
        ['quant', 'brownian-motion-paths'],
        ['quant', 'ito-lemma'],
        ['quant', 'lob-structure'],
        ['quant', 'order-types'],
        ['quant', 'spread-decomposition'],
        ['quant', 'event-time'],
        ['quant', 'market-fragmentation'],
        ['quant', 'kyle-model'],
        ['quant', 'avellaneda-stoikov'],
      ],
    },
    {
      name: '因子、定价与统计',
      note: '截面与时间序列：从 Markowitz 到高频计量。',
      items: [
        ['quant', 'markowitz'],
        ['quant', 'capm'],
        ['quant', 'fama-macbeth'],
        ['quant', 'garch'],
        ['quant', 'har-rv-forecast'],
        ['quant', 'ofi-toxicity'],
      ],
    },
    {
      name: '衍生品与做市',
      note: '从 BSM 到曲面与对冲频率：定价与对冲的日常工作层。',
      items: [
        ['quant', 'bsm'],
        ['quant', 'arb-free-iv'],
        ['quant', 'svi-ssvi'],
        ['quant', 'greeks-hedge'],
        ['quant', 'delta-hedge-freq'],
      ],
    },
    {
      name: '研究纪律',
      note: '这一段是量化研究的职业道德：把自由度锁起来，把检验做足。',
      items: [
        ['quant', 'pbo'],
        ['quant', 'cscv'],
        ['quant', 'white-reality-check'],
        ['quant', 'purge-embargo'],
        ['quant', 'triple-barrier'],
        ['quant', 'research-sim-prod'],
        ['quant', 'late-data-recon'],
      ],
    },
    {
      name: '组合、执行与风险',
      note: '从信号到净值：目标波动、最优执行、风险度量与 Kelly 上界。',
      items: [
        ['quant', 'vol-targeting'],
        ['quant', 'sqrt-impact'],
        ['quant', 'almgren-chriss'],
        ['quant', 'var-methods'],
        ['quant', 'expected-shortfall'],
        ['quant', 'fractional-kelly-bound'],
      ],
    },
    {
      name: '终点：从问题到组合',
      note: '主干收束课：把以上全部走成一次端到端研究。',
      items: [
        ['quant', 'research-question-origination'],
        ['quant', 'hypothesis-preregistration'],
        ['quant', 'data-acquisition-audit'],
        ['quant', 'signal-construction-sandbox'],
        ['quant', 'single-signal-eval'],
        ['quant', 'multitest-discipline-cap'],
        ['quant', 'portfolio-integration-cap'],
        ['quant', 'execution-cost-forecast-cap'],
        ['quant', 'risk-overlay-cap'],
        ['quant', 'paper-to-live-staging'],
        ['quant', 'live-monitoring-handoff'],
        ['quant', 'research-postmortem'],
      ],
    },
  ],
}

export const learnPaths: LearnPath[] = [llmDeployPath, quantResearchPath]

export function getPath(id: string): LearnPath | null {
  return learnPaths.find((p) => p.id === id) ?? null
}
