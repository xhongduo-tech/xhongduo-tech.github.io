import type { Outline } from './schema'

/** 大模型栏第一课：在词表与 SDPA 之前补深度学习缺口。 */
export const llmFoundations: Outline = [
  '深度学习基础',
  [
    [
      '张量与目标',
      [
        [
          '从向量到损失',
          [
            '张量、形状与广播|tensor-shape-broadcast',
            '矩阵乘作为一层|matmul-as-layer',
            '范数、余弦与投影|norm-cosine-projection',
            'Softmax 与数值稳定|softmax-numerics',
            '交叉熵与极大似然|cross-entropy-mle',
            '对数空间与 underflow|logspace-underflow',
            '下一词分类|clm-as-next-token',
          ],
        ],
        [
          '一阶优化',
          [
            '梯度下降与学习率|sgd-learning-rate',
            '小批量噪声|minibatch-noise',
            '动量与自适应学习率|momentum-adaptive-lr',
            '验证集与泛化缺口|val-gen-gap',
            '过拟合、权重衰减与早停|overfit-wd-earlystop',
          ],
        ],
      ],
    ],
    [
      '反向传播',
      [
        [
          '链式法则',
          [
            '链式法则与计算图|chain-rule-graph',
            '反向传播一行一层|backprop-layerwise',
            '自动微分的正向与反向模式|autograd-two-modes',
            '梯度消失与爆炸|vanish-explode-grad',
          ],
        ],
      ],
    ],
    [
      '前馈网络',
      [
        [
          '多层感知机',
          [
            '线性层与偏置|linear-layer-bias',
            '激活：ReLU 与饱和|activation-relu-saturation',
            '初始化为何需要|why-init',
            '参数量：深度与宽度|params-depth-width',
            '残差的动机|why-residual',
            'LayerNorm 的动机|why-layernorm',
            'Dropout 直觉|dropout-intuition',
          ],
        ],
      ],
    ],
    [
      '序列与注意力之前',
      [
        [
          '分布表示',
          [
            '词向量与分布假设|distributional-hypothesis',
            'Word2Vec skip-gram|word2vec-skipgram',
            '负采样|negative-sampling',
            '子词的动机|why-subword',
            '嵌入作为查找|embed-as-lookup-motive',
          ],
        ],
        [
          '循环与对齐',
          [
            'RNN 与长程依赖|rnn-long-range',
            'LSTM 门控|lstm-gating',
            'GRU 对照|gru-vs-lstm',
            'Seq2Seq 编码解码|seq2seq-encode-decode',
            'Teacher forcing|teacher-forcing',
            'Bahdanau 加性注意力|bahdanau-additive-attention',
            '注意力作为内容寻址|attention-content-address',
            '从加性注意力到点积|additive-to-dot-product',
            '位置还没进模型|why-need-position',
          ],
        ],
      ],
    ],
  ],
]
