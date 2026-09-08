import { fromOutline, markAppendix, type Outline } from './schema'

const outline: Outline[] = [
  [
    '成像与分辨率',
    [
      [
        '波动光学',
        [
          [
            '从波到空中像',
            [
              '单色波、折射率与光程|em-wave-index',
              '衍射与空间频率|diffraction-spatial-freq',
              '透镜与数值孔径|lens-na',
              '部分相干照明|partial-coherence',
              '空中像与对比度|aerial-image-contrast',
              '焦深与瑞利焦深|depth-of-focus',
              '偏振与矢量成像|vector-imaging',
            ],
          ],
        ],
      ],
      [
        '产线判据',
        [
          [
            '分辨率',
            [
              '瑞利判据：CD = k₁ λ / NA|rayleigh-litho',
              'k₁ 窗口与分辨率增强|k1-factor-window',
              '波长台阶：g/i 线、KrF、ArF、EUV|litho-wavelengths',
              '掩模、胶、曝光、显影、刻蚀转印|litho-process-flow',
              '套刻 Overlay 与对准|litho-overlay',
              '双工件台 Twinscan 提高产能|twinscan-dual-stage',
              '步进器与扫描投影|stepper-vs-scanner',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '光刻胶与计量',
    [
      [
        '抗蚀剂',
        [
          [
            '化学',
            [
              '化学放大胶|car-resist',
              '酸扩散与线宽粗糙度|acid-diffusion-lwr',
              '显影与衬度曲线|resist-contrast-curve',
              '金属氧化物胶 vs 化学放大胶|euv-resist',
            ],
          ],
        ],
      ],
      [
        '量测',
        [
          [
            '窗口',
            [
              '曝光–散焦工艺窗口|ed-process-window',
              'CD-SEM 与散射测量|cd-sem-scatterometry',
              '套刻标记与衍射套刻|overlay-marks',
              '杂散光与 flare|flare-and-stray',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'DUV 与多重曝光',
    [
      [
        '浸没与分解',
        [
          [
            '193 nm',
            [
              'ArF 浸没：水作介质抬高 NA|arf-immersion',
              '离轴照明与偶极/四极光瞳|off-axis-illumination',
              '相移掩模 PSM|phase-shift-mask',
              '二元掩模与衰减相移|binary-attpsm',
              '掩模 3D 效应|mask-3d-effect',
              'LELE 多次曝光套刻|lele-multipattern',
              'SADP / SAQP 自对准双重/四重图形|sadp-saqp',
              '浸没 DUV + 多重曝光走到 7/5 nm 的代价|duv-multipattern-cost',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'EUV 与 ASML',
    [
      [
        '光源与光学',
        [
          [
            '13.5 nm',
            [
              '真空全反射：Mo/Si 多层膜镜|euv-multilayer-mirror',
              'LPP：CO₂ 激光打锡滴|euv-lpp-tin',
              '预脉冲 + 主脉冲提高转换效率|euv-prepulse',
              '蔡司投影物镜与收集镜|zeiss-euv-optics',
              '氢气流 Dynamic Gas Lock 防污染|euv-hydrogen-dgl',
              'EUV 薄膜 Pellicle|euv-pellicle',
              'EUV 掩模多层与缺陷|euv-mask-defects',
              'NXE：NA 0.33 量产 5/3 nm|asml-nxe',
              'High-NA 0.55 EXE：变形光学与半场|asml-high-na',
              '真空磁浮工件台|euv-maglev-stage',
              '随机效应与光子散粒噪声|euv-stochastics',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '计算光刻与掩模写入',
    [
      [
        '图形修正',
        [
          [
            'OPC 到 ILT',
            [
              '光学邻近修正 OPC|opc',
              '光源掩模协同优化 SMO|smo',
              '逆光刻 ILT 与曲线掩模|ilt-curvilinear',
              '多束电子束写掩模|multibeam-mask-writer',
              'GPU / AI 加速 OPC|computational-litho-gpu',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '制程、封装与管制',
    [
      [
        '前后道',
        [
          [
            '器件与集成',
            [
              'FinFET 到 GAA / nanosheet|finfet-gaa',
              '原子层沉积 ALD 与原子层刻蚀|ald-ale',
              'HBM 堆叠与混合键合|hbm-hybrid-bonding',
              'CoWoS / 2.5D 中介层|cowos-2p5d',
              'Chiplet 与先进封装补光刻极限|chiplet-packaging',
            ],
          ],
        ],
      ],
      [
        '国产与管制',
        [
          [
            '设备与供应链',
            [
              'EUV 出口管制卡住先进逻辑|euv-export-control',
              '国产浸没 DUV 与 28 nm 单次曝光|china-immersion-duv',
              '多重曝光把国产 DUV 往更先进节点推|china-duv-multipattern',
              '国产 EUV 仍处原型|china-euv-prototype',
              '光刻胶、光源、镜头的国产替代|china-litho-supply-chain',
            ],
          ],
        ],
      ],
    ],
  ],
]

const papers: Outline[] = [
  [
    '文献与机台对照',
    [
      [
        '讲义与产品',
        [
          [
            '对照',
            [
              '瑞利判据笔记（Mack）|rayleigh-litho-note',
              'ASML EUV / High-NA 产品线|asml-euv-high-na',
              '计算光刻与 ILT 专文|computational-litho-paper',
            ],
          ],
        ],
      ],
    ],
  ],
]

export const lithoTree = [...fromOutline(outline), ...markAppendix(fromOutline(papers))]
