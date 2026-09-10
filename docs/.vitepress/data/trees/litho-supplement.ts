import type { Outline } from './schema'

/**
 * 光刻补层（2026-09 审视）：接在主干「制程、封装与管制」之后、附录之前。
 * 成像与产线的续篇；不重写瑞利公式。叶子均为待撰写。
 */
export const lithoSupplement: Outline[] = [
  [
    '傅里叶光学与薄膜',
    [
      [
        '变换与传递函数',
        [
          [
            '成像的数学骨架',
            [
              '傅里叶变换对与卷积|fourier-pairs-convolution',
              '采样定理与成像|sampling-theorem-imaging',
              '点扩散函数与 Airy 斑|psf-airy',
              '相干与非相干传递函数|ctf-otf',
              'Sparrow 对 Rayleigh 判据|sparrow-criterion',
              'Abbe 法对 Hopkins 法|abbe-vs-hopkins',
              'SOCS 分解|socs-decomposition',
              '时间相干与空间相干|temporal-spatial-coherence',
              '光源相干性与 speckle|speckle',
              '高 NA 下标量近似失效|scalar-breakdown-high-na',
              '浸没成像的矢量分析|immersion-vector-imaging',
              '像差对 NILS 的敏感度|aberration-nils-sensitivity',
              '焦面倾斜与场曲补偿|focus-tilt-compensation',
            ],
          ],
          [
            '二维图形',
            [
              '接触孔与线端成像|2d-imaging-contacts-lineend',
              '禁止节距|forbidden-pitch',
              '角圆化|corner-rounding',
              '边缘放置误差 EPE|edge-placement-error',
              'EPE 预算|epe-budget',
            ],
          ],
        ],
      ],
      [
        '薄膜与驻波',
        [
          [
            '胶下面的反射',
            [
              '薄膜干涉与反射率|thin-film-reflectivity',
              '摆动曲线|swing-curve',
              '胶厚选择与反射控制|resist-thickness-selection',
              '顶部抗反射 TARC|tarc',
              '多层底部抗反射|multilayer-barc',
              '基底形貌与反射|topography-reflection',
              'EUV 的薄膜效应|euv-thin-film-effects',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '胶物理模型与随机性',
    [
      [
        '曝光与反应',
        [
          [
            '从光子到轮廓',
            [
              'Dill 参数 A、B、C|dill-parameters',
              '曝光动力学与光酸生成|exposure-kinetics-acid',
              'PEB 反应扩散模型|peb-reaction-diffusion',
              '淬灭剂中和动力学|quencher-neutralization',
              '显影速率模型|development-rate-model',
              '集总参数模型|lumped-parameter-model',
              '胶轮廓仿真|resist-profile-simulation',
              '胶的 γ 与对比度|resist-gamma',
              'E₀ 与 E_size|e0-esize',
              '剂量-尺寸曲线|dose-to-size',
              '剂量-焦深耦合|dose-focus-coupling',
              '胶的热流与回流|resist-reflow',
              '玻璃化温度与自由体积|tg-free-volume',
              '胶的老化与保质|resist-shelf-life',
              'PAB 与溶剂残留|pab-solvent',
              '底层与粘附|underlayer-adhesion',
              '聚合物分子量与分布|resist-polymer-mw',
            ],
          ],
        ],
      ],
      [
        '随机性',
        [
          [
            '计数噪声到缺陷',
            [
              '酸与淬灭剂的计数噪声|acid-quencher-counting',
              '二次电子模糊模型|secondary-electron-blur-model',
              '随机性 Monte Carlo 仿真|stochastic-monte-carlo',
              'LER 的 PSD 分析|ler-psd',
              '相关长度与低频粗糙|ler-correlation-length',
              'LER 对器件的影响|ler-device-impact',
              '桥接与断线缺陷|bridge-break-defects',
              '缺陷率与 Z 因子|stochastic-z-factor',
              '随机性与 NILS|stochastics-vs-nils',
              'RLS 三角|rls-tradeoff',
              '随机工艺窗口|stochastic-process-window',
              '化学随机性抑制|chemical-stochastic-mitigation',
              '金属氧化物胶的随机机制|mor-stochastics',
              '干法显影胶|dry-resist-develop',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '涂胶显影与轨道',
    [
      [
        '轨道工艺',
        [
          [
            '曝光前后的每一步',
            [
              '旋涂与厚度均匀性|spin-coating-uniformity',
              '边缘胶珠去除 EBR|ebr-edge-bead',
              '三层堆叠 SOC / SOG|trilayer-soc-sog',
              '软烤与溶剂|soft-bake',
              '显影：搅拌与 puddle|develop-puddle',
              'TMAH 与显影液|tmah-developer',
              '冲洗与表面活性剂|rinse-surfactant',
              '倒塌抑制|collapse-mitigation',
              '去渣|descum',
              '硬烤与 UV 固化|hard-bake-uv-cure',
              '轨道与扫描仪联机|track-scanner-link',
              '热板均匀性|hotplate-uniformity',
              '轨道缺陷来源|track-defect-sources',
              '晶圆边缘曝光|wafer-edge-exposure',
              '背面与斜边清洗|backside-bevel-clean',
              'BARC 开口刻蚀|barc-open-etch',
              '胶去除与灰化|resist-strip-ash',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '曝光机子系统',
    [
      [
        '光源与照明',
        [
          [
            '从激光到狭缝',
            [
              'ArF 准分子激光器|arf-excimer-laser',
              '线宽压窄与带宽|laser-linewidth-narrowing',
              '脉冲能量与重复频率|pulse-energy-rep-rate',
              '激光气体与寿命|laser-gas-lifetime',
              '均光器与复眼|homogenizer-fly-eye',
              '掩模遮挡刀片 REMA|reticle-masking-blades',
              '照明均匀性与狭缝|illumination-uniformity-slit',
              '光瞳测量|pupil-metrology',
              '剂量传感闭环|dose-sensor-loop',
              '光源功率稳定|source-power-stability',
            ],
          ],
        ],
      ],
      [
        '投影物镜',
        [
          [
            '镜头怎么造、怎么坏',
            [
              '折反射物镜设计|catadioptric-lens-design',
              '透镜加热与补偿|lens-heating-compensation',
              'Zernike 漂移与操纵器|aberration-manipulators',
              '透镜致密化与寿命|lens-compaction',
              '熔石英与 CaF₂|fused-silica-caf2',
              '物镜像差测量|lens-aberration-metrology',
              '双折射与偏振像差|birefringence-polarization',
              '物镜污染|lens-contamination',
            ],
          ],
        ],
      ],
      [
        '工件台与传感',
        [
          [
            '纳米级的机械',
            [
              '干涉仪对光栅编码器|interferometer-vs-encoder',
              '调平传感器与空气规|level-sensor-air-gauge',
              '焦点传感与晶圆形貌|focus-sensor-topography',
              '对准传感器 SMASH / ORION|alignment-sensors',
              '对准标记设计|alignment-mark-design',
              '减振与地基|vibration-isolation',
              '工件台加速度与产能|stage-acceleration-throughput',
              'MSD 同步误差|msd-moving-standard-deviation',
              '温控与气流|thermal-airflow-control',
              '晶圆装载与预对准|wafer-load-prealign',
              '掩模库与交换|reticle-library-exchange',
              '机台匹配|tool-matching',
              '稼动率与 MTBF|tool-uptime-mtbf',
              '机台软件与配方|tool-recipe-software',
              '校准周期|scanner-calibration-cycle',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    'EUV 光源、掩模与胶',
    [
      [
        '光源与光学',
        [
          [
            '等离子体到镜面',
            [
              'CO₂ 驱动激光放大链|co2-drive-laser-chain',
              '转换效率的物理|conversion-efficiency-physics',
              '锡等离子体辐射|tin-plasma-emission',
              '光源功率路线 250 W → 1 kW|euv-power-roadmap',
              '收集镜清洁与恢复|collector-cleaning',
              '多层膜粗糙与散射|multilayer-roughness-scatter',
              '镜面形状误差|mirror-figure-error',
              '中频粗糙 MSFR|msfr',
              '反射率与镜数预算|reflectivity-mirror-count',
              '环形场|ring-field',
              '光瞳填充比|pupil-fill-ratio',
              'EUV 照明设置|euv-illumination-settings',
              '氢等离子体与镜面|hydrogen-plasma-mirrors',
              '真空与污染控制|euv-vacuum-contamination',
              'EUV 掩模台与吸盘|euv-reticle-chuck',
              '掩模背面颗粒|reticle-backside-particles',
            ],
          ],
        ],
      ],
      [
        '掩模与胶',
        [
          [
            'EUV 特有的部件',
            [
              '低 n 吸收体|low-n-absorber',
              '相移 EUV 掩模|euv-phase-shift-mask',
              '钌覆盖层|ruthenium-capping',
              '黑边|black-border',
              '掩模 3D 与最佳焦面偏移|mask-3d-best-focus-shift',
              '掩模粗糙度|mask-roughness',
              'EUV 掩模修复|euv-mask-repair',
              '薄膜寿命与更换|pellicle-lifetime',
              '无薄膜运行的风险|pellicle-free-risk',
              'EUV 胶的吸收与厚度|euv-resist-absorption',
              '底层设计与二次电子|underlayer-secondary-electron',
              '高 NA 薄胶的转印|high-na-pattern-transfer',
              'EUV 层数与节点|euv-layers-per-node',
              'EUV 单次曝光极限|euv-single-expose-limit',
              '高 NA 的产能与经济|high-na-throughput-economics',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '掩模制造与数据准备',
    [
      [
        '数据与写入',
        [
          [
            '从版图到掩模',
            [
              'GDSII 与 OASIS|gdsii-oasis',
              '版图分层与流程|layout-layers-flow',
              '掩模数据准备 MDP|mask-data-prep',
              '分割 fracturing|fracturing',
              '掩模工艺修正 MPC|mpc',
              '曲线掩模数据量|curvilinear-data-volume',
              '写入时间与产能|mask-write-time',
              '电子束邻近校正细节|mask-pec-detail',
              '写入网格与 shot 数|write-grid-shot-count',
              '掩模 CD 均匀性|mask-cdu',
              '掩模配准|mask-registration',
            ],
          ],
          [
            '坯、膜与检验',
            [
              '掩模坯与 LTEM 基板|mask-blank-ltem',
              '铬与 MoSi 膜层|chrome-mosi-films',
              '掩模刻蚀与侧壁|mask-etch-sidewall',
              '掩模显影与清洗|mask-develop-clean',
              'die-to-die 与 die-to-database|mask-inspection-d2d-d2db',
              '掩模修复：电子束与纳米机械|mask-repair-ebeam',
              '薄膜安装与检验|pellicle-mount-inspect',
              'AIMS 掩模鉴定|aims-mask-qualification',
              '掩模雾化|mask-haze',
              '掩模成本与周期|mask-cost-cycle',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '仿真、模型与验证',
    [
      [
        '光刻仿真',
        [
          [
            '严格解与紧凑模型',
            [
              'FDTD 掩模仿真|fdtd-mask',
              'RCWA|rcwa',
              '域分解法|domain-decomposition-mask',
              '紧凑胶模型校准|resist-model-calibration',
              '测试图形与 gauge|test-patterns-gauges',
              '模型残差分析|model-residual-analysis',
              '三维胶模型|3d-resist-model',
              '刻蚀模型校准|etch-model-calibration',
              '全芯片仿真的计算量|full-chip-simulation-cost',
            ],
          ],
          [
            '验证与设计规则',
            [
              '光刻友好设计 LFD|lfd',
              '图形匹配|pattern-matching',
              '工艺窗口鉴定 PWQ|pwq',
              'LRC / ORC|lrc-orc',
              'ML 热点预测|ml-hotspot-prediction',
              '分解着色冲突|decomposition-coloring-conflict',
              '设计规则的起源|design-rule-origin',
              '最小面积与 tip-to-tip|min-area-tip-to-tip',
              '标准单元与轨道高度|standard-cell-track-height',
              '缩放助推器|scaling-boosters',
              'SRAM 单元图形化|sram-cell-patterning',
              '金属与 via 协同|metal-via-co-optimization',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '集成、控制与良率',
    [
      [
        '刻蚀与转印',
        [
          [
            '把胶的形状变成硅的形状',
            [
              '等离子体基础|plasma-basics',
              '选择比与 ARDE|selectivity-arde',
              'LER 转移与平滑|ler-transfer-smoothing',
              '硬掩模 SiN / TiN|hardmask-sin-tin',
              '各向异性与侧壁钝化|anisotropy-passivation',
              '刻蚀负载效应|etch-loading-effect',
              '刻蚀偏置与 OPC 回环|etch-bias-opc-loop',
              'CMP 与形貌|cmp-topography',
              '平坦化与焦深|planarization-dof',
              '关键层对准|critical-layer-alignment',
            ],
          ],
        ],
      ],
      [
        '过程控制',
        [
          [
            '闭环',
            [
              '先进过程控制 APC|apc',
              '前馈与反馈 run-to-run|feedforward-feedback-r2r',
              '抽样计划|sampling-plan',
              'SPC 与控制图|spc-control-chart',
              '套刻控制回路|overlay-control-loop',
              '高阶场内校正|high-order-field-correction',
              '批次分派与重工|lot-disposition-rework',
              '计量匹配|metrology-matching',
              '边缘场与部分场|edge-field-partial',
              '晶圆形变与应力|wafer-distortion-stress',
              '机台对机台套刻|tool-to-tool-overlay',
              '厂级数据闭环|fab-data-loop',
              '虚拟计量|virtual-metrology',
            ],
          ],
        ],
      ],
      [
        '良率',
        [
          [
            '缺陷与学习曲线',
            [
              'Poisson 与负二项良率模型|yield-models',
              '关键面积|critical-area',
              '系统性对随机缺陷|systematic-vs-random-defects',
              '随机性良率|stochastic-yield',
              '缺陷检测：光学与电子束|defect-inspection-optical-ebeam',
              '缺陷复检与分类|defect-review-classification',
              '良率学习曲线|yield-learning-curve',
              '参数良率与 CD 分布|parametric-yield-cd',
              '可靠性与图形化缺陷|reliability-patterning-defects',
              '失效分析 TEM / FIB|failure-analysis-tem-fib',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '路线图、经济与替代',
    [
      [
        '节点与经济',
        [
          [
            '光刻怎么算钱',
            [
              '节点命名与实际尺寸|node-naming',
              'CPP 与 MMP|cpp-mmp',
              '晶体管密度度量|transistor-density-metric',
              'IRDS 路线图|irds-roadmap',
              '每层成本与每晶圆成本|cost-per-layer-wafer',
              'EUV 对多重曝光的成本交叉|euv-vs-multipattern-cost-crossover',
              '机台价格与折旧|tool-price-depreciation',
              '产能规划与瓶颈|capacity-planning-bottleneck',
              '晶圆厂资本开支|fab-capex',
              '摩尔定律的光刻视角|moore-litho-view',
              '供应链集中与单一来源|supply-chain-single-source',
              'PFAS 与光刻胶管制|pfas-resist-regulation',
              '代工路线对比|foundry-roadmap-compare',
              '存储器光刻：DRAM 与 NAND|memory-litho-dram-nand',
              '3D NAND 与光刻负担转移|3d-nand-litho-shift',
            ],
          ],
        ],
      ],
      [
        '替代与延伸',
        [
          [
            '光刻的其他形态',
            [
              '多束电子束直写|multibeam-direct-write',
              '干涉光刻|interference-lithography',
              'X 射线与近场|xray-near-field',
              '激光直写与封装光刻|laser-direct-write-packaging',
              '面板级封装光刻|panel-level-litho',
              '显示光刻|display-litho',
              '封装用步进器|packaging-steppers',
              '混合键合的对准|hybrid-bonding-alignment',
              '光子学图形化|photonics-patterning',
              'MEMS 图形化|mems-patterning',
              '灰度光刻|grayscale-litho',
              '双光子 3D 打印|two-photon-3d',
              '生物芯片光刻|biochip-litho',
              '自由电子激光 EUV 源|fel-euv-source',
              'Hyper-NA 之后的物理极限|post-hyper-na-limits',
            ],
          ],
        ],
      ],
    ],
  ],
]
