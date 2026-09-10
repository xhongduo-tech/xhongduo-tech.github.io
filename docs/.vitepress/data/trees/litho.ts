import { fromOutline, markAppendix, type Outline } from './schema'
import { lithoSupplement } from './litho-supplement'
import { lithoFoundations } from './litho-foundations'

const outline: Outline[] = [
  [
    '成像与分辨率',
    [
      [
        '波动光学',
        [
          [
            '波动与衍射',
            [
              '单色波、折射率与光程|em-wave-index',
              '亥姆霍兹与惠更斯–菲涅尔|helmholtz-huygens',
              'Kirchhoff 衍射|kirchhoff-diffraction',
              '菲涅尔与夫琅禾费|fresnel-fraunhofer',
              '衍射与空间频率|diffraction-spatial-freq',
              '阿贝成像原理|abbe-imaging',
              '光瞳滤波与切趾|pupil-apodization',
              '透镜与数值孔径|lens-na',
              '光学扩展量与远心|etendue-telecentricity',
              '远心误差|telecentric-error',
            ],
          ],
          [
            '相干、空中像与焦深',
            [
              '部分相干照明|partial-coherence',
              '相干因子 σ|coherence-sigma',
              'Hopkins TCC|hopkins-tcc',
              '空中像与对比度|aerial-image-contrast',
              'NILS 与 ILS|nils-ils',
              'MTF|mtf-optics',
              '焦深与瑞利焦深|depth-of-focus',
              '离焦作为像差|defocus-as-aberration',
              'Bossung 曲线|bossung-curve',
            ],
          ],
          [
            '偏振与像差',
            [
              '偏振与矢量成像|vector-imaging',
              'TE / TM 对比|te-tm-polarization',
              'Jones 矩阵|jones-polarization',
              'Zernike 像差|zernike-aberrations',
              '球差|spherical-aberration',
              '彗差|coma-aberration',
              '像散与场曲|astigmatism-field',
              '畸变|distortion-field',
              'Strehl 比|strehl-ratio',
              '色差与光源带宽|chromatic-bandwidth',
            ],
          ],
        ],
      ],
      [
        '产线判据',
        [
          [
            '瑞利与邻近',
            [
              '瑞利判据：CD = k₁ λ / NA|rayleigh-litho',
              '半节距对线宽|halfpitch-vs-cd',
              'k₁ 窗口与分辨率增强|k1-factor-window',
              '曝光宽容度|exposure-latitude',
              '掩模误差增强因子 MEEF|meef',
              '光学邻近效应|ope-proximity',
              '疏密偏差|iso-dense-bias',
              '线端缩短|line-end-shortening',
              '波长台阶：g/i 线、KrF、ArF、EUV|litho-wavelengths',
            ],
          ],
          [
            '流程、套刻与产能',
            [
              '掩模、胶、曝光、显影、刻蚀转印|litho-process-flow',
              '显影后对刻蚀后|adi-aei',
              '套刻 Overlay 与对准|litho-overlay',
              '套刻预算分解|overlay-budget',
              '双工件台 Twinscan 提高产能|twinscan-dual-stage',
              '步进器与扫描投影|stepper-vs-scanner',
              '狭缝与扫描平均|slit-scan-average',
              '剂量控制|dose-control',
              '扫描仪产能会计|scanner-throughput',
              'Focus drilling|focus-drilling',
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
            '胶化学',
            [
              '正胶与负胶|positive-vs-negative',
              'DNQ–Novolac 胶|dnq-novolac',
              '化学放大胶|car-resist',
              'PAG 与淬灭剂|pag-quencher',
              '淬灭剂负载|quencher-loading',
              '酸放大器|acid-amplifier',
              'PEB 温度|peb-bake',
              '酸扩散与线宽粗糙度|acid-diffusion-lwr',
              'LWR 与 LCDU|lwr-lcdu',
              '显影与衬度曲线|resist-contrast-curve',
            ],
          ],
          [
            '轮廓、驻波与 EUV 胶',
            [
              'Mack / Notch 溶解模型|dissolution-mack',
              '驻波与 BARC|standing-wave-barc',
              '侧壁角|sidewall-angle',
              '图形倒塌|pattern-collapse',
              '金属氧化物胶 vs 化学放大胶|euv-resist',
              '锡氧胶化学|tin-oxide-resist',
              'EUV 底层|euv-underlayer',
              '二次电子产额|euv-secondary-electron',
              '放气|resist-outgassing',
              '随机缺陷与缺失孔|stochastic-holes',
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
              '焦散矩阵 FEM|fem-matrix',
              'CD 均匀性 CDU|cdu-uniformity',
              'CD-SEM 与散射测量|cd-sem-scatterometry',
              'SEM 收缩与充电|cd-sem-shrink',
              'OCD 与穆勒矩阵|ocd-mueller',
              'AFM 线宽|afm-cd',
              '套刻标记与衍射套刻|overlay-marks',
              'DBO 对 IBO|dbo-ibo',
              '高阶套刻|high-order-overlay',
              'YieldStar 与扫描仪内计量|yieldstar-dbo',
              '杂散光与 flare|flare-and-stray',
              '热点与过程变化|hotspot-pv',
              '刻蚀偏置与 CDU|cdu-etch-bias',
              '显影后缺陷检验|adi-defect-inspect',
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
        '浸没与照明',
        [
          [
            '193 nm 浸没',
            [
              'ArF 浸没：水作介质抬高 NA|arf-immersion',
              '浸没头与水膜|immersion-hood',
              '浸没缺陷：气泡与水印|immersion-defects',
              '面涂与浸出|topcoat-leaching',
              '高 n 浸没液|high-n-immersion',
              '干式层对浸没层|dry-vs-immersion-layer',
              '离轴照明与偶极/四极光瞳|off-axis-illumination',
              '环形、偶极与四极|annular-dipole-quad',
              'FlexRay / 可编程光瞳|flexray-pupil',
              '照明偏振控制|illuminator-polarization',
            ],
          ],
        ],
      ],
      [
        '掩模',
        [
          [
            '透过与三维',
            [
              '相移掩模 PSM|phase-shift-mask',
              '二元掩模与衰减相移|binary-attpsm',
              '交替相移|alt-psm',
              '亚分辨辅助图形 SRAF|sraf-assist',
              '掩模 3D 效应|mask-3d-effect',
              'Kirchhoff 对严格电磁|kirchhoff-vs-emf',
              '掩模阴影|mask-shadowing',
              '掩模 CD 与放置误差|mask-cd-ppe',
              '电子束写掩模邻近效应|ebeam-mask-pec',
              '6 寸掩模|six-inch-reticle',
            ],
          ],
        ],
      ],
      [
        '多重图形化',
        [
          [
            '分解',
            [
              'LELE 多次曝光套刻|lele-multipattern',
              '两次曝光的颜色分解|lele-color-decomposition',
              'LELELE|lelele-triple',
              'SADP / SAQP 自对准双重/四重图形|sadp-saqp',
              '芯轴与侧墙|mandrel-spacer',
              '节距漂移|pitch-walking',
              '奇偶节距|odd-even-pitch',
              '切断掩模|cut-mask-dpt',
              'SAQP 加切断|saqp-plus-cut',
              '浸没 DUV + 多重曝光走到 7/5 nm 的代价|duv-multipattern-cost',
              'EUV 与自对准的混合|euv-sapt-hybrid',
              '互补式 EUV|complementary-euv',
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
        '光源',
        [
          [
            '锡滴',
            [
              'LPP：CO₂ 激光打锡滴|euv-lpp-tin',
              '预脉冲 + 主脉冲提高转换效率|euv-prepulse',
              '锡滴发生器|droplet-generator',
              '锡电离态|tin-ionization',
              '锡碎屑与防护|tin-debris',
              '掠入射收集镜|collector-grazing',
              '收集镜寿命|collector-lifetime',
              '中间焦点 IF|intermediate-focus',
              '功率、剂量与产能|euv-power-throughput',
              'EUV 光源带宽与带外|euv-source-bandwidth',
            ],
          ],
        ],
      ],
      [
        '光学与掩模',
        [
          [
            '多层与缺陷',
            [
              '真空全反射：Mo/Si 多层膜镜|euv-multilayer-mirror',
              '蔡司投影物镜与收集镜|zeiss-euv-optics',
              'EUV flare 与多层散射|euv-flare-ml',
              '斜入射与三维阴影|euv-oblique-shadow',
              '变形光学|anamorphic-projection',
              '氢气流 Dynamic Gas Lock 防污染|euv-hydrogen-dgl',
              '氢起泡|hydrogen-blister',
              '镜面锡沉积|tin-on-mirrors',
              'EUV 薄膜 Pellicle|euv-pellicle',
              '薄膜透过率与热|pellicle-transmission',
              'EUV 掩模多层与缺陷|euv-mask-defects',
              '吸收体材料|euv-absorber-materials',
              '光化检验|actinic-blank-inspect',
              'EUV 吸收与模糊|euv-absorption-blur',
            ],
          ],
        ],
      ],
      [
        '机台',
        [
          [
            'NXE 到 High-NA',
            [
              'NXE：NA 0.33 量产 5/3 nm|asml-nxe',
              'High-NA 0.55 EXE：变形光学与半场|asml-high-na',
              'High-NA 半场拼接|high-na-stitch',
              'High-NA 套刻|high-na-overlay-na',
              'High-NA 胶厚预算|high-na-resist-budget',
              '变形掩模倍率|anamorphic-mask-mag',
              '真空磁浮工件台|euv-maglev-stage',
              '掩模台同步|reticle-stage-sync',
              '掩模夹持|reticle-clamp',
              '随机效应与光子散粒噪声|euv-stochastics',
              '光子散粒噪声预算|photon-shot-budget',
              'EUV 双重曝光|euv-double-pattern',
              'Hyper-NA|hyper-na',
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
            'OPC 与 SMO',
            [
              '光学邻近修正 OPC|opc',
              '规则 OPC 对模型 OPC|rule-vs-model-opc',
              '工艺窗口 OPC|pw-opc',
              '刻蚀模型进 OPC|etch-in-opc',
              '胶紧凑模型|resist-compact-model',
              '光源掩模协同优化 SMO|smo',
              '光瞳图优化|pupil-source-map',
              '波前与 SMO+|smo-plus-wavefront',
              '紧凑模型对严格电磁|compact-vs-rigorous',
            ],
          ],
          [
            'ILT、MRC 与写入',
            [
              '逆光刻 ILT 与曲线掩模|ilt-curvilinear',
              'ILT 正则化|ilt-regularization',
              '掩模规则检查 MRC|mask-rule-check',
              'MRC 修复|mrc-fixup',
              '曲线 MRC|curvilinear-mrc',
              '多束电子束写掩模|multibeam-mask-writer',
              'VSB 对多束|vsb-vs-multibeam',
              'GPU / AI 加速 OPC|computational-litho-gpu',
              '机器学习 OPC|ml-opc',
              '热点修复|hotspot-fix',
              'DTCO|dtco-patterning',
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
              '反应离子刻蚀转印|rie-pattern-transfer',
              '深宽比与刻蚀负荷|etch-loading-ar',
              '接触孔与通孔图形化|contact-hole-patterning',
              '后道波长分层|beol-wavelength-mix',
              '后道 EUV|beol-euv',
              'FinFET 到 GAA / nanosheet|finfet-gaa',
              'nanosheet 间距|nanosheet-spacing',
              '背面供电 BSPDN|backside-power',
              'CFET|cfet',
              '原子层沉积 ALD 与原子层刻蚀|ald-ale',
              '导向自组装 DSA|dsa-dpt',
              '纳米压印对照|nanoimprint-alt',
              'HBM 堆叠与混合键合|hbm-hybrid-bonding',
              'CoWoS / 2.5D 中介层|cowos-2p5d',
              'Chiplet 与先进封装补光刻极限|chiplet-packaging',
            ],
          ],
        ],
      ],
      [
        '对准与热',
        [
          [
            '套刻来源',
            [
              '对准策略与 SPM|alignment-spm',
              '掩模加热|reticle-heating',
              '晶圆热套刻|wafer-thermal-overlay',
              '晶圆吸盘|wafer-chuck',
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
              'Nikon 对 ASML|nikon-vs-asml',
              '国产浸没 DUV 与 28 nm 单次曝光|china-immersion-duv',
              '多重曝光把国产 DUV 往更先进节点推|china-duv-multipattern',
              '国产 EUV 仍处原型|china-euv-prototype',
              '光刻胶、光源、镜头的国产替代|china-litho-supply-chain',
              '掩模厂流程|mask-shop-flow',
              '胶与供应商|resist-vendors',
              '图形化检测设备|patterning-inspection-tools',
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
              'Hopkins 部分相干成像|hopkins-1953-paper',
              'Levinson 光刻原理|levinson-litho-book',
              'van Schoot High-NA 光学|vanschoot-high-na-paper',
              'Goodman 傅里叶光学|goodman-fourier-optics',
              'Naulleau 随机效应|naulleau-stochastics-paper',
            ],
          ],
        ],
      ],
    ],
  ],
]

const [fourier, resistPhys, track, scanner, euvDepth, maskData, sim, integ, roadmap] = lithoSupplement

export const lithoTree = [
  ...fromOutline([
    ...lithoFoundations,
    outline[0],
    fourier,
    outline[1],
    resistPhys,
    track,
    outline[2],
    scanner,
    outline[3],
    euvDepth,
    outline[4],
    maskData,
    sim,
    outline[5],
    integ,
    roadmap,
  ]),
  ...markAppendix(fromOutline(papers)),
]
