import type { Outline } from './schema'

/** 光刻栏前置：电磁/几何光学，以及为何要图形化。 */
export const lithoFoundations: Outline[] = [
  [
    '电磁与几何光学',
    [
      [
        '场与波',
        [
          [
            '从麦克斯韦到平面波',
            [
              '麦克斯韦方程到波动方程|maxwell-to-wave',
              '平面波与波矢|plane-wave-k',
              '偏振：线、圆与椭圆|polarization-states',
              '菲涅尔反射与透射|fresnel-coefficients',
              '布鲁斯特角与全内反射|brewster-tir',
              '薄膜干涉|thin-film-interference',
            ],
          ],
        ],
      ],
      [
        '几何成像',
        [
          [
            '透镜与光瞳',
            [
              '薄透镜成像方程|thin-lens-equation',
              '焦距、物距与横向放大|focal-magnification',
              '光阑、光瞳与视场|stop-pupil-fov',
              '几何像差一览|geometric-aberration-tour',
              '亮度、照度与光学扩展量预备|photometry-etendue-prep',
              '相干长度与光源带宽预备|coherence-length-prep',
              '标量衍射的适用边界|scalar-diffraction-bound',
            ],
          ],
        ],
      ],
    ],
  ],
  [
    '器件与图形化的由来',
    [
      [
        '为什么要做图形',
        [
          [
            '从 MOSFET 到节距',
            [
              'MOSFET 与栅长|mosfet-gate-length',
              'CMOS 反相器为什么成对|cmos-inverter-pair',
              '互连半节距与金属层|interconnect-halfpitch',
              '节点名不再等于栅长|node-vs-gate-length',
              '接触孔、通孔与线|contact-via-line',
              '多重图形化之前的单次曝光极限|single-expose-limit',
              '前道对后道图形化|feol-beol-patterning',
              '良率、缺陷密度与芯片面积|yield-defect-die-area',
            ],
          ],
        ],
      ],
    ],
  ],
]
