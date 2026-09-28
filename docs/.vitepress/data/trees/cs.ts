import { fromOutline, markAppendix } from './schema'
import { csInfo } from './cs-info'
import { csOrg } from './cs-org'
import { csArch } from './cs-arch'
import { csDs } from './cs-ds'
import { csAlgo } from './cs-algo'
import { csCompiler } from './cs-compiler'
import { csOs } from './cs-os'
import { csNet } from './cs-net'
import { csDb } from './cs-db'
import { csSec } from './cs-sec'
import { csPapers } from './cs-papers'
import { csDeepDive } from './cs-deepdive'
import { csSupplement } from './cs-supplement'
import { csProgramming, csNumeric } from './cs-foundations'

const [
  theory,
  digital,
  micro,
  dsAdv,
  algoAdv,
  compilerAdv,
  osAdv,
  netAdv,
  dbAdv,
  dist,
  secAdv,
] = csSupplement

export const csTree = [
  ...fromOutline([
    csProgramming,
    csInfo,
    csNumeric,
    csOrg,
    digital,
    csArch,
    micro,
    csDs,
    dsAdv,
    theory,
    csAlgo,
    algoAdv,
    csCompiler,
    compilerAdv,
    csOs,
    osAdv,
    csNet,
    netAdv,
    csDb,
    dbAdv,
    dist,
    csSec,
    secAdv,
  ]),
  ...fromOutline(csDeepDive),
  ...fromOutline([
    [
      '算法拾遗',
      [
        [
          '动态结构',
          [
            '动态树与 LCT|link-cut-tree',
            '欧拉序与森林维护|euler-tour-tree',
          ],
        ],
        [
          '代数与计数',
          [
            'Berlekamp–Massey 递推识别|berlekamp-massey',
            'Kitamasa 求第 k 项|kitamasa',
            '矩阵树定理|matrix-tree-theorem',
            'Lagrange 插值|lagrange-interpolation',
            'Burnside 与 Polya 计数|burnside-counting',
          ],
        ],
        [
          '组合优化',
          [
            'Slope Trick|slope-trick',
            '拟阵理论|matroid-theory',
            'min-plus 卷积|min-plus-conv',
          ],
        ],
      ],
    ],
  ]),
  ...markAppendix(fromOutline(csPapers)),
]
