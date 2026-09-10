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
  ...markAppendix(fromOutline(csPapers)),
]
