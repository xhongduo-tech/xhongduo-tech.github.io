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

export const csTree = [
  ...fromOutline([csInfo, csOrg, csArch, csDs, csAlgo, csCompiler, csOs, csNet, csDb, csSec]),
  ...markAppendix(fromOutline(csPapers)),
]
