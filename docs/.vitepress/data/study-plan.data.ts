import { readFileSync } from 'node:fs'
import path from 'node:path'
import { selectTodayPlan } from './study-plan-utils.mjs'

export interface StudyTask {
  id: string
  title: string
  minutes: number
}

export interface TodayPlan {
  available: boolean
  date: string
  week?: number
  topic?: string
  totalMinutes?: number
  endDate?: string
  tasks: StudyTask[]
}

declare const data: TodayPlan
export { data }

function readPrivatePlan() {
  if (process.env.STUDY_PLAN_JSON?.trim()) return JSON.parse(process.env.STUDY_PLAN_JSON)
  try {
    return JSON.parse(readFileSync(path.resolve('drafts/year-plan.json'), 'utf8'))
  } catch (error) {
    if ((error as NodeJS.ErrnoException).code === 'ENOENT') return null
    throw error
  }
}

export default {
  load(): TodayPlan {
    return selectTodayPlan(
      readPrivatePlan(),
      new Date(),
      process.env.STUDY_PLAN_TIME_ZONE || 'Asia/Shanghai',
    )
  },
}
