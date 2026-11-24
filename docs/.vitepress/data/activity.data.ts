import { execFileSync } from 'node:child_process'

export interface ActivityDay {
  date: string
  count: number
  future: boolean
}

export interface ActivityData {
  startDate: string
  endDate: string
  today: string
  total: number
  activeDays: number
  longestStreak: number
  days: ActivityDay[]
}

declare const data: ActivityData
export { data }

const DAY = 24 * 60 * 60 * 1000

function isoDate(date: Date): string {
  return date.toISOString().slice(0, 10)
}

function addDays(date: Date, amount: number): Date {
  return new Date(date.getTime() + amount * DAY)
}

export default {
  load(): ActivityData {
    const now = new Date()
    const today = new Date(Date.UTC(now.getUTCFullYear(), now.getUTCMonth(), now.getUTCDate()))
    const start = addDays(today, -today.getUTCDay() - 52 * 7)
    const end = addDays(start, 52 * 7 + 6)
    const startDate = isoDate(start)
    const endDate = isoDate(end)
    const todayDate = isoDate(today)

    const output = execFileSync('git', [
      'log',
      `--since=${startDate}T00:00:00Z`,
      `--until=${todayDate}T23:59:59Z`,
      '--pretty=format:%cI',
      '--',
      '.',
    ], {
      cwd: process.cwd(),
      encoding: 'utf8',
      env: { ...process.env, TZ: 'UTC' },
      maxBuffer: 4 * 1024 * 1024,
    }).trim()

    const counts = new Map<string, number>()
    const commits = output ? output.split('\n').filter(Boolean) : []
    for (const commit of commits) {
      const date = isoDate(new Date(commit))
      counts.set(date, (counts.get(date) ?? 0) + 1)
    }

    const days: ActivityDay[] = []
    for (let offset = 0; offset < 371; offset++) {
      const date = isoDate(addDays(start, offset))
      days.push({ date, count: counts.get(date) ?? 0, future: date > todayDate })
    }

    let longestStreak = 0
    let streak = 0
    for (const day of days) {
      if (day.future) break
      streak = day.count > 0 ? streak + 1 : 0
      longestStreak = Math.max(longestStreak, streak)
    }

    return {
      startDate,
      endDate,
      today: todayDate,
      total: commits.length,
      activeDays: counts.size,
      longestStreak,
      days,
    }
  },
}
