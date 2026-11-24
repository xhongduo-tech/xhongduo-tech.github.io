const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/

export function dateInTimeZone(date, timeZone) {
  const parts = new Intl.DateTimeFormat('en-US', {
    timeZone,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date)
  const values = Object.fromEntries(parts.map(part => [part.type, part.value]))
  return `${values.year}-${values.month}-${values.day}`
}

function utcDay(date) {
  const [year, month, day] = date.split('-').map(Number)
  return Date.UTC(year, month - 1, day) / 86400000
}

export function selectTodayPlan(plan, now = new Date(), fallbackTimeZone = 'Asia/Shanghai') {
  const timeZone = plan?.timeZone || fallbackTimeZone
  const date = dateInTimeZone(now, timeZone)
  if (!plan || !DATE_PATTERN.test(plan.startDate) || !Array.isArray(plan.weeks)) {
    return { available: false, date, tasks: [] }
  }

  const offset = utcDay(date) - utcDay(plan.startDate)
  const weekIndex = Math.floor(offset / 7)
  const dayIndex = ((offset % 7) + 7) % 7
  const week = plan.weeks[weekIndex]
  const day = week?.days?.[dayIndex]
  if (offset < 0 || !week || !day || !Array.isArray(day.tasks)) {
    return { available: false, date, tasks: [] }
  }

  const tasks = day.tasks.map((task, index) => ({
    id: `${date}-w${String(weekIndex + 1).padStart(2, '0')}-${index + 1}`,
    title: String(task.title),
    minutes: Number(task.minutes),
  }))
  if (!tasks.length || tasks.some(task => !task.title || !Number.isFinite(task.minutes) || task.minutes <= 0)) {
    throw new Error(`Invalid study tasks for ${date}`)
  }

  return {
    available: true,
    date,
    week: weekIndex + 1,
    topic: String(week.topic),
    tasks,
    totalMinutes: tasks.reduce((total, task) => total + task.minutes, 0),
    endDate: dateInTimeZone(new Date((utcDay(plan.startDate) + plan.weeks.length * 7 - 1) * 86400000), 'UTC'),
  }
}
