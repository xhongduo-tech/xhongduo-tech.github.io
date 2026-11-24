import test from 'node:test'
import assert from 'node:assert/strict'
import { dateInTimeZone, selectTodayPlan } from '../docs/.vitepress/data/study-plan-utils.mjs'

function makePlan(weekCount = 2) {
  return {
    startDate: '2026-11-23',
    timeZone: 'Asia/Shanghai',
    weeks: Array.from({ length: weekCount }, (_, week) => ({
      topic: `第${week + 1}周`,
      days: Array.from({ length: 7 }, (_, day) => ({
        tasks: [
          { title: `第${week + 1}周第${day + 1}天阅读`, minutes: 60 },
          { title: `第${week + 1}周第${day + 1}天实验`, minutes: 90 },
          { title: `第${week + 1}周第${day + 1}天记录`, minutes: 30 },
        ],
      })),
    })),
  }
}

test('today is selected at the Asia/Shanghai date boundary', () => {
  assert.equal(dateInTimeZone(new Date('2026-11-22T15:59:59Z'), 'Asia/Shanghai'), '2026-11-22')
  assert.equal(dateInTimeZone(new Date('2026-11-22T16:00:00Z'), 'Asia/Shanghai'), '2026-11-23')
  assert.equal(selectTodayPlan(makePlan(), new Date('2026-11-22T15:59:59Z')).available, false)
  assert.equal(selectTodayPlan(makePlan(), new Date('2026-11-22T16:00:00Z')).available, true)
})

test('each calendar day maps to one week day with stable task IDs and total minutes', () => {
  const plan = makePlan()
  for (let day = 0; day < 7; day++) {
    const now = new Date(Date.UTC(2026, 10, 23 + day, 12))
    const today = selectTodayPlan(plan, now)
    assert.equal(today.available, true)
    assert.equal(today.week, 1)
    assert.equal(today.topic, '第1周')
    assert.equal(today.date, `2026-11-${String(23 + day).padStart(2, '0')}`)
    assert.equal(today.totalMinutes, 180)
    assert.equal(today.tasks.length, 3)
    assert.equal(today.tasks[0].id, `${today.date}-w01-1`)
    assert.match(today.tasks[0].title, new RegExp(`第1周第${day + 1}天`))
    assert.deepEqual(selectTodayPlan(plan, now), today)
  }
})

test('week two and dates after the plan are distinguished', () => {
  const plan = makePlan()
  const secondWeek = selectTodayPlan(plan, new Date('2026-11-30T12:00:00Z'))
  assert.equal(secondWeek.week, 2)
  assert.equal(secondWeek.tasks[0].id, '2026-11-30-w02-1')
  assert.equal(secondWeek.endDate, '2026-12-06')
  assert.equal(selectTodayPlan(plan, new Date('2026-12-07T12:00:00Z')).available, false)
})

test('missing plans stay empty and invalid visible tasks fail loudly', () => {
  assert.deepEqual(selectTodayPlan(null, new Date('2026-11-23T12:00:00Z')), {
    available: false, date: '2026-11-23', tasks: [],
  })
  const plan = makePlan(1)
  plan.weeks[0].days[0].tasks[0].minutes = 0
  assert.throws(() => selectTodayPlan(plan, new Date('2026-11-23T12:00:00Z')), /Invalid study tasks/)
})
