// 主控健康检查：扫一遍所有专题，报告 worker 的运行状态。
// 对每个有 `- [ ]` 的专题：worker 进程是否存活？剩余多少条目？
// 输出三类：RUNNING（在写）、STALLED（进程死了但还有剩余，需要重启）、DONE（已写完）。
//
// 用法：
//   node scripts/check-workers.mjs              # 一次性报告
//   node scripts/check-workers.mjs --watch      # 只输出状态变化（供 Monitor 使用，永不退出）
// 退出码：有 STALLED 返回 1，否则 0（--watch 模式恒返回 0，靠事件流通知）。

import { readdirSync, readFileSync } from 'node:fs';
import { execSync } from 'node:child_process';
import { join } from 'node:path';

const WATCH = process.argv.includes('--watch');

function readTopic(idx) {
  const body = readFileSync(idx, 'utf8');
  const remaining = (body.match(/- \[ \]/g) ?? []).length;
  const name = (body.match(/^# (.+)$/m) ?? [])[1] ?? '';
  const tier = idx.replace('docs/posts/', '').split('/')[0];
  const key = idx.replace(`docs/posts/${tier}/`, '').replace('/index.md', '');
  return { tier, key, name, remaining };
}

function workerAlive(tier, key) {
  try {
    execSync(`pgrep -f "run-topic-session.sh ${tier} ${key} "`, { stdio: 'ignore' });
    return true;
  } catch {
    return false;
  }
}

function scan() {
  const topics = [];
  for (const tierDir of readdirSync('docs/posts', { withFileTypes: true })) {
    if (!tierDir.isDirectory()) continue;
    for (const keyDir of readdirSync(join('docs/posts', tierDir.name), { withFileTypes: true })) {
      if (!keyDir.isDirectory()) continue;
      const idx = join('docs/posts', tierDir.name, keyDir.name, 'index.md');
      try {
        const t = readTopic(idx);
        if (t.remaining === 0) continue; // 已完成的不算活跃专题
        t.alive = workerAlive(t.tier, t.key);
        topics.push(t);
      } catch { /* 忽略无 index.md 的目录 */ }
    }
  }
  return topics;
}

if (!WATCH) {
  const topics = scan();
  const stalled = topics.filter((t) => !t.alive);
  const running = topics.filter((t) => t.alive);
  const totalRemaining = topics.reduce((s, t) => s + t.remaining, 0);
  console.log(`运行中 ${running.length} | 停滞(需重启) ${stalled.length} | 剩余条目合计 ${totalRemaining}`);
  for (const t of stalled) console.log(`  [STALLED] ${t.tier}/${t.key} 剩 ${t.remaining}`);
  for (const t of running.slice(0, 5)) console.log(`  [RUNNING] ${t.tier}/${t.key} 剩 ${t.remaining}`);
  process.exit(stalled.length ? 1 : 0);
}

// --watch：只输出状态翻转，供 Monitor 使用
const prev = new Map();
while (true) {
  const topics = scan();
  const sig = new Map(topics.map((t) => [`${t.tier}/${t.key}`, `${t.remaining}:${t.alive ? 'Y' : 'N'}`]));
  for (const [k, v] of sig) {
    if (prev.get(k) !== v) {
      const [rem, al] = v.split(':');
      if (rem === '0') console.log(`DONE ${k}`);
      else if (al === 'N') console.log(`STALLED ${k} 剩 ${rem}`);
      else if (prev.has(k)) console.log(`RESUMED ${k} 剩 ${rem}`);
    }
  }
  prev.clear();
  for (const [k, v] of sig) prev.set(k, v);
  await new Promise((r) => setTimeout(r, 20000));
}
