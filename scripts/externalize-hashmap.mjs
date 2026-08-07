// VitePress 会把 __VP_HASH_MAP__（路由哈希表，10000+ 页时达数百 KB）内联到每一页。
// 10439 页 × ~786KB = 8GB，超 GitHub Pages 1GB 上限。
// 本脚本把它提取为共享的 /hash-map.js，页面只留 ~8KB 引用 → 全站骤降。
// 用法：vitepress build 之后运行：node scripts/externalize-hashmap.mjs

import { readFileSync, writeFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

const dist = join(process.cwd(), 'docs/.vitepress/dist');

function walk(dir, acc = []) {
  for (const e of readdirSync(dir)) {
    const p = join(dir, e);
    statSync(p).isDirectory() ? walk(p, acc) : e.endsWith('.html') && acc.push(p);
  }
  return acc;
}

const files = walk(dist);
// group1=script 属性, group2=JSON.parse 参数(哈希表), group3=同脚本里的其余内容(SITE_DATA 等)
const RE = /<script([^>]*)>window\.__VP_HASH_MAP__=JSON\.parse\(([\s\S]*?)\);([\s\S]*?)<\/script>/;

let extracted = null;
let fixed = 0;
let fail = 0;

for (const f of files) {
  const html = readFileSync(f, 'utf8');
  const m = html.match(RE);
  if (!m) continue;
  try {
    if (extracted === null) {
      // m[2] 是 JSON 字符串字面量 "…"。先解析外层得到内层字符串，再解析成对象。
      const inner = JSON.parse(m[2]);
      const obj = JSON.parse(inner);
      extracted = `window.__VP_HASH_MAP__=${JSON.stringify(obj)};`;
      writeFileSync(join(dist, 'hash-map.js'), extracted);
    }
    // 用 hash-map.js 引用替换哈希表赋值；保留 SITE_DATA 等其余内容
    const newHtml = html.replace(RE, '<script src="/hash-map.js"></script><script$1>$3</script>');
    writeFileSync(f, newHtml);
    fixed++;
  } catch (e) {
    fail++;
    if (fail <= 3) console.log(`解析失败: ${f} => ${e.message.slice(0, 60)}`);
  }
}

if (extracted === null) {
  console.log('未找到 __VP_HASH_MAP__');
  process.exit(1);
}
console.log(`已提取 hash-map.js (${(extracted.length / 1024).toFixed(0)} KB)，更新 ${fixed} 个页面，失败 ${fail}`);
