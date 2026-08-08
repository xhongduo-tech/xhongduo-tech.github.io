// 逐文件复刻 vitepress build 的 Vue 模板编译检查（markdown → HTML → Vue compile）。
// 一次性报出所有「Invalid end tag / Element is missing end tag」类文件，供主控批量修复。
// 用法：node scripts/check-vue.mjs [dir]
// 退出码：有失败文件返回 1。

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';
import { createMarkdownRenderer } from '../node_modules/vitepress/dist/node/index.js';
import { compileTemplate } from '../node_modules/@vue/compiler-sfc/dist/compiler-sfc.cjs.js';
import taskLists from '../node_modules/markdown-it-task-lists/index.js';

const root = process.argv[2] ?? join(process.cwd(), 'docs/posts');

function walk(dir, acc = []) {
  for (const ent of readdirSync(dir)) {
    const p = join(dir, ent);
    const st = statSync(p);
    if (st.isDirectory()) acc.push(...walk(p, []));
    else if (ent.endsWith('.md') && ent !== 'index.md') acc.push(p);
  }
  return acc;
}

// 支持把单个文件当作参数传入（./x.md），用于增量复查
const fileArgs = process.argv.slice(2).filter((a) => a.endsWith('.md'));

// 与 docs/.vitepress/config.mts 的 markdown 配置保持一致（math: true 是关键，
// 否则行内数学被当字面文本、下划线泄漏成 <em>，产生大量误报）
const md = await createMarkdownRenderer(join(process.cwd(), 'docs'), {
  math: true,
  config: (m) => {
    m.use(taskLists);
    // 与 build 一致：客户端 MathJax，输出 \(...\) / \[...\] 并转义 HTML 特殊字符
    const esc = (s) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/\{/g, '&#123;').replace(/\}/g, '&#125;');
    m.renderer.rules.math_inline = (t, i) => '\\(' + esc(t[i].content) + '\\)';
    m.renderer.rules.math_block = (t, i) => '\\[' + esc(t[i].content) + '\\]';
  },
});
const files = fileArgs.length ? fileArgs.map((f) => (f.startsWith('/') ? f : join(process.cwd(), f))) : walk(root);
let bad = 0;

for (const file of files) {
  const src = readFileSync(file, 'utf8');
  let html;
  try {
    html = md.render(src);
  } catch (e) {
    bad++;
    console.log(`✖ ${file.replace(process.cwd() + '/', '')}`);
    console.log(`   RENDER ERROR: ${(e.message ?? String(e)).slice(0, 120)}`);
    continue;
  }
  const errors = [];
  // 用 compiler-sfc 的 compileTemplate 复刻 vitepress build 的编译路径（compiler-dom 漏报 JS 语法类错误）
  try {
    compileTemplate({ source: html, filename: file, id: 'x' });
  } catch (e) {
    errors.push(e);
  }
  if (errors.length) {
    bad++;
    console.log(`✖ ${file.replace(process.cwd() + '/', '')}`);
    for (const e of errors.slice(0, 2)) {
      console.log(`   ${(e.message ?? String(e)).split('\n')[0].slice(0, 140)}`);
    }
  }
}

console.log(`\n${bad === 0 ? '✓ 全部文件通过 Vue 编译检查' : `✖ ${bad} 个文件无法通过 Vue 编译`}`);
process.exit(bad ? 1 : 0);
