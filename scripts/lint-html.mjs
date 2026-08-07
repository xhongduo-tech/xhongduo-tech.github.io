// 构建安全红线扫描：在 vitepress build 之前抓出会让 Vue 模板编译失败的写法。
// 用法：node scripts/lint-html.mjs [dir]   （默认扫描 docs/posts 全部 *.md，跳过 index.md）
// 退出码：有命中返回 1，无命中返回 0。
//
// 覆盖四类已知会炸构建的模式：
//   A. 表格行内联代码里出现裸 `|`（GFM 会把 | 当单元格分隔符，把代码 span 从中间切断，
//      残留的 `<A>`/`<B>` 等被 Vue 当作未闭合标签 → "Element is missing end tag"）
//   B. <span class="marginnote|sidenote">…</span> 内部出现 markdown `**`（导致 <strong> 未闭合）
//   C. 代码围栏外未闭合的 HTML 标签（对每个合法 HTML 标签数开闭）
//   D. 代码围栏外的裸尖括号记号：`<where>`/`<ident>`/`<C>` 这类非 HTML 标签名的记号
//      未用反引号包裹（被 Vue 当作自定义元素 → "Element is missing end tag"）

// HTML5 元素清单（含 SVG/MathML/void），用于 C 的平衡检查与 D 的未知标签判断
const HTML_TAGS = new Set(`
a abbr address area article aside audio b base bdi bdo blockquote body br button canvas caption
cite code col colgroup data datalist dd del details dfn dialog div dl dt em embed fieldset
figcaption figure footer form h1 h2 h3 h4 h5 h6 head header hgroup hr html i iframe img input
ins kbd label legend li link main map mark math menu meta meter nav noscript object ol optgroup
option output p param picture pre progress q rp rt ruby s samp script search section select slot
small source span strong style sub summary sup svg table tbody td template textarea tfoot th
thead time title tr track u ul var video wbr
`.trim().split(/\s+/));

import { readFileSync, readdirSync, statSync } from 'node:fs';
import { join } from 'node:path';

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

// 去掉代码围栏、数学块与行内代码，避免把 `<b>`（数学）、`<digit>`（行内代码）这类
// 会被 markdown-it 正确转义的内容误判为 HTML 标签。B/C/D 三组检查都基于这个输出；
// A 组仍需原始行，因为它正是要检查「行内代码内部」的裸管道。
// 围栏与 $$…$$ 是块级、可能跨行；行内 $…$ / `…` 只处理单行内闭合的。
function stripFences(src) {
  let out = src.replace(/```[\s\S]*?```/g, '');
  out = out.replace(/\$\$[\s\S]*?\$\$/g, '');
  out = out
    .split('\n')
    .map((ln) => ln.replace(/\$[^$\n]*\$/g, '').replace(/`[^`\n]*`/g, ''))
    .join('\n');
  return out;
}

let bad = 0;

for (const file of walk(root)) {
  const src = readFileSync(file, 'utf8');
  const lines = src.split('\n');
  const out = stripFences(src);
  const problems = [];

  // --- A. 表格行内联代码中的裸管道 ---
  for (let i = 0; i < lines.length; i++) {
    const ln = lines[i];
    if (!ln.startsWith('|')) continue;
    // 逐个内联代码 span 检查
    for (const m of ln.matchAll(/`([^`]+)`/g)) {
      const code = m[1];
      // 未转义的 | （\ 转义过的合法）
      if (/(^|[^\\])\|/.test(code)) {
        problems.push(`A: 表格行内联代码含裸 | 需写成 \\|  (行 ${i + 1}): \`${code.slice(0, 50)}\``);
      }
    }
  }

  // --- B. marginnote/sidenote span 内的 ** ---
  for (const m of out.matchAll(/<span class="(marginnote|sidenote)">([\s\S]*?)<\/span>/g)) {
    if (m[2].includes('**')) {
      const pre = out.slice(0, m.index);
      const lineNo = pre.split('\n').length;
      problems.push(`B: marginnote span 内出现 **（应改用 <strong>）约行 ${lineNo}`);
    }
  }

  // --- C/D. 围栏外全部尖括号记号：未知标签 → D；已知标签不平衡 → C ---
  const tagCounts = new Map();
  const unknown = new Set();
  for (const m of out.matchAll(/<(\/?)([A-Za-z][A-Za-z0-9]*)((?:\s[^>]*)?)>/g)) {
    const closing = m[1] === '/';
    const name = m[2].toLowerCase();
    if (!HTML_TAGS.has(name)) {
      unknown.add(name);
      continue;
    }
    const key = closing ? `/${name}` : name;
    tagCounts.set(key, (tagCounts.get(key) ?? 0) + 1);
  }
  for (const name of unknown) {
    problems.push(`D: 疑似裸尖括号记号 <${name}> 未用反引号包裹（应写成 \`<${name}>\`）`);
  }
  const allNames = new Set([...tagCounts.keys()].map((k) => k.replace(/^\//, '')));
  for (const name of allNames) {
    const open = tagCounts.get(name) ?? 0;
    const close = tagCounts.get(`/${name}`) ?? 0;
    if (open !== close) {
      problems.push(`C: <${name}> 开 ${open} / 闭 ${close} 不平衡`);
    }
  }

  if (problems.length) {
    bad++;
    console.log(`\n✖ ${file.replace(process.cwd() + '/', '')}`);
    for (const p of problems) console.log(`   ${p}`);
  }
}

console.log(`\n${bad === 0 ? '✓ 未发现构建红线问题' : `✖ 发现 ${bad} 个文件需要修复`}`);
process.exit(bad ? 1 : 0);
