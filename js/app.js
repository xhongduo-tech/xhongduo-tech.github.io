/* ===================================================
   主题初始化（已在 HTML <script> 中内联执行，这里补
   充按钮文字更新与切换逻辑）
   =================================================== */
document.addEventListener('DOMContentLoaded', () => {
  // 年份
  const yearEl = document.getElementById('year');
  if (yearEl) yearEl.textContent = new Date().getFullYear();

  // 主题按钮
  const themeBtn = document.getElementById('theme-toggle');
  if (themeBtn) {
    const updateBtn = () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      themeBtn.textContent = isDark ? '[light]' : '[dark]';
    };
    updateBtn();
    themeBtn.addEventListener('click', () => {
      const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
      const next = isDark ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateBtn();
      syncHljsTheme(next);
    });
  }

  // 初始化 hljs 主题（跟随系统色，若无手动设置）
  const savedTheme = localStorage.getItem('theme')
    || (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
  syncHljsTheme(savedTheme);

  // 代码块 copy 按钮（事件委托）
  document.addEventListener('click', e => {
    if (e.target.classList.contains('code-copy')) {
      const btn = e.target;
      const code = btn.closest('.code-wrapper')?.querySelector('code');
      if (!code) return;
      navigator.clipboard.writeText(code.textContent).then(() => {
        btn.textContent = 'copied!';
        btn.classList.add('copied');
        setTimeout(() => {
          btn.textContent = 'copy';
          btn.classList.remove('copied');
        }, 2000);
      }).catch(() => {
        btn.textContent = 'failed';
        setTimeout(() => { btn.textContent = 'copy'; }, 2000);
      });
    }
  });

  // 根据当前页面执行对应逻辑
  if (document.getElementById('post-list')) {
    initIndexPage();
  } else if (document.getElementById('post-content')) {
    initPostPage();
  }
});

/* ===== hljs 主题同步 ===== */
function syncHljsTheme(theme) {
  const link = document.getElementById('hljs-theme');
  if (!link) return;
  const base = 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/';
  link.href = theme === 'light'
    ? base + 'github.min.css'
    : base + 'atom-one-dark.min.css';
}

/* ===================================================
   首页：文章列表
   =================================================== */
async function initIndexPage() {
  const CATEGORY_ORDER = ['底层原理', '模型解析', '智能体', '工程实践'];

  let posts = [];
  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error('fetch failed');
    posts = await res.json();
  } catch (e) {
    document.getElementById('post-list').innerHTML =
      '<p class="no-result">// 暂无文章 · 请添加 posts/posts.json</p>';
    return;
  }

  // 按日期降序
  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  // State
  let activeCategory = 'all';
  let activeTag      = null;
  let searchQuery    = '';

  // 渲染分类 Tab
  const categoryTabsEl = document.getElementById('category-tabs');
  ['all', ...CATEGORY_ORDER].forEach(cat => {
    const btn = document.createElement('button');
    btn.className   = 'cat-tab' + (cat === 'all' ? ' active' : '');
    btn.dataset.cat = cat;
    btn.textContent = cat === 'all' ? '全部' : cat;
    categoryTabsEl.appendChild(btn);
  });
  categoryTabsEl.addEventListener('click', e => {
    const btn = e.target.closest('.cat-tab');
    if (!btn) return;
    activeCategory = btn.dataset.cat;
    categoryTabsEl.querySelectorAll('.cat-tab').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    // 切换分类时重置 tag 筛选
    activeTag = null;
    tagFilter.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
    renderList();
  });

  // 收集所有标签，渲染过滤器
  const allTags = [...new Set(posts.flatMap(p => p.tags || []))].sort();
  const tagFilter = document.getElementById('tag-filter');
  allTags.forEach(tag => {
    const btn = document.createElement('button');
    btn.className = 'tag';
    btn.textContent = tag;
    btn.addEventListener('click', () => {
      activeTag = activeTag === tag ? null : tag;
      tagFilter.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
      if (activeTag) btn.classList.add('active');
      renderList();
    });
    tagFilter.appendChild(btn);
  });

  // 搜索
  const searchInput = document.getElementById('search-input');
  searchInput.addEventListener('input', () => {
    searchQuery = searchInput.value.trim().toLowerCase();
    renderList();
  });

  function postCardHtml(p) {
    return `
      <a class="post-card" href="post.html?slug=${encodeURIComponent(p.slug)}">
        <div class="post-card-meta">
          <span class="post-date">${p.date}</span>
          <div class="post-tags">
            ${(p.tags || []).map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}
          </div>
        </div>
        <div class="post-card-title">${escapeHtml(p.title)}</div>
        ${p.summary ? `<div class="post-card-summary">${escapeHtml(p.summary)}</div>` : ''}
      </a>`;
  }

  function renderList() {
    const filtered = posts.filter(p => {
      const matchCat    = activeCategory === 'all' || (p.tags && p.tags[0] === activeCategory);
      const matchTag    = !activeTag || (p.tags && p.tags.includes(activeTag));
      const q           = searchQuery;
      const matchSearch = !q ||
        p.title.toLowerCase().includes(q) ||
        (p.summary || '').toLowerCase().includes(q) ||
        (p.tags || []).some(t => t.toLowerCase().includes(q));
      return matchCat && matchTag && matchSearch;
    });

    const list = document.getElementById('post-list');
    if (filtered.length === 0) {
      list.innerHTML = '<p class="no-result">// 没有找到相关文章</p>';
      return;
    }

    // 「全部」且无任何筛选时：按分类分组展示
    if (activeCategory === 'all' && !activeTag && !searchQuery) {
      const groups = {};
      CATEGORY_ORDER.forEach(cat => { groups[cat] = []; });
      filtered.forEach(p => {
        const cat = p.tags?.[0];
        if (cat && groups[cat]) groups[cat].push(p);
      });
      list.innerHTML = CATEGORY_ORDER
        .filter(cat => groups[cat].length > 0)
        .map(cat => `
          <div class="post-section">
            <div class="post-section-header">
              <span class="section-label">// ${cat}</span>
              <span class="section-count">${groups[cat].length}</span>
            </div>
            ${groups[cat].map(postCardHtml).join('')}
          </div>`)
        .join('');
    } else {
      // 分类 / 标签 / 搜索筛选时：平铺展示
      list.innerHTML = filtered.map(postCardHtml).join('');
    }
  }

  renderList();
}

/* ===================================================
   文章页：渲染单篇文章
   =================================================== */
async function initPostPage() {
  const params    = new URLSearchParams(location.search);
  const slug      = params.get('slug');
  const contentEl = document.getElementById('post-content');

  if (!slug) {
    contentEl.innerHTML = '<p class="no-result">// 未指定文章</p>';
    return;
  }

  // 加载文章索引
  let posts = [];
  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error();
    posts = await res.json();
    posts.sort((a, b) => new Date(b.date) - new Date(a.date));
  } catch (e) {
    contentEl.innerHTML = '<p class="no-result">// 无法加载文章索引</p>';
    return;
  }

  const idx  = posts.findIndex(p => p.slug === slug);
  const post = posts[idx];
  if (!post) {
    contentEl.innerHTML = '<p class="no-result">// 文章不存在</p>';
    return;
  }

  // 加载 Markdown
  let mdText = '';
  try {
    const res = await fetch(`posts/${post.slug}.md`);
    if (!res.ok) throw new Error('404');
    mdText = await res.text();
  } catch (e) {
    contentEl.innerHTML = '<p class="no-result">// 文章文件未找到</p>';
    return;
  }

  // 页面标题
  document.title = `${post.title} · xhd.log`;

  // 配置 marked（用 plain object renderer，避免 new marked.Renderer() 兼容问题）
  if (typeof marked !== 'undefined') {
    marked.setOptions({ breaks: true, gfm: true });

    marked.use({
      renderer: {
        // 代码块：添加语言标签 + copy 按钮
        code(text, lang) {
          let highlighted = escapeHtml(text);
          let language    = 'plaintext';

          if (typeof hljs !== 'undefined' && hljs.getLanguage(lang)) {
            language    = lang;
            highlighted = hljs.highlight(text, { language }).value;
          }

          const langLabel = language !== 'plaintext'
            ? `<span class="code-lang">${language}</span>`
            : `<span class="code-lang">text</span>`;

          return `
        <div class="code-wrapper">
          <div class="code-header">
            ${langLabel}
            <button class="code-copy">copy</button>
          </div>
          <pre><code class="hljs language-${language}">${highlighted}</code></pre>
        </div>`;
        }
      }
    });
  }

  // 阅读时长（中文按字符，英文按单词）
  const charCount = mdText.replace(/\s/g, '').length;
  const readTime  = Math.max(1, Math.ceil(charCount / 400));

  const tagsHtml = (post.tags || [])
    .map(t => `<span class="tag">${escapeHtml(t)}</span>`)
    .join('');

  // 保护数学公式不被 marked 处理，渲染后替换为 KaTeX HTML
  let bodyHtml;
  try {
    if (typeof marked !== 'undefined') {
      const mathStore = [];
      const protectedMd = mdText
        .replace(/\$\$([\s\S]+?)\$\$/g, (_, math) => {
          mathStore.push({ display: true, math });
          return `<!--KMATH${mathStore.length - 1}-->`;
        })
        .replace(/\$([^$\n]+?)\$/g, (_, math) => {
          mathStore.push({ display: false, math });
          return `<!--KMATH${mathStore.length - 1}-->`;
        });
      // await 兼容 marked 在某些配置下返回 Promise<string> 的情况
      let html = await Promise.resolve(marked.parse(protectedMd));
      if (typeof katex !== 'undefined' && mathStore.length > 0) {
        html = html.replace(/<!--KMATH(\d+)-->/g, (_, i) => {
          const { display, math } = mathStore[Number(i)];
          return katex.renderToString(math, { throwOnError: false, displayMode: display });
        });
      }
      bodyHtml = html;
    } else {
      bodyHtml = `<pre>${escapeHtml(mdText)}</pre>`;
    }
  } catch (e) {
    // 降级：不渲染数学公式，但文章内容正常显示
    try {
      bodyHtml = typeof marked !== 'undefined'
        ? await Promise.resolve(marked.parse(mdText))
        : `<pre>${escapeHtml(mdText)}</pre>`;
    } catch (_) {
      bodyHtml = `<pre>${escapeHtml(mdText)}</pre>`;
    }
  }

  contentEl.innerHTML = `
    <header class="post-header">
      <h1 class="post-title">${escapeHtml(post.title)}</h1>
      <div class="post-info">
        <span class="post-date">${post.date}</span>
        <span class="post-read-time">约 ${readTime} 分钟阅读</span>
        <div class="post-tags">${tagsHtml}</div>
      </div>
      <div class="post-author">
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100" width="15" height="15" aria-label="Claude">
          <defs>
            <linearGradient id="claude-icon-grad" x1="0" y1="0" x2="100" y2="100" gradientUnits="userSpaceOnUse">
              <stop offset="0%" stop-color="#E8905F"/>
              <stop offset="100%" stop-color="#C75F2A"/>
            </linearGradient>
          </defs>
          <circle cx="50" cy="50" r="47" fill="url(#claude-icon-grad)"/>
          <path d="M67 35C57 24,37 26,28 44C20 61,29 80,48 82C59 84,68 78,73 67" stroke="white" stroke-width="12" stroke-linecap="round" fill="none"/>
        </svg>
        <span>Claude Code &amp; xuhongduo</span>
      </div>
    </header>
    <div class="post-body">${bodyHtml}</div>
  `;

  // 上一篇 / 下一篇（列表已按日期降序，idx-1 = 更新，idx+1 = 更早）
  const navEl = document.getElementById('post-nav');
  const prev  = posts[idx - 1];
  const next  = posts[idx + 1];

  navEl.innerHTML = [
    prev
      ? `<a class="post-nav-item prev" href="post.html?slug=${encodeURIComponent(prev.slug)}">
           <div class="post-nav-label">← newer</div>
           <div class="post-nav-title">${escapeHtml(prev.title)}</div>
         </a>`
      : '<span></span>',
    next
      ? `<a class="post-nav-item next" href="post.html?slug=${encodeURIComponent(next.slug)}">
           <div class="post-nav-label">older →</div>
           <div class="post-nav-title">${escapeHtml(next.title)}</div>
         </a>`
      : '<span></span>',
  ].join('');

  // 阅读进度条
  const progressBar = document.getElementById('progress-bar');
  if (progressBar) {
    const updateProgress = () => {
      const docH = document.documentElement.scrollHeight - window.innerHeight;
      const pct  = docH > 0 ? Math.min(100, (window.scrollY / docH) * 100) : 0;
      progressBar.style.width = pct + '%';
    };
    window.addEventListener('scroll', updateProgress, { passive: true });
    updateProgress();
  }

  setupFeedback(post);
}

/* ===================================================
   纠错反馈
   =================================================== */
function setupFeedback(post) {
  const navEl = document.getElementById('post-nav');
  if (!navEl) return;

  // 纠错按钮
  const section = document.createElement('div');
  section.className = 'feedback-section';
  section.innerHTML = `<button class="feedback-btn">[ 纠错反馈 ]</button>`;
  navEl.insertAdjacentElement('afterend', section);

  // Modal
  const modal = document.createElement('div');
  modal.className = 'feedback-modal';
  modal.setAttribute('aria-hidden', 'true');
  modal.innerHTML = `
    <div class="feedback-panel">
      <div class="feedback-title">// 纠错反馈</div>
      <div class="feedback-hint">输入 <kbd>@</kbd> 可搜索文章中的标题与段落</div>
      <div class="mention-dropdown" id="mention-dropdown"></div>
      <textarea class="feedback-textarea" id="feedback-textarea"
        placeholder="描述具体错误，例如：@第二章 公式推导有误，正确结果应为..."></textarea>
      <div class="feedback-actions">
        <button class="feedback-cancel">取消</button>
        <button class="feedback-submit">提交到 GitHub Issues →</button>
      </div>
    </div>`;
  document.body.appendChild(modal);

  const textarea    = modal.querySelector('#feedback-textarea');
  const dropdown    = modal.querySelector('#mention-dropdown');
  const submitBtn   = modal.querySelector('.feedback-submit');
  const cancelBtn   = modal.querySelector('.feedback-cancel');

  // 收集文章中所有可引用内容（标题 + 段落首句）
  function getSearchItems() {
    const items = [];
    let h2Text = '';
    let h3Text = '';
    document.querySelectorAll('.post-body h2, .post-body h3, .post-body h4, .post-body p, .post-body li')
      .forEach(el => {
        const text = el.textContent.trim();
        if (!text) return;
        if (el.tagName === 'H2') {
          h2Text = text; h3Text = '';
          items.push({ label: text, ref: text, crumb: '' });
        } else if (el.tagName === 'H3') {
          h3Text = text;
          items.push({ label: text, ref: text, crumb: h2Text });
        } else if (el.tagName === 'H4') {
          items.push({ label: text, ref: text, crumb: h3Text || h2Text });
        } else if (el.tagName === 'P' && text.length > 20) {
          // 段落：截取首句（句号/问号/感叹号前）
          const sentence = text.split(/[。？！.?!]/)[0].slice(0, 50);
          if (sentence.length > 10) {
            const crumb = h3Text || h2Text;
            items.push({ label: sentence + '…', ref: sentence, crumb });
          }
        } else if (el.tagName === 'LI' && text.length > 10 && text.length < 80) {
          const crumb = h3Text || h2Text;
          items.push({ label: text.slice(0, 50) + (text.length > 50 ? '…' : ''), ref: text.slice(0, 50), crumb });
        }
      });
    return items;
  }

  // 显示/隐藏 modal
  function openModal() {
    modal.setAttribute('aria-hidden', 'false');
    modal.classList.add('open');
    textarea.focus();
  }
  function closeModal() {
    modal.setAttribute('aria-hidden', 'true');
    modal.classList.remove('open');
    dropdown.innerHTML = '';
    dropdown.style.display = 'none';
  }

  section.querySelector('.feedback-btn').addEventListener('click', openModal);
  cancelBtn.addEventListener('click', closeModal);
  modal.addEventListener('click', e => { if (e.target === modal) closeModal(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(); });

  // @ 触发下拉
  let mentionStart = -1;
  // 缓存，避免每次输入都重新遍历 DOM
  let _searchCache = null;
  function getCachedItems() {
    if (!_searchCache) _searchCache = getSearchItems();
    return _searchCache;
  }

  textarea.addEventListener('input', () => {
    const val    = textarea.value;
    const cursor = textarea.selectionStart;
    const before = val.slice(0, cursor);
    const atIdx  = before.lastIndexOf('@');

    if (atIdx === -1) { dropdown.style.display = 'none'; mentionStart = -1; return; }

    // 如果 @ 后面已经有空格（说明引用已完成），不弹出
    const afterAt = before.slice(atIdx + 1);
    if (afterAt.includes(' ') && afterAt.length > 10) {
      dropdown.style.display = 'none'; mentionStart = -1; return;
    }

    const query   = afterAt.toLowerCase();
    const matched = getCachedItems().filter(item =>
      item.label.toLowerCase().includes(query) ||
      item.crumb.toLowerCase().includes(query)
    ).slice(0, 8); // 最多展示 8 条

    if (matched.length === 0) { dropdown.style.display = 'none'; mentionStart = -1; return; }

    mentionStart = atIdx;
    dropdown.style.display = 'block';
    dropdown.innerHTML = matched.map(item => `
      <div class="mention-item" data-ref="${escapeHtml(item.ref)}">
        <span class="mention-label">${escapeHtml(item.label)}</span>
        ${item.crumb ? `<span class="mention-crumb">↳ ${escapeHtml(item.crumb)}</span>` : ''}
      </div>`
    ).join('');
  });

  dropdown.addEventListener('click', e => {
    const item = e.target.closest('.mention-item');
    if (!item || mentionStart === -1) return;
    const ref    = item.dataset.ref;
    const val    = textarea.value;
    const cursor = textarea.selectionStart;
    textarea.value = val.slice(0, mentionStart) + '@[' + ref + '] ' + val.slice(cursor);
    textarea.selectionStart = textarea.selectionEnd = mentionStart + ref.length + 4;
    textarea.focus();
    dropdown.style.display = 'none';
    mentionStart = -1;
  });

  // 提交
  submitBtn.addEventListener('click', () => {
    const body = textarea.value.trim();
    if (!body) { textarea.focus(); return; }
    const title  = encodeURIComponent('[纠错] ' + post.title);
    const labels = encodeURIComponent('error-report');
    const bodyEnc = encodeURIComponent(body + '\n\n---\n文章：' + post.slug);
    const url = `https://github.com/xhongduo-tech/xhongduo-tech.github.io/issues/new?title=${title}&labels=${labels}&body=${bodyEnc}`;
    window.open(url, '_blank', 'noopener');
    closeModal();
  });
}

/* ===== 工具函数 ===== */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
