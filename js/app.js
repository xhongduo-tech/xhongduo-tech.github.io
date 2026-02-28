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
  const CATEGORY_ORDER = ['深度学习', '大模型', 'AI Agent', 'AI工具'];

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
}

/* ===== 工具函数 ===== */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
