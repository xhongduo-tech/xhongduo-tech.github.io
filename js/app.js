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
  const CATEGORY_ORDER = ['理论基础', '前沿追踪', '模型训练', '模型微调', '模型部署', '智能体', '系统基础', '工程实践'];
  const CATEGORY_CLASS = {
    '理论基础': 'cat-theory',
    '前沿追踪': 'cat-frontier',
    '模型训练': 'cat-training',
    '模型微调': 'cat-finetune',
    '模型部署': 'cat-deploy',
    '智能体':   'cat-agent',
    '系统基础': 'cat-systems',
    '工程实践': 'cat-engineering',
  };

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

  // 每篇文章取 tags[1]（第一个非分类次级 tag），去重后渲染过滤器
  const allTags = [...new Set(posts.map(p => (p.tags || [])[1]).filter(Boolean))].sort();
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
    const tags = p.tags || [];
    const tagsHtml = tags.map((t, i) => {
      const catCls = i === 0 ? (CATEGORY_CLASS[t] || '') : '';
      return `<span class="tag ${catCls}">${escapeHtml(t)}</span>`;
    }).join('');
    return `
      <a class="post-card" href="post.html?slug=${encodeURIComponent(p.slug)}">
        <div class="post-card-meta">
          <span class="post-date">${p.date}</span>
          <div class="post-tags">${tagsHtml}</div>
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

  // follow 状态（localStorage）
  const followKey       = `follow:${post.slug}`;
  const isFollowing     = localStorage.getItem(followKey) === '1';
  const followBtnClass  = isFollowing ? 'follow-btn following' : 'follow-btn';
  const followBtnText   = isFollowing ? '[ following ]' : '[ + follow ]';

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
        ${post.author === 'both'
          ? `${CLAUDE_SVG}<span>Claude Sonnet/Opus 4.6</span><span class="author-sep">&amp;</span>${OPENAI_SVG}<span>GPT-5.4</span><span class="author-sep">&amp;</span><span>xuhongduo</span>`
          : `${CLAUDE_SVG}<span>Claude Sonnet/Opus 4.6</span>`}
      </div>
      <div class="post-follow">
        <button class="${followBtnClass}" id="follow-btn">${followBtnText}</button>
        <span class="follow-count" id="follow-count"></span>
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

  setupFollow(post);
  setupFeedback(post);
}

/* ===================================================
   关注 (Follow) + 计数
   =================================================== */
async function setupFollow(post) {
  const btn     = document.getElementById('follow-btn');
  const countEl = document.getElementById('follow-count');
  if (!btn) return;

  const key     = `follow:${post.slug}`;
  const apiBase = `https://api.counterapi.dev/v1/xhd-blog/follow-${post.slug}`;

  // 非阻塞加载关注总数
  if (countEl) {
    try {
      const res = await fetch(apiBase);
      if (res.ok) {
        const data = await res.json();
        countEl.textContent = data.count ?? 0;
      } else {
        countEl.textContent = 0;
      }
    } catch (e) {
      countEl.style.display = 'none';
    }
  }

  btn.addEventListener('click', async () => {
    const current = localStorage.getItem(key) === '1';
    if (current) {
      // 取消关注：仅改变本地状态（计数不减）
      localStorage.removeItem(key);
      btn.textContent = '[ + follow ]';
      btn.classList.remove('following');
    } else {
      // 关注：本地保存 + 远端计数 +1
      localStorage.setItem(key, '1');
      btn.textContent = '[ following ]';
      btn.classList.add('following');
      try {
        const res = await fetch(`${apiBase}/up`);
        if (res.ok && countEl) {
          const data = await res.json();
          countEl.textContent = data.count ?? countEl.textContent;
          countEl.style.display = '';
        }
      } catch (e) {
        // 静默失败，按钮状态已正确
      }
    }
  });
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

/* ===== 品牌图标 ===== */
const CLAUDE_SVG = `<svg class="author-icon claude-icon" viewBox="0 0 24 24" fill="currentColor" aria-label="Claude"><path d="m4.7144 15.9555 4.7174-2.6471.079-.2307-.079-.1275h-.2307l-.7893-.0486-2.6956-.0729-2.3375-.0971-2.2646-.1214-.5707-.1215-.5343-.7042.0546-.3522.4797-.3218.686.0608 1.5179.1032 2.2767.1578 1.6514.0972 2.4468.255h.3886l.0546-.1579-.1336-.0971-.1032-.0972L6.973 9.8356l-2.55-1.6879-1.3356-.9714-.7225-.4918-.3643-.4614-.1578-1.0078.6557-.7225.8803.0607.2246.0607.8925.686 1.9064 1.4754 2.4893 1.8336.3643.3035.1457-.1032.0182-.0728-.164-.2733-1.3539-2.4467-1.445-2.4893-.6435-1.032-.17-.6194c-.0607-.255-.1032-.4674-.1032-.7285L6.287.1335 6.6997 0l.9957.1336.419.3642.6192 1.4147 1.0018 2.2282 1.5543 3.0296.4553.8985.2429.8318.091.255h.1579v-.1457l.1275-1.706.2368-2.0947.2307-2.6957.0789-.7589.3764-.9107.7468-.4918.5828.2793.4797.686-.0668.4433-.2853 1.8517-.5586 2.9021-.3643 1.9429h.2125l.2429-.2429.9835-1.3053 1.6514-2.0643.7286-.8196.85-.9046.5464-.4311h1.0321l.759 1.1293-.34 1.1657-1.0625 1.3478-.8804 1.1414-1.2628 1.7-.7893 1.36.0729.1093.1882-.0183 2.8535-.607 1.5421-.2794 1.8396-.3157.8318.3886.091.3946-.3278.8075-1.967.4857-2.3072.4614-3.4364.8136-.0425.0304.0486.0607 1.5482.1457.6618.0364h1.621l3.0175.2247.7892.522.4736.6376-.079.4857-1.2142.6193-1.6393-.3886-3.825-.9107-1.3113-.3279h-.1822v.1093l1.0929 1.0686 2.0035 1.8092 2.5075 2.3314.1275.5768-.3218.4554-.34-.0486-2.2039-1.6575-.85-.7468-1.9246-1.621h-.1275v.17l.4432.6496 2.3436 3.5214.1214 1.0807-.17.3521-.6071.2125-.6679-.1214-1.3721-1.9246L14.38 17.959l-1.1414-1.9428-.1397.079-.674 7.2552-.3156.3703-.7286.2793-.6071-.4614-.3218-.7468.3218-1.4753.3886-1.9246.3157-1.53.2853-1.9004.17-.6314-.0121-.0425-.1397.0182-1.4328 1.9672-2.1796 2.9446-1.7243 1.8456-.4128.164-.7164-.3704.0667-.6618.4008-.5889 2.386-3.0357 1.4389-1.882.929-1.0868-.0062-.1579h-.0546l-6.3385 4.1164-1.1293.1457-.4857-.4554.0608-.7467.2307-.2429 1.9064-1.3114Z"/></svg>`;
const OPENAI_SVG = `<svg class="author-icon openai-icon" viewBox="0 0 24 24" fill="currentColor" aria-label="GPT"><path d="M22.282 9.821a5.985 5.985 0 0 0-.516-4.91 6.046 6.046 0 0 0-6.51-2.9A6.065 6.065 0 0 0 4.981 4.18a5.985 5.985 0 0 0-3.998 2.9 6.046 6.046 0 0 0 .743 7.097 5.98 5.98 0 0 0 .51 4.911 6.051 6.051 0 0 0 6.515 2.9A5.985 5.985 0 0 0 13.26 24a6.056 6.056 0 0 0 5.772-4.206 5.99 5.99 0 0 0 3.997-2.9 6.056 6.056 0 0 0-.747-7.073zm-9.022 12.61a4.476 4.476 0 0 1-2.876-1.04l.141-.081 4.779-2.758a.795.795 0 0 0 .392-.681v-6.737l2.02 1.168a.071.071 0 0 1 .038.052v5.583a4.504 4.504 0 0 1-4.494 4.494zM3.6 18.304a4.47 4.47 0 0 1-.535-3.014l.142.085 4.783 2.759a.771.771 0 0 0 .78 0l5.843-3.369v2.332a.08.08 0 0 1-.033.062L9.74 19.95a4.5 4.5 0 0 1-6.14-1.646zM2.34 7.896a4.485 4.485 0 0 1 2.366-1.973V11.6a.766.766 0 0 0 .388.676l5.815 3.355-2.02 1.168a.076.076 0 0 1-.071 0l-4.83-2.786A4.504 4.504 0 0 1 2.34 7.872zm16.597 3.855l-5.833-3.387L15.119 7.2a.076.076 0 0 1 .071 0l4.83 2.791a4.494 4.494 0 0 1-.676 8.105v-5.678a.79.79 0 0 0-.407-.667zm2.01-3.023l-.141-.085-4.774-2.782a.776.776 0 0 0-.785 0L9.409 9.23V6.897a.066.066 0 0 1 .028-.061l4.83-2.787a4.5 4.5 0 0 1 6.68 4.66zm-12.64 4.135l-2.02-1.164a.08.08 0 0 1-.038-.057V6.075a4.5 4.5 0 0 1 7.375-3.453l-.142.08L8.704 5.46a.795.795 0 0 0-.393.681zm1.097-2.365l2.602-1.5 2.607 1.5v2.999l-2.597 1.5-2.607-1.5z"/></svg>`;

/* ===== 工具函数 ===== */
function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
