/* ===================================================
   Index page: post list, categories, tags, search
   =================================================== */
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('post-list')) {
    initIndexPage();
  }
});

async function initIndexPage() {
  const CATEGORY_ORDER = [
    'LLM 推理',
    'Agents',
    'CUDA / GPU',
    '模型部署',
    '系统工程',
    '训练与微调',
    '论文 / 前沿',
    '工程实践'
  ];

  let posts = [];
  const listEl = document.getElementById('post-list');
  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error('fetch failed');
    posts = await res.json();
  } catch (e) {
    listEl.innerHTML = '<p class="empty-state">// 暂无文章 · 请添加 posts/posts.json</p>';
    return;
  }

  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  // Stats
  const statsEl = document.getElementById('blog-stats');
  if (statsEl) {
    const cats = new Set(posts.map(p => p.tags?.[0]).filter(Boolean));
    const tags = new Set(posts.flatMap(p => p.tags?.slice(1) || []).filter(Boolean));
    statsEl.innerHTML = `
      <div class="stat"><span class="stat-num">${posts.length}</span><span class="stat-label">文章</span></div>
      <div class="stat"><span class="stat-num">${cats.size}</span><span class="stat-label">分类</span></div>
      <div class="stat"><span class="stat-num">${tags.size}</span><span class="stat-label">标签</span></div>
    `;
  }

  let activeCategory = 'all';
  let activeTag = null;
  let searchQuery = '';

  // Category tabs
  const catTabsEl = document.getElementById('category-tabs');
  if (catTabsEl) {
    ['all', ...CATEGORY_ORDER].forEach(cat => {
      const btn = document.createElement('button');
      btn.className = 'category-tab' + (cat === 'all' ? ' active' : '');
      btn.dataset.cat = cat;
      btn.textContent = cat === 'all' ? '全部' : cat;
      catTabsEl.appendChild(btn);
    });
    catTabsEl.addEventListener('click', e => {
      const btn = e.target.closest('.category-tab');
      if (!btn) return;
      activeCategory = btn.dataset.cat;
      catTabsEl.querySelectorAll('.category-tab').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      activeTag = null;
      tagFilterEl?.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
      renderList();
    });
  }

  // Tag filter (use tags[1+] excluding category)
  const tagFilterEl = document.getElementById('tag-filter');
  const allTags = [...new Set(
    posts.flatMap(p => {
      const tags = p.tags || [];
      return tags.slice(1);
    }).filter(Boolean)
  )].sort();
  if (tagFilterEl) {
    allTags.forEach(tag => {
      const btn = document.createElement('button');
      btn.className = 'tag';
      btn.textContent = tag;
      btn.addEventListener('click', () => {
        activeTag = activeTag === tag ? null : tag;
        tagFilterEl.querySelectorAll('.tag').forEach(b => b.classList.remove('active'));
        if (activeTag) btn.classList.add('active');
        renderList();
      });
      tagFilterEl.appendChild(btn);
    });
  }

  // Search
  const searchInput = document.getElementById('search-input');
  if (searchInput) {
    searchInput.addEventListener('input', () => {
      searchQuery = searchInput.value.trim().toLowerCase();
      renderList();
    });
  }

  function renderList() {
    const filtered = posts.filter(p => {
      const cat = p.tags?.[0];
      const matchCat = activeCategory === 'all' || cat === activeCategory;
      const matchTag = !activeTag || (p.tags || []).includes(activeTag);
      const q = searchQuery;
      const matchSearch = !q ||
        (p.title || '').toLowerCase().includes(q) ||
        (p.summary || '').toLowerCase().includes(q) ||
        (p.tags || []).some(t => t.toLowerCase().includes(q));
      return matchCat && matchTag && matchSearch;
    });

    if (filtered.length === 0) {
      listEl.innerHTML = '<p class="empty-state">// 没有找到相关文章</p>';
      return;
    }

    if (activeCategory === 'all' && !activeTag && !searchQuery) {
      const groups = {};
      CATEGORY_ORDER.forEach(c => { groups[c] = []; });
      filtered.forEach(p => {
        const cat = p.tags?.[0];
        if (cat && groups[cat]) groups[cat].push(p);
      });
      listEl.innerHTML = CATEGORY_ORDER
        .filter(c => groups[c].length > 0)
        .map(c => `
          <section class="post-group">
            <h2 class="group-title">${escapeHtml(c)}</h2>
            ${groups[c].map(postCardHtml).join('')}
          </section>
        `)
        .join('');
    } else {
      listEl.innerHTML = `
        <section class="post-group">
          ${filtered.map(postCardHtml).join('')}
        </section>
      `;
    }
  }

  function postCardHtml(p) {
    const tags = p.tags || [];
    const category = tags[0] || '未分类';
    const extraTags = tags.slice(1);
    return `
      <a class="post-card" href="${escapeHtml(p.url || `posts/${p.slug}.html`)}">
        <div class="post-meta">
          <span class="post-date">${escapeHtml(p.date)}</span>
          <span class="post-category">${escapeHtml(category)}</span>
          ${extraTags.length ? `<div class="post-tags-inline">${extraTags.map(t => `<span class="tag">${escapeHtml(t)}</span>`).join('')}</div>` : ''}
        </div>
        <h3 class="post-title">${escapeHtml(p.title)}</h3>
        ${p.summary ? `<p class="post-summary">${escapeHtml(p.summary)}</p>` : ''}
      </a>
    `;
  }

  renderList();
}
