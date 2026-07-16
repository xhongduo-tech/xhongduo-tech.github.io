/* ===================================================
   Index page: featured post, blog grid, categories, tags, search
   =================================================== */
document.addEventListener('DOMContentLoaded', () => {
  if (document.getElementById('post-list')) {
    initIndexPage();
  }
});

async function initIndexPage() {
  const CATEGORY_ORDER = [
    '全部',
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
  const featuredEl = document.getElementById('featured-section');

  try {
    const res = await fetch('posts/posts.json');
    if (!res.ok) throw new Error('fetch failed');
    posts = await res.json();
  } catch (e) {
    listEl.innerHTML = '<p class="empty-state">// 暂无文章 · 请添加 posts/posts.json</p>';
    if (featuredEl) featuredEl.style.display = 'none';
    return;
  }

  posts.sort((a, b) => new Date(b.date) - new Date(a.date));

  let activeCategory = '全部';
  let activeTag = null;
  let searchQuery = '';

  // Featured post: most recent (only when 2+ posts)
  if (featuredEl && posts.length > 1) {
    featuredEl.innerHTML = featuredCardHtml(posts[0]);
  } else if (featuredEl) {
    featuredEl.style.display = 'none';
  }

  // Category tabs
  const catTabsEl = document.getElementById('category-tabs');
  if (catTabsEl) {
    CATEGORY_ORDER.forEach(cat => {
      const btn = document.createElement('button');
      btn.className = 'category-tab' + (cat === '全部' ? ' active' : '');
      btn.dataset.cat = cat;
      btn.textContent = cat;
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

  // Tag filter
  const tagFilterEl = document.getElementById('tag-filter');
  const allTags = [...new Set(
    posts.flatMap(p => (p.tags || []).slice(1)).filter(Boolean)
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
      const matchCat = activeCategory === '全部' || cat === activeCategory;
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

    // Hide featured post from grid when showing all without filters and there are 2+ posts
    const showFeaturedInGrid = activeCategory === '全部' && !activeTag && !searchQuery && posts.length > 1;
    const gridPosts = showFeaturedInGrid ? filtered.slice(1) : filtered;

    listEl.innerHTML = gridPosts.map(blogCardHtml).join('');
  }

  function featuredCardHtml(p) {
    const tags = p.tags || [];
    const category = tags[0] || '未分类';
    return `
      <a class="featured-card" href="${escapeHtml(p.url || `posts/${p.slug}.html`)}">
        <div class="featured-image">
          ${placeholderImage(p.title, p.slug)}
        </div>
        <div class="featured-content">
          <div class="featured-meta">
            <span class="featured-category">${escapeHtml(category)}</span>
            <span class="featured-date">${escapeHtml(p.date)}</span>
          </div>
          <h2 class="featured-title">${escapeHtml(p.title)}</h2>
          <p class="featured-summary">${escapeHtml(p.summary || '')}</p>
          <span class="featured-link">阅读文章 →</span>
        </div>
      </a>
    `;
  }

  function blogCardHtml(p) {
    const tags = p.tags || [];
    const category = tags[0] || '未分类';
    return `
      <a class="blog-card" href="${escapeHtml(p.url || `posts/${p.slug}.html`)}">
        <div class="blog-card-image">
          ${placeholderImage(p.title, p.slug)}
        </div>
        <div class="blog-card-content">
          <div class="blog-card-meta">
            <span class="blog-card-category">${escapeHtml(category)}</span>
            <span class="blog-card-date">${escapeHtml(p.date)}</span>
          </div>
          <h3 class="blog-card-title">${escapeHtml(p.title)}</h3>
          <p class="blog-card-summary">${escapeHtml(p.summary || '')}</p>
          <span class="blog-card-link">阅读文章 →</span>
        </div>
      </a>
    `;
  }

  function placeholderImage(title, seed) {
    const hue = stringHash(seed || title) % 360;
    const hue2 = (hue + 40) % 360;
    const id = 'img-' + Math.abs(stringHash(seed || title)).toString(36);
    const label = escapeHtml(title.slice(0, 18));
    return `
      <svg viewBox="0 0 400 225" preserveAspectRatio="xMidYMid slice" aria-hidden="true">
        <defs>
          <linearGradient id="${id}" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stop-color="hsl(${hue}, 70%, 88%)" />
            <stop offset="100%" stop-color="hsl(${hue2}, 65%, 78%)" />
          </linearGradient>
        </defs>
        <rect width="400" height="225" fill="url(#${id})" />
        <circle cx="320" cy="50" r="80" fill="hsla(${hue2}, 70%, 60%, 0.18)" />
        <circle cx="80" cy="180" r="60" fill="hsla(${hue}, 70%, 55%, 0.14)" />
        <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle" font-family="JetBrains Mono, monospace" font-size="14" fill="hsla(${hue}, 60%, 30%, 0.45)">${label}</text>
      </svg>
    `;
  }

  function stringHash(str) {
    let h = 0;
    for (let i = 0; i < str.length; i++) {
      h = (h << 5) - h + str.charCodeAt(i);
      h |= 0;
    }
    return Math.abs(h);
  }

  renderList();
}
