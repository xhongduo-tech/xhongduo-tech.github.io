/* ===================================================
   Post page behavior: reading progress, math, highlight
   =================================================== */
document.addEventListener('DOMContentLoaded', () => {
  initReadingProgress();
  initMath();
  initHighlight();
});

function initReadingProgress() {
  const bar = document.getElementById('progress-bar');
  if (!bar) return;
  const update = () => {
    const docH = document.documentElement.scrollHeight - window.innerHeight;
    const pct = docH > 0 ? Math.min(100, (window.scrollY / docH) * 100) : 0;
    bar.style.width = pct + '%';
  };
  window.addEventListener('scroll', update, { passive: true });
  update();
}

function initMath() {
  if (typeof renderMathInElement === 'undefined') return;
  const prose = document.querySelector('.prose');
  if (!prose) return;
  renderMathInElement(prose, {
    delimiters: [
      { left: '$$', right: '$$', display: true },
      { left: '$', right: '$', display: false }
    ],
    throwOnError: false
  });
}

function initHighlight() {
  if (typeof hljs !== 'undefined') {
    hljs.highlightAll();
  }
}
