/**
 * Tufted-Blog 交互的 VitePress(SPA) 移植版：
 * 主题切换、返回顶部、代码块行号+复制按钮、左侧 TOC、标题内英文斜体。
 * initEnhancements 每会话执行一次；enhancePage 每次路由变化后执行。
 */

const ICONS = {
  sun: `<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" viewBox="0 0 256 256"><path d="M120,40V16a8,8,0,0,1,16,0V40a8,8,0,0,1-16,0Zm8,24a64,64,0,1,0,64,64A64.07,64.07,0,0,0,128,64ZM58.34,69.66A8,8,0,0,0,69.66,58.34l-16-16A8,8,0,0,0,42.34,53.66Zm0,116.68-16,16a8,8,0,0,0,11.32,11.32l16-16a8,8,0,0,0-11.32-11.32ZM192,72a8,8,0,0,0,5.66-2.34l16-16a8,8,0,0,0-11.32-11.32l-16,16A8,8,0,0,0,192,72Zm5.66,114.34a8,8,0,0,0-11.32,11.32l16,16a8,8,0,0,0-11.32-11.32ZM48,128a8,8,0,0,0-8-8H16a8,8,0,0,0,0,16H40A8,8,0,0,0,48,128Zm80,80a8,8,0,0,0-8,8v24a8,8,0,0,1,16,0V216a8,8,0,0,0-8-8Zm112-88H216a8,8,0,0,0,0,16h24a8,8,0,0,0,0-16Z"></path></svg>`,
  moon: `<svg xmlns="http://www.w3.org/2000/svg" width="1em" height="1em" fill="currentColor" viewBox="0 0 256 256"><path d="M235.54,150.21a104.84,104.84,0,0,1-37,52.91A104,104,0,0,1,32,120,103.09,103.09,0,0,1,52.88,57.48a104.84,104.84,0,0,1,52.91-37,8,8,0,0,1,10,10,88.08,88.08,0,0,0,109.8,109.8,8,8,0,0,1,10,10Z"></path></svg>`,
}
const THEME_KEY = 'theme-preference'

function getStoredTheme() {
  try {
    return sessionStorage.getItem(THEME_KEY)
  } catch {
    return null
  }
}
function systemTheme() {
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}
function updateToggleButton(theme: string) {
  const button = document.getElementById('theme-toggle')
  if (!button) return
  if (theme === 'dark') {
    button.classList.add('is-dark')
    button.setAttribute('aria-label', '切换到浅色模式')
    button.innerHTML = ICONS.sun
  } else {
    button.classList.remove('is-dark')
    button.setAttribute('aria-label', '切换到深色模式')
    button.innerHTML = ICONS.moon
  }
}

/* ---------- 返回顶部 ---------- */
function initBackToTop() {
  const button = document.createElement('button')
  button.id = 'page-jump-btn'
  button.className = 'page-jump-btn'
  button.type = 'button'
  button.setAttribute('aria-label', '返回顶部')
  button.innerHTML =
    '<svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M12 19V5"></path><path d="m5 12 7-7 7 7"></path></svg>'
  button.addEventListener('click', () => window.scrollTo({ top: 0, behavior: 'smooth' }))
  document.body.appendChild(button)
  const update = () => button.classList.toggle('is-visible', window.scrollY > 300)
  update()
  window.addEventListener('scroll', update, { passive: true })
}

export function initEnhancements() {
  // 主题切换
  const button = document.getElementById('theme-toggle')
  button?.addEventListener('click', () => {
    const next =
      (document.documentElement.getAttribute('data-theme') || systemTheme()) === 'dark'
        ? 'light'
        : 'dark'
    try {
      sessionStorage.setItem(THEME_KEY, next)
    } catch {}
    document.documentElement.setAttribute('data-theme', next)
    updateToggleButton(next)
  })
  updateToggleButton(getStoredTheme() || systemTheme())
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', (e) => {
    if (!getStoredTheme()) {
      const t = e.matches ? 'dark' : 'light'
      document.documentElement.setAttribute('data-theme', t)
      updateToggleButton(t)
    }
  })
  initBackToTop()
}

/* ---------- 标题内英文片段斜体（format-headings.js） ---------- */
function formatHeadings() {
  const isAscii = (ch: string) => ch.charCodeAt(0) <= 0x7f
  const hasAscii = (t: string) => Array.from(t).some(isAscii)
  const processNode = (node: ChildNode) => {
    if (node.nodeType === 3) {
      const text = node.nodeValue ?? ''
      if (!hasAscii(text)) return
      const fragment = document.createDocumentFragment()
      let last = 0
      let i = 0
      while (i < text.length) {
        if (!isAscii(text[i])) {
          i++
          continue
        }
        if (i > last) fragment.appendChild(document.createTextNode(text.substring(last, i)))
        let end = i + 1
        while (end < text.length && isAscii(text[end])) end++
        const span = document.createElement('span')
        span.className = 'heading-en'
        span.textContent = text.substring(i, end)
        fragment.appendChild(span)
        last = end
        i = end
      }
      if (last < text.length) fragment.appendChild(document.createTextNode(text.substring(last)))
      node.parentNode?.replaceChild(fragment, node)
    } else if (node.nodeType === 1) {
      Array.from(node.childNodes).forEach(processNode)
    }
  }
  document
    .querySelectorAll('.tuf-article h1, .tuf-article h2, .tuf-article h3, .tuf-article h4')
    .forEach((el) => processNode(el))
}

/* ---------- 代码块：行号 + 复制按钮（code-blocks.js） ---------- */
function enhanceCodeBlocks() {
  document.querySelectorAll('.tuf-article pre > code').forEach((codeBlock) => {
    const pre = codeBlock.parentElement as HTMLElement
    if (!pre.querySelector('.line-numbers-rows')) {
      const clone = codeBlock.cloneNode(true) as HTMLElement
      clone.querySelectorAll('br').forEach((br) => br.replaceWith('\n'))
      const clean = (clone.textContent || '').replace(/\n$/, '')
      const lineCount = clean.split(/\r\n|\r|\n/).length
      const rows = document.createElement('span')
      rows.className = 'line-numbers-rows'
      for (let i = 1; i <= lineCount; i++) {
        const span = document.createElement('span')
        span.textContent = String(i)
        rows.appendChild(span)
      }
      pre.insertBefore(rows, codeBlock)
      pre.classList.add('has-line-numbers')
    }
    if (pre.querySelector('.copy-button')) return
    const copyButton = document.createElement('button')
    copyButton.className = 'copy-button'
    copyButton.textContent = 'Copy'
    copyButton.addEventListener('click', () => {
      const clone = codeBlock.cloneNode(true) as HTMLElement
      clone.querySelectorAll('br').forEach((br) => br.replaceWith('\n'))
      navigator.clipboard
        .writeText(clone.textContent || '')
        .then(() => {
          copyButton.textContent = 'Copied!'
          copyButton.classList.add('copied')
          setTimeout(() => {
            copyButton.textContent = 'Copy'
            copyButton.classList.remove('copied')
          }, 2000)
        })
        .catch(() => {
          copyButton.textContent = 'Error'
        })
    })
    pre.style.position = 'relative'
    pre.appendChild(copyButton)
  })
}

/* ---------- 边注编号：正文中插入上标数字，与边注内容前缀对应 ----------
   复用 tufte-base.css 自带的 sidenote-counter 计数器与 .sidenote-number
   视觉样式；只在正文里插入一个空 <span> 作为编号锚点，具体数字由 CSS
   计数器渲染，因此增删边注会自动重新编号，无需手动维护序号。 */
function numberMarginNotes() {
  document.querySelectorAll('.tuf-article .marginnote').forEach((note) => {
    if (note.previousElementSibling?.classList.contains('sidenote-number')) return
    const marker = document.createElement('span')
    marker.className = 'sidenote-number'
    note.parentNode?.insertBefore(marker, note)
  })
}

/* ---------- 左侧浮动 TOC（toc.js，≥3 个三级标题时启用） ---------- */
function buildToc() {
  document.querySelector('.toc-sidebar')?.remove()
  const section = document.querySelector('.tuf-article > section')
  if (!section) return
  const headings = Array.from(section.querySelectorAll('h2, h3')) as HTMLElement[]
  if (headings.filter((h) => h.tagName === 'H3').length < 3) return

  const nav = document.createElement('nav')
  nav.className = 'toc-sidebar'
  nav.setAttribute('aria-label', '文章目录')
  const list = document.createElement('ol')
  const usedIds = new Set<string>()
  headings.forEach((heading, index) => {
    let id = heading.id || `toc-${index + 1}`
    let suffix = 2
    while (usedIds.has(id)) id = `${id}-${suffix++}`
    heading.id = id
    usedIds.add(id)
    const item = document.createElement('li')
    item.classList.add(`toc-${heading.tagName.toLowerCase()}`)
    if (index > 0) item.classList.add('toc-after-title')
    const link = document.createElement('a')
    link.href = `#${id}`
    link.textContent = heading.textContent?.trim() || ''
    item.appendChild(link)
    list.appendChild(item)
  })
  nav.appendChild(list)

  nav.addEventListener('click', (event) => {
    const link = (event.target as HTMLElement).closest('a')
    if (!link) return
    const target = document.getElementById(link.hash.slice(1))
    if (!target) return
    event.preventDefault()
    target.scrollIntoView({ behavior: 'smooth', block: 'start' })
    history.replaceState(null, '', link.hash)
  })

  const linksById = new Map(
    Array.from(nav.querySelectorAll('a')).map((l) => [l.hash.slice(1), l]),
  )
  const setActive = (id: string) => {
    nav.querySelector('a.is-active')?.classList.remove('is-active')
    linksById.get(id)?.classList.add('is-active')
  }
  setActive(headings[0].id)
  const observer = new IntersectionObserver(
    (entries) => entries.forEach((e) => e.isIntersecting && setActive((e.target as HTMLElement).id)),
    { rootMargin: '0px 0px -70% 0px', threshold: 0 },
  )
  headings.forEach((h) => observer.observe(h))

  document.body.insertBefore(nav, document.querySelector('#app'))
}

export function enhancePage() {
  formatHeadings()
  enhanceCodeBlocks()
  numberMarginNotes()
  buildToc()
}
