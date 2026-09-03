/**
 * 写作需要的页面增强：代码复制、边注编号、长文 TOC、MathJax 重排。
 * enhancePage 在每次路由变化后执行。
 */
export function enhancePage() {
  enhanceCodeBlocks()
  numberMarginNotes()
  buildToc()
  typesetMath()
}

function enhanceCodeBlocks() {
  document.querySelectorAll('.tuf-article pre > code').forEach((codeBlock) => {
    const pre = codeBlock.parentElement as HTMLElement
    if (!pre.querySelector('.copy-button')) {
      const copyButton = document.createElement('button')
      copyButton.className = 'copy-button'
      copyButton.type = 'button'
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
    }

    if (pre.querySelector('.line-numbers-rows')) return
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
  })
}

function numberMarginNotes() {
  document.querySelectorAll('.tuf-article .marginnote').forEach((note) => {
    if (note.previousElementSibling?.classList.contains('sidenote-number')) return
    const marker = document.createElement('span')
    marker.className = 'sidenote-number'
    note.parentNode?.insertBefore(marker, note)
  })
}

let tocCleanup: (() => void) | null = null

function buildToc() {
  document.querySelector('.toc-sidebar')?.remove()
  tocCleanup?.()
  tocCleanup = null

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
    Array.from(nav.querySelectorAll('a')).map((l) => {
      l.classList.add('toc-link')
      return [l.hash.slice(1), l as HTMLAnchorElement]
    }),
  )
  const setActive = (id: string) => {
    nav.querySelector('a.is-active')?.classList.remove('is-active')
    linksById.get(id)?.classList.add('is-active')
  }

  let ticking = false
  const update = () => {
    ticking = false
    const trigger = window.innerHeight * 0.33
    let currentId = headings[0].id
    for (const h of headings) {
      if (h.getBoundingClientRect().top <= trigger) currentId = h.id
      else break
    }
    setActive(currentId)
  }
  const onScroll = () => {
    if (!ticking) {
      requestAnimationFrame(update)
      ticking = true
    }
  }
  update()
  window.addEventListener('scroll', onScroll, { passive: true })
  window.addEventListener('resize', onScroll, { passive: true })
  tocCleanup = () => {
    window.removeEventListener('scroll', onScroll)
    window.removeEventListener('resize', onScroll)
  }

  document.body.insertBefore(nav, document.querySelector('#app'))
}

function typesetMath() {
  const mj = (window as any).MathJax
  if (mj?.typesetPromise) {
    mj.typesetPromise().catch(() => {})
  }
}
