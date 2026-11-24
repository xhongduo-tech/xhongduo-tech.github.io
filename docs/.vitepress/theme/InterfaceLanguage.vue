<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, ref, watch } from 'vue'
import { useData, useRouter, withBase } from 'vitepress'

type Preference = 'auto' | 'zh' | 'en'
type Language = 'zh' | 'en'
const STORAGE_KEY = 'xhd-interface-language-v1'
const { page, frontmatter } = useData()
const router = useRouter()
const preference = ref<Preference>('auto')
const open = ref(false)
const root = ref<HTMLElement | null>(null)
const trigger = ref<HTMLButtonElement | null>(null)
const menu = ref<HTMLElement | null>(null)
const current = computed<Language>(() => page.value.relativePath.startsWith('en/') ? 'en' : 'zh')
const contentLanguage = computed<Language>(() => {
  const declared = String(frontmatter.value.contentLang || '')
  return declared ? (declared.toLowerCase().startsWith('en') ? 'en' : 'zh') : current.value
})
const options = computed(() => [
  { value: 'auto' as const, label: current.value === 'en' ? 'Auto' : '智能', hint: current.value === 'en' ? 'Follow content' : '跟随内容语言' },
  { value: 'zh' as const, label: '中文', hint: current.value === 'en' ? 'Chinese interface' : '中文界面' },
  { value: 'en' as const, label: 'English', hint: current.value === 'en' ? 'English interface' : '英文界面' },
])
const triggerLabel = computed(() => options.value.find(option => option.value === preference.value)?.label || options.value[0].label)

function target(language: Language): string | undefined {
  if (language === current.value) return undefined
  const translations = frontmatter.value.translations
  return translations && typeof translations[language] === 'string' ? translations[language] : undefined
}

function available(language: Preference): boolean {
  return language === 'auto' || language === current.value || Boolean(target(language))
}

function visit(language: Language) {
  const destination = target(language)
  if (destination) router.go(withBase(destination))
}

function select(next: Preference) {
  preference.value = next
  try { localStorage.setItem(STORAGE_KEY, next) } catch { /* Storage is optional. */ }
  visit(next === 'auto' ? contentLanguage.value : next)
}

function menuItems(): HTMLButtonElement[] {
  return Array.from(menu.value?.querySelectorAll<HTMLButtonElement>('[role="menuitemradio"]') || [])
}

function focusItem(index: number) {
  const items = menuItems()
  items[index]?.focus()
}

async function openMenu() {
  if (open.value) return
  open.value = true
  await nextTick()
  const items = menuItems()
  const selected = items.findIndex(item => item.getAttribute('aria-checked') === 'true' && !item.disabled)
  const first = items.findIndex(item => !item.disabled)
  focusItem(selected >= 0 ? selected : first)
}

function closeMenu(restoreFocus = false) {
  open.value = false
  if (restoreFocus) nextTick(() => trigger.value?.focus())
}

function choose(next: Preference) {
  if (!available(next)) return
  select(next)
  closeMenu(true)
}

function moveFocus(event: KeyboardEvent) {
  const items = menuItems()
  if (!items.length) return
  const currentIndex = items.indexOf(document.activeElement as HTMLButtonElement)
  if (event.key === 'Home' || event.key === 'End') {
    const direction = event.key === 'Home' ? 1 : -1
    let index = event.key === 'Home' ? 0 : items.length - 1
    while (items[index]?.disabled) index += direction
    focusItem(index)
    return
  }
  const direction = event.key === 'ArrowDown' ? 1 : -1
  let index = currentIndex
  for (let count = 0; count < items.length; count++) {
    index = (index + direction + items.length) % items.length
    if (!items[index].disabled) {
      focusItem(index)
      break
    }
  }
}

function onMenuKeydown(event: KeyboardEvent) {
  if (['ArrowDown', 'ArrowUp', 'Home', 'End'].includes(event.key)) {
    event.preventDefault()
    moveFocus(event)
  } else if (event.key === 'Escape') {
    event.preventDefault()
    closeMenu(true)
  } else if (event.key === 'Tab') {
    closeMenu()
  }
}

function onOutsidePointer(event: PointerEvent) {
  if (open.value && !root.value?.contains(event.target as Node)) closeMenu()
}

onMounted(() => {
  document.addEventListener('pointerdown', onOutsidePointer)
  try {
    const stored = localStorage.getItem(STORAGE_KEY)
    if (stored === 'auto' || stored === 'zh' || stored === 'en') preference.value = stored
  } catch { /* Keep automatic mode. */ }
  watch(() => page.value.relativePath, () => {
    closeMenu()
    visit(preference.value === 'auto' ? contentLanguage.value : preference.value)
  }, { immediate: true })
})

onBeforeUnmount(() => document.removeEventListener('pointerdown', onOutsidePointer))
</script>

<template>
  <div ref="root" class="interface-language" :class="{ 'is-open': open }">
    <button
      ref="trigger"
      type="button"
      class="language-trigger"
      aria-haspopup="menu"
      :aria-expanded="open"
      :aria-label="current === 'en' ? `Interface language: ${triggerLabel}` : `界面语言：${triggerLabel}`"
      :title="current === 'en' ? 'Interface language' : '界面语言'"
      @click="open ? closeMenu() : openMenu()"
      @keydown.down.prevent="openMenu"
      @keydown.up.prevent="openMenu"
    >
      <svg class="language-globe" viewBox="0 0 20 20" aria-hidden="true"><circle cx="10" cy="10" r="7.25" /><path d="M3 10h14M10 2.75c2 2.05 3 4.47 3 7.25s-1 5.2-3 7.25c-2-2.05-3-4.47-3-7.25s1-5.2 3-7.25Z" /></svg>
      <span class="language-trigger-label">{{ triggerLabel }}</span>
      <svg class="language-chevron" viewBox="0 0 12 12" aria-hidden="true"><path d="m3 4.5 3 3 3-3" /></svg>
    </button>
    <div
      v-if="open"
      ref="menu"
      class="language-menu"
      role="menu"
      :aria-label="current === 'en' ? 'Interface language' : '界面语言'"
      @keydown="onMenuKeydown"
    >
      <button
        v-for="option in options"
        :key="option.value"
        type="button"
        role="menuitemradio"
        :data-language="option.value"
        tabindex="-1"
        :aria-checked="preference === option.value"
        :disabled="!available(option.value)"
        @click="choose(option.value)"
      >
        <span class="language-option-copy">
          <span class="language-option-label">{{ option.label }}</span>
          <span class="language-option-hint">{{ option.hint }}</span>
        </span>
        <svg v-if="preference === option.value" class="language-check" viewBox="0 0 16 16" aria-hidden="true"><path d="m3.5 8.2 2.8 2.8 6.2-6.2" /></svg>
      </button>
    </div>
  </div>
</template>
