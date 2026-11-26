<script setup lang="ts">
import { computed, nextTick, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useData, withBase } from 'vitepress'
import TopicIcon from './TopicIcon.vue'
import { routeTopicEdge } from './topic-routing.mjs'
import { buildFocusedDomainLayout } from './topic-layout.mjs'
import { data as articles } from '../data/articles.data'
import { domains, domainProgress } from '../data/domain-progress.mjs'
import {
  topicNodes,
  topicEdges,
  topicGroups,
  topicResearch,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
  TOPIC_WORLD_WIDTH,
} from '../data/topic-topology.mjs'

type TopicNode = {
  id: string
  domain: string
  group: string
  title: string
  titleEn: string
  order: number
  maturity: string
  x: number
  y: number
}
type TopicGroup = {
  id: string
  domain: string
  title: string
  titleEn: string
  summary: string
  summaryEn: string
  maturity: string
  index: number
  count: number
  x: number
  y: number
  width: number
  height: number
}
type Point = { x: number; y: number }
type ResearchReference = {
  group: string
  title: string
  source: string
  year: number
  kind: 'paper' | 'report' | 'engineering' | 'position'
  url: string
}
type DragState = {
  pointerId: number
  type: 'pan' | 'node'
  start: Point
  origin: Point
  nodeId?: string
  moved: boolean
}

const WIDTH = TOPIC_NODE_WIDTH as number
const HEIGHT = TOPIC_NODE_HEIGHT as number
const WORLD_WIDTH = TOPIC_WORLD_WIDTH as number
const nodes = topicNodes as TopicNode[]
const groups = topicGroups as TopicGroup[]
const research = topicResearch as ResearchReference[]
const relations = topicEdges as [string, string][]
const { lang } = useData()
const english = computed(() => lang.value === 'en-US')
const svg = ref<SVGSVGElement | null>(null)
const viewport = reactive({ width: 900, height: 620 })
const zoom = ref(0.9)
const pan = reactive<Point>({ x: 24, y: 16 })
const positions = reactive<Record<string, Point>>({})
const selectedId = ref(nodes[0]?.id || '')
const activeDomain = ref('foundations')
const emphasis = ref(false)
const drag = ref<DragState | null>(null)
const draggedNodes = new Set<string>()
const activeLayout = ref(buildFocusedDomainLayout(nodes, groups, 'foundations', 1))
let suppressNextClick = false
let resizeObserver: ResizeObserver | undefined

for (const node of nodes) positions[node.id] = { x: node.x, y: node.y }
const byId = new Map(nodes.map(node => [node.id, node]))
const groupById = new Map(groups.map(group => [group.id, group]))
const domainById = new Map(domains.map(domain => [domain.id, domain]))
const orderedNodes = [...nodes].sort((a, b) => a.order - b.order)
const prerequisitesById = new Map(nodes.map(node => [node.id, [] as string[]]))
const dependentsById = new Map(nodes.map(node => [node.id, [] as string[]]))
for (const [source, target] of relations) {
  prerequisitesById.get(target)?.push(source)
  dependentsById.get(source)?.push(target)
}
const domainLayouts = domains.map((domain, index) => {
  const domainNodes = nodes.filter(node => node.domain === domain.id)
  const domainGroups = groups.filter(group => group.domain === domain.id)
  const y = 74
  const bottom = Math.max(...domainGroups.map(group => group.y + group.height)) + 22
  return {
    domain,
    index,
    count: domainNodes.length,
    x: (domainGroups[0]?.x || 0) - 12,
    y,
    width: WIDTH + 52,
    height: bottom - y,
  }
})
const overviewDomainLayouts = domains.map((domain, index) => {
  const domainGroups = groups.filter(group => group.domain === domain.id)
  const y = 74
  const bottom = 110 + (domainGroups.length - 1) * 100 + 76 + 22
  return {
    domain,
    index,
    count: nodes.filter(node => node.domain === domain.id).length,
    x: (domainGroups[0]?.x || 0) - 12,
    y,
    width: WIDTH + 52,
    height: bottom - y,
  }
})
const visibleNodes = computed(() => activeDomain.value ? nodes.filter(node => node.domain === activeDomain.value) : [])
const visibleGroups = computed(() => activeDomain.value ? activeLayout.value.groups : [])
const visibleDomainLayouts = computed(() => {
  if (!activeDomain.value) return overviewDomainLayouts
  const layout = domainLayouts.find(candidate => candidate.domain.id === activeDomain.value)
  const bounds = activeLayout.value.domain
  return layout && bounds ? [{ ...layout, ...bounds }] : []
})
const selected = computed(() => byId.get(selectedId.value) || nodes[0])
const selectedGroup = computed(() => groupById.get(selected.value?.group))
const completedCount = computed(() => Object.values(domainProgress).filter(item => item.completed).length)
const articlesByTopic = computed(() => {
  const grouped = new Map<string, typeof articles>()
  for (const article of articles) {
    if (!article.topicId) continue
    grouped.set(article.topicId, [...(grouped.get(article.topicId) || []), article])
  }
  return grouped
})
const publishedCount = computed(() => nodes.filter(node => articlesByTopic.value.has(node.id)).length)
const selectedArticles = computed(() => articlesByTopic.value.get(selectedId.value) || [])
const selectedResearch = computed(() => research.filter(reference => reference.group === selected.value?.group))
const prerequisites = computed(() => (prerequisitesById.get(selectedId.value) || []).map(id => byId.get(id)!).filter(Boolean))
const dependents = computed(() => (dependentsById.get(selectedId.value) || []).map(id => byId.get(id)!).filter(Boolean))
const relatedIds = computed(() => new Set([
  selectedId.value,
  ...(prerequisitesById.get(selectedId.value) || []),
  ...(dependentsById.get(selectedId.value) || []),
]))
const selectedOrderIndex = computed(() => orderedNodes.findIndex(node => node.id === selectedId.value))
const previousNode = computed(() => orderedNodes[selectedOrderIndex.value - 1])
const nextNode = computed(() => orderedNodes[selectedOrderIndex.value + 1])
const edges = computed(() => {
  const visible = visibleNodes.value.map(node => ({ id: node.id, ...positions[node.id] }))
  const visibleIds = new Set(visible.map(node => node.id))
  return relations.filter(([source, target]) => visibleIds.has(source) && visibleIds.has(target)).map(([sourceId, targetId]) => {
    const source = visible.find(node => node.id === sourceId)!
    const target = visible.find(node => node.id === targetId)!
    const route = routeTopicEdge(source, target, visible, WIDTH, HEIGHT)
    return {
      id: `${sourceId}-${targetId}`,
      path: route.path,
      detour: route.points.length > 2,
      active: sourceId === selectedId.value || targetId === selectedId.value,
      muted: emphasis.value && sourceId !== selectedId.value && targetId !== selectedId.value,
    }
  })
})

function title(node: TopicNode): string {
  return english.value ? node.titleEn : node.title
}

function domainTitle(node: TopicNode): string {
  const domain = domainById.get(node.domain)
  return english.value ? domain?.titleEn || node.domain : domain?.title || node.domain
}

function groupTitle(group?: TopicGroup): string {
  return group ? (english.value ? group.titleEn : group.title) : ''
}

function groupSummary(group?: TopicGroup): string {
  return group ? (english.value ? group.summaryEn : group.summary) : ''
}

function maturityLabel(maturity: string, short = false): string {
  const labels: Record<string, [string, string, string, string]> = {
    core: ['主干', '基础主干', 'Core', 'Core foundation'],
    advanced: ['进阶', '进阶方法', 'Advanced', 'Advanced method'],
    frontier: ['前沿', '前沿研究', 'Frontier', 'Frontier research'],
    applied: ['实验', '实验验证', 'Lab', 'Experimental validation'],
  }
  return labels[maturity]?.[english.value ? (short ? 2 : 3) : (short ? 0 : 1)] || maturity
}

function researchKindLabel(kind: ResearchReference['kind']): string {
  const labels = {
    paper: ['论文', 'Paper'],
    report: ['技术报告', 'Technical report'],
    engineering: ['工程实践', 'Engineering'],
    position: ['立场论文', 'Position paper'],
  }
  return labels[kind][english.value ? 1 : 0]
}

function overviewY(group: TopicGroup): number {
  return 110 + group.index * 100
}

function stepLabel(node: TopicNode): string {
  return `#${String(node.order).padStart(3, '0')}`
}

function iconFor(node: TopicNode): string {
  const id = node.id
  if (node.domain === 'projects') return 'experiment'
  if (/linear|probability|optimization|information|scaling|statistical/.test(id)) return 'math'
  if (/python|tokenization|tokenizer|structured|template/.test(id)) return 'code'
  if (/embedding|vector|retrieval|reranking|rag/.test(id)) return 'retrieval'
  if (/data|contamination|cache|prefix/.test(id)) return 'database'
  if (/reward|preference/.test(id)) return 'reward'
  if (/safety|refusal|guardrail/.test(id)) return 'shield'
  if (/roofline|metric|observability|benchmark/.test(id)) return 'performance'
  if (/distillation/.test(id)) return 'distill'
  if (/parallel|distributed|rollout|sync|disaggregation/.test(id)) return 'parallel'
  if (/attention|transformer|mixture|positional|normalization|multimodal/.test(id)) return 'network'
  if (/pretraining|finetuning|rlhf|rl-post|rlvr|policy|ppo|grpo|sft|lora/.test(id)) return 'training'
  if (/precision|quantization|kernel|paged|batching|prefill|gpu/.test(id)) return 'chip'
  if (/decoding|sampling|speculative|multi-token/.test(id)) return 'decode'
  if (/benchmark|judge|evaluation|robustness|reproduction|regression/.test(id)) return 'evaluation'
  if (/prompt|tool|agent|memory|context/.test(id)) return 'agent'
  return node.domain
}

function hasArticles(node: TopicNode): boolean {
  return articlesByTopic.value.has(node.id)
}

function selectNode(node: TopicNode) {
  selectedId.value = node.id
  emphasis.value = true
}

function prepareLayout(domain: string, force = false) {
  const columns = viewport.width >= 680 ? 2 : 1
  if (!force && activeLayout.value.id === domain && activeLayout.value.columns === columns) return
  activeLayout.value = buildFocusedDomainLayout(nodes, groups, domain, columns)
  for (const [id, position] of Object.entries(activeLayout.value.positions)) {
    if (!draggedNodes.has(id)) positions[id] = { ...position }
  }
}

function focusNode(node: TopicNode, emphasize = true) {
  if (activeDomain.value !== node.domain) {
    activeDomain.value = node.domain
    prepareLayout(node.domain)
  }
  selectedId.value = node.id
  emphasis.value = emphasize
  zoom.value = Math.max(zoom.value, 0.9)
  const position = positions[node.id]
  const bounds = activeLayout.value.domain
  const centerX = bounds ? bounds.x + bounds.width / 2 : position.x + WIDTH / 2
  pan.x = viewport.width / 2 - centerX * zoom.value
  pan.y = viewport.height * 0.3 - (position.y + HEIGHT / 2) * zoom.value
}

function focusDomain(domain: (typeof domains)[number]) {
  const first = nodes.find(node => node.domain === domain.id)
  if (!first) return
  activeDomain.value = domain.id
  prepareLayout(domain.id)
  emphasis.value = false
  selectedId.value = first.id
  nextTick(() => focusNode(first, false))
}

function focusGroup(group: TopicGroup) {
  const first = nodes.find(node => node.group === group.id)
  if (!first) return
  activeDomain.value = group.domain
  prepareLayout(group.domain)
  nextTick(() => focusNode(first, false))
}

function showAll() {
  activeDomain.value = ''
  emphasis.value = false
  nextTick(fitGraph)
}

function clampZoom(value: number): number {
  return Math.min(2.4, Math.max(0.12, value))
}

function zoomAt(nextValue: number, center?: Point) {
  const next = clampZoom(nextValue)
  const point = center || { x: viewport.width / 2, y: viewport.height / 2 }
  const ratio = next / zoom.value
  pan.x = point.x - (point.x - pan.x) * ratio
  pan.y = point.y - (point.y - pan.y) * ratio
  zoom.value = next
}

function fitGraph() {
  const layouts = visibleDomainLayouts.value
  const current = visibleNodes.value.map(node => positions[node.id])
  if (!layouts.length || (activeDomain.value && !current.length)) return
  const minX = activeDomain.value ? Math.min(...layouts.map(layout => layout.x), ...current.map(point => point.x - 16)) : 0
  const minY = activeDomain.value ? Math.min(...layouts.map(layout => layout.y), ...current.map(point => point.y - 16)) : 0
  const maxX = activeDomain.value ? Math.max(...layouts.map(layout => layout.x + layout.width), ...current.map(point => point.x + WIDTH + 16)) : WORLD_WIDTH
  const maxY = Math.max(...layouts.map(layout => layout.y + layout.height), ...current.map(point => point.y + HEIGHT + 16))
  const width = maxX - minX
  const height = maxY - minY
  const next = clampZoom(Math.min((viewport.width - 36) / width, (viewport.height - 56) / height, 1))
  zoom.value = next
  pan.x = (viewport.width - width * next) / 2 - minX * next
  pan.y = (viewport.height - height * next) / 2 - minY * next + 12
}

function resetGraph() {
  draggedNodes.clear()
  for (const node of nodes) {
    positions[node.id].x = node.x
    positions[node.id].y = node.y
  }
  activeDomain.value = 'foundations'
  prepareLayout('foundations', true)
  selectedId.value = nodes[0]?.id || ''
  emphasis.value = false
  zoom.value = 0.9
  nextTick(() => focusNode(nodes[0], false))
}

function svgPoint(event: PointerEvent | WheelEvent): Point {
  if (!svg.value) return { x: 0, y: 0 }
  const point = svg.value.createSVGPoint()
  point.x = event.clientX
  point.y = event.clientY
  const matrix = svg.value.getScreenCTM()
  const transformed = matrix ? point.matrixTransform(matrix.inverse()) : point
  return { x: transformed.x, y: transformed.y }
}

function onWheel(event: WheelEvent) {
  event.preventDefault()
  zoomAt(zoom.value * Math.exp(-event.deltaY * 0.001), svgPoint(event))
}

function startPan(event: PointerEvent) {
  if (event.button !== 0 || drag.value) return
  event.preventDefault()
  svg.value?.focus()
  svg.value?.setPointerCapture(event.pointerId)
  drag.value = { pointerId: event.pointerId, type: 'pan', start: svgPoint(event), origin: { ...pan }, moved: false }
}

function startNodeDrag(event: PointerEvent, node: TopicNode) {
  if (event.button !== 0 || drag.value) return
  event.preventDefault()
  selectedId.value = node.id
  emphasis.value = true
  ;(event.currentTarget as SVGGElement).focus()
  svg.value?.setPointerCapture(event.pointerId)
  drag.value = {
    pointerId: event.pointerId,
    type: 'node',
    start: svgPoint(event),
    origin: { ...positions[node.id] },
    nodeId: node.id,
    moved: false,
  }
}

function onPointerMove(event: PointerEvent) {
  const active = drag.value
  if (!active || active.pointerId !== event.pointerId) return
  event.preventDefault()
  const point = svgPoint(event)
  const dx = point.x - active.start.x
  const dy = point.y - active.start.y
  if (Math.hypot(dx, dy) > 3) active.moved = true
  if (active.type === 'pan') {
    pan.x = active.origin.x + dx
    pan.y = active.origin.y + dy
  } else if (active.nodeId) {
    positions[active.nodeId].x = active.origin.x + dx / zoom.value
    positions[active.nodeId].y = active.origin.y + dy / zoom.value
  }
}

function endDrag(event: PointerEvent) {
  const active = drag.value
  if (!active || active.pointerId !== event.pointerId) return
  suppressNextClick = active.type === 'node' && active.moved
  if (active.type === 'node' && active.moved && active.nodeId) draggedNodes.add(active.nodeId)
  drag.value = null
  if (svg.value?.hasPointerCapture(event.pointerId)) svg.value.releasePointerCapture(event.pointerId)
}

function onNodeClick(node: TopicNode) {
  if (suppressNextClick) {
    suppressNextClick = false
    return
  }
  selectNode(node)
}

function onCanvasKeydown(event: KeyboardEvent) {
  if (event.target !== event.currentTarget) return
  const moves: Record<string, Point> = {
    ArrowLeft: { x: 40, y: 0 },
    ArrowRight: { x: -40, y: 0 },
    ArrowUp: { x: 0, y: 40 },
    ArrowDown: { x: 0, y: -40 },
  }
  if (moves[event.key]) {
    event.preventDefault()
    pan.x += moves[event.key].x
    pan.y += moves[event.key].y
  } else if (['+', '=', '-'].includes(event.key)) {
    event.preventDefault()
    zoomAt(zoom.value + (event.key === '-' ? -0.15 : 0.15))
  } else if (event.key === '0') {
    event.preventDefault()
    resetGraph()
  }
}

onMounted(() => {
  if (!svg.value) return
  resizeObserver = new ResizeObserver(entries => {
    const rect = entries[0]?.contentRect
    if (!rect) return
    viewport.width = rect.width
    viewport.height = rect.height
    if (activeDomain.value && activeLayout.value.columns !== (rect.width >= 680 ? 2 : 1)) {
      prepareLayout(activeDomain.value, true)
      nextTick(() => selected.value && focusNode(selected.value, emphasis.value))
    }
  })
  resizeObserver.observe(svg.value)
})

onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<template>
  <div class="topic-topology">
    <div class="topic-learning-guide">
      <span class="topic-learning-guide-label"><TopicIcon name="path" :size="17" />{{ english ? 'LEARNING PATH' : '学习主线' }}</span>
      <p>{{ english ? 'Follow the numbered topics and return to any missing prerequisite. Your daily tasks remain in Today’s plan.' : '按编号推进，前置不足时随时回跳；每天的具体任务以“今日 Todo”为准。' }}</p>
    </div>

    <div class="topic-topology-summary">
      <p><strong>{{ nodes.length }}</strong> {{ english ? 'topics' : '个主题' }}<span>·</span>{{ groups.length }} {{ english ? 'tracks' : '个专业方向' }}<span>·</span>{{ publishedCount }} {{ english ? 'published' : '已发布' }}<span>·</span>{{ completedCount }} / {{ domains.length }} {{ english ? 'domains accepted' : '领域已验收' }}</p>
      <div class="topic-canvas-controls" role="group" :aria-label="english ? 'Topology viewport controls' : '拓扑视图控制'">
        <button type="button" :aria-label="english ? 'Zoom out' : '缩小'" :disabled="zoom <= 0.12" @click="zoomAt(zoom - 0.15)">−</button>
        <span aria-live="polite">{{ Math.round(zoom * 100) }}%</span>
        <button type="button" :aria-label="english ? 'Zoom in' : '放大'" :disabled="zoom >= 2.4" @click="zoomAt(zoom + 0.15)">+</button>
        <button type="button" class="topic-fit" @click="fitGraph">{{ english ? 'Fit view' : '适配' }}</button>
        <button type="button" class="topic-reset" @click="resetGraph">{{ english ? 'Reset' : '复位' }}</button>
      </div>
    </div>

    <div class="topic-domain-jump" role="group" :aria-label="english ? 'Focus a domain' : '定位到领域'">
      <button type="button" class="topic-domain-overview" :aria-pressed="activeDomain === ''" @click="showAll">
        <span class="topic-domain-jump-icon"><TopicIcon name="path" :size="19" /></span>
        <span class="topic-domain-jump-copy"><strong>{{ english ? 'Domain overview' : '领域总览' }}</strong><small>{{ groups.length }} {{ english ? 'tracks' : '个专业方向' }}</small></span>
      </button>
      <button
        v-for="layout in domainLayouts" :key="layout.domain.id" type="button"
        :aria-pressed="activeDomain === layout.domain.id"
        :aria-label="`${english ? layout.domain.titleEn : layout.domain.title}，${layout.count} ${english ? 'topics' : '个主题'}`"
        @click="focusDomain(layout.domain)"
      >
        <span class="topic-domain-jump-icon"><TopicIcon :name="layout.domain.id" :size="19" /></span>
        <span class="topic-domain-jump-copy"><strong>{{ english ? layout.domain.titleEn : layout.domain.title }}</strong><small>{{ layout.count }} {{ english ? 'topics' : '个主题' }}</small></span>
        <i class="topic-domain-state" :class="{ complete: domainProgress[layout.domain.id].completed }" aria-hidden="true" />
      </button>
    </div>

    <div v-if="activeDomain && selected" class="topic-group-navigation">
      <p><span>{{ english ? 'DOMAIN' : '当前领域' }}</span><strong>{{ domainTitle(selected) }}</strong></p>
      <div class="topic-group-jump" role="group" :aria-label="english ? 'Focus a specialization' : '定位到专业方向'">
        <button
          v-for="group in visibleGroups" :key="group.id" type="button"
          :aria-pressed="selected.group === group.id"
          @click="focusGroup(group)"
        >
          <span>{{ String(group.index + 1).padStart(2, '0') }}</span>
          <strong>{{ groupTitle(group) }}</strong>
          <small>{{ group.count }}</small>
        </button>
      </div>
    </div>

    <div class="topic-canvas-shell" :class="{ 'is-dragging': drag }">
      <svg
        ref="svg"
        class="topic-canvas"
        :viewBox="`0 0 ${viewport.width} ${viewport.height}`"
        role="group"
        tabindex="0"
        :aria-label="english ? 'Hierarchical LLM topic dependency graph' : '分层大模型主题依赖图'"
        @pointerdown="startPan"
        @pointermove="onPointerMove"
        @pointerup="endDrag"
        @pointercancel="endDrag"
        @wheel="onWheel"
        @keydown="onCanvasKeydown"
      >
        <title>{{ english ? 'Hierarchical LLM topic dependency graph' : '分层大模型主题依赖图' }}</title>
        <desc>{{ english ? 'Choose a domain and specialization above. Drag to pan, drag nodes to rearrange, and use the wheel or buttons to zoom. Enter selects a focused topic.' : '通过上方领域和专业方向定位。拖拽空白处平移，拖拽节点调整布局，使用滚轮或按钮缩放。按 Enter 选中键盘聚焦的主题。' }}</desc>
        <defs>
          <pattern id="topic-dot-grid" width="22" height="22" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="0.8" class="topic-grid-dot" /></pattern>
          <marker id="topic-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 8 4 0 8Z" /></marker>
        </defs>
        <rect class="topic-canvas-background" :width="viewport.width" :height="viewport.height" fill="url(#topic-dot-grid)" />
        <g :transform="`translate(${pan.x} ${pan.y}) scale(${zoom})`" :class="{ 'is-emphasizing': emphasis }">
          <g v-for="layout in visibleDomainLayouts" :key="layout.domain.id" class="topic-domain-group" aria-hidden="true">
            <rect class="topic-domain-lane" :x="layout.x" :y="layout.y" :width="layout.width" :height="layout.height" rx="13" />
            <text class="topic-domain-label" :x="layout.x + 4" y="40">0{{ layout.index + 1 }} / {{ english ? layout.domain.titleEn : layout.domain.title }}</text>
            <text class="topic-domain-count" :x="layout.x + layout.width - 4" y="40" text-anchor="end">{{ layout.count }}</text>
            <line :x1="layout.x + 4" :x2="layout.x + layout.width - 4" y1="57" y2="57" />
          </g>
          <g v-for="group in visibleGroups" :key="group.id" class="topic-track-group" aria-hidden="true">
            <rect class="topic-track-lane" :x="group.x" :y="group.y" :width="group.width" :height="group.height" rx="10" />
            <text class="topic-track-index" :x="group.x + 12" :y="group.y + 21">{{ String(group.index + 1).padStart(2, '0') }}</text>
            <text class="topic-track-title" :x="group.x + 40" :y="group.y + 21">{{ groupTitle(group) }}</text>
            <text class="topic-track-count" :x="group.x + group.width - 12" :y="group.y + 21" text-anchor="end">{{ group.count }}</text>
            <line :x1="group.x + 12" :x2="group.x + group.width - 12" :y1="group.y + 31" :y2="group.y + 31" />
          </g>
          <g v-if="!activeDomain" class="topic-overview-tracks">
            <g
              v-for="group in groups" :key="group.id"
              class="topic-overview-track"
              :transform="`translate(${group.x} ${overviewY(group)})`"
              role="button" tabindex="0"
              :aria-label="`${groupTitle(group)}，${group.count} ${english ? 'topics' : '个主题'}`"
              @pointerdown.stop
              @click.stop="focusGroup(group)"
              @keydown.enter.prevent="focusGroup(group)"
              @keydown.space.prevent="focusGroup(group)"
            >
              <rect class="topic-overview-track-box" :width="group.width" height="76" rx="10" />
              <text class="topic-track-index" x="14" y="27">{{ String(group.index + 1).padStart(2, '0') }}</text>
              <text class="topic-overview-track-title" x="42" y="28">{{ groupTitle(group) }}</text>
              <text class="topic-overview-track-meta" x="42" y="51">{{ english ? `${group.count} topics · ${maturityLabel(group.maturity, true)}` : `${group.count} 个主题 · ${maturityLabel(group.maturity, true)}` }}</text>
              <text class="topic-overview-track-arrow" :x="group.width - 14" y="44" text-anchor="end">→</text>
            </g>
          </g>
          <path
            v-for="edge in edges" :key="edge.id" :d="edge.path"
            class="topic-edge" :class="{ active: edge.active && emphasis, detour: edge.detour, muted: edge.muted }" :marker-end="edge.active && emphasis ? 'url(#topic-arrow)' : undefined"
          />
          <g
            v-for="node in visibleNodes" :key="node.id"
            class="topic-node"
            :class="{
              'is-selected': selectedId === node.id,
              'is-published': hasArticles(node),
              'is-frontier': node.maturity === 'frontier',
              'is-related': emphasis && relatedIds.has(node.id),
              'is-muted': emphasis && !relatedIds.has(node.id),
            }"
            :transform="`translate(${positions[node.id].x} ${positions[node.id].y})`"
            role="button" tabindex="0"
            :aria-label="`${stepLabel(node)} ${title(node)}，${groupTitle(groupById.get(node.group))}，${maturityLabel(node.maturity)}，${hasArticles(node) ? (english ? 'published' : '已发布') : (english ? 'no published article' : '尚无公开文章')}`"
            :aria-pressed="selectedId === node.id"
            @pointerdown.stop="startNodeDrag($event, node)"
            @click.stop="onNodeClick(node)"
            @keydown.enter.prevent="selectNode(node)"
            @keydown.space.prevent="selectNode(node)"
          >
            <rect class="topic-node-box" :width="WIDTH" :height="HEIGHT" rx="9" />
            <rect class="topic-node-accent" width="3" :height="HEIGHT - 20" x="0" y="10" rx="1.5" />
            <rect class="topic-node-icon-bg" x="12" y="17" width="34" height="34" rx="9" />
            <TopicIcon class="topic-node-icon" :name="iconFor(node)" :x="20" :y="25" :size="18" />
            <text class="topic-node-title" :class="{ 'is-long': title(node).length > 24 }" x="56" y="34">{{ title(node) }}</text>
            <text class="topic-node-subtitle" x="56" y="53">{{ english ? `${maturityLabel(node.maturity, true)} · ${prerequisitesById.get(node.id)?.length || 0} prereq · ${dependentsById.get(node.id)?.length || 0} next` : `${maturityLabel(node.maturity, true)} · 先修 ${prerequisitesById.get(node.id)?.length || 0} · 后续 ${dependentsById.get(node.id)?.length || 0}` }}</text>
            <text class="topic-node-state" :x="WIDTH - 12" y="18" text-anchor="end">{{ stepLabel(node) }}</text>
            <circle v-if="hasArticles(node)" class="topic-node-published" :cx="WIDTH - 13" :cy="HEIGHT - 12" r="3" />
            <circle class="topic-node-port" cx="0" :cy="HEIGHT / 2" r="2.5" />
            <circle class="topic-node-port" :cx="WIDTH" :cy="HEIGHT / 2" r="2.5" />
            <circle class="topic-node-port" :cx="WIDTH / 2" cy="0" r="2.5" />
            <circle class="topic-node-port" :cx="WIDTH / 2" :cy="HEIGHT" r="2.5" />
          </g>
        </g>
      </svg>
      <p class="topic-canvas-hint">{{ english ? 'Drag to pan · drag nodes to rearrange · wheel to zoom · arrow keys on canvas to pan' : '拖拽空白平移 · 拖动节点调整布局 · 滚轮缩放 · 画布聚焦后可用方向键平移' }}</p>
    </div>

    <section v-if="selected" class="topic-detail" aria-live="polite" :aria-label="english ? 'Selected topic' : '当前主题'">
      <div class="topic-detail-main">
        <div class="topic-detail-heading">
          <span class="topic-detail-icon"><TopicIcon :name="iconFor(selected)" :size="22" /></span>
          <div class="topic-detail-title">
            <p>{{ domainTitle(selected) }} / {{ groupTitle(selectedGroup) }}</p>
            <h3>{{ title(selected) }}</h3>
            <span v-if="!english" lang="en">{{ selected.titleEn }}</span>
          </div>
          <div class="topic-detail-badges">
            <span class="topic-detail-maturity" :data-maturity="selected.maturity">{{ maturityLabel(selected.maturity) }}</span>
            <span class="topic-detail-step">{{ stepLabel(selected) }}</span>
          </div>
        </div>
        <p class="topic-detail-summary">{{ groupSummary(selectedGroup) }}</p>
      </div>

      <div class="topic-detail-sequence" :aria-label="english ? 'Learning sequence' : '学习顺序'">
        <button type="button" :disabled="!previousNode" @click="previousNode && focusNode(previousNode)">
          <span>← {{ english ? 'Previous' : '上一主题' }}</span><strong>{{ previousNode ? title(previousNode) : (english ? 'Start' : '起点') }}</strong>
        </button>
        <button type="button" :disabled="!nextNode" @click="nextNode && focusNode(nextNode)">
          <span>{{ english ? 'Next' : '下一主题' }} →</span><strong>{{ nextNode ? title(nextNode) : (english ? 'Complete' : '终点') }}</strong>
        </button>
      </div>

      <div class="topic-detail-relations">
        <section>
          <p><strong>{{ english ? 'Prerequisites' : '先修这些主题' }}</strong></p>
          <div v-if="prerequisites.length" class="topic-relation-links">
            <button v-for="node in prerequisites" :key="node.id" type="button" @click="focusNode(node)"><span>{{ stepLabel(node) }}</span>{{ title(node) }}</button>
          </div>
          <p v-else class="topic-no-prerequisite">{{ english ? 'Foundational topic; start here.' : '基础主题，可以从这里开始。' }}</p>
        </section>
        <section>
          <p><strong>{{ english ? 'What this unlocks' : '掌握后继续' }}</strong></p>
          <div v-if="dependents.length" class="topic-relation-links">
            <button v-for="node in dependents" :key="node.id" type="button" @click="focusNode(node)"><span>{{ stepLabel(node) }}</span>{{ title(node) }}</button>
          </div>
          <p v-else class="topic-no-prerequisite">{{ english ? 'No direct dependent topics.' : '暂无直接后续主题。' }}</p>
        </section>
      </div>

      <div v-if="selectedResearch.length" class="topic-detail-research">
        <p><strong>{{ english ? 'Research evidence' : '研究证据' }}</strong><span>{{ english ? 'Sources support the mechanism, not personal reproduction.' : '资料支持机制学习，不代表本人复现。' }}</span></p>
        <div class="topic-research-links">
          <a v-for="reference in selectedResearch" :key="reference.url" :href="reference.url" target="_blank" rel="noopener noreferrer">
            <strong>{{ reference.title }} <span aria-hidden="true">↗</span></strong>
            <span>{{ reference.source }} · {{ reference.year }} · {{ researchKindLabel(reference.kind) }}</span>
          </a>
        </div>
      </div>

      <div class="topic-detail-reading">
        <template v-if="selectedArticles.length">
          <a v-for="article in selectedArticles" :key="article.url" :href="withBase(article.url)">{{ article.title }} <span aria-hidden="true">↗</span></a>
        </template>
        <p v-else>{{ english ? 'No public article is linked yet. Pending status is not a mastery record.' : '尚无对应的公开文章；“待发布”不代表学习或掌握状态。' }}</p>
      </div>
    </section>
    <p class="topic-topology-note">{{ english ? 'Numbers are a recommended order computed from public prerequisites, not a calendar or mastery record. Daily tasks come only from Today’s plan; domain acceptance still requires public evidence.' : '编号由公开前置关系计算，是推荐学习顺序，不是日程或掌握记录。每日任务仅来自“今日 Todo”；领域验收仍须公开证据。' }}</p>
  </div>
</template>
