<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useData, withBase } from 'vitepress'
import TopicIcon from './TopicIcon.vue'
import { data as articles } from '../data/articles.data'
import { domains, domainProgress } from '../data/domain-progress.mjs'
import {
  topicNodes,
  topicEdges,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
  TOPIC_WORLD_WIDTH,
  TOPIC_WORLD_HEIGHT,
} from '../data/topic-topology.mjs'

type TopicNode = { id: string; domain: string; title: string; titleEn: string; x: number; y: number }
type Point = { x: number; y: number }
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
const WORLD_HEIGHT = TOPIC_WORLD_HEIGHT as number
const nodes = topicNodes as TopicNode[]
const relations = topicEdges as [string, string][]
const { lang } = useData()
const english = computed(() => lang.value === 'en-US')
const svg = ref<SVGSVGElement | null>(null)
const viewport = reactive({ width: 900, height: 600 })
const zoom = ref(0.9)
const pan = reactive<Point>({ x: 24, y: 16 })
const positions = reactive<Record<string, Point>>({})
const selectedId = ref(nodes[0]?.id || '')
const drag = ref<DragState | null>(null)
let suppressNextClick = false
let resizeObserver: ResizeObserver | undefined

for (const node of nodes) positions[node.id] = { x: node.x, y: node.y }
const byId = new Map(nodes.map(node => [node.id, node]))
const domainById = new Map(domains.map(domain => [domain.id, domain]))
const domainLayouts = domains.map((domain, index) => {
  const domainNodes = nodes.filter(node => node.domain === domain.id)
  return {
    domain,
    index,
    count: domainNodes.length,
    x: (domainNodes[0]?.x || 0) - 16,
    width: WIDTH + 32,
    height: domainNodes.length * 88 + 30,
  }
})
const selected = computed(() => byId.get(selectedId.value) || nodes[0])
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
const prerequisites = computed(() => relations
  .filter(([, target]) => target === selectedId.value)
  .map(([source]) => byId.get(source)!)
  .filter(Boolean))
const dependents = computed(() => relations
  .filter(([source]) => source === selectedId.value)
  .map(([, target]) => byId.get(target)!)
  .filter(Boolean))
const edges = computed(() => relations.map(([sourceId, targetId]) => {
  const a = positions[sourceId]
  const b = positions[targetId]
  let path: string
  if (Math.abs(b.x - a.x) > 40) {
    const forward = b.x > a.x
    const direction = forward ? 1 : -1
    const startX = a.x + (forward ? WIDTH : 0)
    const endX = b.x + (forward ? 0 : WIDTH)
    const startY = a.y + HEIGHT / 2
    const endY = b.y + HEIGHT / 2
    const midX = (startX + endX) / 2
    if (Math.abs(endY - startY) < 1) {
      path = `M ${startX} ${startY} H ${endX}`
    } else {
      const vertical = endY > startY ? 1 : -1
      const radius = Math.min(7, Math.abs(endX - startX) / 4, Math.abs(endY - startY) / 2)
      path = `M ${startX} ${startY} H ${midX - direction * radius} Q ${midX} ${startY} ${midX} ${startY + vertical * radius} V ${endY - vertical * radius} Q ${midX} ${endY} ${midX + direction * radius} ${endY} H ${endX}`
    }
  } else {
    const down = b.y > a.y
    const direction = down ? 1 : -1
    const startX = a.x + WIDTH / 2
    const endX = b.x + WIDTH / 2
    const startY = a.y + (down ? HEIGHT : 0)
    const endY = b.y + (down ? 0 : HEIGHT)
    if (Math.abs(endX - startX) < 1) {
      path = `M ${startX} ${startY} V ${endY}`
    } else {
      const midY = (startY + endY) / 2
      const horizontal = endX > startX ? 1 : -1
      const radius = Math.min(7, Math.abs(endY - startY) / 4, Math.abs(endX - startX) / 2)
      path = `M ${startX} ${startY} V ${midY - direction * radius} Q ${startX} ${midY} ${startX + horizontal * radius} ${midY} H ${endX - horizontal * radius} Q ${endX} ${midY} ${endX} ${midY + direction * radius} V ${endY}`
    }
  }
  return {
    id: `${sourceId}-${targetId}`,
    path,
    active: sourceId === selectedId.value || targetId === selectedId.value,
  }
}))

function title(node: TopicNode): string {
  return english.value ? node.titleEn : node.title
}

function domainTitle(node: TopicNode): string {
  const domain = domainById.get(node.domain)
  return english.value ? domain?.titleEn || node.domain : domain?.title || node.domain
}

function iconFor(node: TopicNode): string {
  const id = node.id
  if (/linear|probability|optimization|information|scaling|statistical/.test(id)) return 'math'
  if (/python|tokenization|tokenizer|structured/.test(id)) return 'code'
  if (/embedding|vector|retrieval|reranking|rag/.test(id)) return 'retrieval'
  if (/training-data|contamination/.test(id)) return 'database'
  if (/attention|transformer|mixture|positional|normalization|multimodal/.test(id)) return 'network'
  if (/pretraining|finetuning|preference|rlhf|rl-post|sft/.test(id)) return 'training'
  if (/precision|quantization|kernel|cache|paged|parallel|serving|batching/.test(id)) return 'chip'
  if (/decoding|sampling|speculative|multi-token/.test(id)) return 'decode'
  if (/benchmark|metric|judge|evaluation|robustness|safety|guardrails|reproduction/.test(id)) return 'evaluation'
  if (/prompt|tool|agent|memory/.test(id)) return 'agent'
  if (/lab|experiment|simulation|ablation/.test(id)) return 'experiment'
  return node.domain
}

function kind(node: TopicNode): string {
  const labels: Record<string, [string, string]> = {
    foundations: ['概念与机制', 'Concept & mechanism'],
    training: ['训练方法', 'Training method'],
    inference: ['推理机制与系统', 'Inference & systems'],
    evaluation: ['评测方法', 'Evaluation method'],
    applications: ['应用模式', 'Application pattern'],
    projects: ['可复现实验', 'Reproducible experiment'],
  }
  return labels[node.domain]?.[english.value ? 1 : 0] || node.domain
}

function hasArticles(node: TopicNode): boolean {
  return articlesByTopic.value.has(node.id)
}

function selectNode(node: TopicNode) {
  selectedId.value = node.id
}

function focusNode(node: TopicNode) {
  selectedId.value = node.id
  zoom.value = Math.max(zoom.value, 0.9)
  const position = positions[node.id]
  pan.x = viewport.width / 2 - (position.x + WIDTH / 2) * zoom.value
  pan.y = viewport.height * 0.42 - (position.y + HEIGHT / 2) * zoom.value
}

function focusDomain(domain: (typeof domains)[number]) {
  const first = nodes.find(node => node.domain === domain.id)
  if (first) focusNode(first)
}

function clampZoom(value: number): number {
  return Math.min(2.4, Math.max(0.3, value))
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
  const current = nodes.map(node => positions[node.id])
  const minX = Math.min(0, ...current.map(point => point.x))
  const minY = Math.min(0, ...current.map(point => point.y))
  const maxX = Math.max(WORLD_WIDTH, ...current.map(point => point.x + WIDTH))
  const maxY = Math.max(WORLD_HEIGHT, ...current.map(point => point.y + HEIGHT))
  const width = maxX - minX
  const height = maxY - minY
  const next = clampZoom(Math.min((viewport.width - 32) / width, (viewport.height - 64) / height, 1))
  zoom.value = next
  pan.x = (viewport.width - width * next) / 2 - minX * next
  pan.y = (viewport.height - height * next) / 2 - minY * next + 16
}

function resetGraph() {
  for (const node of nodes) {
    positions[node.id].x = node.x
    positions[node.id].y = node.y
  }
  selectedId.value = nodes[0]?.id || ''
  zoom.value = 0.9
  pan.x = 24
  pan.y = 16
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
  })
  resizeObserver.observe(svg.value)
})

onBeforeUnmount(() => resizeObserver?.disconnect())
</script>

<template>
  <div class="topic-topology">
    <div class="topic-topology-summary">
      <p><strong>{{ nodes.length }}</strong> {{ english ? 'topics' : '个主题' }}<span>·</span>{{ publishedCount }} {{ english ? 'published' : '已发布' }}<span>·</span>{{ completedCount }} / {{ domains.length }} {{ english ? 'domains accepted' : '领域已验收' }}</p>
      <div class="topic-canvas-controls" role="group" :aria-label="english ? 'Topology viewport controls' : '拓扑视图控制'">
        <button type="button" :aria-label="english ? 'Zoom out' : '缩小'" :disabled="zoom <= 0.3" @click="zoomAt(zoom - 0.15)">−</button>
        <span aria-live="polite">{{ Math.round(zoom * 100) }}%</span>
        <button type="button" :aria-label="english ? 'Zoom in' : '放大'" :disabled="zoom >= 2.4" @click="zoomAt(zoom + 0.15)">+</button>
        <button type="button" class="topic-fit" @click="fitGraph">{{ english ? 'Fit all' : '全图' }}</button>
        <button type="button" class="topic-reset" @click="resetGraph">{{ english ? 'Reset' : '复位' }}</button>
      </div>
    </div>

    <div class="topic-domain-jump" role="group" :aria-label="english ? 'Focus a domain' : '定位到领域'">
      <button
        v-for="layout in domainLayouts" :key="layout.domain.id" type="button"
        :aria-pressed="selected?.domain === layout.domain.id"
        :aria-label="`${english ? layout.domain.titleEn : layout.domain.title}，${layout.count} ${english ? 'topics' : '个主题'}`"
        @click="focusDomain(layout.domain)"
      >
        <span class="topic-domain-jump-icon"><TopicIcon :name="layout.domain.id" :size="19" /></span>
        <span class="topic-domain-jump-copy"><strong>{{ english ? layout.domain.titleEn : layout.domain.title }}</strong><small>{{ layout.count }} {{ english ? 'topics' : '个主题' }}</small></span>
        <i class="topic-domain-state" :class="{ complete: domainProgress[layout.domain.id].completed }" aria-hidden="true" />
      </button>
    </div>

    <div class="topic-canvas-shell" :class="{ 'is-dragging': drag }">
      <svg
        ref="svg"
        class="topic-canvas"
        :viewBox="`0 0 ${viewport.width} ${viewport.height}`"
        role="group"
        tabindex="0"
        :aria-label="english ? 'Draggable LLM topic dependency graph' : '可拖拽的大模型主题依赖图'"
        @pointerdown="startPan"
        @pointermove="onPointerMove"
        @pointerup="endDrag"
        @pointercancel="endDrag"
        @wheel="onWheel"
        @keydown="onCanvasKeydown"
      >
        <title>{{ english ? 'LLM topic dependency graph' : '大模型主题依赖图' }}</title>
        <desc>{{ english ? 'Drag the background to pan, drag nodes to rearrange, and use the wheel or buttons to zoom. Enter selects a focused topic.' : '拖拽空白处平移，拖拽节点调整布局，使用滚轮或按钮缩放。按 Enter 选中键盘聚焦的主题。' }}</desc>
        <defs>
          <pattern id="topic-dot-grid" width="20" height="20" patternUnits="userSpaceOnUse"><circle cx="1.5" cy="1.5" r="0.85" class="topic-grid-dot" /></pattern>
          <marker id="topic-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto"><path d="M 0 0 8 4 0 8Z" /></marker>
        </defs>
        <rect class="topic-canvas-background" :width="viewport.width" :height="viewport.height" fill="url(#topic-dot-grid)" />
        <g :transform="`translate(${pan.x} ${pan.y}) scale(${zoom})`">
          <g v-for="layout in domainLayouts" :key="layout.domain.id" class="topic-domain-group" aria-hidden="true">
            <rect class="topic-domain-lane" :x="layout.x" y="74" :width="layout.width" :height="layout.height" rx="10" />
            <text class="topic-domain-label" :x="layout.x + 4" y="38">0{{ layout.index + 1 }} / {{ english ? layout.domain.titleEn : layout.domain.title }}</text>
            <text class="topic-domain-count" :x="layout.x + layout.width - 4" y="38" text-anchor="end">{{ layout.count }}</text>
            <line :x1="layout.x + 4" :x2="layout.x + layout.width - 4" y1="55" y2="55" />
          </g>
          <path
            v-for="edge in edges" :key="edge.id" :d="edge.path"
            class="topic-edge" :class="{ active: edge.active }" :marker-end="edge.active ? 'url(#topic-arrow)' : undefined"
          />
          <g
            v-for="node in nodes" :key="node.id"
            class="topic-node" :class="{ 'is-selected': selectedId === node.id, 'is-published': hasArticles(node) }"
            :transform="`translate(${positions[node.id].x} ${positions[node.id].y})`"
            role="button" tabindex="0"
            :aria-label="`${title(node)}，${domainTitle(node)}，${hasArticles(node) ? (english ? 'published' : '已发布') : (english ? 'no published article' : '尚无公开文章')}`"
            :aria-pressed="selectedId === node.id"
            @pointerdown.stop="startNodeDrag($event, node)"
            @click.stop="onNodeClick(node)"
            @keydown.enter.prevent="selectNode(node)"
            @keydown.space.prevent="selectNode(node)"
          >
            <rect class="topic-node-box" :width="WIDTH" :height="HEIGHT" rx="8" />
            <rect class="topic-node-icon-bg" x="12" y="17" width="34" height="34" rx="8" />
            <TopicIcon class="topic-node-icon" :name="iconFor(node)" :x="20" :y="25" :size="18" />
            <text class="topic-node-title" :class="{ 'is-long': title(node).length > 22 }" x="56" y="34">{{ title(node) }}</text>
            <text class="topic-node-subtitle" x="56" y="53">{{ kind(node) }}</text>
            <text class="topic-node-state" :x="WIDTH - 12" y="18" text-anchor="end">{{ hasArticles(node) ? (english ? 'Published' : '已发布') : (english ? 'Pending' : '待发布') }}</text>
            <circle class="topic-node-port" cx="0" :cy="HEIGHT / 2" r="3" />
            <circle class="topic-node-port" :cx="WIDTH" :cy="HEIGHT / 2" r="3" />
          </g>
        </g>
      </svg>
      <p class="topic-canvas-hint">{{ english ? 'Drag to pan · drag nodes to rearrange · wheel to zoom · arrow keys on canvas to pan' : '拖拽空白平移 · 拖动节点调整布局 · 滚轮缩放 · 画布聚焦后可用方向键平移' }}</p>
    </div>

    <section v-if="selected" class="topic-detail" aria-live="polite" :aria-label="english ? 'Selected topic' : '当前主题'">
      <div class="topic-detail-heading">
        <span class="topic-detail-icon"><TopicIcon :name="iconFor(selected)" :size="22" /></span>
        <div class="topic-detail-title">
          <p>{{ domainTitle(selected) }} · {{ selectedArticles.length ? (english ? 'Published' : '已发布') : (english ? 'Unpublished' : '待发布') }}</p>
          <h3>{{ title(selected) }}</h3>
          <span v-if="!english" lang="en">{{ selected.titleEn }}</span>
        </div>
      </div>
      <div class="topic-detail-relations">
        <p><strong>{{ english ? 'Prerequisites' : '直接前置' }}</strong></p>
        <div v-if="prerequisites.length" class="topic-relation-links">
          <button v-for="node in prerequisites" :key="node.id" type="button" @click="focusNode(node)">{{ title(node) }}</button>
        </div>
        <p v-else class="topic-no-prerequisite">{{ english ? 'Foundational topic' : '基础主题，无直接前置' }}</p>
        <p class="topic-dependent-count">{{ english ? `${dependents.length} dependent topics` : `${dependents.length} 个后续主题` }}</p>
      </div>
      <div class="topic-detail-reading">
        <template v-if="selectedArticles.length">
          <a v-for="article in selectedArticles" :key="article.url" :href="withBase(article.url)">{{ article.title }} <span aria-hidden="true">↗</span></a>
        </template>
        <p v-else>{{ english ? 'No public article is linked to this topic yet.' : '这个主题还没有对应的公开文章。' }}</p>
      </div>
    </section>
    <p class="topic-topology-note">{{ english ? 'This is a public knowledge index, not a schedule or mastery record. Only real published articles are linked; domain acceptance still requires public evidence.' : '这是公共知识索引，不是日程或掌握记录。只关联真实已发布文章；领域验收仍须公开证据。' }}</p>
  </div>
</template>
