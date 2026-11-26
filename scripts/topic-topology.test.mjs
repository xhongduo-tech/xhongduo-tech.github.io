import test from 'node:test'
import assert from 'node:assert/strict'
import {
  topicNodes,
  topicEdges,
  topicGroups,
  topicResearch,
  TOPIC_NODE_WIDTH,
  TOPIC_NODE_HEIGHT,
  TOPIC_WORLD_WIDTH,
  TOPIC_WORLD_HEIGHT,
} from '../docs/.vitepress/data/topic-topology.mjs'

const DOMAIN_X = {
  foundations: 60,
  training: 360,
  inference: 660,
  evaluation: 960,
  applications: 1260,
  projects: 1560,
}

const ORIGINAL_TOPIC_IDS = [
  'linear-algebra', 'probability-statistics', 'optimization', 'python-pytorch',
  'information-theory', 'tokenization', 'embeddings', 'attention-mechanism',
  'positional-encoding', 'normalization-residuals', 'transformer-architecture',
  'mixture-of-experts', 'pretraining-objectives', 'training-data', 'mixed-precision',
  'distributed-training', 'scaling-laws', 'supervised-finetuning', 'parameter-efficient-ft',
  'preference-modeling', 'rlhf', 'direct-preference-opt', 'rl-post-training',
  'autoregressive-decoding', 'sampling-strategies', 'kv-cache', 'attention-kernels',
  'batching-scheduling', 'paged-attention', 'quantization', 'speculative-decoding',
  'multi-token-prediction', 'model-parallel-inference', 'serving-systems',
  'benchmark-design', 'automatic-metrics', 'llm-as-judge', 'human-evaluation',
  'statistical-significance', 'robustness-evaluation', 'safety-evaluation',
  'contamination-analysis', 'evaluation-reproduction', 'prompt-engineering',
  'structured-generation', 'vector-retrieval', 'reranking', 'retrieval-augmented-gen',
  'tool-use', 'agent-planning', 'agent-memory', 'multimodal-fusion', 'guardrails',
  'tokenizer-lab', 'tiny-transformer-lab', 'attention-kernel-bench', 'kv-cache-simulation',
  'decoding-ablation', 'quantization-benchmark', 'sft-experiment', 'preference-learning-lab',
  'rag-evaluation-lab', 'inference-serving-bench',
]

const FRONTIER_TOPIC_IDS = [
  'linear-attention', 'selective-state-spaces', 'state-space-duality',
  'hybrid-sequence-models', 'test-time-memory', 'superposition-features',
  'sparse-autoencoders', 'cross-layer-transcoders', 'attribution-graphs',
  'activation-steering', 'interpretability-evaluation', 'instruction-hierarchy',
  'deliberative-alignment', 'safe-completions', 'reward-tampering',
  'alignment-faking', 'agentic-misalignment', 'parallel-sampling',
  'verifier-guided-search', 'beam-lookahead-search', 'tool-integrated-reasoning',
  'sequential-revision', 'evolutionary-inference', 'adaptive-reasoning-budget',
  'test-time-compute-scaling', 'scalable-oversight', 'cot-faithfulness-monitoring',
  'evaluation-awareness', 'agent-trajectory-evaluation', 'computer-use',
  'multi-agent-orchestration', 'multimodal-reasoning', 'visual-tool-reasoning',
]

const MATURITIES = new Set(['core', 'advanced', 'frontier', 'applied'])

test('topic nodes form a professional bilingual index with unique stable IDs', () => {
  assert.equal(topicNodes.length, 185)
  assert.deepEqual(new Set(topicNodes.map(node => node.domain)), new Set(Object.keys(DOMAIN_X)))
  const ids = new Set(topicNodes.map(node => node.id))
  for (const id of ORIGINAL_TOPIC_IDS) assert.ok(ids.has(id), `Original topic ID was removed: ${id}`)
  for (const id of ['ppo', 'grpo', 'rlvr', 'reward-hacking', 'prefill-decode', 'continuous-batching', 'prefix-caching', 'speculative-correctness', 'train-inference-weight-sync']) {
    assert.ok(ids.has(id), `Missing professional core topic: ${id}`)
  }
  for (const id of FRONTIER_TOPIC_IDS) assert.ok(ids.has(id), `Missing frontier topic: ${id}`)

  for (const node of topicNodes) {
    assert.deepEqual(Object.keys(node).sort(), ['domain', 'group', 'id', 'maturity', 'order', 'title', 'titleEn', 'x', 'y'].sort())
    assert.match(node.id, /^[a-z0-9]+(?:-[a-z0-9]+)*$/)
    assert.ok(Object.hasOwn(DOMAIN_X, node.domain), `Invalid domain: ${node.domain}`)
    assert.ok(MATURITIES.has(node.maturity), `Invalid maturity: ${node.id} -> ${node.maturity}`)
    assert.ok(node.title.trim().length > 0)
    assert.ok(node.titleEn.trim().length > 0)
    assert.ok(node.titleEn.length <= 44, `English title is too long: ${node.titleEn}`)
  }
})

test('maturity labels distinguish foundational, advanced, frontier and experimental work', () => {
  const byId = new Map(topicNodes.map(node => [node.id, node]))
  assert.equal(byId.get('linear-algebra').maturity, 'core')
  assert.equal(byId.get('ppo').maturity, 'advanced')
  assert.equal(byId.get('sparse-autoencoders').maturity, 'frontier')
  assert.equal(byId.get('tokenizer-lab').maturity, 'applied')
  assert.equal(byId.get('latent-reasoning').maturity, 'frontier')
  assert.equal(byId.get('reasoning-internalization').maturity, 'frontier')
  assert.equal(byId.get('multimodal-reasoning').maturity, 'frontier')
})

test('specialization groups are bilingual, ordered and contain their declared nodes', () => {
  assert.equal(topicGroups.length, 36)
  const groupIds = new Set()
  for (const group of topicGroups) {
    assert.ok(!groupIds.has(group.id), `Duplicate group: ${group.id}`)
    groupIds.add(group.id)
    assert.ok(Object.hasOwn(DOMAIN_X, group.domain))
    assert.ok(group.title && group.titleEn && group.summary && group.summaryEn)
    const nodes = topicNodes.filter(node => node.group === group.id)
    assert.equal(group.count, nodes.length)
    assert.ok(nodes.length > 0)
    assert.ok(nodes.every(node => node.domain === group.domain))
    const siblings = topicGroups.filter(candidate => candidate.domain === group.domain)
    assert.equal(group.index, siblings.findIndex(candidate => candidate.id === group.id))
  }
  for (const node of topicNodes) assert.ok(groupIds.has(node.group), `Unknown group: ${node.group}`)
})

test('topic nodes fit their specialization lanes and the world bounds', () => {
  assert.equal(TOPIC_NODE_WIDTH, 240)
  assert.equal(TOPIC_NODE_HEIGHT, 68)
  assert.equal(TOPIC_WORLD_WIDTH, 1850)

  for (const group of topicGroups) {
    const nodes = topicNodes.filter(node => node.group === group.id)
    nodes.forEach((node, index) => {
      assert.equal(node.x, DOMAIN_X[group.domain], `Unexpected column for ${node.id}`)
      assert.equal(node.y, group.y + 42 + index * 84, `Unexpected row for ${node.id}`)
      assert.ok(node.x >= group.x && node.x + TOPIC_NODE_WIDTH <= group.x + group.width)
      assert.ok(node.y >= group.y && node.y + TOPIC_NODE_HEIGHT <= group.y + group.height)
    })
  }

  for (const node of topicNodes) {
    assert.ok(Number.isFinite(node.x) && Number.isFinite(node.y))
    assert.ok(node.x >= 0 && node.y >= 0)
    assert.ok(node.x + TOPIC_NODE_WIDTH <= TOPIC_WORLD_WIDTH, `Node exceeds world width: ${node.id}`)
    assert.ok(node.y + TOPIC_NODE_HEIGHT <= TOPIC_WORLD_HEIGHT, `Node exceeds world height: ${node.id}`)
  }
  assert.equal(TOPIC_WORLD_HEIGHT, Math.max(...topicGroups.map(group => group.y + group.height)) + 64)
})

test('topic edges are unique, non-reflexive and reference existing nodes', () => {
  assert.equal(topicEdges.length, 387)
  const ids = new Set(topicNodes.map(node => node.id))
  const edges = new Set()
  const connected = new Set()
  for (const edge of topicEdges) {
    assert.ok(Array.isArray(edge) && edge.length === 2)
    const [source, target] = edge
    assert.ok(ids.has(source), `Missing edge source: ${source}`)
    assert.ok(ids.has(target), `Missing edge target: ${target}`)
    assert.notEqual(source, target, `Self-edge: ${source}`)
    const key = `${source}\0${target}`
    assert.ok(!edges.has(key), `Duplicate edge: ${source} -> ${target}`)
    edges.add(key)
    connected.add(source)
    connected.add(target)
  }
  for (const node of topicNodes) assert.ok(connected.has(node.id), `Isolated topic: ${node.id}`)
})

test('learning numbers are unique, contiguous and follow an acyclic prerequisite order', () => {
  const byId = new Map(topicNodes.map(node => [node.id, node]))
  assert.deepEqual([...topicNodes.map(node => node.order)].sort((a, b) => a - b), Array.from({ length: topicNodes.length }, (_, index) => index + 1))
  const adjacency = new Map(topicNodes.map(node => [node.id, []]))
  const indegree = new Map(topicNodes.map(node => [node.id, 0]))
  for (const [source, target] of topicEdges) {
    assert.ok(byId.get(source).order < byId.get(target).order, `Learning order violates ${source} -> ${target}`)
    adjacency.get(source).push(target)
    indegree.set(target, indegree.get(target) + 1)
  }

  const queue = [...indegree].filter(([, degree]) => degree === 0).map(([id]) => id)
  let visited = 0
  for (let index = 0; index < queue.length; index += 1) {
    const source = queue[index]
    visited += 1
    for (const target of adjacency.get(source)) {
      indegree.set(target, indegree.get(target) - 1)
      if (indegree.get(target) === 0) queue.push(target)
    }
  }
  assert.equal(visited, topicNodes.length, 'Topic dependency graph contains a cycle')
})

test('research references are attributable, typed and attached to real specialization groups', () => {
  assert.equal(topicResearch.length, 43)
  const groupIds = new Set(topicGroups.map(group => group.id))
  const referencedGroups = new Set()
  const references = new Set()
  const kinds = new Set(['paper', 'report', 'engineering', 'position'])
  const currentYear = new Date().getUTCFullYear()

  for (const reference of topicResearch) {
    assert.deepEqual(Object.keys(reference).sort(), ['group', 'kind', 'source', 'title', 'url', 'year'].sort())
    assert.ok(groupIds.has(reference.group), `Unknown research group: ${reference.group}`)
    assert.ok(kinds.has(reference.kind), `Invalid research kind: ${reference.kind}`)
    assert.ok(reference.title.trim().length > 0)
    assert.ok(reference.source.trim().length > 0)
    assert.ok(Number.isInteger(reference.year) && reference.year >= 2020 && reference.year <= currentYear, `Invalid research year: ${reference.year}`)
    const url = new URL(reference.url)
    assert.equal(url.protocol, 'https:', `Research URL must use HTTPS: ${reference.url}`)
    const key = `${reference.group}\0${reference.url}`
    assert.ok(!references.has(key), `Duplicate research reference in ${reference.group}: ${reference.url}`)
    references.add(key)
    referencedGroups.add(reference.group)
  }
  assert.equal(referencedGroups.size, 16)
})
