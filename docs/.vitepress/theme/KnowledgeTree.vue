<script setup>
import { ref } from 'vue'
import { withBase } from 'vitepress'
import { trees } from '../data/knowledge-tree.mjs'

const expanded = ref({}) // treeId -> bool
const expandedBranch = ref({}) // treeId.branchIdx -> bool

function toggleTree(id) {
  expanded.value[id] = !expanded.value[id]
}
function toggleBranch(key) {
  expandedBranch.value[key] = !expandedBranch.value[key]
}
function isOpen(key) {
  return expandedBranch.value[key] !== false // 默认展开
}
function href(node) {
  return node.path ? withBase(`/posts/${node.path}/`) : null
}
function tagLabel(tag) {
  return tag === 'ref' ? '引用' : tag === 'add' ? '待建' : ''
}
const tagClass = (t) => (t === 'add' ? 'kt-tag-add' : 'kt-tag-ref')
</script>

<template>
  <div class="kt">
    <div class="kt-intro">
      <p>
        全人类知识按领域划分为 <strong>12 棵知识树</strong>，每棵从基础 → 核心 → 进阶 → 专业 → 前沿逐级展开。
        跨树重合的节点以「引用」呈现（不重复建专题）；「待建」为规划补充的专题。
      </p>
    </div>

    <div v-for="tree in trees" :key="tree.id" class="kt-tree">
      <div class="kt-tree-head" :class="{ open: expanded[tree.id] }" @click="toggleTree(tree.id)">
        <span class="kt-caret">{{ expanded[tree.id] ? '▾' : '▸' }}</span>
        <h3 class="kt-tree-name">{{ tree.name }}</h3>
        <span class="kt-tree-count">{{ tree.branches.reduce((s, b) => s + b.nodes.length, 0) }} 节点</span>
      </div>

      <p v-if="expanded[tree.id]" class="kt-tree-desc">{{ tree.desc }}</p>

      <div v-show="expanded[tree.id]" class="kt-branches">
        <div
          v-for="(branch, bi) in tree.branches"
          :key="bi"
          class="kt-branch"
          :class="{ open: isOpen(tree.id + '.' + bi) }"
        >
          <div class="kt-branch-head" @click="toggleBranch(tree.id + '.' + bi)">
            <span class="kt-caret">{{ isOpen(tree.id + '.' + bi) ? '▾' : '▸' }}</span>
            <span class="kt-level">{{ branch.level }}</span>
            <span class="kt-branch-count">{{ branch.nodes.length }}</span>
          </div>
          <ul v-show="isOpen(tree.id + '.' + bi)" class="kt-nodes">
            <li v-for="(node, ni) in branch.nodes" :key="ni" class="kt-node">
              <a v-if="href(node)" :href="href(node)" class="kt-link">{{ node.name }}</a>
              <span v-else class="kt-link kt-link-add">{{ node.name }}</span>
              <span
                v-if="node.tag"
                class="kt-tag"
                :class="tagClass(node.tag)"
                >{{ tagLabel(node.tag) }}</span
              >
            </li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.kt { max-width: 72rem; margin: 0 auto; padding: 1rem 0 4rem; }
.kt-intro { color: #666; font-size: 0.95rem; margin-bottom: 1.5rem; }
.kt-tree { border: 1px solid #e3e1dc; border-radius: 6px; margin-bottom: 1rem; overflow: hidden; }
.kt-tree-head { display: flex; align-items: baseline; gap: 0.5rem; padding: 0.8rem 1rem; cursor: pointer; background: #faf9f7; }
.kt-tree-head:hover { background: #f3f1ec; }
.kt-tree-name { margin: 0; font-size: 1.15rem; font-weight: 600; }
.kt-tree-count { margin-left: auto; color: #999; font-size: 0.8rem; }
.kt-tree-desc { padding: 0.4rem 1rem; color: #777; font-size: 0.9rem; }
.kt-caret { color: #b0aca3; font-size: 0.8rem; width: 1em; }
.kt-branches { padding: 0.3rem 0 0.6rem; }
.kt-branch-head { display: flex; align-items: baseline; gap: 0.5rem; padding: 0.45rem 1.2rem; cursor: pointer; }
.kt-branch-head:hover { background: #f7f5f1; }
.kt-level { font-weight: 600; font-size: 0.95rem; color: #444; }
.kt-branch-count { color: #bbb; font-size: 0.75rem; }
.kt-nodes { margin: 0; padding: 0.2rem 1rem 0.6rem 2.4rem; list-style: none; }
.kt-node { padding: 0.18rem 0; font-size: 0.92rem; }
.kt-link { color: #1a4f8b; text-decoration: none; border-bottom: 1px dotted #9fb7d0; }
.kt-link:hover { border-bottom-style: solid; }
.kt-link-add { color: #888; border-bottom: none; cursor: default; }
.kt-tag { font-size: 0.7rem; padding: 0.05rem 0.4rem; border-radius: 3px; margin-left: 0.4rem; vertical-align: 1px; }
.kt-tag-ref { background: #eef2f7; color: #3a6ea5; }
.kt-tag-add { background: #f7efdc; color: #a07d2d; }
@media (prefers-color-scheme: dark) {
  .kt-intro, .kt-tree-desc { color: #9a9a9a; }
  .kt-tree { border-color: #3a3a3a; }
  .kt-tree-head { background: #262626; }
  .kt-tree-head:hover { background: #2e2e2e; }
  .kt-branch-head:hover { background: #2a2a2a; }
  .kt-level { color: #cfcfcf; }
  .kt-caret { color: #666; }
  .kt-link { color: #6ba3d8; border-bottom-color: #3d5a77; }
  .kt-link-add { color: #777; }
  .kt-tag-ref { background: #26303a; color: #7ba9d6; }
  .kt-tag-add { background: #3a3320; color: #c9a45a; }
}
</style>
