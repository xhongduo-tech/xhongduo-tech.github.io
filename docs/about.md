---
layout: page
title: 关于徐鸿铎
description: 大模型部署与平台开发实践，以及对模型原理、推理系统和训练方法的持续学习。
kind: about
status: published
sidebar: false
aside: false
outline: false
---

<div class="profile-page">

<header class="profile-header">
  <p class="profile-overline" lang="en">About</p>

# 徐鸿铎 {#profile}

  <p class="profile-bio">我在工作中参与大模型部署与模型管理平台开发。部署时遇到的延迟、资源与稳定性问题，让我开始追踪模型如何生成、训练如何改变输出，以及怎样用实验定位变化的来源。</p>
  <nav class="profile-links" aria-label="作者链接">
    <a href="https://github.com/xhongduo-tech" target="_blank" rel="noopener noreferrer">GitHub <span aria-hidden="true">↗</span></a>
    <a href="/roadmap/">完整知识地图 <span aria-hidden="true">→</span></a>
  </nav>
</header>

<section class="profile-section" aria-labelledby="profile-interests">

## 关注方向 {#profile-interests}

  <ul class="profile-focus">
    <li>
      <a href="/inference/">
        <h3>推理算法与服务系统</h3>
        <p>从一次生成的状态出发，追踪解码、KV 缓存与请求调度。</p>
        <span class="profile-arrow" aria-hidden="true">→</span>
      </a>
    </li>
    <li>
      <a href="/training/">
        <h3>数据与监督微调</h3>
        <p>关注训练样本的标签位置、LoRA 的更新路径和独立样本上的表现。</p>
        <span class="profile-arrow" aria-hidden="true">→</span>
      </a>
    </li>
    <li>
      <a href="/evaluation/">
        <h3>偏好优化与模型评测</h3>
        <p>比较偏好优化前后的逐例差异，追查收益、退化与评测条件。</p>
        <span class="profile-arrow" aria-hidden="true">→</span>
      </a>
    </li>
  </ul>
</section>

<section class="profile-section" aria-labelledby="profile-notebook">

## 关于这里 {#profile-notebook}

  <p>我用这个博客记录具体问题。例如同一个 token 为什么能在两条推理路径上算出相同结果，或微调后 loss 下降时怎样检查任务能力。专题里会保留推导、源码入口与可复算的例子，项目页记录待验证的假设和对照。</p>
  <p>实测会注明模型版本、输入、配置与环境，并留下失败样例和结论适用的范围。</p>
  <nav class="profile-links" aria-label="博客说明">
    <a href="/writing/">写作与证据规范 <span aria-hidden="true">→</span></a>
    <a href="https://github.com/xhongduo-tech/xhongduo-tech.github.io" target="_blank" rel="noopener noreferrer">本站源码 <span aria-hidden="true">↗</span></a>
  </nav>
</section>

</div>
