---
title: rollout 引擎与权重同步
date: 2026-09-08
section: llm
---

# rollout 引擎与权重同步

<div class="epigraph">
<p>生成用推理引擎、训练用 Megatron，中间那条权重边的延迟与一致性，决定 logprob 还能不能当 π_old。</p>
<footer>—— 对照 HybridFlow 的重切分、vLLM / SGLang 作为 rollouter、OpenRLHF 与 verl 的广播与 sleep 模式</footer>
</div>

[上一课](/llm/off-policy-correction)要求每条轨迹有可信的 $\log\pi_{\mathrm{beh}}$。缺口落到系统：**推理引擎里的权重是不是 learner 刚推送的那一份，精度与模板是否一致。** 异步架构课写时序契约；本课写同步边上的工程：广播、sleep 共置、重算。算法仍是 [PPO](/llm/ppo-llm) / [GRPO](/llm/grpo)。

## 问题

训练布局（FSDP / Megatron 切分）与生成布局（TP + paged KV）不同。每次更新后要把 $\theta$ 变成生成引擎可加载的副本。共置 sleep：同一组 GPU 时分复用，权重切分转换，延迟低、实现重。分置：learner 训练、rollouter 常驻，NCCL / 参数服务器广播，生成可持续，但推送瞬间版本跃迁。推送频率低，staleness 升；频率高，气泡升。

失败模式：引擎仍用旧权生成，却把当前 $\theta$ 的 logprob 当成 $\pi_{\mathrm{beh}}$——IS 全错。或生成用 chat 模板 A，训练前向用模板 B，奖励打在另一字符串上。

### 缓存的 logprob 属于哪一版

生成时应在**同一次前向**里记下 $\log\pi_{\mathrm{beh}}$，与发出的 token 对齐，含温度、top-$p$ 是否作用于采样但不作用于要存储的裸 logprob。存储裸 $\log\pi$（softmax 前的分布对数），采样约束另记。训练重算必须关 dropout、对齐精度。

<span class="marginnote">LoRA 只同步 A/B 矩阵时，基座必须双方锁定同一份。漏同步一份 adapter，就是静默 off-policy。</span>

## 方法

最小协议：`sync_id` 单调递增；rollouter 加载成功才接受新请求；正在解码的序列要么跑完旧版，要么中断重算 KV。Learner 在 `sync_id` 写入样本元数据。评测路径与训练生成共用同一引擎配置。重切分（3D-HybridEngine）把训练切分转成推理切分，避免 CPU 落盘。

```mermaid
flowchart LR
  L["Learner θ"] --> S["sync_id 广播 / 重切分"]
  S --> R["Rollouter 加载"]
  R --> Y["采样 y + logπ_beh"]
  Y --> Q["带 sync_id 入队"]
  Q --> L
```

与奖励模型同步：RM 冻结则少一条边；RM 迭代时要第二份 `sync_id`，否则策略与裁判版本错位，像在优化旧口味。

## 机制

权重边的延迟进入 staleness $\eta$。数学上 $\eta$ 已在上一课定义；本课决定 $\eta$ 的物理下限：广播 70B 权重要多少毫秒。部分同步（只推 embedding）会造成层间版本撕裂，比整网落后一拍更难校正。要么全量原子换，要么明确冻结哪些层。

<span class="marginnote">CUDA graph / 编译缓存按形状绑定。换权后必须使缓存失效。旧 graph 跑新权是未定义行为。</span>

## 边界与工程取舍

小模型共置同步更简单，不必上参数服务器。长链推理才值得分置。MoE 还要同步路由专家，漏专家比漏一层更致命。同步完成应以 rollouter 的校验哈希为准，不要信发送端「已推送」。下一课采样温度：即使权重一致，解码随机性仍改变 $\pi_{\mathrm{beh}}$ 的定义——温度是策略的一部分，不是 UI。


sync_id 要以 rollouter 加载成功为准；模板、LoRA、MoE 专家漏同步会让比率在版本号正确时仍然全错。

## 小结

- 训练与生成之间的权重同步必须原子、带 sync_id，logπ_beh 与该版对齐。
- 共置重切分延迟低；分置广播吞吐高、staleness 要配额。
- 模板、LoRA、MoE 专家漏同步 = 静默错误 IS。
- RM 若迭代，裁判也要版本号。
- 出处：Sheng 等 HybridFlow；OpenRLHF / verl / slime 的 rollouter 实践。
