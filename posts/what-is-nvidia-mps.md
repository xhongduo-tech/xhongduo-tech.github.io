## 一句话定义

NVIDIA Multi-Process Service（MPS）是一个运行在 GPU 与用户 CUDA 进程之间的守护进程，它让多个进程共享同一个 CUDA Context，从而把原本被上下文切换和串行执行浪费掉的算力利用起来。

---

## 为什么需要 MPS

默认情况下，每个 CUDA 进程独占一个 CUDA Context。GPU 在同一时刻只能挂起一个 Context，当多个进程都要提交内核时，驱动只能在它们之间快速切换。这种切换本身就会带来开销，而且很多小型内核因为无法真正并行，导致 SM 利用率低下。

典型场景：

- 一个推理服务同时承载多个小 batch 的请求，每个请求一个进程
- 训练框架把每个 worker 作为独立进程启动，单卡需要服务多个 worker
- HPC 作业由多个 MPI rank 组成，每个 rank 提交的核函数规模很小

在这些场景下，单进程的算力需求往往填不满整卡，而多进程之间又互相抢占 Context，结果 GPU 很忙但有效吞吐不高。

---

## MPS 的工作方式

MPS 的核心结构是一个 MPS Server 和一组 MPS Client。

| 组件 | 作用 |
|------|------|
| MPS Server | 持有真正的 GPU Context，接收并调度多个 Client 的内核与显存请求 |
| MPS Client | 用户进程，通过 MPS 提供的 IPC 通道把 CUDA 调用转发给 Server |
| CUDA Driver | 看到的是一个统一上下文，因此不同 Client 的内核可以在不同 SM 上并发执行 |

Client 进程启动时设置环境变量 `CUDA_VISIBLE_DEVICES` 和 `CUDA_MPS_PIPE_DIRECTORY`，即可把调用路由到 Server。Server 负责合并提交队列、管理显存分配、并在内核满足条件时并发调度。

```bash
# 启动 MPS 控制守护进程
export CUDA_VISIBLE_DEVICES=0
nvidia-cuda-mps-control -d

# Client 进程在环境中指定 MPS 管道目录后启动
cd /your/app && CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps ./your_program

# 关闭 MPS
echo quit | nvidia-cuda-mps-control
```

---

## 能带来什么收益

MPS 的收益高度依赖负载特征：

| 负载特征 | 预期效果 |
|----------|----------|
| 大量小内核、低占用 | 显著提升 SM 利用率和吞吐 |
| 单进程已占满整卡 | 几乎无收益，反而可能引入调度开销 |
| 内核规模差异大 | 大内核可能阻塞小内核，收益不稳定 |
| 需要严格的时延保证 | MPS 不做抢占式 QoS，时延可控性有限 |

实际观察中，MPS 最常见的作用是提升 **多进程小 batch 推理** 的总吞吐，而不是降低单个请求的延迟。

---

## 限制与风险

MPS 不是虚拟化，也不是隔离方案。使用前需要明确以下限制：

| 限制项 | 说明 |
|--------|------|
| 显存共享 | 所有 Client 共享同一块物理显存，任一进程 OOM 会导致整卡受影响 |
| 故障传播 | 一个 Client 崩溃可能拖垮 MPS Server，进而影响所有 Client |
| 单 Server 限制 | 一块 GPU 通常只能绑定一个 MPS Server |
| 计算能力要求 | 需要 Compute Capability 3.5 及以上 |
| 与 MIG 互斥 | MPS 不能与 NVIDIA MIG 在同一 GPU 上同时使用 |

因此，MPS 更适合内部可控的推理集群或训练 worker，不适合多租户、强隔离环境。

---

## MPS 与 MIG 的对比

两者都用于提升单卡利用率，但设计目标不同。

| 维度 | MPS | MIG |
|------|-----|-----|
| 隔离级别 | 软件层共享，无硬隔离 | 硬件级分区，显存与计算单元隔离 |
| 适用场景 | 同信任域内的多进程 | 多租户、需要 QoS 保障的环境 |
| 显存 | 共享 | 固定切分 |
| 调度粒度 | 内核级并发 | 实例级独占 |
| 可用硬件 | Kepler 及以后 | Ampere 及以后 |

如果目标是让多个进程安全地共用一张卡，优先评估 MIG；如果硬件不支持 MIG 且负载可信任，MPS 是更轻量的替代。

---

## 总结

MPS 通过共享 CUDA Context 让多个进程的内核并发执行，解决的是「单进程填不满 GPU，多进程又互相切换」的问题。它不是资源隔离方案，也不能凭空增加算力，但在受控环境下，对提升小负载并发吞吐有明显价值。选型时应当先确认负载是否以小内核为主、是否能接受共享显存带来的故障传播风险，再决定是否启用。
