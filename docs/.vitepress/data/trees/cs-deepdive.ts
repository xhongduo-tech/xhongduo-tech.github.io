import { fromOutline, type Outline } from './schema'

/** 深钻层：主干收束之后的研究向加深课程。挂载于主干之后、附录之前。 */
const courses: [string, [string, (readonly [string, readonly string[]] | readonly string[])[]][]] = [["GPU 编程深入", [[["执行模型", ["CUDA 执行模型的深入|gpu-execution-model-deep", "内存层次的实践|gpu-memory-hierarchy-practice", "warp 级原语|gpu-warp-primitives", "共享内存与 bank 冲突|gpu-smem-bank-conflicts"]]], [["张量核与工具", ["张量核的编程直觉|gpu-tensorcore-programming", "CUTLASS 的结构直觉|gpu-cutlass-intuition", "性能计数与 profiling|gpu-profiling", "GPU 调试|gpu-debugging", "ROCm 的对照|gpu-rocm-comparison", "GPU 编程案例与收束|gpu-case-map"]]]]], ["并行计算模型", [[["模型", ["从 PRAM 到实践|par-pram-to-practice", "OpenMP 与 MPI 的直觉|par-openmp-mpi", "数据并行的模式|par-data-parallel-patterns", "任务并行|par-task-parallel"]]], [["性能", ["lock-free 并行|par-lockfree", "SIMD 与向量化|par-simd-vectorization", "NUMA|par-numa", "roofline 的深化与收束|par-roofline-map"]]]]], ["高性能网络编程", [[["路径", ["epoll 的深化|hpc-epoll-deep", "io_uring|hpc-io-uring", "内核旁路：DPDK 与 AF_XDP|hpc-kernel-bypass", "RDMA|hpc-rdma", "零拷贝与延迟测量|hpc-zerocopy-latency", "高性能网络收束|hpc-map"]]]]], ["存储系统深入", [[["结构", ["LSM 的深入|st-lsm-deep", "B+ 树的实现细节|st-btree-implementation", "SSD 与 FTL|st-ssd-ftl"]]], [["格式与系统", ["列存格式：Parquet 解剖|st-parquet-anatomy", "压缩列存|st-compressed-columnar", "缓存的层次|st-cache-tiers", "分布式存储的案例|st-distributed-cases", "存储深化收束|st-map"]]]]], ["虚拟化与容器深入", [[["机制", ["hypervisor 的类型|virt-hypervisor-types", "KVM 与 QEMU|virt-kvm-qemu", "容器运行时|virt-container-runtime", "namespace 与 cgroup 的深化|virt-ns-cgroup-deep"]]], [["扩展与边界", ["网络虚拟化|virt-network-virtualization", "GPU 虚拟化|virt-gpu", "安全边界|virt-security-boundary", "性能开销|virt-performance-overhead", "虚拟化案例与收束|virt-case-map"]]]]], ["可观测性与调试", [[["工具", ["指标、日志与追踪|obs-metrics-logs-traces", "eBPF 的入门与应用|obs-ebpf", "perf 的深化|obs-perf-deep", "火焰图|obs-flamegraphs", "内存调试|obs-memory-debugging"]]], [["工程", ["分布式追踪|obs-distributed-tracing", "SLO 工程|obs-slo-engineering", "事故指挥|obs-incident-command", "postmortem 文化与案例|obs-postmortem-culture", "可观测性收束|obs-map"]]]]], ["编译器后端深化", [[["核心", ["寄存器分配的深入|cg-register-allocation-deep", "指令选择|cg-instruction-selection", "指令调度|cg-instruction-scheduling", "SIMD 自动向量化|cg-simd-autovec"]]], [["链接与运行", ["链接器的深入|cg-linker-deep", "JIT 的深化|cg-jit-deep", "MLIR 的结构直觉|cg-mlir-intuition", "优化器的调试与收束|cg-debugging-map"]]]]], ["流处理", [[["语义", ["流处理模型|stream-model", "窗口语义|stream-windows", "watermark|stream-watermark", "exactly-once|stream-exactly-once"]]], [["系统", ["流 join|stream-joins", "状态管理|stream-state", "背压|stream-backpressure", "Flink 与 Kafka Streams 的对照与收束|stream-systems-map"]]]]], ["分布式共识深化", [[["实现", ["Raft 的实现细节|con-raft-implementation", "Multi-Raft|con-multi-raft", "Paxos 谱系|con-paxos-family"]]], [["验证与优化", ["线性一致性的测试|con-linearity-testing", "共识与复制状态机|con-consensus-rsm", "etcd 案例走读|con-etcd-case", "共识的性能优化与收束|con-perf-map"]]]]], ["查询优化深化", [[["代价与执行", ["代价模型|qo-cost-model", "join 算法的实现|qo-join-implementation", "统计信息|qo-statistics", "计划空间与枚举|qo-plan-space"]]], [["现代执行", ["自适应执行|qo-adaptive-execution", "向量化执行的深入|qo-vectorized-deep", "物化视图|qo-materialized-views", "查询优化案例与收束|qo-case-map"]]]]], ["分布式事务深化", [[["协议", ["2PC 的实现细节|dt-2pc-implementation", "Percolator 与 TiDB 案例|dt-percolator-tidb", "Calvin 与确定型数据库|dt-calvin-deterministic"]]], [["体系", ["NewSQL 谱系|dt-newsql-family", "隔离级别的实现|dt-isolation-implementation", "分布式事务案例与收束|dt-case-map"]]]]], ["密码学工程", [[["实践", ["对称加密模式的实践|ce-symmetric-modes", "AEAD 的深入|ce-aead-deep", "签名方案的实践|ce-signature-practice", "KEM 与混合加密|ce-kem-hybrid"]]], [["陷阱与前沿", ["协议实现的陷阱|ce-protocol-pitfalls", "侧信道|ce-side-channels", "随机数|ce-randomness", "后量子概览|ce-postquantum-overview", "密码学工程收束|ce-map"]]]]], ["系统安全深化", [[["攻防", ["内核漏洞类别|ss-kernel-vuln-classes", "沙箱逃逸|ss-sandbox-escape", "软件供应链安全|ss-supply-chain", "模糊测试的深入|ss-fuzzing-deep"]]], [["体系", ["内存安全语言|ss-memory-safe-langs", "硬件安全的视角|ss-hardware-perspective", "移动安全|ss-mobile", "云安全模型|ss-cloud-model", "系统安全案例与收束|ss-case-map"]]]]], ["数值与科学计算", [[["基础", ["浮点数的深入|sc-float-deep", "BLAS 的层次|sc-blas-levels", "稀疏计算|sc-sparse-computing", "自动微分的实现|sc-autodiff-implementation"]]], [["应用", ["随机数生成|sc-random-generation", "优化库的对照|sc-optimizer-libraries", "精度与稳定性的案例与收束|sc-case-map"]]]]], ["软件工程实践", [[["测试与版本", ["测试策略：单元、集成与属性|swe-testing-strategy", "git 的内部模型|swe-git-internals", "分支与变更管理|swe-branching-changes"]]], [["构建与交付", ["构建系统：make 到 bazel|swe-build-systems", "CI/CD|swe-ci-cd", "软件工程实践收束|swe-map"]]]]]] as const

export const csDeepDive: Outline = [
  [
    'GPU 编程深入',
    [
      [
        '执行模型',
        [
          'CUDA 执行模型的深入|gpu-execution-model-deep',
          '内存层次的实践|gpu-memory-hierarchy-practice',
          'warp 级原语|gpu-warp-primitives',
          '共享内存与 bank 冲突|gpu-smem-bank-conflicts',
        ],
      ],
      [
        '张量核与工具',
        [
          '张量核的编程直觉|gpu-tensorcore-programming',
          'CUTLASS 的结构直觉|gpu-cutlass-intuition',
          '性能计数与 profiling|gpu-profiling',
          'GPU 调试|gpu-debugging',
          'ROCm 的对照|gpu-rocm-comparison',
          'GPU 编程案例与收束|gpu-case-map',
        ],
      ],
    ],
  ],
  [
    '并行计算模型',
    [
      [
        '模型',
        [
          '从 PRAM 到实践|par-pram-to-practice',
          'OpenMP 与 MPI 的直觉|par-openmp-mpi',
          '数据并行的模式|par-data-parallel-patterns',
          '任务并行|par-task-parallel',
        ],
      ],
      [
        '性能',
        [
          'lock-free 并行|par-lockfree',
          'SIMD 与向量化|par-simd-vectorization',
          'NUMA|par-numa',
          'roofline 的深化与收束|par-roofline-map',
        ],
      ],
    ],
  ],
  [
    '高性能网络编程',
    [
      [
        '路径',
        [
          'epoll 的深化|hpc-epoll-deep',
          'io_uring|hpc-io-uring',
          '内核旁路：DPDK 与 AF_XDP|hpc-kernel-bypass',
          'RDMA|hpc-rdma',
          '零拷贝与延迟测量|hpc-zerocopy-latency',
          '高性能网络收束|hpc-map',
        ],
      ],
    ],
  ],
  [
    '存储系统深入',
    [
      [
        '结构',
        [
          'LSM 的深入|st-lsm-deep',
          'B+ 树的实现细节|st-btree-implementation',
          'SSD 与 FTL|st-ssd-ftl',
        ],
      ],
      [
        '格式与系统',
        [
          '列存格式：Parquet 解剖|st-parquet-anatomy',
          '压缩列存|st-compressed-columnar',
          '缓存的层次|st-cache-tiers',
          '分布式存储的案例|st-distributed-cases',
          '存储深化收束|st-map',
        ],
      ],
    ],
  ],
  [
    '虚拟化与容器深入',
    [
      [
        '机制',
        [
          'hypervisor 的类型|virt-hypervisor-types',
          'KVM 与 QEMU|virt-kvm-qemu',
          '容器运行时|virt-container-runtime',
          'namespace 与 cgroup 的深化|virt-ns-cgroup-deep',
        ],
      ],
      [
        '扩展与边界',
        [
          '网络虚拟化|virt-network-virtualization',
          'GPU 虚拟化|virt-gpu',
          '安全边界|virt-security-boundary',
          '性能开销|virt-performance-overhead',
          '虚拟化案例与收束|virt-case-map',
        ],
      ],
    ],
  ],
  [
    '可观测性与调试',
    [
      [
        '工具',
        [
          '指标、日志与追踪|obs-metrics-logs-traces',
          'eBPF 的入门与应用|obs-ebpf',
          'perf 的深化|obs-perf-deep',
          '火焰图|obs-flamegraphs',
          '内存调试|obs-memory-debugging',
        ],
      ],
      [
        '工程',
        [
          '分布式追踪|obs-distributed-tracing',
          'SLO 工程|obs-slo-engineering',
          '事故指挥|obs-incident-command',
          'postmortem 文化与案例|obs-postmortem-culture',
          '可观测性收束|obs-map',
        ],
      ],
    ],
  ],
  [
    '编译器后端深化',
    [
      [
        '核心',
        [
          '寄存器分配的深入|cg-register-allocation-deep',
          '指令选择|cg-instruction-selection',
          '指令调度|cg-instruction-scheduling',
          'SIMD 自动向量化|cg-simd-autovec',
        ],
      ],
      [
        '链接与运行',
        [
          '链接器的深入|cg-linker-deep',
          'JIT 的深化|cg-jit-deep',
          'MLIR 的结构直觉|cg-mlir-intuition',
          '优化器的调试与收束|cg-debugging-map',
        ],
      ],
    ],
  ],
  [
    '流处理',
    [
      [
        '语义',
        [
          '流处理模型|stream-model',
          '窗口语义|stream-windows',
          'watermark|stream-watermark',
          'exactly-once|stream-exactly-once',
        ],
      ],
      [
        '系统',
        [
          '流 join|stream-joins',
          '状态管理|stream-state',
          '背压|stream-backpressure',
          'Flink 与 Kafka Streams 的对照与收束|stream-systems-map',
        ],
      ],
    ],
  ],
  [
    '分布式共识深化',
    [
      [
        '实现',
        [
          'Raft 的实现细节|con-raft-implementation',
          'Multi-Raft|con-multi-raft',
          'Paxos 谱系|con-paxos-family',
        ],
      ],
      [
        '验证与优化',
        [
          '线性一致性的测试|con-linearity-testing',
          '共识与复制状态机|con-consensus-rsm',
          'etcd 案例走读|con-etcd-case',
          '共识的性能优化与收束|con-perf-map',
        ],
      ],
    ],
  ],
  [
    '查询优化深化',
    [
      [
        '代价与执行',
        [
          '代价模型|qo-cost-model',
          'join 算法的实现|qo-join-implementation',
          '统计信息|qo-statistics',
          '计划空间与枚举|qo-plan-space',
        ],
      ],
      [
        '现代执行',
        [
          '自适应执行|qo-adaptive-execution',
          '向量化执行的深入|qo-vectorized-deep',
          '物化视图|qo-materialized-views',
          '查询优化案例与收束|qo-case-map',
        ],
      ],
    ],
  ],
  [
    '分布式事务深化',
    [
      [
        '协议',
        [
          '2PC 的实现细节|dt-2pc-implementation',
          'Percolator 与 TiDB 案例|dt-percolator-tidb',
          'Calvin 与确定型数据库|dt-calvin-deterministic',
        ],
      ],
      [
        '体系',
        [
          'NewSQL 谱系|dt-newsql-family',
          '隔离级别的实现|dt-isolation-implementation',
          '分布式事务案例与收束|dt-case-map',
        ],
      ],
    ],
  ],
  [
    '密码学工程',
    [
      [
        '实践',
        [
          '对称加密模式的实践|ce-symmetric-modes',
          'AEAD 的深入|ce-aead-deep',
          '签名方案的实践|ce-signature-practice',
          'KEM 与混合加密|ce-kem-hybrid',
        ],
      ],
      [
        '陷阱与前沿',
        [
          '协议实现的陷阱|ce-protocol-pitfalls',
          '侧信道|ce-side-channels',
          '随机数|ce-randomness',
          '后量子概览|ce-postquantum-overview',
          '密码学工程收束|ce-map',
        ],
      ],
    ],
  ],
  [
    '系统安全深化',
    [
      [
        '攻防',
        [
          '内核漏洞类别|ss-kernel-vuln-classes',
          '沙箱逃逸|ss-sandbox-escape',
          '软件供应链安全|ss-supply-chain',
          '模糊测试的深入|ss-fuzzing-deep',
        ],
      ],
      [
        '体系',
        [
          '内存安全语言|ss-memory-safe-langs',
          '硬件安全的视角|ss-hardware-perspective',
          '移动安全|ss-mobile',
          '云安全模型|ss-cloud-model',
          '系统安全案例与收束|ss-case-map',
        ],
      ],
    ],
  ],
  [
    '数值与科学计算',
    [
      [
        '基础',
        [
          '浮点数的深入|sc-float-deep',
          'BLAS 的层次|sc-blas-levels',
          '稀疏计算|sc-sparse-computing',
          '自动微分的实现|sc-autodiff-implementation',
        ],
      ],
      [
        '应用',
        [
          '随机数生成|sc-random-generation',
          '优化库的对照|sc-optimizer-libraries',
          '精度与稳定性的案例与收束|sc-case-map',
        ],
      ],
    ],
  ],
  [
    '软件工程实践',
    [
      [
        '测试与版本',
        [
          '测试策略：单元、集成与属性|swe-testing-strategy',
          'git 的内部模型|swe-git-internals',
          '分支与变更管理|swe-branching-changes',
        ],
      ],
      [
        '构建与交付',
        [
          '构建系统：make 到 bazel|swe-build-systems',
          'CI/CD|swe-ci-cd',
          '软件工程实践收束|swe-map',
        ],
      ],
    ],
  ],
  [
    '数据库内核实践',
    [
      [
        '存储与索引',
        [
          '页结构与缓冲池的实现|dbk-pages-bufferpool',
          'B+ 树的并发控制|dbk-btree-concurrency',
          'LSM 引擎的实现|dbk-lsm-implementation',
        ],
      ],
      [
        '执行与事务',
        [
          '执行器的实现|dbk-executor-implementation',
          '事务管理器的实现|dbk-tx-manager',
          'WAL 的实现|dbk-wal-implementation',
          '数据库内核实践收束|dbk-map',
        ],
      ],
    ],
  ],
  [
    '操作系统内核实践',
    [
      [
        '核心子系统',
        [
          '进程与调度的实现|osk-process-scheduler',
          '虚拟内存的实现|osk-vm-implementation',
          '文件系统的实现|osk-fs-implementation',
        ],
      ],
      [
        '工程',
        [
          '系统调用的路径|osk-syscall-path',
          '中断与驱动的骨架|osk-interrupt-drivers',
          '内核调试的方法|osk-kernel-debugging',
          '内核实践收束|osk-map',
        ],
      ],
    ],
  ],
  [
    '网络协议栈实践',
    [
      [
        '收发路径',
        [
          '收包路径的解剖|npk-rx-path',
          'TCP 状态机的实现|npk-tcp-state-machine',
          '拥塞控制的实现|npk-congestion-implementation',
        ],
      ],
      [
        '工程',
        [
          '套接字层的实现|npk-socket-layer',
          '协议栈的调试工具|npk-debugging-tools',
          '协议栈实践收束|npk-map',
        ],
      ],
    ],
  ],
  [
    '容器编排',
    [
      [
        '机制',
        [
          '调度器的架构|orch-scheduler-arch',
          '声明式 API 与控制循环|orch-declarative-controller',
          '服务发现|orch-service-discovery',
          '滚动发布|orch-rolling-release',
        ],
      ],
      [
        '资源与多租户',
        [
          '资源模型与 QoS|orch-resource-qos',
          '多租户与命名空间策略|orch-multitenant-policy',
          '网络策略|orch-network-policy',
          '存储编排|orch-storage-orchestration',
          '编排案例与收束|orch-case-map',
        ],
      ],
    ],
  ],
  [
    '嵌入式系统',
    [
      [
        '基础',
        [
          '裸机启动|emb-bare-metal-boot',
          '中断实践|emb-interrupt-practice',
          '外设协议：I2C 与 SPI|emb-i2c-spi',
        ],
      ],
      [
        '系统与生产',
        [
          '实时操作系统 RTOS|emb-rtos',
          '功耗管理|emb-power-management',
          '看门狗与失效安全|emb-watchdog-failsafe',
          'OTA 更新|emb-ota-updates',
          '嵌入式安全|emb-security',
          '测试与仿真|emb-testing-simulation',
          '嵌入式案例与收束|emb-case-map',
        ],
      ],
    ],
  ],
  [
    '性能工程案例',
    [
      [
        '方法',
        [
          '性能需求的工程化|pe3-requirements',
          '基准设计的陷阱|pe3-benchmark-pitfalls',
          '微基准与真实负载|pe3-microbench-vs-real',
        ],
      ],
      [
        '案例',
        [
          '性能回归的排查案例|pe3-regression-case',
          '容量规划案例|pe3-capacity-case',
          '延迟长尾的追猎案例|pe3-tail-latency-case',
          '性能与成本的联合优化案例|pe3-cost-case',
          '性能工程案例收束|pe3-map',
        ],
      ],
    ],
  ],
  [
    '算法工程实践',
    [
      [
        '工程',
        [
          '算法选型的工程视角|ae-selection-engineering',
          '常数优化|ae-constant-optimization',
          '缓存友好的算法|ae-cache-friendly',
          '位运算技巧|ae-bit-tricks',
        ],
      ],
      [
        '验证与案例',
        [
          '随机化算法的实践|ae-randomized-practice',
          '算法基准的方法|ae-benchmark-method',
          '算法库的对照与使用|ae-library-comparison',
          '算法工程案例与收束|ae-case-map',
        ],
      ],
    ],
  ],
]