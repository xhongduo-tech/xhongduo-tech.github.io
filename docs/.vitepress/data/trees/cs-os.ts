import type { Outline } from './schema'

/**
 * 操作系统：把内核、进程、调度、同步、虚存、文件、IPC、
 * 启动与隔离上成千上万个词条归并成可递进的课序。百级叶子，不是词条清单。
 */
export const csOs: Outline = [
  '操作系统',
  [
    [
      '内核边界',
      [
        [
          '特权与陷入',
          [
            '内核与用户态|kernel-user',
            '系统调用 ABI|syscall-abi',
            'vDSO|vdso',
            'trapframe|trapframe',
            '系统调用路径|syscall-path',
            'copy_from_user|copy-from-user',
            '重启系统调用|restart-syscall',
            '跟踪与 ptrace 入口|syscall-trace',
          ],
        ],
        [
          '中断与下半部',
          [
            '中断下半部|interrupt-bottom-half',
            '硬中断与软中断|hardirq-softirq',
            'tasklet 与 workqueue|tasklet-workqueue',
            '中断线程化|irq-thread',
            '中断亲和|irq-affinity',
            'NMI|nmi',
            '内核抢占|kernel-preempt',
            'per-CPU 数据|percpu',
          ],
        ],
      ],
    ],
    [
      '进程与线程',
      [
        [
          '生命周期',
          [
            '进程映像|process-image',
            'PCB 与 task_struct|pcb-task-struct',
            'fork|fork',
            'execve|execve',
            'wait 与僵尸|wait-zombie',
            '孤儿与 init|orphan-init',
            'exit 与回收|exit-reap',
            '进程组与会话|process-groups',
          ],
        ],
        [
          '线程与上下文',
          [
            '线程与共享地址空间|thread-shared-addr',
            '用户线程对内核线程|ult-klt',
            '线程局部存储|thread-tls',
            'clone 标志|clone-flags',
            '上下文切换|context-switch',
            '内核栈|kernel-stack',
            'FPU 惰性保存|fpu-lazy',
            'idle 线程|idle-thread',
          ],
        ],
      ],
    ],
    [
      '调度',
      [
        [
          '经典策略',
          [
            '调度指标|scheduling-metrics',
            'CPU 型与 I/O 型|cpu-vs-io-bound',
            '运行队列|runqueue',
            'FCFS 与 SJF|fcfs-sjf',
            '时间片轮转|rr-quantum',
            '多级反馈队列|mlfq',
            '优先级反转|priority-inversion',
            '优先级继承与天花板|pi-pcp',
          ],
        ],
        [
          '公平与实时',
          [
            '时间片与公平调度|timeslice-cfs',
            'vruntime|cfs-vruntime',
            'nice 与权重|nice-weight',
            '负载均衡与迁移|load-balance-migrate',
            '实时调度对照|realtime-sched',
            '速率单调|rate-monotonic',
            'EDF|edf-sched',
            'CPU 亲和|cpu-affinity',
            'NUMA 调度|numa-sched',
          ],
        ],
      ],
    ],
    [
      '同步',
      [
        [
          '原语',
          [
            '竞争与临界区|race-critical',
            '内存屏障|memory-barrier-os',
            '原子读改写|atomic-rmw',
            '锁与关中断|lock-irq',
            '自旋锁|spinlock',
            'MCS 锁|mcs-lock',
            'mutex 与休眠|mutex-sleep',
            'futex|futex',
            '读写锁|rwlock',
            'seqlock|seqlock',
          ],
        ],
        [
          '睡眠与无锁',
          [
            '信号量|semaphore',
            '管程与条件变量|monitor-condvar',
            'completion|completion',
            'RCU 读者|rcu-read',
            'RCU 宽限期|rcu-gp',
            '无锁与 ABA|lockfree-aba',
          ],
        ],
        [
          '死锁',
          [
            '死锁四个条件|deadlock-coffman',
            '预防与避免|deadlock-prevent',
            '银行家算法|banker-algorithm',
            '活锁与饥饿|livelock-starvation',
            '锁顺序|lock-ordering',
          ],
        ],
      ],
    ],
    [
      '虚存',
      [
        [
          '地址与缺页',
          [
            '地址空间布局|addrspace-layout',
            '用户/内核分裂|user-kernel-split',
            'brk 与堆|brk-heap',
            '按需调页|demand-paging',
            '缺页路径|page-fault-path',
            '大页|huge-pages',
            'TLB shootdown|tlb-shootdown',
          ],
        ],
        [
          '置换与交换',
          [
            '缺页与置换|page-replace',
            'CLOCK|clock-replace',
            '工作集|working-set',
            '抖动|thrashing',
            '匿名页对文件页|anon-vs-file-page',
            'swap|swap-device',
            'OOM|oom-killer',
            '过度提交|overcommit',
          ],
        ],
        [
          '映射与内核分配',
          [
            '写时复制|cow-fork',
            'mmap|mmap',
            'mprotect|mprotect',
            'buddy|buddy-allocator',
            'slab|slab-allocator',
            '回收 shrinker|memory-reclaim',
          ],
        ],
      ],
    ],
    [
      '文件与存储',
      [
        [
          '接口与 VFS',
          [
            '文件作为字节流|file-bytestream',
            '打开文件表|file-table',
            'dentry 缓存|dcache',
            'inode 与目录|inode-dir',
            '路径查找|path-lookup',
            '硬链接与符号链接|hard-symlink',
            '虚拟文件系统|vfs',
            '挂载与 superblock|mount-super',
          ],
        ],
        [
          '缓存与一致性',
          [
            '缓冲与脏页|buffer-dirty',
            '页缓存|page-cache',
            'writeback|writeback',
            'fsync|fsync',
            '日志模式|ext4-journal',
            '崩溃一致性|crash-consistency',
            '模式位|file-mode-bits',
            'setuid 与粘滞位|setuid-sticky',
          ],
        ],
        [
          '块层',
          [
            '磁盘调度|disk-sched',
            'blk-mq|blk-mq',
            'bio|bio-block',
            'I/O 与 DMA|io-dma',
            'IOMMU|iommu',
          ],
        ],
      ],
    ],
    [
      'IPC 与事件',
      [
        [
          '信号与管道',
          [
            '信号|signals',
            '信号掩码|sigmask',
            '管道与 IPC|ipc-pipe',
            '命名管道|named-pipe',
            'Unix 域套接字|unix-socket',
            'POSIX 共享内存|posix-shm',
          ],
        ],
        [
          '多路复用',
          [
            'select / poll|select-poll',
            'epoll|epoll',
            'io_uring|io-uring',
          ],
        ],
      ],
    ],
    [
      '时间、设备、启动',
      [
        [
          '钟、终端与 init',
          [
            'jiffies|jiffies',
            'hrtimer|hrtimer',
            'tty 与 PTY|tty-pty',
            '内核模块|kernel-module',
            'proc 与 sysfs|sysfs-proc',
            '早期启动|early-boot',
            'init 与用户空间|init-userspace',
          ],
        ],
      ],
    ],
    [
      '隔离与虚拟化',
      [
        [
          '命名空间到陷入',
          [
            'namespaces|namespaces',
            'cgroups|cgroups',
            '容器的 OS 含义|container-os',
            'seccomp 过滤|seccomp-filter',
            '陷阱与模拟|trap-and-emulate',
            'EPT / NPT|ept-npt',
            'virtio|virtio',
          ],
        ],
      ],
    ],
  ],
]
