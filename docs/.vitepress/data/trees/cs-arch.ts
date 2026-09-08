import type { Outline } from './schema'

/** 计算机体系结构：流水、存储层次、并行微结构。 */
export const csArch: Outline = [
  '计算机体系结构',
  [
    [
      '流水',
      [
        [
          '重叠执行',
          [
            '流水线五级|pipeline-five-stage',
            '流水线寄存器|pipeline-registers',
            '结构冒险|structural-hazard',
            '数据冒险与转发|data-hazard-forward',
            'load-use 气泡|load-use-stall',
            '控制冒险与分支预测|control-hazard-predict',
            'BTB 与返回栈|btb-ras',
            '两级与锦标赛预测|tournament-predict',
            '流水线异常|pipeline-exception',
            '延迟槽对照|delay-slot',
            'CPI 与阿姆达尔|cpi-amdahl',
            'IPC 与利用率|ipc-util',
          ],
        ],
      ],
    ],
    [
      '存储层次',
      [
        [
          '延迟与局部性',
          [
            '局部性原理|locality-principle',
            '直接映射 Cache|direct-mapped-cache',
            '组相联与替换|set-associative-replace',
            '全相联对照|fully-associative',
            '写回与写分配|write-back-allocate',
            '写穿与写周围|write-through',
            '缺失分类|cache-miss-types',
            'MSHR 与缺失下继续|mshr',
            '预取|cache-prefetch',
            '包含与互斥层次|inclusive-exclusive-cache',
            '虚拟内存分页|paging-vm',
            'TLB|tlb-translate',
            '多级页表|multi-level-page-table',
            '硬件页表游走|hw-page-walk',
            'ASID / PCID|asid-pcid',
            '一致性问题引入|coherence-intro',
          ],
        ],
      ],
    ],
    [
      '并行微结构',
      [
        [
          '指令级到多核',
          [
            '超标量发射|superscalar-issue',
            '乱序与 ROB|ooo-rob',
            'Tomasulo|tomasulo',
            '保留站与公共数据总线|rs-cdb',
            '存储缓冲与 load 绕过|store-buffer',
            'SMT|smt',
            'SIMD 与向量|simd-vector',
            '多核与共享缓存|multicore-shared-cache',
            'MESI|mesi-protocol',
            '侦听与目录|snoop-vs-directory',
            '存储一致性模型|memory-consistency',
            'TSO 与弱序|tso-weak',
            '互连与 NUMA|interconnect-numa',
            '内存控制器与 DRAM 时序|dram-timing',
          ],
        ],
      ],
    ],
  ],
]
