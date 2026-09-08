import type { Outline } from './schema'

/** 数字逻辑与计算机组成：门 → 时序 → ISA 与单周期。 */
export const csOrg: Outline = [
  '数字逻辑与计算机组成',
  [
    [
      '组合',
      [
        [
          '从门到运算器',
          [
            'CMOS 反相器|cmos-inverter',
            '与或非与万能门|nand-universal',
            '传输门与三态|transmission-tristate',
            '组合逻辑时延|gate-delay',
            '关键路径|critical-path',
            '多路选择器|mux-select',
            '译码器与编码器|decoder-encoder',
            '加法器与超前进位|adder-cla',
            '行波进位对照|ripple-adder',
            '进位选择|carry-select',
            '阵列乘法|array-multiplier',
            '移位与桶形移位|barrel-shifter',
            'ALU 数据通路|alu-datapath',
            '比较器|comparator-circuit',
            'PLA 与 ROM 作组合|pla-rom',
          ],
        ],
      ],
    ],
    [
      '时序',
      [
        [
          '状态与时钟',
          [
            '锁存与触发器|latch-flipflop',
            'D / T / JK 对照|flipflop-types',
            '建立保持时间|setup-hold',
            '亚稳态与同步器|metastability',
            '寄存器与移位|register-shift',
            '有限状态机|fsm-control',
            'Moore 与 Mealy|moore-mealy',
            '计数器|counter-circuit',
            'SRAM 与 DRAM 阵列|memory-array-sram-dram',
            'DRAM 刷新|dram-refresh',
            '闪存对照|flash-memory',
            '时钟域|clock-domain',
            '时钟树与抖动|clock-jitter',
          ],
        ],
      ],
    ],
    [
      'ISA 与单周期',
      [
        [
          '机器如何执行指令',
          [
            '存储程序|stored-program',
            '哈佛与冯·诺依曼|harvard-von-neumann',
            '指令格式|instruction-format',
            '寻址方式|addressing-modes',
            'RISC 与 CISC|risc-cisc',
            'RISC-V 整数指令|riscv-int-isa',
            'RISC-V CSR 与 ecall|riscv-csr-ecall',
            '伪指令|pseudo-instructions',
            '寄存器堆|register-file',
            '单周期数据通路|single-cycle-datapath',
            '控制器真值表|control-truth-table',
            '多周期与微程序直觉|multicycle-microcode',
            '总线与 MMIO|bus-mmio',
            '调用约定与栈|calling-convention-stack',
            '异常与中断入口|exception-interrupt-entry',
            'PLIC 与中断号|plic-irq',
            '特权级|privilege-rings',
          ],
        ],
      ],
    ],
  ],
]
