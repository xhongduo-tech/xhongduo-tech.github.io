import type { Outline } from './schema'

/** 程序语言与编译：前端结构 → 中后端降到机器。 */
export const csCompiler: Outline = [
  '程序语言与编译',
  [
    [
      '前端',
      [
        [
          '从文本到结构',
          [
            '编译器通行证|compiler-passes',
            '正则与词法|regex-lexer',
            'NFA 与 DFA|nfa-dfa',
            'Thompson 构造|thompson-nfa',
            '上下文无关文法|cfg-grammar',
            '二义与左递归|ambiguity-leftrec',
            '递归下降|recursive-descent',
            'LL 与 FIRST/FOLLOW|ll-first-follow',
            'LR 与移进归约|lr-shift-reduce',
            'SLR 与 LALR|slr-lalr',
            '抽象语法树|ast',
            '访问者与遍|ast-visitor',
            '作用域与符号表|scope-symtab',
            '名字解析|name-resolution',
            '类型检查|typecheck',
            '类型推导直觉|type-inference',
            'overload 与强制|overload-coerce',
          ],
        ],
      ],
    ],
    [
      '中后端',
      [
        [
          '降到机器',
          [
            '三地址中间表示|ir-three-address',
            '控制流图|cfg-ir',
            'SSA|ssa-form',
            '支配与 φ|dominator-phi',
            '活跃变量与数据流|dataflow-liveness',
            '到达定义|reaching-def',
            '可用表达式|available-expr',
            '指令选择|instruction-select',
            '指令调度|instruction-sched',
            '寄存器分配着色|regalloc-color',
            '溢出与重物化|spill-remat',
            '窥孔与窥视窗|peephole',
            'ABI 与代码生成|abi-codegen',
            '位置无关代码|pic',
            '链接与重定位|link-reloc',
            'GOT 与 PLT|got-plt',
            '加载与动态链接|load-dynlink',
            '运行时与 GC 直觉|runtime-gc',
            '标记清除与分代|gc-mark-gen',
            '异常表|exception-table',
          ],
        ],
      ],
    ],
  ],
]
