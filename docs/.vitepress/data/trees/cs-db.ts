import type { Outline } from './schema'

/** 数据库：关系与查询 → 存储、日志与事务。 */
export const csDb: Outline = [
  '数据库',
  [
    [
      '模型与查询',
      [
        [
          '关系',
          [
            '关系模型|relational-model',
            'ER 到关系|er-to-rel',
            '键与完整性|keys-integrity',
            '外键与参照动作|fk-actions',
            '关系代数|relational-algebra',
            '选择投影连接的顺序|algebra-reorder',
            'SQL 声明性|sql-declarative',
            '空值与三值逻辑|sql-null',
            '连接算法|join-algorithms',
            '嵌套循环与哈希连接|nlj-hash-join',
            '排序归并连接|sort-merge-join',
            '查询计划|query-plan',
            '代价估计|cost-estimate',
            '直方图与基数|histogram-card',
            '查询改写|query-rewrite',
            '谓词下推|predicate-pushdown',
            '范式与分解|normal-forms',
            'BCNF 与 3NF 取舍|bcnf-3nf',
            '视图与物化|views-materialize',
            '覆盖索引扫描|covering-index',
          ],
        ],
      ],
    ],
    [
      '存储与事务',
      [
        [
          '持久与并发',
          [
            '页与槽|page-slot',
            '堆文件与聚簇|heap-cluster',
            'B+ 树索引|bplus-index',
            'B+ 分裂与合并|bplus-split',
            '哈希索引|hash-index',
            'WAL|wal',
            'steal / no-force|steal-noforce',
            'ARIES 直觉|aries-recovery',
            '检查点|checkpoint',
            '事务 ACID|acid',
            '冲突可串行|conflict-serializable',
            '前向与后向冲突|rw-wr-ww',
            '两阶段锁|two-pl',
            '严格与强严格 2PL|strict-2pl',
            'MVCC|mvcc',
            '快照隔离|snapshot-isolation',
            '幻读与写偏斜|phantom-write-skew',
            '隔离级别|isolation-levels',
            '库内死锁|db-deadlock',
            'wait-die 与 wound-wait|wait-die',
            '两阶段提交|two-pc',
            '复制与日志传送|replication-log',
            '分片直觉|sharding',
          ],
        ],
      ],
    ],
  ],
]
