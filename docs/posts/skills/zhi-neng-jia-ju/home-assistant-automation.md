---
title: Home Assistant 自动化与脚本配置
date: 2026-08-07
---

# Home Assistant 自动化与脚本配置

<div class="epigraph">
<p>自动化是对重复劳动最优雅的复仇。</p>
<footer>—— 自动化爱好者格言</footer>
</div>

<div class="article-byline">
<p>第十级 · 智能家居设备安装与调试 ｜ Home Assistant 官方文档 ｜ 2026-08-07</p>
</div>

## 为什么 HA 自动化是「规则表达的天花板」

米家能拼「当…就…」，而 Home Assistant 的自动化几乎是**完整的编程语言**：变量、模板、嵌套、多条件、服务调用、调试日志一应俱全。米家做不到的「温度 > 30℃ 且（客厅有人或阳台门开）就开空调」，在 HA 里是小菜一碟。这一篇讲 HA 自动化的骨架、图形化与 YAML 两种写法、以及脚本与蓝图的进阶能力。

## 1 自动化 YAML 骨架

HA 的自动化由一个 YAML 文件（或 UI 中录入）描述，核心是三个字段，对应 TCA 模型：

```yaml
alias: 回家亮灯
triggers:
  - trigger: state
    entity_id: binary_sensor.front_door
    to: "on"
conditions:
  - condition: time
    after: "17:30:00"
actions:
  - action: light.turn_on
    target:
      entity_id: light.living_room
    data:
      brightness_pct: 80
mode: single
```

**triggers（触发）**：何时开始评估。常见类型：`state`（状态变化）、`time`（定时）、`sun`（日出日落）、`numeric_state`（数值范围）、`event`（事件）。

**conditions（条件）**：触发后还需满足什么才执行。`time`（时间窗口）、`state`（某设备状态）、`numeric_state`（数值）、`and/or/not`（逻辑组合）、`template`（模板表达式）。

**actions（动作）**：执行什么。调用**服务（Service）**——`light.turn_on`、`climate.set_temperature`、`notify.mobile_app` 等。<span class="marginnote">理解「服务」是看懂 HA 自动化的钥匙：<strong>服务是 HA 的「可调用操作」，域（domain）是设备类别</strong>——`light.turn_on` 就是对「灯」域调用「开」服务。配置界面里的每个开关，底层都是服务调用。</span>

**mode（模式）**：`single`（单实例，默认）、`restart`、`queued`、`parallel`。决定自动化被再次触发时旧实例怎么处理——**这是防「重复触发堆积」的关键字段**。

## 2 图形化编辑器 vs YAML

HA 提供两套写法：

**图形化编辑器（UI）**：在「设置 → 自动化与场景」里用可视化表单搭触发/条件/动作，不写代码。适合简单场景，所见即所得，也是新手入口。

**YAML 模式**：切换「编辑 YAML」，直接写代码。适合复杂逻辑，可读性差但表达力强。**注意：** 一个自动化要么用 UI 模式要么用 YAML 模式，不能混用（用 YAML 编辑过的自动化会锁定为「YAML 模式」）。

**最佳实践**：UI 搭骨架，复杂条件（模板、嵌套）用 YAML 补。给客户交付的自动化，养成**写 alias 注释**的习惯——alias 是排障时一眼看懂「这条规则干嘛的」的唯一入口。<span class="marginnote">alias 命名是 HA 工程化的第一课：<strong>「回家亮灯-晚间」「离家布防-全屋」比「Auto 1」「Auto 2」好排障一百倍</strong>。alias 还能作为脚本的「人名」，供其他自动化调用。</span>

## 3 脚本与蓝图

**脚本（Script）**：一组动作的**命名序列**，可被自动化、仪表盘、语音重复调用。区别在于：自动化是「等触发」，脚本是「被调用」。

```yaml
script:
  evening_movie:
    alias: 观影模式
    sequence:
      - action: light.turn_on
        target: { entity_id: light.living_room }
        data: { brightness_pct: 20 }
      - action: cover.close_cover
        target: { entity_id: cover.living_room_curtain }
      - action: media_player.select_source
        target: { entity_id: media_player.tv }
        data: { source: "HDMI 1" }
```

**蓝图（Blueprint）**：可复用的**自动化模板**。社区分享一份蓝图，填入自己的设备即可用，避免重复劳动。HA 社区（Community Blueprint Exchange）有大量现成蓝图（如「按门锁触发回家场景」）。<span class="marginnote">蓝图是「自动化的函数库」：<strong>把「基于门锁状态的回家场景」这类高频逻辑做成一模板，填设备 ID 就复用</strong>。给客户搭同质化场景（多个房间的入睡模式），一份蓝图循环套用，效率翻倍。</span>

## 4 实战：写一个带模板条件的分段回家自动化

把米家做不到的「多条件 + 模板」在 HA 里实现一遍：

**需求**：18 点后，门磁打开，且（客厅温度 > 28℃ 或 湿度 > 70%）时，开客厅空调制冷 26℃。

```yaml
alias: 回家-高温高湿开空调
triggers:
  - trigger: state
    entity_id: binary_sensor.front_door
    to: "on"
conditions:
  - condition: time
    after: "18:00:00"
  - condition: or
    conditions:
      - condition: numeric_state
        entity_id: sensor.living_temp
        above: 28
      - condition: numeric_state
        entity_id: sensor.living_humidity
        above: 70
actions:
  - action: climate.turn_on
    target: { entity_id: climate.living_ac }
  - action: climate.set_temperature
    target: { entity_id: climate.living_ac }
    data: { temperature: 26 }
mode: single
```

**调试方法**：在「开发者工具 → 模板」页可以直接测模板表达式；自动化旁边有「运行」按钮可手动触发；「记录器」里能看到自动化触发日志与执行耗时。<span class="marginnote">HA 排障三件套：<strong>开发者工具（测模板/调服务）、自动化运行按钮（手动触发）、记录器日志（看触发链）</strong>。自动化「没反应」时，先手动运行一次——手动能跑，说明规则对、触发没到；手动都跑不了，规则本身有错。</span>

## 5 核心对比表：自动化 vs 脚本 vs 场景

纯技能主题，用对比表替代公式解析：

| 对比维度 | 自动化 | 脚本 | 场景 |
| --- | --- | --- | --- |
| 运行方式 | 等触发自动运行 | 被手动/他方调用 | 一键/被调用设置状态 |
| 是否带条件 | 是（trigger+condition） | 可含条件 | 否（纯状态快照） |
| 复用性 | 低 | 高（命名调用） | 高（快照引用） |
| 复杂逻辑 | 最强 | 中 | 无 |
| 典型用途 | 场景自动触发 | 观影模式等固定流程 | 一键全屋状态 |

## 6 小结

- HA 自动化骨架：**triggers + conditions + actions + mode**，对应 TCA 模型。
- 两种写法：**UI 搭骨架、YAML 补复杂条件**；alias 命名是排障第一课。
- **脚本**是命名动作序列，被调用执行；**蓝图**是自动化模板，一份复用多场景。
- 模板条件（`numeric_state`、`or`、`and`）让「多条件判断」成为可能，这是米家做不到的。
- 排障三件套：**开发者工具、手动运行按钮、记录器日志**。

在下一节，我们把「触发源」做一次专门梳理——讲语音控制与传感器触发条件设置。
