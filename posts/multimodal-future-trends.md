## 核心结论

多模态的下一阶段，不是“把图片、文本、音频一起喂给模型”这么简单，而是把多种感官和动作统一进一个闭环系统。这个闭环可以概括为 DP-TA：感知与对齐、世界建模、策略与执行三层结构。它的价值在于把“看到了什么”“现在处在什么状态”“下一步该做什么”拆成稳定接口，减少系统耦合失控的问题。根据 Frontiers 在 2025 年的综述，这类结构正在成为具身智能系统的通用组织方式。  
来源：<https://www.frontiersin.org/articles/10.3389/frobt.2025.1668910>

“具身智能”第一次出现时可以把它理解成：模型不只回答问题，还能通过机器人、车辆、机械臂等“身体”真正作用于环境。这样一来，多模态系统的目标就从“理解内容”扩展成“理解并行动”。视觉、听觉、触觉甚至未来的嗅觉，不再是并列输入通道，而是共同定义一个统一状态，再由模型同时生成语言、动作和安全反馈。

一个直观的玩具例子是看护机器人。它读入摄像头画面、病人的呼吸声、机械臂末端的力觉信息，先合成为“病人正在翻身、床边空间狭窄、当前接触力偏大”的房间状态，再输出两类结果：一类是语言提示，例如“请扶住左肩”；另一类是动作控制，例如把机械臂速度降到安全阈值以下。这才是未来多模态的核心方向：从多输入模型升级为多感官闭环系统。

下表可以先把整体结构看清：

| 层级 | 主要职责 | 典型输入 | 典型输出 |
| --- | --- | --- | --- |
| 感知与对齐 | 融合视觉、语音、触觉等异构信号 | RGB、音频、力觉、文本 | 统一状态 token |
| 世界建模 | 建立空间、因果、任务结构 | 统一状态 | 任务图、环境预测、风险估计 |
| 策略与执行 | 生成动作计划并实时修正 | 任务图、目标约束、安全规则 | 语言指令、动作轨迹、执行反馈 |

---

## 问题定义与边界

“多模态未来趋势”至少包含四条主线：多感官融合、具身智能、实时多模态理解、跨语言多模态模型。这里的“多感官融合”指系统同时处理视觉、听觉、触觉等不同来源的信息；“跨语言多模态”指模型不仅看图说英语，也能在中文、英文、日文等语言之间共享视觉和动作语义；“实时理解”强调系统要在毫秒到秒级预算内持续更新状态，而不是离线分析一段数据。

边界也必须说清。本文讨论的是“感知-建模-决策-执行”的系统趋势，不展开生成式图像、纯语音助手或传统单模态分类器。换句话说，只有当多模态输入真的改变了系统决策，甚至改变了物理动作，才属于这里的重点。

真正难的地方不是“能不能接入更多传感器”，而是“不同模态在时间、空间和语义上能不能对齐”。“对齐”第一次出现时可以把它理解成：同一时刻、同一对象、同一事件，来自不同传感器的数据能否被系统识别为同一件事。比如摄像头看到“前方有行人”，热成像看到“前方有热源”，雷达看到“前方有移动目标”，模型必须把三者合成一个稳定实体，而不是三条互相冲突的证据。

自动驾驶是最典型的真实工程例子。现实系统往往同时使用 RGB、LiDAR、毫米波雷达、热成像。RGB 语义丰富，但夜间和雨雾容易失真；LiDAR 深度精确，但点云稀疏；雷达抗恶劣天气能力强，但语义弱；热成像在低能见度下有效，但很难直接对齐到 RGB 的细粒度语义。因此，多模态边界不是“加得越多越好”，而是“保留哪些模态、允许哪些误差、在哪一层融合”。

| 模态 | 主要优势 | 主要短板 | 常见对齐风险 |
| --- | --- | --- | --- |
| RGB | 纹理、颜色、类别语义丰富 | 低光、逆光、遮挡敏感 | 夜间失真导致语义漂移 |
| 音频 | 事件触发明显，远距离有效 | 噪声大、定位弱 | 时间同步误差 |
| 触觉/力觉 | 接触状态直接、对抓取关键 | 只能局部感知 | 与视觉目标错配 |
| LiDAR/雷达 | 深度和速度信息稳定 | 稀疏或分辨率低 | 坐标系漂移 |
| 热成像 | 低能见度下可用 | 细粒度语义差 | 与 RGB 无法直接一一对应 |

---

## 核心机制与推导

从抽象上看，多模态具身系统要学的是一个映射：

$$
f:(x_{\text{text}}, x_{\text{vision}}, x_{\text{audio}}, x_{\text{touch}})\mapsto (a_{t_1}, a_{t_2}, ..., a_{t_n}, y)
$$

其中，$x$ 表示不同模态输入，$a_{t_i}$ 表示动作序列，$y$ 可以是语言解释、安全告警或任务状态。这个公式的重点不是“输出动作”本身，而是动作不再是模型外部的手工模块，而成为与语言、视觉同级的输出形式。WTA-H 对 2025 年 Gemini Robotics 的总结就强调了这一点：动作被视为新的输出模态，系统通过统一架构把感知、推理和执行绑定起来。  
来源：<https://www.wta-h.com/ai-2025-q3-multimodal_and_embodied_ai.html>

为什么还需要世界模型？因为仅靠当前帧融合，不足以支持多步任务。世界模型第一次出现时可以理解成：系统内部对“环境接下来会怎样变化”的预测器。比如机械臂抓杯子，不只要知道“杯子现在在哪”，还要预测“如果从这个角度夹，杯子会不会滑”“桌上其他物体会不会被碰倒”。DP-TA 的中间层正是在做这件事，它把统一状态进一步组织成任务图、因果关系或空间结构，再交给策略层。

可以把推导链条写成：

$$
s_t = \Phi(x_t^{(1)}, x_t^{(2)}, ..., x_t^{(m)})
$$

$$
g_t = W(s_t, h_{1:t-1})
$$

$$
a_t = \pi(g_t, c_t)
$$

这里，$\Phi$ 是感知融合与对齐模块，把多模态输入变成统一状态 $s_t$；$W$ 是世界模型，用当前状态和历史信息生成结构表示 $g_t$；$\pi$ 是策略函数，在任务约束 $c_t$ 下输出动作 $a_t$。如果加入实时代码合成模块 $\mathcal{C}$，那么策略层还能把抽象计划变成可执行控制脚本：

$$
\text{code}_t = \mathcal{C}(g_t, a_t, \text{safety})
$$

这就是“感知到动作”的可执行闭环。白话说，模型不是只说“去拿杯子”，而是要把“靠近、张开夹爪、施加 2N 力、上抬 8cm”这些步骤落到执行器接口上。

玩具例子可以用“抓杯子”说明。摄像头看到桌上有杯子，麦克风听到用户说“把红杯子递给我”，触觉传感器报告夹爪当前没有接触。融合后得到统一状态：目标是红杯子，位置在桌面右侧，当前夹持为空。世界模型预测从正上方抓取更稳，因为杯柄朝左。策略层于是输出动作序列，而不是一句自然语言。

---

## 代码实现

下面给一个可运行的简化版示例。它不依赖大模型，只演示三件事：多模态编码、按不确定性加权融合、基于统一状态生成动作。这里的“不确定性”可以理解成“这个模态此刻有多不可靠”。

```python
from dataclasses import dataclass

@dataclass
class SensorObs:
    value: float
    confidence: float  # 0~1, 越大越可信

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def encode_sensors(rgb: SensorObs, audio: SensorObs, touch: SensorObs):
    # 统一状态：目标存在概率、紧急程度、接触稳定度
    target_prob = 0.6 * rgb.value + 0.4 * audio.value
    urgency = audio.value
    contact = touch.value
    return {
        "target_prob": clamp(target_prob, 0.0, 1.0),
        "urgency": clamp(urgency, 0.0, 1.0),
        "contact": clamp(contact, 0.0, 1.0),
        "conf": {
            "rgb": rgb.confidence,
            "audio": audio.confidence,
            "touch": touch.confidence,
        }
    }

def feature_conditioned_align(state):
    conf = state["conf"]
    total = conf["rgb"] + conf["audio"] + conf["touch"]
    weights = {k: v / total for k, v in conf.items()}

    # 简化版 F-CMA：按当前可靠性动态调整状态贡献
    fused_score = (
        weights["rgb"] * state["target_prob"] +
        weights["audio"] * state["urgency"] +
        weights["touch"] * state["contact"]
    )
    state["fused_score"] = clamp(fused_score, 0.0, 1.0)
    state["weights"] = weights
    return state

def generate_plan(aligned):
    if aligned["target_prob"] < 0.3:
        return "search_target"
    if aligned["weights"]["rgb"] < 0.2:
        return "slow_approach_with_audio_touch_priority"
    if aligned["contact"] < 0.4:
        return "approach_and_grasp"
    return "lift_and_deliver"

def plan_to_action(plan):
    mapping = {
        "search_target": ["scan_scene"],
        "slow_approach_with_audio_touch_priority": ["reduce_speed", "approach_carefully"],
        "approach_and_grasp": ["move_to_target", "close_gripper"],
        "lift_and_deliver": ["lift_object", "move_to_user"],
    }
    return mapping[plan]

rgb = SensorObs(value=0.9, confidence=0.8)
audio = SensorObs(value=0.7, confidence=0.6)
touch = SensorObs(value=0.2, confidence=0.9)

state = encode_sensors(rgb, audio, touch)
aligned = feature_conditioned_align(state)
plan = generate_plan(aligned)
actions = plan_to_action(plan)

assert 0.0 <= aligned["fused_score"] <= 1.0
assert abs(sum(aligned["weights"].values()) - 1.0) < 1e-9
assert plan == "approach_and_grasp"
assert actions == ["move_to_target", "close_gripper"]
print(plan, actions)
```

这段代码对应真实系统里的三个阶段：

1. 感知编码：把 RGB、音频、触觉转成统一状态字段。
2. 条件对齐：根据模态可信度动态改变融合权重。
3. 行动生成：根据统一状态选择计划，再翻译成执行器动作。

如果放到真实工程中，`encode_sensors` 会变成各自的编码器，例如视觉 backbone、音频事件检测器、触觉时序模型；`feature_conditioned_align` 会变成跨模态 attention 或门控网络；`generate_plan` 往往由世界模型加策略模型共同完成；`plan_to_action` 则连接机器人 API、车辆控制器或医疗设备接口。

真实工程例子可以看仓储机器人分拣。摄像头识别货物条码与外形，力觉检测吸盘是否吸稳，麦克风可用于异常声学信号监测，比如电机异响。系统先融合状态，再预测“当前抓取是否稳定”“目标物体是否会滑落”，最后生成放置路径。如果视觉置信度降低，策略层会主动减速甚至触发复检，而不是照常执行。

---

## 工程权衡与常见坑

第一类坑是模态丢失。比如摄像头被雨水挡住、麦克风被环境噪声淹没、触觉传感器出现漂移。多模态系统的误区是“多一个模态就更稳”，但如果没有可靠的置信度建模，错误模态反而会把结果拖偏。解决办法通常是动态加权、冗余设计和缺失模态训练，让系统学会“少一个输入也能退化运行”。

第二类坑是语义模糊。视觉认为前方是“软包装”，触觉认为接触面“偏硬”，这不一定是冲突，也可能是包装里装着硬物。没有世界模型时，系统只能看到局部矛盾；有了世界模型，才可能把这些证据解释成更稳定的中间状态。

第三类坑是分布漂移。分布漂移第一次出现时可以理解成：训练环境和真实部署环境不是同一种数据分布。实验室里灯光稳定、背景干净，但真实仓库会有反光膜、异形纸箱、嘈杂声源。很多演示系统在 benchmark 上指标高，落地后表现差，根因不在模型参数量，而在传感器校准、时钟同步和长尾场景覆盖不够。

第四类坑是实时性。跨模态 attention、3D 重建、世界模型滚动预测都很吃算力。如果每一步都依赖大型模型，延迟可能直接超过控制周期。工程上常用的做法是两级系统：前端轻量感知负责高频刷新，后端大模型负责低频规划；危险情况下优先走硬规则和安全控制，不等待大模型完整推理。

| 工程维度 | 常见风险 | 直接后果 | 常见规避策略 |
| --- | --- | --- | --- |
| 模态丢失 | 遮挡、低光、传感器故障 | 错检、漏检、动作失误 | 不确定性加权、缺失模态训练 |
| 时间同步 | 音视频或视觉触觉不同步 | 事件错配 | 硬件时钟同步、缓冲对齐 |
| 语义漂移 | 训练集和部署场景不同 | 泛化下降 | 在线校准、域适应、任务反馈 |
| 实时性不足 | 模型过重、链路过长 | 控制延迟过大 | 分层推理、模型裁剪、提前停止 |
| 安全约束缺失 | 只追求成功率 | 物理风险升高 | 规则护栏、动作限幅、人工接管 |

一个面向新手也很重要的判断标准是：不要只看融合精度，还要看系统在“坏条件下怎么退化”。例如自动驾驶遇到暴雨，合理行为不是继续输出高置信控制，而是降低 RGB 权重、提高雷达和热成像占比，同时明确输出“感知不确定，需要保守策略”。能保守退化，比平均分更重要。

---

## 替代方案与适用边界

不是所有问题都需要完整的具身闭环。若任务只是“看懂视频并回答问题”，那么以视频世界模型为核心的方案往往更合适。比如 Meta 的 V-JEPA 2 路线，重点是从视频中学到时空结构和因果变化，再把这种表征迁移到理解或规划任务。它适合数据标注少、暂时没有真实执行器、但需要预测未来状态的场景。

另一类替代方案是 Caption-Assisted Reasoning 或类似 Emma-X 的语言辅助方法。这类系统先把视觉内容转成较结构化的文本或中间描述，再由语言模型做分步推理。优点是可解释性强、接入成本低，适合教育辅助、客服问答、科学图表解释等任务；缺点是它对真实动作控制帮助有限，因为文本中间层会丢掉大量细粒度时空信息。

所以，方案选择取决于你要解决的问题是不是“必须作用现实世界”。

| 方案 | 核心能力 | 适合场景 | 不足 |
| --- | --- | --- | --- |
| Embodied Closed-loop（DP-TA / Gemini Robotics 类） | 多感官融合 + 世界建模 + 动作执行 | 机器人、自动驾驶、医疗设备 | 系统复杂、成本高、安全要求高 |
| Video World Model（V-JEPA 2 类） | 从视频学习时空与因果结构 | 视频推理、离线规划、低标注学习 | 缺少真实执行闭环 |
| Caption-Assisted Reasoning | 语言中间层规划与解释 | 教育、客服、科研辅助 | 细粒度动作控制弱 |
| 传统单模态系统 | 单一路径稳定优化 | 约束清晰、场景简单的任务 | 泛化和鲁棒性有限 |

可以用一句话区分适用边界：如果目标是“解释世界”，视频世界模型和语言辅助方案就够用；如果目标是“改变世界”，具身闭环几乎不可避免。

---

## 参考资料

- Zhang, Tian, Xiong. “A review of embodied intelligence systems: a three-layer framework integrating multimodal perception, world modeling, and structured strategies.” Frontiers in Robotics and AI, 2025. <https://www.frontiersin.org/articles/10.3389/frobt.2025.1668910>
- Emergent Mind. “Multimodal Perception & Fusion.” 更新于 2026-02-09. <https://www.emergentmind.com/topics/multimodal-perception-and-fusion>
- Thilo Hofmeister. “Multimodal & Embodied AI - Q3 2025.” WTA-H, 2025. <https://www.wta-h.com/ai-2025-q3-multimodal_and_embodied_ai.html>
