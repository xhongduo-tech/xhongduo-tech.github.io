## 核心结论

坐标预测精度，指模型给出的点击位置与真实目标中心之间的像素偏差，是计算机使用型智能体能否稳定完成 GUI 操作的关键中间变量。这里的 GUI 是图形用户界面，也就是按钮、菜单、输入框、标签页、图标这些可视化控件。对“看图点按钮”的多模态模型来说，只要点击偏差超过目标的可点击范围，哪怕模型已经理解对了语义，动作仍然会失败。

先把结论说清楚。基于题目给出的研究摘要，以及 GUI 交互本身的几何约束，可以先抓住三条主线：

| 因素 | 典型现象 | 对精度的影响 |
|---|---|---|
| 分辨率下降 | 截图被缩小、压缩后再送入模型 | 坐标量化误差放大，点击成功率下降 |
| 目标过小 | 按钮、图标边长小于 20-24px | 少量偏差就会越界，失败率明显上升 |
| 界面更难看清 | 深色主题、元素密集、弹窗叠加、低对比度 | 目标边界更模糊，更容易误点邻近控件 |

如果把“是否点中”当成一个几何问题，那么最直接的误差定义就是欧氏距离：

$$
E=\sqrt{(x_{\text{pred}}-x_{\text{true}})^2+(y_{\text{pred}}-y_{\text{true}})^2}
$$

其中：

| 符号 | 含义 |
|---|---|
| $x_{\text{pred}}, y_{\text{pred}}$ | 模型输出的点击坐标 |
| $x_{\text{true}}, y_{\text{true}}$ | 真实目标中心坐标 |
| $E$ | 预测点到真实中心的像素误差 |

只要

$$
E>r_{\text{target}}
$$

也就是误差大于目标的有效命中半径 $r_{\text{target}}$，这次点击就应该视为失败。

这件事对新手最容易误判的地方在于：视觉上“差一点”不等于交互上“还能成功”。看一个最小例子。假设一个收藏按钮大小是 $16\times 16$ px，中心点在 $(500,300)$。模型预测到 $(514,309)$，那么误差为：

$$
E=\sqrt{14^2+9^2}=\sqrt{277}\approx 16.64
$$

而这个按钮的保守命中半径近似只有：

$$
r_{\text{target}}=\frac{16}{2}=8
$$

于是有：

$$
16.64 > 8
$$

所以这次点击一定失败。它不是“有点偏”，而是已经完全点出了按钮范围。

再看真实工程场景。题目给出的 OSWorld 摘要指出，Claude 3.5 Sonnet 在纯截图配置下的坐标偏差中位数大约在 15-30px；同一摘要还指出，当按钮尺寸小于 20px 时，失败率可达到约 35%。这说明现代桌面界面中的典型问题不是“模型完全不会点”，而是“模型常常知道该点谁，但落点不够准”。

把这三类因素放在一起，可以得到一个足够实用的工程判断：

| 场景 | 语义理解 | 坐标精度 | 结果 |
|---|---|---|---|
| 大按钮、高分辨率、界面干净 | 对 | 够用 | 往往成功 |
| 小按钮、高分辨率、界面密集 | 对 | 边缘 | 易误点 |
| 小按钮、低分辨率、深色主题 | 对 | 不够 | 高概率失败 |
| 有 a11y 或 bbox 辅助 | 对 | 由系统计算 | 稳定性明显提升 |

结论可以压缩成一句话：纯截图 GUI 代理的瓶颈，常常不是“看不懂”，而是“点不准”。

---

## 问题定义与边界

这篇文章讨论的不是“模型会不会规划任务”，而是更窄、更底层的问题：模型已经知道要点击哪个控件时，它输出的像素坐标是否足够准确。

这个边界必须先划清楚，否则“任务失败”会把很多不同原因混在一起，最后既无法分析，也无法优化。

| 维度 | 本文讨论 | 不讨论或弱讨论 |
|---|---|---|
| 输入信息 | 纯截图视觉输入 | 完整 a11y 树、DOM、原生控件树 |
| 输出动作 | 点击坐标 $(x,y)$ | 长链规划、工具调用编排 |
| 目标类型 | 按钮、图标、菜单项、标签页等离散目标 | 拖拽、自由绘图、复杂手势 |
| 错误来源 | 坐标偏差、缩放映射、边界识别失败 | 网络延迟、系统卡顿、权限弹窗 |

这里的 a11y 是 accessibility 的缩写，可以理解为系统暴露给辅助功能或自动化工具的结构化界面信息，例如：

- 按钮名称
- 控件角色
- 是否可点击
- 边界框位置
- 层级关系

一旦系统拿到了这类信息，问题就不再是“从像素里猜按钮中心”，而变成“从结构化对象中选择正确目标”。这是完全不同的难度等级。

为了讨论精度，成功判定也必须定义清楚。最常见的近似是把矩形目标转成一个保守的圆形命中区。对于宽 $w$、高 $h$ 的矩形按钮，可以取：

$$
r_{\text{target}}=\frac{\min(w,h)}{2}
$$

于是成功条件变成：

$$
\sqrt{(x_{\text{pred}}-x_{\text{true}})^2+(y_{\text{pred}}-y_{\text{true}})^2}\le\frac{\min(w,h)}{2}
$$

这个定义虽然保守，但工程上很好用，因为它把“按钮越小越脆弱”写成了明确的数学条件。

看两个例子：

| 按钮大小 | 保守半径 $r_{\text{target}}$ | 含义 |
|---|---|---|
| $48\times 32$ px | $16$ px | 容错相对较高 |
| $24\times 24$ px | $12$ px | 已经比较敏感 |
| $16\times 16$ px | $8$ px | 很容易失败 |

如果一个 $40\times 20$ px 的按钮中心在 $(800,200)$，模型点击到了 $(811,200)$，那么误差是 11px；而命中半径只有 10px，所以这次点击仍然失败。新手在这里最容易犯的错，是只看“偏差不大”，不看“目标到底有多小”。

把题目中的边界条件整理后，可以得到一张更适合工程评估的风险表：

| 条件 | 风险级别 | 原因 |
|---|---|---|
| 1920×1080 原始截图 | 中 | 视觉细节较充分，但仍会存在 15-30px 级偏差 |
| 960×540 或更低 | 高 | 一个预测单位对应更大的真实像素块 |
| 目标小于 24px | 高 | 命中区太小，容错显著下降 |
| 深色主题 | 中到高 | 边界、阴影、文本对比度更难稳定识别 |
| 界面密集 | 高 | 相邻控件距离小，轻微偏差就会误触 |
| 无 a11y 信息 | 高 | 只能依赖视觉纹理推断点击中心 |

还要再强调一个常见误解：“只要模型读懂了按钮文字，就应该能点中。”这不成立。因为这里至少包含两个不同子任务：

| 子任务 | 本质问题 |
|---|---|
| 语义识别 | 它知道应该点哪个控件吗？ |
| 坐标定位 | 它知道该点这个控件的哪一个像素区域吗？ |

前者回答“点谁”，后者回答“点哪里”。很多 GUI 代理恰恰是在第二步失败。

---

## 核心机制与推导

多模态模型在截图上做坐标预测时，至少要经过三层映射：

1. 原始屏幕先被压缩到模型可处理的输入分辨率。
2. 模型在压缩后的视觉表征中识别目标区域。
3. 模型再把预测位置映射回原始屏幕坐标。

这三步里，第一步和第三步共同决定了一个核心现象：同样大小的内部误差，在低分辨率输入下会被放大成更大的真实像素误差。

这里定义下采样比例 downsample ratio 为 $r$，表示“模型输入尺寸 / 原始屏幕尺寸”。例如：

- 从 1920×1080 缩到 960×540，则 $r=0.5$
- 从 1920×1080 缩到 480×270，则 $r=0.25$

如果模型在缩略图空间中的感知误差约为 $\delta$ 个单位，那么映射回原图后的误差近似变成：

$$
\Delta \approx \frac{\delta}{r}
$$

这就是为什么 $r$ 越小，真实误差越大。不是模型突然更“笨”，而是同样的内部定位偏差在映射回原图时被放大了。

下面把这个过程写成明确推导。

假设原图坐标为 $(x,y)$，缩放后的坐标为：

$$
(x',y')=(rx,ry)
$$

模型在缩放图上的预测误差为 $(\epsilon_x,\epsilon_y)$。也就是说，它实际输出的是：

$$
(x'+\epsilon_x,\ y'+\epsilon_y)
$$

再映射回原图：

$$
x_{\text{pred}}=\frac{x'+\epsilon_x}{r}=x+\frac{\epsilon_x}{r}
$$

$$
y_{\text{pred}}=\frac{y'+\epsilon_y}{r}=y+\frac{\epsilon_y}{r}
$$

所以原图上的误差项是：

$$
\Delta_x=\frac{\epsilon_x}{r},\quad \Delta_y=\frac{\epsilon_y}{r}
$$

对应的总误差近似为：

$$
\Delta=\sqrt{\left(\frac{\epsilon_x}{r}\right)^2+\left(\frac{\epsilon_y}{r}\right)^2}
=\frac{1}{r}\sqrt{\epsilon_x^2+\epsilon_y^2}
$$

这条式子非常关键，因为它直接给出工程含义：

| 下采样比例 $r$ | 误差放大倍数 $1/r$ |
|---|---|
| 1.0 | 1 倍 |
| 0.5 | 2 倍 |
| 0.4 | 2.5 倍 |
| 0.25 | 4 倍 |

如果缩略图里只偏了 6px：

- 当 $r=1.0$ 时，原图偏差约 6px
- 当 $r=0.5$ 时，原图偏差约 12px
- 当 $r=0.25$ 时，原图偏差约 24px

这个量级一旦碰到小按钮，失败几乎是必然的。

可以配合一个示意图理解：

```text
原图 1920x1080
+--------------------------------------+
|                                      |
|                [按钮]                |
|                  ^                   |
|               真中心点               |
|                                      |
+--------------------------------------+

缩略图 960x540
+------------------+
|                  |
|      [按钮]      |
|        ^         |
|   模型在这里估计   |
|                  |
+------------------+

若缩略图上偏 7px
回到原图后约偏 14px
```

题目给出的研究摘要提到，点击成功率 $\mathrm{SR}(r)$ 会随分辨率上升而上升，并且在 $r<0.4$ 后下降更明显。工程上可以把它写成一个粗糙但有用的一阶近似：

$$
\mathrm{SR}(r)\approx k\cdot r
$$

这里要明确，这不是物理定律，只是表达单调关系的简化模型。它的意义只有一个：其他条件大致不变时，更高的输入分辨率通常带来更高的命中率。

如果想把“命中失败”写得再直观一点，可以把误差和目标半径直接合并成一个裕量指标：

$$
M=r_{\text{target}}-E
$$

其中：

- $M>0$ 表示还有命中余量
- $M=0$ 表示刚好命中边界
- $M<0$ 表示已经越界失败

这样做的好处是，在日志分析时可以直接比较不同目标的“安全裕量”，而不是只看误差本身。

看一个更完整的玩具例子。原始分辨率下一个按钮大小为 $24\times24$ px，中心在 $(1000,400)$，内部误差约为 6 个缩略图像素。

| $r$ | 原图误差近似 | 按钮半径 | 结果 |
|---|---|---|---|
| 1.0 | 6 px | 12 px | 可能成功 |
| 0.5 | 12 px | 12 px | 处于边界 |
| 0.25 | 24 px | 12 px | 基本必然失败 |

真实工程例子则更典型。题目给出的 GIMP 类案例中，模型虽然知道要找 “Brightness-Contrast”，但只能以固定步长估计大概位置。结果是它的大方向没错，却会因为 15-30px 的误差落在空白区域、标签栏边缘或者邻近菜单项上。最麻烦的不是“一次点歪”，而是点歪之后会触发新的界面状态，例如：

- 打开错误菜单
- 焦点跳到错误控件
- 触发无效弹窗
- 需要多一次截图和重试

于是整个 agent loop，也就是“观察-思考-行动”循环，会被浪费在纠错上。

---

## 代码实现

下面给出一个可以直接运行的 Python 示例，用来说明怎么测量坐标误差、统计成功率，并同时记录分辨率、主题、目标大小等变量。这个例子不依赖任何第三方库，只用标准库，重点是评估方法而不是模型本身。

```python
from dataclasses import dataclass
from math import sqrt
from statistics import mean, median


@dataclass
class Sample:
    scene: str
    resolution: str
    theme: str
    target_xy: tuple[float, float]
    target_w: int
    target_h: int
    downsample_ratio: float
    internal_error: tuple[float, float]


def euclidean_error(pred: tuple[float, float], target: tuple[float, float]) -> float:
    px, py = pred
    tx, ty = target
    return sqrt((px - tx) ** 2 + (py - ty) ** 2)


def target_radius(target_w: int, target_h: int) -> float:
    return min(target_w, target_h) / 2.0


def hit_success(
    pred: tuple[float, float],
    target: tuple[float, float],
    target_w: int,
    target_h: int,
) -> bool:
    return euclidean_error(pred, target) <= target_radius(target_w, target_h)


def simulate_scaled_prediction(
    true_xy: tuple[float, float],
    base_internal_error: tuple[float, float],
    downsample_ratio: float,
) -> tuple[float, float]:
    if downsample_ratio <= 0:
        raise ValueError("downsample_ratio must be > 0")
    tx, ty = true_xy
    ex, ey = base_internal_error
    # 模型在缩略图空间中的误差会在回映到原图时按 1/r 放大。
    return (tx + ex / downsample_ratio, ty + ey / downsample_ratio)


def evaluate(samples: list[Sample]) -> list[dict]:
    rows = []
    for sample in samples:
        pred = simulate_scaled_prediction(
            sample.target_xy,
            sample.internal_error,
            sample.downsample_ratio,
        )
        err = euclidean_error(pred, sample.target_xy)
        radius = target_radius(sample.target_w, sample.target_h)
        ok = hit_success(pred, sample.target_xy, sample.target_w, sample.target_h)

        rows.append(
            {
                "scene": sample.scene,
                "resolution": sample.resolution,
                "theme": sample.theme,
                "target_size": f"{sample.target_w}x{sample.target_h}",
                "downsample_ratio": sample.downsample_ratio,
                "pred_xy": (round(pred[0], 2), round(pred[1], 2)),
                "error_px": round(err, 2),
                "radius_px": round(radius, 2),
                "margin_px": round(radius - err, 2),
                "success": ok,
            }
        )
    return rows


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must not be empty")
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")

    pos = (len(sorted_values) - 1) * q
    left = int(pos)
    right = min(left + 1, len(sorted_values) - 1)
    frac = pos - left
    return sorted_values[left] * (1 - frac) + sorted_values[right] * frac


def summarize(rows: list[dict]) -> dict:
    errors = sorted(row["error_px"] for row in rows)
    success_rate = sum(1 for row in rows if row["success"]) / len(rows)
    return {
        "count": len(rows),
        "pass_at_1": round(success_rate, 3),
        "avg_error": round(mean(errors), 2),
        "med_error": round(median(errors), 2),
        "p90_error": round(quantile(errors, 0.9), 2),
    }


def print_table(rows: list[dict]) -> None:
    headers = [
        "scene",
        "resolution",
        "theme",
        "target_size",
        "ratio",
        "pred_xy",
        "error_px",
        "radius_px",
        "margin_px",
        "success",
    ]
    widths = {h: len(h) for h in headers}
    for row in rows:
        widths["scene"] = max(widths["scene"], len(str(row["scene"])))
        widths["resolution"] = max(widths["resolution"], len(str(row["resolution"])))
        widths["theme"] = max(widths["theme"], len(str(row["theme"])))
        widths["target_size"] = max(widths["target_size"], len(str(row["target_size"])))
        widths["ratio"] = max(widths["ratio"], len(str(row["downsample_ratio"])))
        widths["pred_xy"] = max(widths["pred_xy"], len(str(row["pred_xy"])))
        widths["error_px"] = max(widths["error_px"], len(str(row["error_px"])))
        widths["radius_px"] = max(widths["radius_px"], len(str(row["radius_px"])))
        widths["margin_px"] = max(widths["margin_px"], len(str(row["margin_px"])))
        widths["success"] = max(widths["success"], len(str(row["success"])))

    header_line = " | ".join(h.ljust(widths[h]) for h in headers)
    sep_line = "-+-".join("-" * widths[h] for h in headers)
    print(header_line)
    print(sep_line)

    for row in rows:
        values = {
            "scene": row["scene"],
            "resolution": row["resolution"],
            "theme": row["theme"],
            "target_size": row["target_size"],
            "ratio": row["downsample_ratio"],
            "pred_xy": row["pred_xy"],
            "error_px": row["error_px"],
            "radius_px": row["radius_px"],
            "margin_px": row["margin_px"],
            "success": row["success"],
        }
        print(" | ".join(str(values[h]).ljust(widths[h]) for h in headers))


samples = [
    Sample(
        scene="large_button_light",
        resolution="1920x1080",
        theme="light",
        target_xy=(500, 300),
        target_w=40,
        target_h=24,
        downsample_ratio=1.0,
        internal_error=(6, 4),
    ),
    Sample(
        scene="small_button_light",
        resolution="1920x1080",
        theme="light",
        target_xy=(500, 300),
        target_w=16,
        target_h=16,
        downsample_ratio=1.0,
        internal_error=(10, 8),
    ),
    Sample(
        scene="dense_dark_halfres",
        resolution="960x540",
        theme="dark",
        target_xy=(1000, 420),
        target_w=20,
        target_h=20,
        downsample_ratio=0.5,
        internal_error=(7, 5),
    ),
    Sample(
        scene="small_button_quarterres",
        resolution="480x270",
        theme="dark",
        target_xy=(1000, 420),
        target_w=20,
        target_h=20,
        downsample_ratio=0.25,
        internal_error=(4, 3),
    ),
]

rows = evaluate(samples)
summary = summarize(rows)

assert round(euclidean_error((3, 4), (0, 0)), 2) == 5.00
assert target_radius(16, 16) == 8.0
assert hit_success((10, 10), (12, 12), 8, 8) is True
assert hit_success((10, 10), (20, 20), 8, 8) is False
assert summary["count"] == 4
assert summary["avg_error"] > 0
assert summary["p90_error"] >= summary["med_error"]

print_table(rows)
print()
print("summary =", summary)
```

这段代码有三个特点，对新手尤其重要。

第一，它把“误差”和“是否点中”拆开了。很多人只统计任务是否成功，但不知道失败到底是语义错误还是坐标错误。这里单独记录 `error_px`、`radius_px`、`margin_px`，就能看出失败是否只是因为偏差略大。

第二，它把环境变量写进了样本。下面三个字段在真实系统里几乎是必须记录的：

| 字段 | 含义 | 为什么必须记录 |
|---|---|---|
| `resolution` | 截图分辨率 | 用来分析缩放与误差关系 |
| `theme` | 明亮/深色主题 | 用来分析外观变体影响 |
| `target_size` | 目标大小 | 用来分析小目标失败率 |

第三，它明确区分了“内部误差”和“原图误差”。`internal_error` 是模型在缩略图空间里的偏差，`downsample_ratio` 决定这个偏差映射回原图时会被放大多少。这正好对应前一节的推导。

如果你直接运行上面代码，会得到类似下面的输出结构：

```text
scene                  | resolution | theme | target_size | ratio | pred_xy         | error_px | radius_px | margin_px | success
-----------------------+------------+-------+-------------+-------+-----------------+----------+-----------+-----------+--------
large_button_light     | 1920x1080  | light | 40x24       | 1.0   | (506.0, 304.0)  | 7.21     | 12.0      | 4.79      | True
small_button_light     | 1920x1080  | light | 16x16       | 1.0   | (510.0, 308.0)  | 12.81    | 8.0       | -4.81     | False
dense_dark_halfres     | 960x540    | dark  | 20x20       | 0.5   | (1014.0, 430.0) | 17.2     | 10.0      | -7.2      | False
small_button_quarterres| 480x270    | dark  | 20x20       | 0.25  | (1016.0, 432.0) | 20.0     | 10.0      | -10.0     | False
```

从这张表可以直接看出：

- 大按钮在原始分辨率下，即使有 7px 左右误差，也仍可能成功。
- 同样量级的偏差放到 16px 小按钮上，就会直接失败。
- 当分辨率降低到一半或四分之一时，即使内部误差不算大，回到原图后也会变成 17px、20px 甚至更大的偏差。

如果要在真实系统里埋点评估，最小日志格式可以写成：

```text
timestamp,task_id,resolution,theme,target_w,target_h,x_true,y_true,x_pred,y_pred,error_px,success
2026-03-08T10:01:00Z,t1,1920x1080,light,24,24,500,300,514,309,16.64,false
2026-03-08T10:02:00Z,t2,960x540,dark,20,20,1000,420,1014,430,17.20,false
```

真实工程里建议至少再补两个字段：

| 字段 | 作用 |
|---|---|
| `nearest_neighbor_dist` | 记录最近邻可点击目标距离，便于分析误点风险 |
| `step_index` | 记录这是任务中的第几步，便于分析长链任务中的误差累积 |

几百到几千条点击样本后，就可以稳定算出：

- `pass@1`：第一次点击是否命中
- `avg_error`：平均像素偏差
- `p50/p90 error`：中位数和高分位误差
- 按主题、分辨率、控件大小分组后的失败率

其中 `pass@1` 可以简单理解为“一次命中率”，也就是第一次出手是否成功，不看后续补救动作。

---

## 工程权衡与常见坑

工程里最常见的错误，不是完全不做评估，而是只看最终任务成功率，不拆解失败来源。坐标问题尤其容易被掩盖，因为任务最后可能靠多轮重试做成，但真实成本已经上升很多，例如：

- 截图轮数增加
- token 消耗增加
- 任务时延增加
- 状态被误操作污染
- 后续步骤建立在错误上下文上

先看最核心的权衡。

| 选择 | 好处 | 代价 |
|---|---|---|
| 降低截图分辨率 | 省带宽、省 token、响应更快 | 坐标量化误差变大 |
| 只用视觉截图 | 接入简单、跨平台 | 无法直接拿到可靠边界框 |
| 保留原始高分辨率 | 点击更准 | 成本更高、推理更慢 |
| 引入 a11y/边界框 | 精度和稳定性更高 | 系统接入复杂度增加 |
| 扩大目标点击区 | 容错更高 | 需要改 UI 或额外封装点击逻辑 |

这个权衡不是抽象的，而是会直接影响长任务的最终通过率。假设单次点击成功率是 $p$，一个任务需要连续点击 $n$ 次，那么理想化的一次通过概率近似是：

$$
P_{\text{task}}=p^n
$$

这说明单步误差会被任务长度放大。

看两个数字：

$$
0.75^{10}\approx 5.6\%
$$

$$
0.69^{10}\approx 2.4\%
$$

题目给出的 OpenApps 摘要里提到，暗色主题会让某些代理的 pass@1 从约 75% 降到 69%。单看只是 6 个百分点，但如果任务需要 10 次连续稳定点击，整体一次通过概率会近乎腰斩。这就是为什么“单次只差一点”在长链任务中会变成明显的稳定性问题。

常见坑可以分成下面几类。

| 常见坑 | 典型后果 | 规避方式 |
|---|---|---|
| 截图压得太小 | 模型看得懂但点不准 | 尽量保持 1080p 级别输入，至少保留关键区域清晰度 |
| 目标太小 | 少量偏差直接失败 | 扩大 hit area，参考 24px 以上 |
| 元素太密集 | 误点相邻控件 | 记录邻近目标距离，单独评估密集布局 |
| 深色主题未测试 | 线上稳定性突然下降 | 明暗主题分别做回归测试 |
| 只看最终成功率 | 坐标问题被重试掩盖 | 单独记录点击级指标 |
| 没有结构化信息 | 只能“猜中心点” | 接入 a11y、DOM 或显式 bbox |
| 坐标映射链路不一致 | 明明预测对了仍点偏 | 统一截图、缩放、显示、执行的坐标系 |

最后一类坑最容易被忽略。很多系统里的失败不是模型本身带来的，而是工程链路把坐标搞错了。例如：

- 截图按设备像素渲染，但点击按 CSS 像素执行
- 浏览器缩放比例不是 100%
- Retina 屏使用了逻辑坐标和物理坐标两套体系
- 多显示器场景下坐标原点不一致

这类问题和模型能力无关，但结果看起来和“模型点不准”完全一样。因此工程上必须做一条单独的校验：

$$
(x_{\text{screen}}, y_{\text{screen}})
\overset{\text{capture}}{\longrightarrow}
(x_{\text{model}}, y_{\text{model}})
\overset{\text{decode}}{\longrightarrow}
(x_{\text{exec}}, y_{\text{exec}})
$$

要求在没有模型误差时满足：

$$
(x_{\text{exec}}, y_{\text{exec}})=(x_{\text{screen}}, y_{\text{screen}})
$$

如果这条链路本身不守恒，后续所有精度分析都会失真。

WCAG 2.5.8 提到的 24×24 CSS 像素目标尺寸，也可以直接放进这里理解。它原本是无障碍规范，不是给智能体写的，但对 UI 代理同样成立，因为它本质上是在提高命中容错。目标越大、间距越足，越不容易因为几像素偏差而误触邻近控件。

---

## 替代方案与适用边界

如果系统只靠截图预测坐标，那么它天然受限于三件事：

- 视觉分辨率
- 目标大小
- 界面对比度与密度

因此要提升精度，最有效的路线通常不是“让模型更努力看图”，而是给它更多可定位信息。

最常见的替代方案是“visual + metadata”，也就是视觉加结构化元数据。这里的 metadata 可以理解为一切机器可读、可直接参与定位的信息，例如：

- 控件边界框
- 控件类型
- 文本标签
- DOM 节点
- a11y 树中的 role / name / state

把几种常见方案放在一起看：

| 方案 | 输入 | 优势 | 适用边界 |
|---|---|---|---|
| Visual-only | 截图 | 接入最简单，跨平台 | 原型验证、低风险任务 |
| Visual + a11y | 截图 + 可访问性树 | 目标边界更准，动作更稳定 | 原生应用、桌面自动化、部分浏览器环境 |
| Visual + SoM | 截图 + 场景标注框 | 易解释、易调试 | 研究评估、代理框架 |
| DOM/API 直控 | 结构化页面对象 | 精度最高、确定性最强 | Web 场景、可控环境 |
| Hybrid 执行 | 视觉负责找目标，系统负责算中心 | 保留泛化能力，同时降低像素误差 | 通用代理平台 |

这里的 SoM 可以理解为 Set-of-Marks，即先由系统把可交互区域显式框出来、编号，再交给模型做选择。这样做的关键价值不在“让模型看更多”，而在“减少信息损失”。模型不再需要从复杂纹理中直接回归一个自由坐标，而是从一组候选框里做离散选择。

一个最小伪代码如下：

```text
system:
当前屏幕分辨率为 1920x1080。
候选可点击区域：
1. 收藏按钮 bbox=(488,292,16,16)
2. 分享按钮 bbox=(520,292,16,16)
请只输出要点击的候选编号。

agent output:
1
```

系统再根据边界框计算中心：

$$
x_{\text{click}}=x+\frac{w}{2},\quad y_{\text{click}}=y+\frac{h}{2}
$$

如果 `bbox=(488,292,16,16)`，那么中心点是：

$$
(496,300)
$$

这个设计把原本的自由坐标回归问题，转成了“目标选择”问题。二者的难度完全不同：

| 任务形式 | 输出空间 | 容错性 |
|---|---|---|
| 直接输出坐标 | 连续空间 | 低 |
| 选择候选框编号 | 离散空间 | 高 |

看一个玩具例子。只给模型一张有三个相邻图标的截图时，它需要同时完成两件事：

1. 判断哪个图标是“收藏”
2. 估计它中心点的精确像素位置

如果额外给出候选框，模型只需要完成第一件事，第二件事由系统精确计算。因此自由坐标中的 15-30px 误差，通常会显著下降。

真实工程里常见的做法也是这样：

- 截图仍然保留，让模型做语义判断
- 系统同时提供候选框、a11y 节点或 DOM 节点
- 模型选择对象 ID
- 执行层使用对象的中心点或热点区域完成点击

不过替代方案也有边界，不能说接了 a11y 就万事大吉。

| 场景 | 为什么替代方案会受限 |
|---|---|
| Canvas、视频画面、远程桌面 | 没有稳定结构化对象，只能回到像素级定位 |
| 动画频繁、布局实时变化 | 静态边界框容易过期 |
| 自定义控件或伪元素 | a11y / DOM 信息不完整 |
| 游戏界面、图像编辑器画布 | 目标本身不是标准 GUI 控件 |
| 多层遮挡或浮层切换很快 | 检测到的候选框未必对应当前可交互目标 |

所以工程上更现实的策略不是“彻底抛弃视觉坐标”，而是分层处理：

1. 能用结构化边界时，优先用边界。
2. 只能看图时，尽量提高分辨率并记录误差分布。
3. 对小目标、深色主题、密集界面做专项回归测试。
4. 把点击失败拆解成“选错目标”和“点偏目标”两类。
5. 对高风险步骤提供重试或确认机制，而不是盲点下一次。

这套分层思路的核心是：让模型负责它擅长的语义判断，让系统负责它更擅长的几何执行。

---

## 参考资料

下面的资料分成三类：基准与论文、设计规范、交互设计研究。前两类可以直接支撑本文的结论，第三类用于补充“小目标为什么更难点中”的人机交互背景。

| 来源 | 类型 | 主要发现 | 可复现数据类型 |
|---|---|---|---|
| [OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments](https://os-world.github.io/) | 基准 / 论文 | 多模态计算机使用代理在真实 GUI 环境中仍显著受限，GUI grounding 是关键瓶颈之一 | 任务成功率、环境配置、操作日志 |
| [OSWorld 论文条目（arXiv 2404.07972）](https://arxiv.gg/abs/2404.07972) | 论文 | 给出真实桌面和 Web 环境中的开放式任务评估框架 | 模型对比、任务级指标 |
| [OpenApps: Simulating Environment Variations to Measure UI-Agent Reliability](https://beta.hyper.ai/en/papers/2511.20766) | 论文 / 环境 | UI 主题、内容和布局变体会显著改变代理可靠性，任务成功率可随环境版本大幅波动 | 主题切换、任务通过率、loop 行为 |
| [OpenApps 项目页](https://facebookresearch.github.io/OpenApps/) | 项目 | 提供可配置外观与内容的应用环境，适合测量 UI 变体下的稳定性 | 环境变体、批量任务评测 |
| [W3C WCAG 2.2: Understanding SC 2.5.8 Target Size (Minimum)](https://www.w3.org/WAI/WCAG22/Understanding/target-size-minimum) | 规范 | 指针输入目标应至少达到 24×24 CSS 像素，或提供足够间距 | 目标最小尺寸、间距约束 |
| [W3C WCAG 2.2 Guidelines](https://w3c.github.io/wcag/guidelines/22/) | 规范 | 给出正式条文与适用例外 | 设计约束、合规边界 |
| [Nielsen Norman Group: Touch Target Size 等相关研究入口](https://www.nngroup.com/) | UX 研究 | 小目标和近距离目标会显著增加误触与导航成本 | 目标尺寸、交互错误率、可用性观察 |

把这些资料与本文的结论一一对应，大致可以这样理解：

| 本文结论 | 对应资料 |
|---|---|
| 纯截图代理的瓶颈之一是 GUI grounding | OSWorld |
| 环境主题与外观变化会影响稳定性 | OpenApps |
| 小目标天然更难稳定命中 | WCAG 2.5.8、NN/g 相关可用性研究 |
| 更大的目标与间距能提升容错 | WCAG 2.5.8 |

如果要进一步扩展这篇文章，最值得补的不是更多口号式结论，而是更细的点击级数据，例如：

- 不同目标尺寸分桶后的 `pass@1`
- 不同分辨率下的 `p50/p90 error`
- 深浅主题分别统计的误差分布
- “识别对但点偏”与“识别错目标”的占比

因为只有把失败拆开，才能知道后续应该优化模型、优化 UI，还是直接改执行链路。
