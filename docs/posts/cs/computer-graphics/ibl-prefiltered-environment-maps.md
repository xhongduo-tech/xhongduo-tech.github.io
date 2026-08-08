---
title: 基于图像的光照（IBL）与预滤波环境贴图
date: 2026-08-08
---

# 基于图像的光照（IBL）与预滤波环境贴图

<div class="epigraph">
<p>环境就是光源——把天空存进贴图，物体就「浸」在了光里。</p>
<footer>—— 图形学课堂常谈</footer>
</div>

<div class="article-byline">
<p>第三级 · 计算机图形学 ｜ GAMES101 & 虎书（Fundamentals of Computer Graphics）§20.5 ｜ 2026-08-08</p>
</div>

## 为什么环境贴图能当光源

《环境光贴图》一节我们见过：环境贴图让物体「反射出天空」。但它的威力不止于此——**环境贴图还能作为整个场景的光源**：天空、周围建筑、反光板发出的光，全部「从环境方向照进来」，为物体的漫反射与高光提供能量。这就是**基于图像的光照（Image-Based Lighting, IBL）**。<span class="marginnote">IBL 的思想：<strong>把「环境」当成一个巨大的光源——它从四面八方发光，照到物体的每个表面</strong>。渲染时不再用「几个点光源」，而是用一张环境贴图（天空/HDRI 全景）提供「全方向的光」。这比点光源真实得多——物体「浸在环境里」，天然获得环境色的反射与漫反射。</span>

IBL 的工程核心是**预滤波（prefiltering）**：环境贴图太大、方向太多，逐方向采样太贵——把「光的贡献」预计算成几张查找表，运行时一次查询。

## 1 漫反射 IBL：辐照度图

漫反射 IBL 要计算：**环境光对表面点 $p$（法线 $\vec{n}$）的漫反射贡献**——即「从法线半球来的所有环境光」的余弦加权平均：

$$
L_{\text{diffuse}}(p) = \int_{H^2} L_{\text{env}}(\omega_i)\, \cos\theta_i\, d\omega_i
$$

这个积分对每个法线方向 $\vec{n}$ 都要算一次——无法逐像素实时算。**预计算**：把「对每个法线方向的积分结果」存成一张**辐照度图（irradiance map）**（也叫漫反射卷积图）：

$$
E(\vec{n}) = \int_{H^2} L_{\text{env}}(\omega_i) \cos\theta_i\, d\omega_i
$$

运行时：用表面法线 $\vec{n}$ 查辐照度图，一次查询得到漫反射间接光。<span class="marginnote">「辐照度图 = 把环境光的『半球积分』预计算成按法线查询的表」：<strong>离线对每个法线方向，把环境贴图在法线半球上做余弦加权积分，结果存成一张很模糊的贴图——因为积分把环境光「抹平」了</strong>。运行时只需用法线查一次，就得到「该朝向的漫反射环境光」。这是「预计算换取运行时查询」的教科书案例。</span>

## 2 高光 IBL：预滤波环境贴图

高光 IBL 更复杂：高光的反射依赖**反射方向 + 粗糙度**——粗糙度越大，反射越「糊」（要采样一片方向的平均）：

$$
L_{\text{specular}}(p, \omega_o) = \int_{H^2} f_r(\omega_i, \omega_o)\, L_{\text{env}}(\omega_i)\, \cos\theta_i\, d\omega_i
$$

**预滤波（prefiltered）环境贴图**：把环境贴图按粗糙度模糊成 Mipmap 链——粗糙度 0 是原图（锐利反射），粗糙度 1 是极度模糊（粗糙反射）：

- 每一级 Mipmap = 「该粗糙度下的反射采样平均」。
- 运行时：按粗糙度选 Mipmap 层，沿反射方向 $\vec{r}$ 采样一次——「一次采样 = 该粗糙度的反射平均」。

加上 BRDF 的预计算查找表（split-sum 近似），高光 IBL 变成「两次查找 + 一次乘」——完全实时。<span class="marginnote">「预滤波环境贴图 = 把『按粗糙度模糊的反射』预计算成 Mipmap」：<strong>粗糙度 0 的镜面反射采原图、粗糙度 0.3 采第 2 级（稍糊）、粗糙度 1 采最糊层——反射的「模糊度」直接映射到 Mipmap 层级</strong>。这就是《各向异性过滤》里 Mipmap 思想在 IBL 的复用：预滤波 + 按需选层，一次采样拿到「一片方向的平均」。</span>

## 3 公式解析：split-sum 近似

高光 IBL 的积分含「BRDF × 环境光」两项，无法直接预计算（BRDF 依赖观察方向）。**split-sum 近似**把它拆成两个可预计算的积分：

$$
\int_{H^2} f_r L_{\text{env}} \cos\theta\, d\omega \approx \left(\int_{H^2} f_r \cos\theta\, d\omega\right) \left(\int_{H^2} L_{\text{env}} \cos\theta\, d\omega\right)
$$

- **第一步，分离**：把「BRDF 部分」与「环境光部分」拆成两个独立的积分——用「近似可分离」假设。
- **第二步，第一项（环境光）**：$\int L_{\text{env}} \cos\theta\, d\omega$ = 预滤波环境贴图（按粗糙度），沿反射方向采样。
- **第三步，第二项（BRDF）**：$\int f_r \cos\theta\, d\omega$ 只依赖粗糙度、法线、观察角——预计算成一张 **BRDF 查找表**（输入 roughness 与 $\cos\theta$，输出缩放与偏移）。
- **第四步，合并**：两次查找（预滤波贴图 + BRDF 表）相乘，得到高光 IBL。

**辨析｜易错点：** split-sum 是**近似**（不是精确分解）——它在「环境光变化平缓」时误差小、在「环境光剧烈变化」（小面积强光源）时误差大。另一个坑：BRDF 查找表的分辨率与精度要够，否则高光出现「带状」伪影。

## 4 IBL 的完整实现流程

现代 PBR 的 IBL 管线：

```text
// 预处理（加载时算一次）
irradiance_map    = convolve(env_map, diffuse_kernel);    // 漫反射：辐照度图
prefiltered_env   = prefilter(env_map, roughness_levels); // 高光：按粗糙度模糊成 Mipmap
brdf_lut          = integrate_brdf();                     // BRDF 查找表

// 运行时（每像素查四次）
vec3 n        = normalize(normal);
vec3 r        = reflect(-view_dir, n);
vec3 diffuse  = texture(irradiance_map, n).rgb;                        // 查 1
vec3 specular = textureLod(prefiltered_env, r, roughness * levels).rgb;// 查 2
vec2 brdf     = texture(brdf_lut, vec2(ndotv, roughness)).rg;          // 查 3
vec3 ibl      = diffuse * albedo + specular * (F0 * brdf.x + brdf.y);
```

「预处理做一次、运行时查四次」——IBL 把「环境光积分」从运行时彻底搬到预处理，是实时 PBR 的标配光照源。<span class="marginnote">「IBL 的『预处理 - 运行时』分工」：<strong>四张查找表（辐照度、预滤波、BRDF）在加载时算好，运行时每像素只做几次纹理采样——环境光的全部「积分成本」被预计算消化</strong>。这让 IBL 成为 PBR 材质「默认照亮方式」：任何物体放进环境，天然获得匹配环境的漫反射与高光——游戏场景、产品可视化、电影预览全靠它。</span>

## 5 IBL 与直接光照的配合

IBL 提供「环境光」（间接光），直接光照（点/方向光）提供「直接光」——完整的光照 = 两者相加：

$$
L_{\text{total}} = \underbrace{L_{\text{direct}}}_{\text{点/方向光}} + \underbrace{L_{\text{IBL-diffuse}} + L_{\text{IBL-specular}}}_{\text{环境光（IBL）}}
$$

- **直接光**：阴影映射 + 光源采样。
- **IBL**：环境的漫反射 + 高光。

IBL 让场景「无光也亮」——环境本身提供基础照明；直接光在环境之上叠加。两者配合，物体既被环境「浸透」，又被直接光照「照亮」。

**IBL 的局限**：环境光假设「环境在无穷远处」——室内小空间、物体近距离的颜色渗透（红墙映沙发）是 IBL 无法精确表达的（那是实时 GI 的领域，上一节）。IBL 是「无穷远环境的 GI」，实时 GI 是「场景内几何的 GI」——两者互补。<span class="marginnote">「IBL 管无穷远、实时 GI 管场景内」是两类间接光的分工：<strong>IBL 的环境来自「无限远的天空/全景」（查贴图），实时 GI（RSM/LPV/VXGI）的光来自「场景内的几何」（算传播）</strong>——真实渲染两者都要：天空光用 IBL，物体间的颜色渗透用 GI。现代引擎（UE Lumen）把它们融合：IBL 提供天空基础光，Lumen 补充场景内的间接反射。</span>

## 7 IBL 与实时渲染的关系：环境光的「默认值」

IBL 在实时渲染里的地位，值得一句话点透：**它是「环境光的默认值」**。

- 一个刚搭好的 PBR 场景，没有 IBL 时「黑得像宇宙深处」——加了 IBL，物体立刻「浸在光里」。
- IBL 是「懒人的 GI」：不追踪场景内的光线弹射，直接用「无穷远环境」的预计算光——「够用、便宜、实时」。
- 真正的场景内 GI（RSM/LPV/VXGI/Lumen）在 IBL 之上「补充」——IBL 管天空、GI 管室内——**「IBL 是基座，GI 是增强」**。

**为什么 IBL 是 PBR 的标配**：PBR 材质需要「环境光」来体现「金属反射环境、粗糙表面漫反射环境」——IBL 一次预计算 + 几次采样，把环境光「免费」给了 PBR——这是 PBR 材质系统「开箱即亮」的原因。

**辨析｜易错点：** 一个认知提醒——「IBL ≠ 全局光照」——IBL 假设环境在无穷远（没有场景几何的遮挡与弹射），真实室内环境（墙挡住天空、墙互相反射）IBL 表达不了——「室内还是靠 GI，IBL 只在室外/天空场景接近真实」。

## 8 小结

- **IBL = 环境贴图当光源**：环境光从四面八方照进物体，漫反射 + 高光都有。
- **漫反射 IBL**：辐照度图 = 对每个法线方向的余弦加权积分，按法线查一次。
- **高光 IBL**：预滤波环境贴图（按粗糙度模糊成 Mipmap）+ BRDF 查找表。
- **split-sum 近似**把高光积分拆成「环境光项 × BRDF 项」两个可预计算的积分。
- 完整流程：四张查找表预处理一次，运行时每像素查几次——环境光成本被预消化。
- IBL 管无穷远环境、实时 GI 管场景内几何，两者互补。

在下一节，我们进入第十一篇现代专题——从**GPU 渲染管线深入：可编程着色器与 GLSL/HLSL**开始。
