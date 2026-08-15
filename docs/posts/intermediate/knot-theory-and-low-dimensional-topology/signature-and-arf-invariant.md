
---

title: 结的签名与 Arf 不变量

date: 2026-08-07

---



# 结的签名与 Arf 不变量



<div class="epigraph">

<p>一个结的签名，是它藏在水面下的代数身份的指纹。</p>

<footer>—— 本文作者按</footer>

</div>



<div class="article-byline">

<p>第二级 · 纽结理论与低维拓扑 ｜ Lickorish《An Introduction to Knot Theory》第8–9、17章 ｜ 2026-08-07</p>

</div>



## 为什么从「整数不变量」开始



多项式不变量信息量大，但计算繁琐。签名与 Arf 不变量是**整数**（或模 2 整数）不变量——计算简洁、性质漂亮，而且天生是**共轭不变量（concordance invariant）**：它们在「四维里的光滑等价」下保持不变。这让它们成为研究「结能否在四维里解开」这一深层问题的关键工具。



**签名（signature）** $\sigma(K)$ 由 Trotter（1962）与 Murasugi 引入，从 Seifert 矩阵的对称化取符号差；**Arf 不变量** $\operatorname{Arf}(K)$ 来自 Seifert 曲面上的二次型，取值 $\mathbb{Z}/2$，其值可由 Alexander 多项式在 $t = -1$ 处的取值读出。<span class="marginnote">「Arf」得名于土耳其数学家 C. Arf，他在 1941 年研究代数曲线的二次型时定义了这族不变量。结的 Arf 不变量是「二次型不变量」在低维拓扑里的化身——一条从代数几何通向结理论的暗线。</span>



## 1 签名：Seifert 矩阵的对称化



**签名（signature）**：设 $V$ 是结 $K$ 的 Seifert 矩阵（见第2篇之一），则



$$

\sigma(K) = \operatorname{sgn}\big(V + V^{\mathsf{T}}\big),

$$



即对称矩阵 $V + V^{\mathsf{T}}$ 的**符号差**——正特征值个数减去负特征值个数。



- $V + V^{\mathsf{T}}$ 是实对称矩阵，特征值全为实数，符号差定义良好。

- $\sigma(K)$ 与 Seifert 曲面、基的选取无关——是不变量。



**例**：三叶结 $3_1$ 的 Seifert 矩阵 $V = \begin{pmatrix} -1 & 1 \\ 0 & -1 \end{pmatrix}$，则 $V + V^{\mathsf{T}} = \begin{pmatrix} -2 & 1 \\ 1 & -2 \end{pmatrix}$，特征值为 $-1, -3$，符号差 $\sigma(3_1) = -2$。八字结 $4_1$ 的 $\sigma = 0$。



**辨析｜签名 vs 亏格**：签名是「整数」，亏格也是「整数」，但来源完全不同：亏格是曲面的「洞数」（几何），签名是矩阵的「正负特征值差」（代数）。它们各自捕捉结的不同侧面——签名对镜像敏感（$\sigma(K^*) = -\sigma(K)$），亏格对镜像不敏感。



## 2 公式解析：符号差怎么算



对 $n \times n$ 对称实矩阵 $M$，符号差 $\operatorname{sgn} M$ = （正特征值数）−（负特征值数）。



$$

\sigma(K) = \operatorname{sgn}(V + V^{\mathsf{T}}) = \#\{\lambda_i > 0\} - \#\{\lambda_i \lt  0\}.

$$



- **第一步，为什么要对称化**：Seifert 矩阵 $V$ 本身不对称（$V_{ij} = \operatorname{lk}(a_i, a_j^+) \neq V_{ji}$），不能直接谈特征值符号。$V + V^{\mathsf{T}}$ 对称化，把「前向缠绕」与「后向缠绕」合并。

- **第二步，特征值符号的含义**：$V + V^{\mathsf{T}}$ 正定（全正特征值）意味着「缠绕往一个方向偏」，负定意味着「偏反方向」。签名就是「偏正的程度减偏负的程度」。

- **第三步，镜像翻转**：镜像结的 Seifert 矩阵变号（$V \to -V$），于是 $V + V^{\mathsf{T}} \to -(V+V^{\mathsf{T}})$，特征值全变号，$\sigma \to -\sigma$。所以**签名是手性的直接探测器**：$\sigma \neq 0$ 的结必为手性结。



## 3 签名的基本性质



- **镜像**：$\sigma(K^*) = -\sigma(K)$。

- **连通和**：$\sigma(K_1 \# K_2) = \sigma(K_1) + \sigma(K_2)$。

- **定向**：翻转整个定向，$\sigma$ 不变；对链环翻转一个分量则 $\sigma$ 可能改变。

- **共轭不变量**：若 $K$ 是光滑共轭（slice）的，则 $\sigma(K) = 0$——签名是「四维可解性」的必要条件。

- **判别手性**：$\sigma(3_1) = -2 \neq 0$，三叶结手性；$\sigma(4_1) = 0$，八字结两性（符合已知）。<span class="marginnote">签名与 Jones 多项式判手性互为补充：Jones 对某些结失效（非手性结但多项式不对称），签名同样不是万能的。但两者的「失效集合」不同，合起来覆盖面更广。经典结表里「手性判定」同时列 Jones 多项式与签名。</span>



## 4 Arf 不变量：模 2 的二次型



**Arf 不变量（Arf invariant）**：设 $F$ 是结 $K$ 的 Seifert 曲面。$H_1(F; \mathbb{Z}/2)$ 上有一个二次型 $q$（对每个模 2 同调类，取「沿曲线的自缠绕数 mod 2」）。对非退化二次型，**Arf 不变量** $\operatorname{Arf}(K) \in \mathbb{Z}/2$ 定义为该二次型的 Arf 不变式。



对结而言，Arf 不变量可以**免去二次型直接算**——Robertello（1965）给出惊人简洁的公式：



$$

\operatorname{Arf}(K) = \begin{cases} 0, & \Delta_K(-1) \equiv \pm 1 \pmod 8, \\ 1, & \Delta_K(-1) \equiv \pm 3 \pmod 8. \end{cases}

$$



即只需把 Alexander 多项式代入 $t = -1$，看模 8 的剩余。



- 三叶结 $\Delta_{3_1}(t) = t^2 - t + 1$，$\Delta(-1) = 3$，$\pm 3 \pmod 8$，故 $\operatorname{Arf}(3_1) = 1$。

- 八字结 $\Delta_{4_1}(t) = t^2 - 3t + 1$，$\Delta(-1) = 5$，$5 \equiv -3 \pmod 8$，故 $\operatorname{Arf}(4_1) = 1$。

- 平凡结 $\Delta = 1$，$\Delta(-1) = 1$，$\operatorname{Arf} = 0$。



**易错点｜Arf 的模 8 陷阱**：公式里是「$\Delta_K(-1)$ 的取值模 8」，不是「多项式本身模 8」——必须先把 $t = -1$ 代入得到整数，再看该整数模 8 是 $\pm 1$ 还是 $\pm 3$。而且 $\Delta(-1)$ 必为奇数（因为 $\Delta_K(t)$ 的系数与 $t=1$ 处的约束），所以模 8 分类总是合法。



## 5 Arf 不变量的性质与用途



- **镜像与定向**：$\operatorname{Arf}(K)$ 对镜像、定向都不变（它是 $\mathbb{Z}/2$ 值且不关心手性）。

- **连通和**：$\operatorname{Arf}(K_1 \# K_2) = \operatorname{Arf}(K_1) + \operatorname{Arf}(K_2) \pmod 2$。

- **共轭不变量**：slice 结的 Arf 不变量为 0。

- **奇偶区分**：Arf 不变量只取 0 或 1，是最粗糙的「二分类」，却常是决定性的——例如「三叶结不是 slice」可由 $\sigma \neq 0$ 或 $\operatorname{Arf} = 1$ 双双判定。<span class="marginnote">Arf 不变量的「切片/非切片」判定力：一个结若 $\operatorname{Arf} = 1$ 则绝不 slice；反之 $\operatorname{Arf} = 0$ 也不保证 slice。它与签名构成「slice 障碍」的两道闸门。四维拓扑里的 slice 问题是低维拓扑最活跃的方向之一，这里看到的是它的第一道算术约束。</span>



**辨析｜签名与 Arf 的分工**：签名是整数（能分辨镜像、反映「缠绕的净方向」），Arf 是模 2（不分辨镜像、捕捉「二次型是否可拆解」）。两者都来自 Seifert 曲面的代数结构，但回答不同问题：签名问「缠绕偏哪个方向」，Arf 问「缠绕能否被抵消」。



### 签名与四维拓扑：slice 障碍



签名是「共轭不变量」的意义在四维：一个结若能「在四维球里铺成一张光滑圆盘」（slice 结），它的签名必须为 0。**为什么？**



- 若 $K$ slice，则存在四维中的光滑圆盘 $D^2$ 以 $K$ 为边界。

- 取 $K$ 的一个 Seifert 曲面与这个圆盘「配对」，用四维拓扑的相交形式（intersection form）论证——「圆盘存在」会强迫 Seifert 矩阵的对称化 $V + V^{\mathsf{T}}$ 与某个「双曲形式」同构，其符号差为 0。

- 所以 $\sigma(K) \neq 0 \Rightarrow K$ 不是 slice——签名是「四维可解性」的第一道闸门。



**例**：三叶结 $\sigma = -2 \neq 0$，所以三叶结**不 slice**——它无法在四维里被光滑地解开。这是「签名的四维应用」最常被引用的结论。



### Arf 不变量与「可切性」的更精细判定



Arf 不变量补上签名漏掉的「模 2」信息：



- $\operatorname{Arf}(K) = 1 \Rightarrow K$ 不 slice（签名可能为 0 但 Arf 挡路）。

- 签名与 Arf 合起来给出「slice 障碍」的两个独立判据——「$\sigma = 0$ 且 $\operatorname{Arf} = 0$」是 slice 的必要条件，但不是充分条件。



**签名族（signature obstructions）**：现代研究把单个签名推广为「签名函数」$\sigma_\omega(K)$（$\omega$ 在单位圆上变化），给出无穷多 slice 障碍——签名从「一个整数」升级为「一族整数」。这是四维拓扑里「量子化」签名的标准手法。



### 签名与手性的再确认



签名判手性不是孤例，把它与其他判据对照：



| 结 | 签名 $\sigma$ | Arf | 手性 |

| --- | --- | --- | --- |

| 三叶结 $3_1$ | $-2$ | 1 | 手性 |

| 八字结 $4_1$ | 0 | 1 | 两性 |

| 平凡结 | 0 | 0 | 两性 |



- 八字结 $\sigma = 0$ 但 Arf = 1——「签名看不出、Arf 看得出」。

- 两条整数不变量互补：一个看「缠绕净方向」，一个看「缠绕可拆性」。



## 6 小结



- **签名** $\sigma(K) = \operatorname{sgn}(V + V^{\mathsf{T}})$：Seifert 矩阵对称化的符号差，整数不变量。

- 签名对镜像取反（$\sigma(K^*) = -\sigma(K)$），对连通和相加；$\sigma \neq 0$ 判手性。

- **Arf 不变量**取 $\mathbb{Z}/2$，可由 $\Delta_K(-1)$ 模 8 读出：$\pm 1$ → 0，$\pm 3$ → 1。

- 三叶结 $\operatorname{Arf} = 1$、八字结 $\operatorname{Arf} = 1$、平凡结 $\operatorname{Arf} = 0$。

- 两者都是**共轭不变量**：slice 结必有 $\sigma = 0$ 且 $\operatorname{Arf} = 0$——它们连接着四维理论（slice 结的判据）。
- 签名与 Arf 是「整数 / 模 2」两种粒度的代数印记：签名判手性与 slice，Arf 判四维切片性。

在下一节，我们进入第3篇——**Tangles 与 skein 理论**：把结切碎成带端点的积木，把「构造结」变成「代数的运算」。