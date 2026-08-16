// 待建专题详细主题 · 依据权威书籍章节
// 共 669 个待建专题（2026-08 第一轮 42 个、第二轮 233 个、定向增补 5+11+20+21+3+5 个），约 7998 个章节级子主题
// 每个专题包含: title(中文标题) + books(权威书籍) + chapters(章节级子主题)

export const treeDetails = {
  'foundations/elementary-number-theory': {
    title: "初等数论",
    books: [
          "G.H. Hardy and E.M. Wright, \"An Introduction to the Theory of Numbers\" (6th ed., 2008)",
          "Kenneth H. Rosen, \"Elementary Number Theory and Its Applications\" (6th ed., 2011)",
          "Ivan Niven, H.S. Zuckerman, H.L. Montgomery, \"An Introduction to the Theory of Numbers\" (5th ed., 1991)"
    ],
    chapters: [
      "整除性与欧几里得算法 (Hardy & Wright §1.1-1.7)",
      "素数与算术基本定理 (Hardy & Wright §1.3-2.1)",
      "素数分布与素数定理 (Hardy & Wright §1-2, 22；Rosen 素数分布章)",
      "同余、剩余类与中国剩余定理 (Niven-Zuckerman §2.3-2.6)",
      "费马小定理、欧拉定理与 Wilson 定理 (Hardy & Wright §6.1-6.5)",
      "二次剩余与 Gauss 互反律 (Hardy & Wright §9)",
      "原根与指标 (Niven-Zuckerman §4.2-4.4)",
      "连分数与最佳逼近 (Hardy & Wright §10.1-10.9)",
      "丢番图方程：线性丢番图、勾股三元组、四平方和与 Pell 方程 (Hardy & Wright §15, 18；Niven-Zuckerman §5)",
      "数论函数与 Möbius 反演 (Hardy & Wright §13)",
      "数论在密码学中的应用 (Rosen §8-9)"
    ],
  },
  'foundations/vectors-tensors': {
    title: "向量与张量初步",
    books: [
          "George B. Arfken, H.J. Weber, F.E. Harris, \"Mathematical Methods for Physicists\" (7th ed., 2013)",
          "Mary L. Boas, \"Mathematical Methods in the Physical Sciences\" (3rd ed., 2005)"
    ],
    chapters: [
      "向量代数：点积、叉积与多重积 (Arfken §1.1-1.7)",
      "向量分析：梯度、散度与旋度 (Arfken §1.6-1.9)",
      "线积分、面积分与体积分 (Boas §6.3-6.5)",
      "积分定理：Gauss 散度定理与 Stokes 旋度定理 (Arfken Ch 1-2 末节；Boas §6.5-6.6)",
      "正交曲线坐标与度量系数 (Arfken §2.1-2.5)",
      "张量定义与指标记号 (Arfken §2.6-2.7)",
      "协变、逆变张量与商律 (Arfken §2.6-2.8)",
      "赝张量与 Levi-Civita 记号、反对称张量 (Arfken Ch 2 张量章节)",
      "二阶张量的本征值与本征轴 (Arfken §3.5)",
      "张量微积分：协变导数与张量的散度/旋度 (Arfken Ch 2 末节)",
      "张量在连续介质力学中的应用 (Boas §10.4)"
    ],
  },
  'intermediate/matrix-theory': {
    title: "矩阵论",
    books: [
          "Roger A. Horn and Charles R. Johnson, \"Matrix Analysis\" (2nd ed., 2013)",
          "Peter Lancaster and Miron Tismenetsky, \"The Theory of Matrices\" (2nd ed., 1985)"
    ],
    chapters: [
      "矩阵的基本运算与分块 (Horn & Johnson Ch. 0-1)",
      "特征值、特征向量与谱定理 (Horn & Johnson Ch. 1-2)",
      "酉等价与正规矩阵 (Horn & Johnson Ch. 2)",
      "Schur 三角化与奇异值分解 (Horn & Johnson Ch. 2-3)",
      "Hermite 矩阵与复对称矩阵 (Horn & Johnson Ch. 4)",
      "矩阵范数 (Horn & Johnson Ch. 5)",
      "特征值定位与扰动理论：Gershgorin 圆盘定理、Weyl 不等式与 Hoffman-Wielandt 定理 (Horn & Johnson Ch. 6)",
      "正定矩阵与平方根 (Horn & Johnson Ch. 7)",
      "非负矩阵与 Perron-Frobenius 理论 (Horn & Johnson Ch. 8)",
      "Jordan 标准型与广义特征向量 (Horn §3.1-3.2)",
      "Cayley-Hamilton 定理与最小多项式 (Horn §3.3)",
      "矩阵函数：矩阵指数与对数 (Lancaster & Tismenetsky Ch. 9)",
      "Kronecker 积与张量积 (Horn §4.3)",
      "线性矩阵方程（Sylvester/Lyapunov） (Horn §4.4)",
      "广义逆与 Moore-Penrose 伪逆 (Lancaster & Tismenetsky Ch. 12)"
    ],
  },
  'intermediate/algebraic-geometry': {
    title: "代数几何",
    books: [
          "Robin Hartshorne, \"Algebraic Geometry\" (GTM 52, 1977)",
          "Qing Liu, \"Algebraic Geometry and Arithmetic Curves\" (2002)"
    ],
    chapters: [
      "仿射簇与代数集 (Hartshorne Ch. I §1-2)",
      "射影簇与正则函数 (Hartshorne Ch. I §2-3)",
      "态射与有理映射 (Hartshorne Ch. I §4)",
      "光滑性与正规簇、切空间与维数 (Hartshorne Ch. I §5)",
      "层与概形 (Hartshorne Ch. II §1-2)",
      "概形的态射与纤维积 (Hartshorne Ch. II §3)",
      "分离态射与真态射 (Hartshorne Ch. II §4)",
      "模层与凝聚层 (Hartshorne Ch. II §5)",
      "除子、线性系与微分形式 (Hartshorne Ch. II §6-8)",
      "层上同调与 Čech 上同调 (Hartshorne Ch. III §2-4)",
      "Serre 对偶定理 (Hartshorne Ch. III §7)",
      "Riemann-Roch 定理 (Hartshorne Ch. IV §1)",
      "Riemann-Hurwitz 公式与曲线论深化：曲线上的映射、亏格与椭圆曲线分类 (Hartshorne Ch. IV §2-3)",
      "双有理几何（blow-up/有理映射） (Hartshorne Ch. V)",
      "曲面的相交理论与曲面上的 Riemann-Roch (Hartshorne Ch. V §1-2)",
      "GAGA 原理（选读） (Hartshorne Ch. III §5)"
    ],
  },
  'intermediate/lie-algebra': {
    title: "李代数与李群",
    books: [
          "James E. Humphreys, \"Introduction to Lie Algebras and Representation Theory\" (GTM 9, 1972)",
          "William Fulton and Joe Harris, \"Representation Theory: A First Course\" (GTM 129, 1991)",
          "Brian C. Hall, \"Lie Groups, Lie Algebras, and Representations\" (2nd ed., 2015)"
    ],
    chapters: [
      "李代数的定义与基本性质 (Humphreys §1-2)",
      "可解与幂零李代数 (Humphreys §3)",
      "Engel 定理与 Lie 定理 (Humphreys §4)",
      "半单李代数与 Killing 型 (Humphreys §5)",
      "表示的完全可约性：Weyl 定理 (Humphreys §6)",
      "sl(2,C) 的表示论 (Humphreys §7)",
      "Cartan 子代数与根空间分解 (Humphreys §8)",
      "根系公理与 Weyl 群 (Humphreys §9)",
      "根系分类与 Dynkin 图 (Humphreys §10-12)",
      "万有包络代数与 PBW 定理 (Humphreys §13)",
      "最高权表示与 Verma 模 (Humphreys §20-21)",
      "Weyl 特征标公式 (Humphreys §22-24)",
      "SU(2) 与 SU(3) 的表示 (Fulton & Harris Part I-II)",
      "李群-李代数对应与指数映射 (Hall Ch. 5-9)"
    ],
  },
  'intermediate/harmonic-analysis': {
    title: "调和分析",
    books: [
          "Elias M. Stein and Rami Shakarchi, \"Fourier Analysis: An Introduction\" (Princeton, 2003)",
          "Elias M. Stein and Guido Weiss, \"Introduction to Fourier Analysis on Euclidean Spaces\" (PMS 32, 1971)",
          "Walter Rudin, \"Fourier Analysis on Groups\" (1962)"
    ],
    chapters: [
      "Fourier 级数与收敛性 (Stein & Shakarchi Book I Ch. 1-2)",
      "极大函数与 Hardy-Littlewood 定理 (Stein & Shakarchi Book III Ch. 3)",
      "Calderón-Zygmund 分解 (Stein & Shakarchi Book IV Ch. 2)",
      "Marcinkiewicz 插值定理 (Stein & Shakarchi Book III Ch. 5)",
      "Fourier 变换与反演公式 (Stein & Shakarchi Book I Ch. 5-6)",
      "Poisson 求和公式与采样定理 (Stein & Shakarchi Book I Ch. 5)",
      "Paley-Wiener 定理 (Stein & Shakarchi Book I Ch. 6)",
      "局部紧群上的 Haar 测度与调和分析 (Rudin Ch. 1)",
      "Hilbert 变换与奇异积分算子 (Stein & Weiss Ch. VI)",
      "Riesz 变换、乘子理论与分数次积分 (Stein & Weiss Ch. V-VI)",
      "Pontryagin 对偶与局部紧 Abel 群上的 Fourier 分析 (Rudin Ch. 2-4)"
    ],
  },
  'intermediate/homological-algebra': {
    title: "同调代数",
    books: [
          "Charles A. Weibel, \"An Introduction to Homological Algebra\" (Cambridge, 1994)",
          "Henri Cartan and Samuel Eilenberg, \"Homological Algebra\" (1956)"
    ],
    chapters: [
      "复形与同调群 (Weibel Ch. 1)",
      "蛇引理与连接同态 (Weibel Ch. 1.3)",
      "导出函子 (Weibel Ch. 2)",
      "Ext 与 Tor (Weibel Ch. 3)",
      "同调维数与整体维数：射影/内射维数与 Hilbert 合冲定理 (Weibel Ch. 4)",
      "谱序列 (Weibel Ch. 5)",
      "双复形与 Künneth 公式 (Weibel Ch. 2.7, 5.6)",
      "群同调与 Lie 代数同调 (Weibel Ch. 6-7)",
      "单纯方法：Simplicial Methods 与 Dold-Kan 等价 (Weibel Ch. 8)",
      "Hochschild 与循环同调（选读） (Weibel Ch. 9)",
      "导出范畴简介（选读） (Weibel 附录)"
    ],
  },
  'intermediate/commutative-algebra': {
    title: "交换代数",
    books: [
          "Michael F. Atiyah and Ian G. MacDonald, \"Introduction to Commutative Algebra\" (1969)",
          "Hideyuki Matsumura, \"Commutative Ring Theory\" (Cambridge, 1989)",
          "David Eisenbud, \"Commutative Algebra with a View Toward Algebraic Geometry\" (GTM 150, 1995)"
    ],
    chapters: [
      "理想、模与环同态 (Atiyah-Macdonald Ch. 1-2)",
      "局部化与分数理想 (Atiyah-Macdonald Ch. 3, 9)",
      "准素分解 (Atiyah-Macdonald Ch. 4)",
      "链条件与 Noether 环、Artin 环 (Atiyah-Macdonald Ch. 6-8)",
      "Hilbert 零点定理与 Zariski 拓扑 (Atiyah-Macdonald Ch. 7)",
      "离散赋值环与 Dedekind 整环 (Atiyah-Macdonald Ch. 9)",
      "完备化与 Hensel 引理 (Atiyah-Macdonald Ch. 10)",
      "维数理论：Krull 维数、高度、Hilbert 函数与 Hilbert-Samuel 多项式 (Atiyah-Macdonald Ch. 11)",
      "维数理论深化与正则局部环 (Matsumura Ch. 5-6)",
      "Koszul 复形与正则环 (Matsumura Ch. 6 / Eisenbud Ch. 17)",
      "张量积与平坦模 (Atiyah-Macdonald Ch. 2-3)",
      "整扩张（Going-up/Going-down） (Atiyah-Macdonald Ch. 5)",
      "深度与正则序列 (Matsumura Ch. 6 / Eisenbud Ch. 17-18)",
      "Cohen-Macaulay 模与 Gorenstein 环 (Matsumura Ch. 6 / Eisenbud Ch. 18)",
      "相伴素与支集 (Atiyah-Macdonald Ch. 4)",
      "局部上同调（选读） (Matsumura 补充章 / Eisenbud Ch. 18)"
    ],
  },
  'intermediate/riemannian-geometry': {
    title: "黎曼几何",
    books: [
          "Manfredo P. do Carmo, \"Riemannian Geometry\" (Birkhäuser, 1992)",
          "John M. Lee, \"Riemannian Manifolds: An Introduction to Curvature\" (GTM 176, 1997)",
          "Peter Petersen, \"Riemannian Geometry\" (3rd ed., 2016)"
    ],
    chapters: [
      "黎曼度量与 Levi-Civita 联络 (do Carmo Ch. 1-2)",
      "协变导数与平行移动 (do Carmo Ch. 2)",
      "测地线与指数映射 (do Carmo Ch. 3)",
      "曲率：截面、Ricci 与标量曲率 (do Carmo Ch. 4)",
      "Jacobi 场与共轭点 (do Carmo Ch. 5)",
      "等距浸入与第二基本形式、Gauss-Codazzi 方程 (do Carmo Ch. 6)",
      "完备性与 Hopf-Rinow 定理 (do Carmo Ch. 7)",
      "Cartan-Hadamard 定理 (do Carmo Ch. 7)",
      "常曲率空间与空间形式：双曲空间 (do Carmo Ch. 8)",
      "变分公式与 Bonnet-Myers 定理 (do Carmo Ch. 9)",
      "Rauch 比较定理与体积比较：Bishop-Gromov (do Carmo Ch. 10; Petersen Ch. 6-7)",
      "Morse 指标定理（选读） (do Carmo Ch. 11)"
    ],
  },
  'intermediate/calculus-of-variations': {
    title: "变分法",
    books: [
          "Bruce van Brunt, \"The Calculus of Variations\" (Universitext, 2004)",
          "Lawrence C. Evans, \"Partial Differential Equations\" (2nd ed., 2010, Ch. 8)",
          "Ivar Ekeland and Roger Témam, \"Convex Analysis and Variational Problems\" (SIAM, 1999)"
    ],
    chapters: [
      "Euler-Lagrange 方程推导 (van Brunt Ch. 2)",
      "等周问题与约束变分 (van Brunt Ch. 4)",
      "Hamilton 原理与最小作用量原理 (van Brunt Ch. 3)",
      "Noether 对称性定理 (van Brunt Ch. 7)",
      "Weierstrass 过分函数与角条件 (van Brunt Ch. 6)",
      "二阶变分、Legendre 条件与 Jacobi 条件：共轭点与充分条件 (van Brunt Ch. 5)",
      "Hilbert 积分不变与场论方法：Weierstrass 场与 Mayer 场 (van Brunt Ch. 8)",
      "直接方法与下半连续性 (Evans Ch. 8.2)",
      "Hamilton-Jacobi 方程 (Evans Ch. 10)",
      "极小曲面问题与平均曲率 (Evans Ch. 8.5)",
      "凸分析与变分问题的对偶理论：Legendre-Fenchel 变换与 Fenchel 对偶 (Ekeland-Témam Ch. III-IV)"
    ],
  },
  'intermediate/elliptic-curves': {
    title: "椭圆曲线与模形式",
    books: [
          "Joseph H. Silverman, \"The Arithmetic of Elliptic Curves\" (2nd ed., GTM 106, 2009)",
          "Neil Koblitz, \"Introduction to Elliptic Curves and Modular Forms\" (2nd ed., 1993)",
          "Jean-Pierre Serre, \"A Course in Arithmetic\" (1973)"
    ],
    chapters: [
      "代数曲线基础：除子与曲线上的 Riemann-Roch (Silverman Ch. II §1-6)",
      "Weierstrass 方程与椭圆曲线群律 (Silverman Ch. III)",
      "椭圆曲线上的几何加法 (Silverman Ch. III §3)",
      "形式群与形式对数 (Silverman Ch. IV)",
      "有限域上的椭圆曲线 (Silverman Ch. V)",
      "局部域上的椭圆曲线与约化理论：good/multiplicative/additive reduction、Tate 曲线与 Tamagawa 数 (Silverman Ch. VII)",
      "挠点与 Mordell-Weil 定理 (Silverman Ch. VIII)",
      "整数点与 Siegel 定理 (Silverman Ch. IX)",
      "复乘法与模函数 (Silverman Ch. VI)",
      "模形式与 Eisenstein 级数 (Serre Ch. VII)",
      "j-不变量与模曲线 (Serre Ch. VII-VIII)",
      "Hecke 算子与 L-函数 (Koblitz Ch. III)",
      "模形式深化：Hecke 算子新形式理论、Atkin-Lehner 与模曲线的 Galois 表示 (Koblitz Ch. III 延伸 / Diamond & Shurman Ch. 5)",
      "同源与对偶同源 (Silverman Ch. III §4)",
      "椭圆曲线上的复结构（格与一致化） (Silverman Ch. VI)",
      "BSD 猜想 (Silverman Ch. X §6)",
      "模性定理（Taniyama-Shimura-Weil） (Silverman Ch. C §16)"
    ],
  },
  'intermediate/analytic-number-theory': {
    title: "解析数论",
    books: [
          "Tom M. Apostol, \"Introduction to Analytic Number Theory\" (1976)",
          "Hugh L. Montgomery and Robert C. Vaughan, \"Multiplicative Number Theory I\" (2007)",
          "Gerald Tenenbaum, \"Introduction to Analytic and Probabilistic Number Theory\" (3rd ed., 2015)"
    ],
    chapters: [
      "Dirichlet 级数与欧拉乘积 (Apostol Ch. 11-13)",
      "黎曼 Zeta 函数 (Apostol Ch. 12-13)",
      "解析延拓与函数方程 (Apostol Ch. 12)",
      "素数定理 (Apostol Ch. 13)",
      "Dirichlet 特征与等差数列中的素数 (Apostol Ch. 6-7)",
      "Gauss 和与特征和、二次互反律的解析侧 (Apostol Ch. 9-10)",
      "零区域与 Deuring-Heilbronn 现象 (Montgomery & Vaughan Ch. 8-9)",
      "Riemann 假设与零点分布 (Montgomery & Vaughan Ch. 13-14)",
      "特征函数与均值估计 (Tenenbaum Ch. II)",
      "均值定理深化：Selberg-Delange 方法与算术函数均值 (Montgomery & Vaughan Ch. 13-14; Tenenbaum Ch. II)",
      "大筛法与 Bombieri-Vinogradov 定理 (Montgomery & Vaughan Ch. 27)",
      "加法数论：Waring 问题与 Hardy-Littlewood 圆法 (Tenenbaum Ch. III / Vaughan《The Hardy-Littlewood Method》)"
    ],
  },
  'intermediate/algebraic-number-theory': {
    title: "代数数论",
    books: [
          "Jürgen Neukirch, \"Algebraic Number Theory\" (Springer, 1999)",
          "Kenneth Ireland and Michael Rosen, \"A Classical Introduction to Modern Number Theory\" (2nd ed., 1990)",
          "Serge Lang, \"Algebraic Number Theory\" (2nd ed., 1994)"
    ],
    chapters: [
      "代数整数与数环 (Neukirch Ch. I §1-3)",
      "Dedekind 整环与理想唯一分解 (Neukirch Ch. I §2-3)",
      "类数与理想类群 (Neukirch Ch. I §6)",
      "Dirichlet 单位定理 (Neukirch Ch. I §7)",
      "赋值、完备化与 p-adic 数 (Neukirch Ch. II §1-5)",
      "分歧理论深入：惯性群、分歧群与高阶分歧 (Neukirch Ch. II §7-10)",
      "Minkowski 几何与类数有限性定理 (Neukirch Ch. I §5-6)",
      "差积与判别式 (Neukirch Ch. III §2)",
      "局部类域论 (Neukirch Ch. V)",
      "Artin 互反律 (Neukirch Ch. VI)",
      "Chebotarev 密度定理 (Neukirch Ch. VI §3 / VII)",
      "Dedekind Zeta 函数与类数公式 (Neukirch Ch. VII)",
      "Gauss 和与 Jacobi 和 (Ireland & Rosen Ch. 8-9)"
    ],
  },
  'intermediate/dynamical-systems': {
    title: "动力系统",
    books: [
          "Morris W. Hirsch, Stephen Smale, Robert L. Devaney, \"Differential Equations, Dynamical Systems, and an Introduction to Chaos\" (2nd ed., 2004)",
          "John Guckenheimer and Philip Holmes, \"Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields\" (1983)",
          "Lawrence Perko, \"Differential Equations and Dynamical Systems\" (3rd ed., 2001)"
    ],
    chapters: [
      "一维线性与非线性流 (Hirsch-Smale-Devaney Ch. 1-2)",
      "不动点与稳定性分析 (Hirsch-Smale-Devaney Ch. 2-3)",
      "相平面与保守系统 (Perko Ch. 2-3)",
      "分岔理论：鞍结与 Hopf 分岔 (Guckenheimer & Holmes Ch. 3)",
      "Lorenz 方程与混沌 (Hirsch-Smale-Devaney Ch. 14)",
      "Poincaré 映射与周期轨道 (Guckenheimer & Holmes Ch. 1-2)",
      "Hamilton 系统与 KAM 简介 (Guckenheimer & Holmes Ch. 7)",
      "奇怪吸引子与 Lyapunov 指数 (Hirsch-Smale-Devaney Ch. 15)",
      "离散动力系统与迭代映射：logistic 映射、周期倍化与混沌 (Hirsch-Smale-Devaney Ch. 15)",
      "全局分岔：同宿/异宿轨道与 Shilnikov 混沌 (Guckenheimer & Holmes Ch. 6, Hirsch-Smale-Devaney Ch. 16)"
    ],
  },
  'intermediate/fractal-geometry': {
    title: "分形几何",
    books: [
          "Kenneth Falconer, \"Fractal Geometry: Mathematical Foundations and Applications\" (3rd ed., 2014)",
          "Gerald A. Edgar, \"Measure, Topology, and Fractal Geometry\" (2nd ed., 2008)",
          "Michael F. Barnsley, \"Fractals Everywhere\" (3rd ed., 2012)"
    ],
    chapters: [
      "Hausdorff 测度与维数 (Falconer Ch. 2-3)",
      "Box-counting 维数与 Packing 维数 (Falconer Ch. 3)",
      "自相似集与迭代函数系统 (Falconer Ch. 9, Barnsley Ch. 3-4)",
      "分形的投影与乘积 (Falconer Ch. 7-8)",
      "Julia 集与 Mandelbrot 集 (Falconer Ch. 14)",
      "Brown 运动的分形性质 (Falconer Ch. 16)",
      "函数图象的分形维数 (Falconer Ch. 11)",
      "分形在物理中的应用 (Falconer Ch. 18)",
      "分形维数的计算技巧：交叠条件与开集条件 (Falconer Ch. 4)",
      "自仿集与自共形集 (Falconer Ch. 10)",
      "随机分形 (Falconer Ch. 15)",
      "分形测度与多重分形 (Falconer Ch. 17)"
    ],
  },
  'intermediate/fuzzy-mathematics': {
    title: "模糊数学",
    books: [
          "Hans-Jürgen Zimmermann, \"Fuzzy Set Theory—and Its Applications\" (4th ed., 2001)",
          "George J. Klir and Bo Yuan, \"Fuzzy Sets and Fuzzy Logic: Theory and Applications\" (1995)",
          "Hung T. Nguyen and Elbert A. Walker, \"A First Course in Fuzzy Logic\" (3rd ed., 2006)"
    ],
    chapters: [
      "模糊集合与隶属函数 (Zimmermann Ch. 2-3)",
      "模糊集运算与代数性质 (Zimmermann Ch. 3)",
      "扩展原理与模糊数 (Zimmermann Ch. 5)",
      "模糊关系与模糊关系方程 (Zimmermann Ch. 6)",
      "模糊逻辑与近似推理 (Klir & Yuan Ch. 4-5)",
      "可能性理论 (Zimmermann Ch. 8)",
      "模糊控制 (Zimmermann Ch. 11)",
      "模糊聚类与模式识别 (Zimmermann Ch. 14)",
      "模糊测度与模糊积分：Sugeno 积分与模糊测度公理 (Zimmermann Ch. 4)",
      "模糊决策分析：多准则决策 (Zimmermann Ch. 12)",
      "模糊数学规划 (Zimmermann Ch. 13)"
    ],
  },
  'intermediate/mathematical-modeling': {
    title: "数学建模",
    books: [
          "Frank R. Giordano, Maurice D. Weir, William P. Fox, \"A First Course in Mathematical Modeling\" (5th ed., 2014)",
          "Mark M. Meerschaert, \"Mathematical Modeling\" (4th ed., 2013)"
    ],
    chapters: [
      "建模过程与比例模型 (Giordano Ch. 2)",
      "拟合、回归与插值 (Giordano Ch. 3)",
      "离散与连续模型 (Giordano Ch. 5-7)",
      "常微分方程模型 (Giordano Ch. 11)",
      "概率与蒙特卡洛模拟 (Meerschaert Ch. 7-8)",
      "图论与组合优化模型 (Giordano Ch. 8)",
      "优化模型与线性规划 (Giordano Ch. 13)",
      "人口、生态与传染病模型 (Giordano Ch. 12)",
      "差分方程与离散动力系统建模 (Giordano Ch. 1, 6)",
      "博弈论模型 (Giordano Ch. 10)",
      "决策论模型 (Giordano Ch. 9)"
    ],
  },
  'intermediate/operations-research': {
    title: "运筹学（线性/整数/网络规划）",
    books: [
          "Frederick S. Hillier and Gerald J. Lieberman, \"Introduction to Operations Research\" (10th ed., 2015)",
          "Hamdy A. Taha, \"Operations Research: An Introduction\" (10th ed., 2017)",
          "George B. Dantzig and Mukund N. Thapa, \"Linear Programming\" (1997)"
    ],
    chapters: [
      "线性规划与单纯形法 (Hillier & Lieberman Ch. 4)",
      "对偶理论与对偶单纯形法 (Hillier & Lieberman Ch. 6)",
      "整数规划与分支定界 (Hillier & Lieberman Ch. 12)",
      "网络优化与最短路、最大流 (Hillier & Lieberman Ch. 10)",
      "动态规划 (Hillier & Lieberman Ch. 11)",
      "非线性规划与 KKT 条件 (Hillier & Lieberman Ch. 13)",
      "排队论 (Hillier & Lieberman Ch. 17)",
      "决策分析与博弈论 (Hillier & Lieberman Ch. 15-16)",
      "运输与指派问题 (Hillier & Lieberman Ch. 9)",
      "元启发式算法 (Hillier & Lieberman Ch. 14)",
      "库存论 (Hillier & Lieberman Ch. 18)",
      "马尔可夫决策过程 (Hillier & Lieberman Ch. 19)",
      "模拟方法 (Hillier & Lieberman Ch. 20)"
    ],
  },
  'intermediate/queueing-theory': {
    title: "排队论与可靠性数学",
    books: [
          "Leonard Kleinrock, \"Queueing Systems, Volume 1: Theory\" (1975)",
          "Donald Gross and Carl M. Harris, \"Fundamentals of Queueing Theory\" (4th ed., 2008)",
          "Sheldon M. Ross, \"Introduction to Probability Models\" (11th ed., 2014)"
    ],
    chapters: [
      "Poisson 过程与指数分布 (Ross Ch. 5)",
      "生灭过程与 M/M/1 队列 (Kleinrock Ch. 2-3)",
      "M/M/c 与 Markov 排队网络 (Kleinrock Ch. 4)",
      "M/G/1 与 Pollaczek-Khinchine 公式 (Kleinrock Ch. 5)",
      "优先权与多类队列 (Gross & Harris Ch. 8)",
      "可靠性函数与故障分布 (Ross Ch. 14)",
      "系统可用性、MTBF 与 MTTR (Ross Ch. 14)",
      "更新过程与更新奖酬 (Ross Ch. 7)",
      "G/G/1 队列与近似方法（扩散近似）(Kleinrock Ch. 6)",
      "批量到达与批量服务队列 (Gross & Harris Ch. 4-5)",
      "马尔可夫更新过程与半马尔可夫过程 (Ross Ch. 7)",
      "冗余系统与 k-out-of-n 可靠性、可维修系统 (Ross Ch. 14)"
    ],
  },
  'intermediate/cryptography-math': {
    title: "密码学数学基础",
    books: [
          "Jeffrey Hoffstein, Jill Pipher, Joseph H. Silverman, \"An Introduction to Mathematical Cryptography\" (2nd ed., 2014)",
          "Douglas R. Stinson, \"Cryptography: Theory and Practice\" (3rd ed., 2006)",
          "Neal Koblitz, \"A Course in Number Theory and Cryptography\" (2nd ed., 1994)"
    ],
    chapters: [
      "RSA 密码与模幂运算 (HPS Ch. 3)",
      "离散对数与 Diffie-Hellman 密钥交换 (HPS Ch. 2)",
      "整数分解与 Miller-Rabin 素性检验 (HPS Ch. 3)",
      "椭圆曲线密码学 (HPS Ch. 6)",
      "椭圆曲线数字签名算法 ECDSA (HPS Ch. 6)",
      "格与 NTRU 公钥加密 (HPS Ch. 7)",
      "数字签名与 Hash 函数 (Stinson Ch. 7-8)",
      "秘密共享与零知识证明 (Stinson Ch. 13)",
      "Shannon 保密理论与信息论基础：熵、完美保密与一次一密 (HPS Ch. 1, 4; Stinson Ch. 2)",
      "流密码与伪随机性 (Stinson Ch. 8)"
    ],
  },
  'intermediate/numerical-linear-algebra': {
    title: "数值线性代数",
    books: [
          "Lloyd N. Trefethen and David Bau III, \"Numerical Linear Algebra\" (SIAM, 1997)",
          "Gene H. Golub and Charles F. Van Loan, \"Matrix Computations\" (4th ed., 2013)",
          "James W. Demmel, \"Applied Numerical Linear Algebra\" (1997)"
    ],
    chapters: [
      "矩阵-向量乘法与正交化 (Trefethen & Bau Ch. 1-7)",
      "QR 分解与 Householder 反射 (Trefethen & Bau Ch. 7-10)",
      "最小二乘问题与 SVD (Trefethen & Bau Ch. 4-5, 11)",
      "条件数与向后稳定性 (Trefethen & Bau Ch. 12-14)",
      "特征值算法与 Schur 分解 (Trefethen & Bau Ch. 24-25)",
      "QR 算法 (Trefethen & Bau Ch. 28-30)",
      "Krylov 子空间迭代方法 (Trefethen & Bau Ch. 31-35)",
      "共轭梯度法与 GMRES (Trefethen & Bau Ch. 35, 38)",
      "高斯消去与 LU 分解：选主元、数值稳定性与迭代细化 (Trefethen & Bau Lectures 20-22)",
      "Cholesky 分解与对称正定系统 (Trefethen & Bau Lecture 23)",
      "带状与稀疏矩阵直接法 (Golub & Van Loan Ch. 11)"
    ],
  },
  'intermediate/numerical-pde': {
    title: "偏微分方程数值解",
    books: [
          "Randall J. LeVeque, \"Finite Difference Methods for Ordinary and Partial Differential Equations\" (SIAM, 2007)",
          "John C. Strikwerda, \"Finite Difference Schemes and Partial Differential Equations\" (2nd ed., 2004)",
          "Claes Johnson, \"Numerical Solution of Partial Differential Equations by the Finite Element Method\" (1987)"
    ],
    chapters: [
      "有限差分法基础 (LeVeque Ch. 1-3)",
      "椭圆方程与迭代求解 (LeVeque Ch. 4)",
      "一维抛物方程的稳定性分析 (LeVeque Ch. 9)",
      "双曲方程与 CFL 条件 (LeVeque Ch. 10)",
      "Lax-Richtmyer 等价定理 (Strikwerda Ch. 2)",
      "有限元变分形式 (C. Johnson Ch. 1-2)",
      "Sobolev 空间与误差估计 (C. Johnson Ch. 4-5)",
      "多维问题与边界条件处理 (LeVeque Ch. 13)",
      "非线性守恒律与激波捕捉：Riemann 问题与 Godunov 型格式 (LeVeque Ch. 11-12)",
      "有限体积法 (LeVeque Part V)",
      "抛物方程分裂方法：ADI 与算子分裂 (LeVeque Ch. 6-8)"
    ],
  },
  'humanities/history-of-mathematics': {
    title: "数学史",
    books: [
          "Victor J. Katz, \"A History of Mathematics: An Introduction\" (3rd ed., 2018)",
          "Carl B. Boyer and Uta C. Merzbach, \"A History of Mathematics\" (3rd ed., 2011)",
          "Howard Eves, \"An Introduction to the History of Mathematics\" (6th ed., 1990)"
    ],
    chapters: [
      "古代埃及与巴比伦数学 (Katz Ch. 1)",
      "古希腊数学：泰勒斯到欧几里得 (Katz Ch. 2-3)",
      "中国与印度古代数学 (Katz Ch. 6)",
      "中世纪伊斯兰数学 (Katz Ch. 7)",
      "文艺复兴与代数兴起 (Katz Ch. 9)",
      "微积分的创立：牛顿与莱布尼茨 (Katz Ch. 13-14)",
      "18 世纪数学：欧拉、伯努利家族与分析学的扩张 (Katz Ch. 16-17)",
      "非欧几何的发现与几何学革命 (Katz Ch. 18-19)",
      "概率论与数理统计史 (Katz Ch. 20)",
      "19 世纪抽象代数与分析严密化 (Katz Ch. 21)",
      "20 世纪数学与公理化方法 (Katz Ch. 25)"
    ],
  },
  'intermediate/category-theory': {
    title: "范畴论",
    books: [
          "Saunders Mac Lane, \"Categories for the Working Mathematician\" (2nd ed., GTM 5, 1998)",
          "Emily Riehl, \"Category Theory in Context\" (2016)",
          "Tom Leinster, \"Basic Category Theory\" (2014)"
    ],
    chapters: [
      "范畴、函子与自然变换 (Mac Lane Ch. I-II)",
      "极限与余极限 (Mac Lane Ch. III, V)",
      "伴随函子 (Mac Lane Ch. IV)",
      "特殊极限与完备范畴：filtered 余极限与伴随函子定理 (Mac Lane Ch. V §5-6, IX)",
      "单子与代数 (Mac Lane Ch. VI)",
      "单范畴与对称单范畴、辫结构 (Mac Lane Ch. VII, XI)",
      "加性范畴与 Abel 范畴 (Mac Lane Ch. VIII)",
      "Yoneda 引理与可表函子 (Riehl Ch. 2-3)",
      "Kan 扩张 (Mac Lane Ch. X)",
      "拓扑斯与层简介 (Mac Lane Ch. IV §9-10)"
    ],
  },
  'intermediate/stochastic-calculus': {
    title: "随机分析（Itô 微积分）",
    books: [
          "Ioannis Karatzas and Steven E. Shreve, \"Brownian Motion and Stochastic Calculus\" (2nd ed., 1991)",
          "Philip E. Protter, \"Stochastic Integration and Differential Equations\" (2nd ed., 2005)",
          "Bernt Øksendal, \"Stochastic Differential Equations\" (6th ed., 2003)"
    ],
    chapters: [
      "Brown 运动、鞅与停时 (Karatzas & Shreve Ch. 1-2)",
      "Itô 随机积分 (Karatzas & Shreve Ch. 3)",
      "Itô 公式与 Lévy 特征 (Protter Ch. 2)",
      "Itô 随机微分方程 (Karatzas & Shreve Ch. 5)",
      "Girsanov 定理与测度变换 (Karatzas & Shreve Ch. 3.5)",
      "局部鞅与半鞅理论 (Protter Ch. 2-3)",
      "Feynman-Kac 公式 (Karatzas & Shreve Ch. 5.7)",
      "Black-Scholes 模型与应用 (Øksendal Ch. 12)",
      "鞅表示定理 (Karatzas & Shreve Ch. 3)",
      "局部时与 Tanaka 公式 (Karatzas & Shreve Ch. 3)",
      "最优停止与随机控制应用 (Øksendal Ch. 10-11)"
    ],
  },
  'intermediate/bayesian-statistics': {
    title: "贝叶斯统计",
    books: [
          "Andrew Gelman et al., \"Bayesian Data Analysis\" (3rd ed., 2013)",
          "James O. Berger, \"Statistical Decision Theory and Bayesian Analysis\" (2nd ed., 1985)",
          "José M. Bernardo and Adrian F.M. Smith, \"Bayesian Theory\" (2000)"
    ],
    chapters: [
      "贝叶斯推断基础 (Gelman Ch. 1-2)",
      "先验分布与共轭先验 (Gelman Ch. 2)",
      "后验推断与 MCMC 方法 (Gelman Ch. 11)",
      "Gibbs 抽样与 Metropolis 算法 (Gelman Ch. 11)",
      "分层模型 (Gelman Ch. 5)",
      "模型选择与贝叶斯因子 (Gelman Ch. 7, Berger Ch. 5)",
      "贝叶斯决策理论 (Berger Ch. 2-4)",
      "贝叶斯计算与 Stan (Gelman Ch. 13)",
      "多参数模型：多变量正态与缺失数据 (Gelman Ch. 3, 8)",
      "模型检验与后验预测检验 (Gelman Ch. 6)",
      "贝叶斯回归与广义线性模型 (Gelman Ch. 14-15)",
      "贝叶斯渐近理论 (Gelman Ch. 4)"
    ],
  },
  'intermediate/high-dimensional-statistics': {
    title: "高维统计分析",
    books: [
          "Martin J. Wainwright, \"High-Dimensional Statistics: A Non-Asymptotic Viewpoint\" (Cambridge, 2019)",
          "Sara van de Geer, \"Estimation and Testing Under Sparsity\" (2016)",
          "Peter Bühlmann and Sara van de Geer, \"Statistics for High-Dimensional Data\" (2011)"
    ],
    chapters: [
      "集中不等式 (Wainwright Ch. 2)",
      "Rademacher 复杂性与经验过程 (Wainwright Ch. 4)",
      "高维线性回归与 Lasso (Wainwright Ch. 7)",
      "压缩感知与稀疏恢复 (Wainwright Ch. 7.5)",
      "非渐近方法与样本复杂性 (Wainwright Ch. 5-6)",
      "矩阵低秩恢复与 RPCA (Wainwright Ch. 9)",
      "偏差-方差权衡与 oracle 不等式 (Wainwright Ch. 12)",
      "图形模型学习 (Bühlmann & van de Geer Ch. 13)",
      "高维 minimax 下界与信息论下界 (Wainwright Ch. 14-15)",
      "协方差矩阵估计与随机矩阵理论 (Bühlmann & van de Geer Ch. 6)",
      "非参数回归与 RKHS 方法 (Wainwright Ch. 13)"
    ],
  },
  'intermediate/time-series-analysis': {
    title: "时间序列分析",
    books: [
          "George E.P. Box, Gwilym M. Jenkins, Gregory C. Reinsel, \"Time Series Analysis: Forecasting and Control\" (4th ed., 2008)",
          "James D. Hamilton, \"Time Series Analysis\" (1994)",
          "Robert H. Shumway and David S. Stoffer, \"Time Series Analysis and Its Applications\" (4th ed., 2017)"
    ],
    chapters: [
      "平稳性与自相关函数 (Box-Jenkins-Reinsel Ch. 2-3)",
      "ARMA 与 ARIMA 模型 (Box-Jenkins-Reinsel Ch. 4-6)",
      "模型识别与参数估计 (Box-Jenkins-Reinsel Ch. 6-7)",
      "季节性 SARIMA 模型 (Box-Jenkins-Reinsel Ch. 9)",
      "单位根检验与协整 (Hamilton Ch. 19)",
      "GARCH 与波动率模型 (Hamilton Ch. 21)",
      "谱分析与周期图 (Shumway & Stoffer Ch. 4)",
      "Kalman 滤波与状态空间模型 (Shumway & Stoffer Ch. 6)",
      "向量自回归 VAR 与多元时间序列 (Hamilton Ch. 10-11)",
      "预测方法：最小均方误差预测 (Box-Jenkins-Reinsel Ch. 5)",
      "传递函数与干预分析 (Box-Jenkins-Reinsel Ch. 10-11)",
      "非线性与机制转换模型：马尔可夫切换 (Hamilton Ch. 22)",
      "长记忆与 ARFIMA 模型 (Box-Jenkins-Reinsel Ch. 12)"
    ],
  },
  'intermediate/topological-data-analysis': {
    title: "拓扑数据分析",
    books: [
          "Herbert Edelsbrunner and John L. Harer, \"Computational Topology: An Introduction\" (AMS, 2010)",
          "Afra Zomorodian, \"Topology for Computing\" (Cambridge, 2005)",
          "Gunnar Carlsson, \"Topological Pattern Recognition\" (surveys, 2009)"
    ],
    chapters: [
      "单纯复形与同调 (Edelsbrunner & Harer Ch. III-IV)",
      "Betti 数与 Euler 特征 (Edelsbrunner & Harer Ch. IV)",
      "Vietoris-Rips 与 Čech 复形 (Zomorodian Ch. 5-6)",
      "持续同调 (Edelsbrunner & Harer Ch. VII)",
      "持续图与持续景观 (Carlsson surveys)",
      "Bottleneck 与 Wasserstein 距离 (Edelsbrunner & Harer Ch. VIII)",
      "点云数据的拓扑特征提取 (Edelsbrunner & Harer Ch. IX)",
      "Mapper 算法与应用 (Carlsson surveys)",
      "Morse 函数与 Reeb 图 (Edelsbrunner & Harer Ch. VI)",
      "对偶理论：Poincaré 对偶与 Alexander 对偶 (Edelsbrunner & Harer Ch. V)",
      "持续同调的矩阵化简算法 (Edelsbrunner & Harer Ch. VII)",
      "多重持续同调与 zigzag 持久性 (前沿文献)"
    ],
  },
  'intermediate/information-geometry': {
    title: "信息几何",
    books: [
          "Shun-ichi Amari, \"Information Geometry and Its Applications\" (Springer, 2016)",
          "Shun-ichi Amari and Hiroshi Nagaoka, \"Methods of Information Geometry\" (AMS, 2000)",
          "Frank Nielsen, \"An Elementary Introduction to Information Geometry\" (2020)"
    ],
    chapters: [
      "概率分布族与统计流形 (Amari 2016 Ch. 1-2)",
      "Fisher 信息度量 (Amari & Nagaoka Ch. 2)",
      "α-联络与对偶平坦结构 (Amari 2016 Ch. 2-3)",
      "e-平坦与 m-平坦族 (Amari 2016 Ch. 3)",
      "广义 Pythagoras 定理 (Amari 2016 Ch. 3)",
      "投影定理与 EM 算法 (Amari 2016 Ch. 4)",
      "渐近推断与 Cramér-Rao 界 (Amari & Nagaoka Ch. 5)",
      "流形上的优化与自然梯度 (Amari 2016 Ch. 12)",
      "散度理论：f-散度、Bregman 散度与 KL 散度的几何 (Amari 2016 Ch. 1-2)",
      "高阶渐近推断理论：Edgeworth 展开与 Amari-Chentsov 张量 (Amari & Nagaoka Ch. 4-5)",
      "对偶平坦流形在机器学习中的应用 (Amari 2016 Ch. 12)"
    ],
  },
  'intermediate/modern-physics': {
    title: "近代物理（相对论/量子引论）",
    books: [
          "Paul A. Tipler and Ralph A. Llewellyn, \"Modern Physics\" (6th ed., 2012)",
          "Kenneth S. Krane, \"Modern Physics\" (3rd ed., 2012)",
          "Robert Eisberg and Robert Resnick, \"Quantum Physics of Atoms, Molecules, Solids, Nuclei, and Particles\" (2nd ed., 1985)"
    ],
    chapters: [
      "狭义相对论与洛伦兹变换 (Tipler Ch. 1-2)",
      "光的量子化：黑体辐射与光电效应 (Tipler Ch. 3)",
      "氢原子玻尔模型 (Tipler Ch. 4)",
      "物质波与德布罗意假设 (Tipler Ch. 5)",
      "薛定谔方程与一维势阱 (Tipler Ch. 6)",
      "量子隧穿 (Tipler Ch. 6)",
      "角动量与自旋 (Tipler Ch. 7)",
      "多电子原子与元素周期表 (Tipler Ch. 7)",
      "统计物理初步 (Tipler Ch. 8)",
      "分子结构与光谱 (Tipler Ch. 9)",
      "固体物理初步 (Tipler Ch. 10)",
      "核物理基础 (Tipler Ch. 11)",
      "粒子物理标准模型引论 (Tipler Ch. 12)",
      "天体物理与宇宙学 (Tipler Ch. 13)"
    ],
  },
  'intermediate/fluid-mechanics': {
    title: "流体力学",
    books: [
          "G.K. Batchelor, \"An Introduction to Fluid Dynamics\" (Cambridge, 1967)",
          "L.D. Landau and E.M. Lifshitz, \"Fluid Mechanics\" (2nd ed., 1987, Course Vol. 6)",
          "Horace Lamb, \"Hydrodynamics\" (6th ed., 1932)"
    ],
    chapters: [
      "流体运动学与连续性方程 (Batchelor Ch. 2-3)",
      "应力张量与本构关系 (Batchelor Ch. 4)",
      "Navier-Stokes 方程 (Batchelor Ch. 5)",
      "粘性流动精确解：Couette 与 Poiseuille 流 (Landau & Lifshitz Ch. II)",
      "粘性流动与边界层 (Batchelor Ch. 5-6)",
      "边界层理论系统 (Landau & Lifshitz Ch. IV)",
      "不可压缩无粘流动与势流理论 (Batchelor Ch. 7)",
      "涡量、环量与 Kelvin 定理 (Batchelor Ch. 7-8)",
      "低雷诺数流动（Stokes 流） (Landau & Lifshitz Ch. II §20-24)",
      "表面波与重力波 (Landau & Lifshitz §9-12)",
      "声波 (Landau & Lifshitz Ch. VII)",
      "可压缩流动与气体动力学（激波/Mach 数） (Landau & Lifshitz Ch. VIII-IX)",
      "流体中的热传导与扩散 (Landau & Lifshitz Ch. V-VI)",
      "势流理论与复势 (Landau & Lifshitz §III)",
      "湍流简介与 Reynolds 应力 (Landau & Lifshitz §31-34)",
      "流体稳定性（Rayleigh-Taylor/Kelvin-Helmholtz） (Landau & Lifshitz §VIII)",
      "计算流体力学（CFD）方法 (Anderson §8)"
    ],
  },
  'intermediate/nonlinear-dynamics-chaos': {
    title: "非线性动力学与混沌",
    books: [
          "Steven H. Strogatz, \"Nonlinear Dynamics and Chaos: With Applications to Physics, Biology, Chemistry, and Engineering\" (2nd ed., 2015)",
          "John Guckenheimer and Philip Holmes, \"Nonlinear Oscillations, Dynamical Systems, and Bifurcations of Vector Fields\" (1983)",
          "Robert C. Hilborn, \"Chaos and Nonlinear Dynamics\" (2nd ed., 2000)"
    ],
    chapters: [
      "一维流与不动点稳定性 (Strogatz Ch. 2-3)",
      "分岔：鞍结、跨临界与音叉分岔 (Strogatz Ch. 3-4)",
      "二维线性系统与相平面 (Strogatz Ch. 5-6)",
      "极限环与 Poincaré-Bendixson 定理 (Strogatz Ch. 7-8)",
      "Lorenz 方程与混沌吸引子 (Strogatz Ch. 9)",
      "一维映射与倍周期分岔 (Strogatz Ch. 10)",
      "分形几何与奇怪吸引子 (Strogatz Ch. 11)",
      "Hamilton 系统与 KAM 定理 (Guckenheimer & Holmes Ch. 7)"
    ],
  },
  'advanced/atomic-molecular-optical-physics': {
    title: "原子分子与光物理",
    books: [
          "Christopher J. Foot, \"Atomic Physics\" (Oxford, 2005)",
          "Gordon W.F. Drake (ed.), \"Springer Handbook of Atomic, Molecular, and Optical Physics\" (2006)",
          "B.H. Bransden and C.J. Joachain, \"Physics of Atoms and Molecules\" (2nd ed., 2003)"
    ],
    chapters: [
      "氢原子精细结构 (Foot Ch. 2-4)",
      "多电子原子与 LS 耦合 (Foot Ch. 4)",
      "塞曼效应与斯塔克效应 (Foot Ch. 3-5)",
      "超精细结构与同位素位移 (Foot Ch. 6)",
      "跃迁概率与选择定则 (Foot Ch. 7)",
      "分子转动光谱 (Bransden & Joachain Ch. 10)",
      "分子振动与电子光谱 (Bransden & Joachain Ch. 10)",
      "激光原理与光泵浦 (Foot Ch. 7-8)",
      "激光冷却与磁光阱 (Foot Ch. 9)",
      "磁阱、蒸发冷却与 BEC (Foot Ch. 10)",
      "精密光谱与原子钟 (Foot Ch. 11)"
    ],
  },
  'advanced/nuclear-physics': {
    title: "核物理",
    books: [
          "Kenneth S. Krane, \"Introductory Nuclear Physics\" (Wiley, 1987)",
          "Samuel S.M. Wong, \"Introductory Nuclear Physics\" (2nd ed., 1998)",
          "John D. Walecka, \"Theoretical Nuclear and Subnuclear Physics\" (2nd ed., 2004)"
    ],
    chapters: [
      "核子性质 (Krane Ch. 2)",
      "核自旋与电磁矩 (Krane Ch. 2)",
      "核力与双核子系统 (Krane Ch. 3)",
      "壳模型与集体模型 (Krane Ch. 4)",
      "放射性衰变 (Krane Ch. 5)",
      "α 衰变动力学 (Krane Ch. 6)",
      "β 衰变与费米理论 (Krane Ch. 7)",
      "γ 衰变与内转换 (Krane Ch. 8)",
      "核反应 (Krane Ch. 9)",
      "中子物理与散射 (Krane Ch. 9)",
      "核谱学与激发态 (Krane Ch. 10)",
      "裂变与核能 (Krane Ch. 11)",
      "聚变 (Krane Ch. 12)",
      "恒星核合成 (Krane Ch. 13)",
      "辐射探测器与核技术应用 (Krane Ch. 14)"
    ],
  },
  'advanced/plasma-physics': {
    title: "等离子体物理",
    books: [
          "Francis F. Chen, \"Introduction to Plasma Physics and Controlled Fusion\" (3rd ed., 2016)",
          "Robert J. Goldston and Paul H. Rutherford, \"Introduction to Plasma Physics\" (1995)",
          "Nicholas A. Krall and Alvin W. Trivelpiece, \"Principles of Plasma Physics\" (1973)"
    ],
    chapters: [
      "等离子体定义与 Debye 屏蔽 (Chen Ch. 1-2)",
      "单粒子轨道与漂移运动 (Chen Ch. 2)",
      "磁流体力学方程 (Chen Ch. 3)",
      "等离子体振荡与色散关系 (Chen Ch. 4)",
      "等离子体波分类（EM 波与磁化等离子体色散） (Chen Ch. 4)",
      "输运过程与碰撞 (Chen Ch. 5)",
      "等离子体不稳定性 (Chen Ch. 6)",
      "动理学理论：Vlasov 方程与朗道阻尼 (Chen Ch. 7)",
      "非线性效应与等离子体湍流 (Chen Ch. 8)",
      "磁约束与托卡马克 (Chen Ch. 9)",
      "惯性约束聚变 (Chen Ch. 9)",
      "等离子体诊断方法 (Goldston & Rutherford)"
    ],
  },
  'advanced/computational-physics': {
    title: "计算物理",
    books: [
          "Mark E.J. Newman, \"Computational Physics\" (2012)",
          "Nicholas J. Giordano and Hisao Nakanishi, \"Computational Physics\" (2nd ed., 2006)",
          "Harvey Gould, Jan Tobochnik, Wolfgang Christian, \"An Introduction to Computer Simulation Methods\" (3rd ed., 2006)"
    ],
    chapters: [
      "数值积分与求根 (Newman Ch. 5-6)",
      "数据拟合与最小二乘/误差分析 (Newman Ch. 4, 11)",
      "矩阵运算与本征值问题 (Newman Ch. 6)",
      "常微分方程数值解 (Newman Ch. 8)",
      "偏微分方程数值解 (Newman Ch. 9)",
      "快速傅里叶变换 (Newman Ch. 7)",
      "蒙特卡洛方法与随机数 (Newman Ch. 10)",
      "随机游走与马尔可夫链蒙特卡洛 (Newman Ch. 10)",
      "随机过程模拟：自相关与功率谱 (Newman)",
      "分子动力学模拟 (Giordano & Nakanishi Ch. 8-9)",
      "有限元方法 (Giordano & Nakanishi Ch. 10)"
    ],
  },
  'advanced/semiconductor-physics': {
    title: "半导体物理",
    books: [
          "Simon M. Sze and Kwok K. Ng, \"Physics of Semiconductor Devices\" (3rd ed., 2007)",
          "Marius Grundmann, \"The Physics of Semiconductors\" (3rd ed., 2016)",
          "Karl W. Böer, \"Introduction to Space Charge Effects in Semiconductors\" (2010)"
    ],
    chapters: [
      "能带、有效质量与态密度 (Sze Ch. 1)",
      "载流子统计与 Fermi-Dirac 分布 (Sze Ch. 1)",
      "漂移、扩散与复合 (Sze Ch. 1)",
      "p-n 结二极管 (Sze Ch. 2)",
      "金属-半导体接触与肖特基结 (Sze Ch. 3)",
      "MIS/MOS 电容与界面态 (Sze Ch. 4)",
      "双极型晶体管 (Sze Ch. 5)",
      "MOS 场效应晶体管 (Sze Ch. 6)",
      "微波器件：JFET/MESFET/MODFET (Sze Ch. 7)",
      "异质结与量子阱结构 (Grundmann Ch. 12)",
      "半导体激光器 (Sze Ch. 12)",
      "光电器件：LED 与太阳能电池 (Sze Ch. 13)"
    ],
  },
  'advanced/mesoscopic-physics': {
    title: "介观物理与介观输运",
    books: [
          "Yuli V. Nazarov and Yaroslav M. Blanter, \"Quantum Transport: Introduction to Nanoscience\" (Cambridge, 2009)",
          "Thomas Dittrich, \"Quantum Transport and Dissipation\" (Wiley, 1998)",
          "D.K.K. de Jong and C.W.J. Beenakker, \"Mesoscopic Electron Transport\" (1997, NATO ASI)"
    ],
    chapters: [
      "介观尺度与相位相干长度 (Nazarov & Blanter Ch. 1)",
      "Landauer-Büttiker 公式 (Nazarov & Blanter Ch. 1-2)",
      "散射矩阵与电导量子化 (Nazarov & Blanter Ch. 1-2)",
      "量子霍尔效应 (Nazarov & Blanter Ch. 2)",
      "Aharonov-Bohm 效应与 Berry 相位 (Nazarov & Blanter Ch. 1)",
      "库仑阻塞与单电子隧穿 (Nazarov & Blanter Ch. 5)",
      "量子点能级与输运谱 (Nazarov & Blanter Ch. 5)",
      "退相干与量子噪声 (Nazarov & Blanter Ch. 6)",
      "超导介观输运与 Andreev 反射 (Nazarov & Blanter)",
      "非平衡 Green 函数方法 (Nazarov & Blanter)",
      "自旋输运与自旋电子学 (Nazarov & Blanter)"
    ],
  },
  'advanced/quantum-information': {
    title: "量子信息基础",
    books: [
          "Michael A. Nielsen and Isaac L. Chuang, \"Quantum Computation and Quantum Information\" (Cambridge, 2000)",
          "Mark M. Wilde, \"Quantum Information Theory\" (2nd ed., 2017)",
          "Stephen M. Barnett, \"Quantum Information\" (Oxford, 2009)"
    ],
    chapters: [
      "量子比特与量子门 (Nielsen & Chuang Ch. 4)",
      "量子测量与 POVM 形式 (Nielsen & Chuang Ch. 2)",
      "量子纠缠与 Bell 不等式 (Nielsen & Chuang Ch. 2)",
      "量子算法：Shor 与 Grover (Nielsen & Chuang Ch. 5-6)",
      "量子错误更正码 (Nielsen & Chuang Ch. 10)",
      "量子密码与 BB84 协议 (Nielsen & Chuang Ch. 12)",
      "量子噪声与量子信道 (Nielsen & Chuang Ch. 8)",
      "量子信息熵与 Schumacher 压缩 (Nielsen & Chuang Ch. 11)"
    ],
  },
  'advanced/particle-physics-experiments': {
    title: "粒子物理实验方法",
    books: [
          "Glenn F. Knoll, \"Radiation Detection and Measurement\" (4th ed., 2010)",
          "William R. Leo, \"Techniques for Nuclear and Particle Physics Experiments\" (2nd ed., 1994)",
          "Claus Grupen and Boris Shwartz, \"Particle Detectors\" (2nd ed., 2008)"
    ],
    chapters: [
      "辐射与物质相互作用 (Knoll Ch. 1-2)",
      "计数统计与误差分析 (Knoll Ch. 3)",
      "气体探测器：电离室与正比计数管 (Knoll Ch. 4-5)",
      "径迹探测器：漂移室、TPC 与硅微条 (Grupen & Shwartz)",
      "半导体探测器 (Knoll Ch. 11-12)",
      "闪烁探测器 (Knoll Ch. 8-9)",
      "量能器：电磁与强子量能器 (Grupen & Shwartz)",
      "光电倍增管与读出电子学 (Knoll Ch. 9, Grupen Ch. 16)",
      "粒子鉴别方法 (Grupen & Shwartz Ch. 14)",
      "中子探测器 (Knoll)",
      "触发与数据获取系统 (Grupen & Shwartz Ch. 17)",
      "加速器原理与对撞机 (Leo Ch. 3)"
    ],
  },
  'advanced/string-theory': {
    title: "弦论与量子引力",
    books: [
          "Joseph Polchinski, \"String Theory, Volume 1: An Introduction to the Bosonic String\" (Cambridge, 1998)",
          "Barton Zwiebach, \"A First Course in String Theory\" (2nd ed., 2009)",
          "Katrin Becker, Melanie Becker, John H. Schwarz, \"String Theory and M-Theory: A Modern Introduction\" (2007)"
    ],
    chapters: [
      "相对论粒子的作用量 (Zwiebach Ch. 2-5)",
      "Nambu-Goto 作用量与弦运动学 (Zwiebach Ch. 6)",
      "弦量子化与谱（光锥量子化） (Zwiebach Ch. 7-11)",
      "弦的约束与临界维数 (Polchinski Vol. 1 Ch. 1-2)",
      "共形场论（CFT） (Polchinski Vol. 1 Ch. 3)",
      "弦相互作用与散射振幅 (Polchinski Vol. 1 Ch. 6)",
      "圈图与模不变性 (Polchinski Vol. 1 Ch. 7)",
      "开弦与 D-膜 (Polchinski Vol. 1 Ch. 9)",
      "T-对偶性 (Polchinski Vol. 1 Ch. 8)",
      "超弦与 GSO 投影 (Polchinski Vol. 2 Ch. 10)",
      "Calabi-Yau 紧致化 (Becker-Becker-Schwarz Ch. 9)",
      "M 理论与 S 对偶 (Becker-Becker-Schwarz Ch. 6-7)",
      "D 膜世界体积规范理论 (Polchinski Vol. 2 Ch. 13)",
      "AdS/CFT 对应 (Becker-Becker-Schwarz Ch. 15)",
      "量子引力与黑洞熵 (Becker-Becker-Schwarz Ch. 14)"
    ],
  },
  'humanities/history-of-physics': {
    title: "物理学史",
    books: [
          "Emilio Segrè, \"From Falling Bodies to Radio Waves: Classical Physicists and Their Discoveries\" (W.H. Freeman, 1984)",
          "Abraham Pais, \"Subtle Is the Lord: The Science and the Life of Albert Einstein\" (Oxford, 1982)",
          "Max Jammer, \"The Conceptual Development of Quantum Mechanics\" (McGraw-Hill, 1966)"
    ],
    chapters: [
      "古希腊到牛顿力学的建立 (Segrè Part I)",
      "科学革命：哥白尼—伽利略—开普勒 (Segrè Part I)",
      "光学史与波动说之争 (Segrè Part III)",
      "热学与统计力学的发展 (Segrè Part II)",
      "电磁学的建立：从奥斯特到麦克斯韦 (Segrè Part III)",
      "相对论的诞生 (Pais Part II)",
      "量子力学的建立：普朗克到海森堡 (Jammer Ch. 3-5)",
      "原子核物理与粒子物理的兴起 (Segrè, From X-rays to Quarks §4)",
      "凝聚态物理的兴起 (Segrè, From X-rays to Quarks §10)",
      "20 世纪物理学的统一 (Pais, Inward Bound §20)"
    ],
  },
  'advanced/topological-quantum-computing': {
    title: "拓扑量子计算",
    books: [
          "Jiannis K. Pachos, \"Introduction to Topological Quantum Computation\" (Cambridge, 2012)",
          "Zhenghan Wang, \"Topological Quantum Computation\" (CBMS Regional Conference, 2010)",
          "C. Nayak, S.H. Simon, A. Stern, M. Freedman, S. Das Sarma, \"Non-Abelian Anyons and Topological Quantum Computation\" (Rev. Mod. Phys. 80, 2008)"
    ],
    chapters: [
      "任意子统计与辫子群 (Pachos Ch. 1-2)",
      "拓扑相与拓扑简并 (Pachos Ch. 3)",
      "Aharonov-Bohm 效应与 Berry 相位 (Pachos Ch. 4)",
      "阿贝尔任意子模型 (Pachos Ch. 5)",
      "非阿贝尔任意子 (Pachos Ch. 6)",
      "分数量子霍尔效应 (Pachos Ch. 7)",
      "量子计算中的拓扑保护 (Pachos Ch. 8)",
      "Fibonacci 任意子模型 (Pachos Ch. 9)",
      "Kitaev 环面码与表面码 (Pachos)",
      "拓扑容错阈值 (Pachos)",
      "拓扑超导体与 Majorana 零模 (Pachos / Nayak et al. RMP)"
    ],
  },
  'advanced/cold-atoms-quantum-simulation': {
    title: "冷原子与量子模拟",
    books: [
          "Christopher J. Pethick and Henrik Smith, \"Bose-Einstein Condensation in Dilute Gases\" (2nd ed., 2008)",
          "H.J. Metcalf and Peter van der Straten, \"Laser Cooling and Trapping\" (Springer, 1999)",
          "I. Bloch, J. Dalibard, W. Zwerger, \"Many-body physics with ultracold gases\" (Rev. Mod. Phys. 80, 2008)"
    ],
    chapters: [
      "玻色-爱因斯坦凝聚基础 (Pethick & Smith Ch. 2-3)",
      "Gross-Pitaevskii 方程与 Bogoliubov 元激发 (Pethick & Smith Ch. 4-6)",
      "Doppler 与 Sisyphus 激光冷却 (Metcalf & van der Straten Ch. 3-6)",
      "磁光阱与磁阱 (Metcalf & van der Straten Ch. 7-11)",
      "蒸发冷却与光学势阱 (Metcalf & van der Straten)",
      "Feshbach 共振 (Pethick & Smith Ch. 5)",
      "光晶格与 Bose-Hubbard 模型 (Bloch-Dalibard-Zwerger §III)",
      "旋量凝聚体 (Pethick & Smith Ch. 12)",
      "超冷费米气体与 BCS-BEC 穿越 (Pethick & Smith Ch. 16)",
      "量子模拟：磁性相变与 Hubbard 模型 (Bloch-Dalibard-Zwerger §V)"
    ],
  },
  'advanced/gravitational-wave-astronomy': {
    title: "引力波天文学",
    books: [
          "Michele Maggiore, \"Gravitational Waves, Volume 1: Theory and Experiments\" (Oxford, 2008)",
          "Jolien D. Creighton and Warren G. Anderson, \"Gravitational-Wave Physics and Astronomy: An Introduction\" (Wiley, 2011)",
          "Charles W. Misner, Kip S. Thorne, John A. Wheeler, \"Gravitation\" (Freeman, 1973)"
    ],
    chapters: [
      "线性引力波与 TT 规范 (Maggiore Ch. 1)",
      "引力波能量与四极矩公式 (Maggiore Ch. 1)",
      "后牛顿近似与双星波形 (Creighton & Anderson Ch. 4)",
      "致密天体作为引力波源 (Maggiore Ch. 4)",
      "双星并合波形建模 (Creighton & Anderson Ch. 5)",
      "LIGO 与干涉仪原理 (Maggiore Ch. 7)",
      "噪声分析与灵敏度曲线 (Creighton & Anderson Ch. 6)",
      "匹配滤波与信号检测 (Creighton & Anderson Ch. 7)",
      "参数估计与贝叶斯推断 (Creighton & Anderson Ch. 8)",
      "多信使天文学与并合事件 (Creighton & Anderson Ch. 9)",
      "LISA 与空间引力波探测 (Maggiore)",
      "脉冲星计时阵列与引力波背景 (Creighton & Anderson)"
    ],
  },
  'intermediate/element-chemistry': {
    title: "元素化学",
    books: [
          "Catherine E. Housecroft and Alan G. Sharpe, \"Inorganic Chemistry\" (5th ed., 2018)",
          "F. Albert Cotton, Geoffrey Wilkinson, Carlos A. Murillo, Manfred Bochmann, \"Advanced Inorganic Chemistry\" (6th ed., 1999)",
          "John Emsley, \"Nature's Building Blocks: An A-Z Guide to the Elements\" (2nd ed., 2011)"
    ],
    chapters: [
      "s 区元素：碱金属与碱土金属 (Housecroft Ch. 11-12)",
      "p 区元素：第 13-15 族 (Housecroft Ch. 13-15)",
      "p 区第 16-18 族：氧族、卤素与稀有气体 (Housecroft Ch. 16-18)",
      "d 区元素与配位化学 (Housecroft Ch. 19-22)",
      "配位化合物的电子光谱与磁性（晶体场/配位场理论、Jahn-Teller 效应） (Housecroft Ch. 20)",
      "d 区配合物的反应机理 (Housecroft Ch. 26)",
      "d 区有机金属化学 (Housecroft Ch. 24)",
      "过渡金属的氧化态与颜色 (Cotton Ch. 17-18)",
      "f 区元素：镧系与锕系 (Housecroft Ch. 27)",
      "主族氢化物与卤化物 (Cotton Ch. 5-7)",
      "金属簇化合物 (Cotton Ch. 18)",
      "固态结构与离子固体 (Housecroft Ch. 6)"
    ],
  },
  'intermediate/chemical-thermodynamics': {
    title: "化学热力学",
    books: [
          "Peter Atkins and Julio de Paula, \"Physical Chemistry\" (11th ed., 2018)",
          "Kenneth S. Pitzer, \"Thermodynamics\" (3rd ed., 1995)",
          "George C. Pimentel and Richard D. Spratley, \"Understanding Chemical Thermodynamics\" (1969)"
    ],
    chapters: [
      "气体的性质与状态方程（理想/真实气体、维里方程、对应状态原理） (Atkins Focus 1)",
      "热力学第一定律与焓 (Atkins Focus 2)",
      "热力学第二定律与熵 (Atkins Focus 3)",
      "热力学第三定律与统计熵 (Atkins Focus 3)",
      "化学势与相平衡 (Atkins Focus 5)",
      "理想与实际溶液：Raoult 与 Henry 定律 (Atkins Focus 5)",
      "化学平衡与平衡常数 (Atkins Focus 6)",
      "电化学与 Nernst 方程 (Atkins Focus 6)",
      "纯物质物理变化与相变（Clapeyron/Clausius-Clapeyron 方程、纯物质相图） (Atkins Focus 4)",
      "相图（二元/三元体系） (Atkins Focus 5)",
      "表面热力学与 Kelvin 方程 (Atkins Focus 19)",
      "统计热力学（配分函数/Boltzmann 分布） (Atkins Focus 13)",
      "热化学（Hess 定律/键焓） (Atkins Focus 2)",
      "非平衡热力学（Onsager 倒易关系） (Atkins Focus 16)"
    ],
  },
  'intermediate/chemical-kinetics': {
    title: "化学动力学（深化）",
    books: [
          "Peter Atkins and Julio de Paula, \"Physical Chemistry\" (11th ed., 2018)",
          "James H. Espenson, \"Chemical Kinetics and Reaction Mechanisms\" (2nd ed., 1995)",
          "Keith J. Laidler, \"Chemical Kinetics\" (3rd ed., 1987)"
    ],
    chapters: [
      "反应速率与速率方程 (Atkins Focus 17)",
      "一级、二级反应动力学 (Atkins Focus 17)",
      "Arrhenius 方程与活化能 (Atkins Focus 17)",
      "复合反应与稳态近似 (Atkins Focus 17)",
      "反应机理与中间体 (Espenson Ch. 4)",
      "催化：均相与多相 (Atkins Focus 17)",
      "溶液中的反应与盐效应（离子强度、Brønsted-Bjerrum、Debye-Hückel 极限律） (Laidler Ch. 4, 8)",
      "快速反应技术（停流、温度跃变、闪光光解、脉冲辐解） (Atkins Focus 17C)",
      "扩散控制反应 (Atkins Focus 18B)",
      "光化学反应动力学 (Laidler Ch. 11)",
      "链反应与爆炸机理 (Laidler Ch. 5)",
      "过渡态理论（Eyring 方程） (Atkins Focus 18)",
      "单分子反应理论（Lindemann/RRKM） (Atkins Focus 18)",
      "酶动力学（Michaelis-Menten） (Atkins Focus 17)",
      "振荡反应（Belousov-Zhabotinsky） (Atkins Focus 17)",
      "飞秒化学与实时观测 (Atkins Focus 18)"
    ],
  },
  'intermediate/surface-chemistry': {
    title: "表面与界面化学",
    books: [
          "Gabor A. Somorjai and Yimin Li, \"Introduction to Surface Chemistry and Catalysis\" (2nd ed., 2010)",
          "Hans-Jürgen Butt, Karlheinz Graf, Michael Kappl, \"Physics and Chemistry of Interfaces\" (3rd ed., 2013)",
          "Arthur W. Adamson and Alice P. Gast, \"Physical Chemistry of Surfaces\" (6th ed., 1997)"
    ],
    chapters: [
      "表面结构与吸附等温线 (Somorjai Ch. 2-3)",
      "Langmuir 与 BET 吸附理论 (Adamson Ch. 17)",
      "吸附动力学（粘附系数、表面扩散、前驱态） (Somorjai Ch. 3-4)",
      "表面电子结构与功函数（金属表面、d 带中心） (Somorjai Ch. 4-5)",
      "表面张力与毛细现象 (Butt-Graf-Kappl Ch. 2)",
      "表面能与润湿性 (Butt-Graf-Kappl Ch. 3, 6)",
      "双电层与 Zeta 电势 (Butt-Graf-Kappl Ch. 4)",
      "表面光谱技术 (Somorjai Ch. 6)",
      "多相催化原理 (Somorjai Ch. 8-9)",
      "自组装单层 SAM (Butt-Graf-Kappl Ch. 9)",
      "Langmuir-Blodgett 膜 (Adamson Ch. 15)",
      "表面活性剂与胶束 (Butt-Graf-Kappl Ch. 11)",
      "纳米颗粒表面化学 (Somorjai Ch. 7)"
    ],
  },
  'intermediate/organic-synthesis': {
    title: "有机合成",
    books: [
          "Jonathan Clayden, Nick Greeves, Stuart Warren, \"Organic Chemistry\" (2nd ed., 2012)",
          "Michael B. Smith, \"March's Advanced Organic Chemistry\" (8th ed., 2020)",
          "K.C. Nicolaou and E.J. Sorensen, \"Classics in Total Synthesis\" (Wiley, 1996)"
    ],
    chapters: [
      "有机反应性与极性反转 (Clayden Ch. 6-12)",
      "烯醇化学与 aldol 反应 (Clayden Ch. 20, 25-26)",
      "立体选择性反应 (Clayden Ch. 32-33)",
      "周环反应：Diels-Alder 与电环化 (Clayden Ch. 34-35)",
      "逆合成分析与保护基 (Clayden Ch. 23, 28)",
      "芳香化学与亲电取代 (Clayden Ch. 21-22)",
      "杂环化学 (Clayden Ch. 29-31)",
      "全合成实例：天然产物 (Nicolaou & Sorensen Ch. 1-12)",
      "氧化与还原反应（Swern/PCC/NaBH4/LiAlH4） (March Ch. 19)",
      "过渡金属催化 C-C 偶联（Suzuki/Heck/Sonogashira） (Clayden Ch. 40)",
      "不对称合成（手性辅助/Sharpless 环氧化） (Clayden Ch. 41)",
      "保护基策略与正交性 (Clayden Ch. 23)",
      "有机催化（Organocatalysis） (Clayden Ch. 20, 41)",
      "有机金属试剂（Grignard、有机锂、有机锌） (Clayden Ch. 9)",
      "Wittig 反应与叶立德化学（磷/硫/砷叶立德） (Clayden Ch. 27)",
      "烯烃复分解（Grubbs/Hoveyda、ROMP/CM） (Clayden Ch. 40)",
      "自由基反应（Barton-McCombie 脱氧、自由基聚合） (Clayden Ch. 37)",
      "卡宾化学与 Simmons-Smith 环丙烷化 (Clayden Ch. 38)"
    ],
  },
  'intermediate/stereochemistry': {
    title: "立体化学",
    books: [
          "Ernest L. Eliel, Samuel H. Wilen, Lewis N. Mander, \"Stereochemistry of Organic Compounds\" (Wiley, 1994)",
          "David G. Morris, \"Stereochemistry\" (RSC, 2001)",
          "Bernard Testa, \"Principles of Organic Stereochemistry\" (Dekker, 1982)"
    ],
    chapters: [
      "立体异构与手性中心 (Eliel Ch. 1-3)",
      "对映体与非对映体 (Eliel Ch. 3, 6)",
      "构型命名与 CIP 规则 (Eliel Ch. 5)",
      "前手性与前立体异构性（prochirality） (Eliel Ch. 8)",
      "构象分析：无环与环状体系（端基异构效应、立体电子效应） (Eliel Ch. 10-11)",
      "立体异构体的分离：拆分与外消旋化 (Eliel Ch. 7)",
      "手性识别与对映体过量测定 (Eliel Ch. 6)",
      "旋光色散 ORD 与圆二色 CD（Cotton 效应、八区律） (Eliel Ch. 13)",
      "手性的光谱与 X 射线测定 (Eliel Ch. 13)",
      "立体选择性与立体专一性 (Eliel Ch. 12)",
      "不对称合成方法（手性辅基/催化） (Eliel Ch. 12)",
      "外消旋体拆分方法深化（动力学拆分、酶/结晶拆分） (Eliel Ch. 7)"
    ],
  },
  'intermediate/spectroscopy': {
    title: "波谱学（NMR/IR/MS）",
    books: [
          "Robert M. Silverstein, Francis X. Webster, David Kiemle, \"Spectrometric Identification of Organic Compounds\" (7th ed., 2005)",
          "Dudley H. Williams and Ian Fleming, \"Spectroscopic Methods in Organic Chemistry\" (6th ed., 2007)",
          "Peter Atkins and Julio de Paula, \"Physical Chemistry\" (11th ed., 2018, Part 3)"
    ],
    chapters: [
      "紫外-可见光谱与共轭体系 (Silverstein Ch. 7)",
      "红外光谱与官能团鉴定 (Silverstein Ch. 2)",
      "质谱与分子量测定 (Silverstein Ch. 1)",
      "质谱碎裂规律与 McLafferty 重排（高分辨质谱） (Silverstein Ch. 1)",
      "核磁共振：1H 与 13C NMR (Silverstein Ch. 3-5)",
      "自旋-自旋耦合与化学位移 (Williams & Fleming Ch. 3)",
      "二维 NMR：COSY、HSQC、NOESY (Williams & Fleming Ch. 4)",
      "动态 NMR 与化学交换（变温 NMR） (Williams & Fleming Ch. 3)",
      "电子顺磁共振 EPR (Atkins Focus 12)",
      "结构鉴定综合应用 (Silverstein Ch. 7-8)",
      "X 射线衍射原理与晶体结构（Bragg 定律、结构因子、相问题） (Atkins Focus 16)",
      "拉曼光谱与 SERS (Atkins Focus 11)",
      "荧光光谱 (Atkins Focus 11)",
      "圆二色光谱（CD） (Atkins Focus 11)"
    ],
  },
  'advanced/photochemistry': {
    title: "光化学",
    books: [
          "Nicholas J. Turro, V. Ramamurthy, J.C. Scaiano, \"Principles of Molecular Photochemistry\" (University Science Books, 2009)",
          "Howard E. Zimmerman, \"Molecular Photochemistry\" (Benjamin, 1966)",
          "C.H. Bamford and C.F.H. Tipper (eds.), \"Photochemistry\" (Comprehensive Chemical Kinetics, Vol. 14-15, 1972)"
    ],
    chapters: [
      "光吸收与激发态 (Turro Ch. 1-3)",
      "Jablonski 图与辐射跃迁 (Turro Ch. 4)",
      "无辐射跃迁与系间窜越 (Turro Ch. 5)",
      "单重态与三重态动力学 (Turro Ch. 2-3)",
      "光致电子转移与 Marcus 理论 (Turro Ch. 7)",
      "光敏化与猝灭 (Turro Ch. 7)",
      "三重态-三重态湮灭上转换 TTA (Turro Ch. 7)",
      "光化学反应：周环与异构化 (Turro Ch. 6)",
      "命名光化学反应：Norrish I/II、Paternò-Büchi 与光环加成 (Turro, Modern Molecular Photochemistry of Organic Molecules)",
      "激光化学与光催化 (Turro Ch. 1)",
      "化学发光与生物发光 (Turro, Modern Molecular Photochemistry of Organic Molecules)"
    ],
  },
  'advanced/colloid-chemistry': {
    title: "胶体与界面化学",
    books: [
          "Paul C. Hiemenz and Raj Rajagopalan, \"Principles of Colloid and Surface Chemistry\" (3rd ed., 1997)",
          "Robert J. Hunter, \"Foundations of Colloid Science\" (2nd ed., 2001)",
          "Duncan J. Shaw, \"Introduction to Colloid and Surface Chemistry\" (4th ed., 1992)"
    ],
    chapters: [
      "胶体分类与稳定性 (Hiemenz Ch. 1)",
      "胶体制备与成核生长（Ostwald 熟化、溶胶-凝胶） (Hiemenz Ch. 1)",
      "DLVO 理论 (Hiemenz Ch. 10-11, 13)",
      "双电层与 Zeta 电势 (Hiemenz Ch. 11-12)",
      "空间位阻稳定（steric stabilization、聚合物刷） (Hiemenz Ch. 13)",
      "表面活性剂与胶束 (Hiemenz Ch. 8)",
      "乳液与微乳液 (Hunter Ch. 9)",
      "泡沫与气泡 (Hunter Ch. 9)",
      "高分子溶液 (Hiemenz Ch. 3)",
      "凝胶与网络结构 (Hunter Ch. 15)",
      "流变学性质 (Hunter Ch. 15)",
      "光散射表征（静态/动态光散射 DLS、SAXS/SANS） (Hiemenz Ch. 5)"
    ],
  },
  'advanced/cheminformatics': {
    title: "化学信息学",
    books: [
          "Johann Gasteiger and Thomas Engel (eds.), \"Chemoinformatics: A Textbook\" (2nd ed., Wiley, 2003)",
          "Andrew R. Leach and Valerie J. Gillet, \"An Introduction to Chemoinformatics\" (Springer, 2003)",
          "Jürgen Bajorath (ed.), \"Chemoinformatics and Computational Chemical Biology\" (2010)"
    ],
    chapters: [
      "化学结构的计算机表示 (Gasteiger & Engel Ch. 2)",
      "化学数据库与信息检索 (Leach & Gillet Ch. 3)",
      "子结构搜索与 SMARTS 模式语言 (Leach & Gillet Ch. 1)",
      "分子描述符与拓扑指标 (Gasteiger & Engel Ch. 4)",
      "构象生成与 3D 结构预测 (Gasteiger & Engel Ch. 5)",
      "分子相似性与分子指纹（Tanimoto、MACCS、ECFP/Morgan） (Leach & Gillet Ch. 5)",
      "化学空间与分子多样性分析 (Leach & Gillet Ch. 6)",
      "定量构效关系 QSAR (Leach & Gillet Ch. 4)",
      "药效团识别 (Leach & Gillet Ch. 8)",
      "分子对接与虚拟筛选 (Leach & Gillet Ch. 8)",
      "逆合成分析与计算机辅助合成设计 CAOS（LHASA/Chematica/Reaxys） (Gasteiger & Engel)",
      "机器学习与化学数据挖掘 (Bajorath Ch. 4)"
    ],
  },
  'advanced/radiochemistry': {
    title: "放射化学",
    books: [
          "Gregory R. Choppin, Jan-Olov Liljenzin, Jan Rydberg, \"Radiochemistry and Nuclear Chemistry\" (4th ed., 2013)",
          "Walter D. Loveland, David J. Morrissey, Glenn T. Seaborg, \"Modern Nuclear Chemistry\" (Wiley, 2006)",
          "József Kónya and Noémi M. Nagy, \"Nuclear and Radiochemistry\" (Elsevier, 2012)"
    ],
    chapters: [
      "核稳定性与放射性衰变 (Choppin Ch. 3-5)",
      "α、β、γ 衰变模式 (Choppin Ch. 5)",
      "放射性活度与半衰期 (Choppin Ch. 5)",
      "核结构与核模型（结合能、液滴模型、壳模型） (Choppin Ch. 6)",
      "核反应与中子活化 (Choppin Ch. 10)",
      "核裂变与核聚变（裂变产物、核反应堆原理） (Choppin Ch. 19-20)",
      "辐射化学（水的辐解、辐射对物质的作用） (Choppin Ch. 8)",
      "辐射防护与剂量学（剂量当量、防护标准、内照射） (Choppin Ch. 15)",
      "放射性核素的化学分离 (Choppin Ch. 18, 21)",
      "放射性测量技术 (Choppin Ch. 9)",
      "同位素示踪剂应用 (Loveland Ch. 17)",
      "核废料管理与环境影响 (Choppin Ch. 21-22)"
    ],
  },
  'advanced/chromatography-separation': {
    title: "色谱与分离分析",
    books: [
          "Douglas A. Skoog, F. James Holler, Stanley R. Crouch, \"Principles of Instrumental Analysis\" (7th ed., 2018)",
          "Colin F. Poole, \"The Essence of Chromatography\" (Elsevier, 2003)",
          "Joseph Sherma and Bernard Fried (eds.), \"Handbook of Thin-Layer Chromatography\" (3rd ed., 2003)"
    ],
    chapters: [
      "色谱分离原理与保留理论 (Skoog Ch. 26)",
      "气相色谱 GC (Skoog Ch. 27)",
      "高效液相色谱 HPLC (Skoog Ch. 28)",
      "离子交换色谱 (Poole Ch. 4)",
      "离子色谱 IC (Poole Ch. 4)",
      "尺寸排阻色谱 (Poole Ch. 4)",
      "亲和色谱 (Poole Ch. 4)",
      "薄层色谱 TLC (Poole Ch. 6)",
      "色谱-质谱联用（GC-MS、LC-MS/MS） (Poole Ch. 9)",
      "毛细管电泳 (Skoog Ch. 30)",
      "超临界流体色谱 (Skoog Ch. 29)",
      "样品前处理（固相萃取 SPE、QuEChERS） (Poole Ch. 1)"
    ],
  },
  'advanced/green-chemistry': {
    title: "绿色化学",
    books: [
          "Paul T. Anastas and John C. Warner, \"Green Chemistry: Theory and Practice\" (Oxford, 1998)",
          "Mike Lancaster, \"Green Chemistry: An Introductory Text\" (2nd ed., RSC, 2010)",
          "Paul T. Anastas and Robert H. Crabtree (eds.), \"Handbook of Green Chemistry\" (Wiley, 2009-2014)"
    ],
    chapters: [
      "12 条绿色化学原则 (Anastas & Warner Ch. 2)",
      "原子经济性与 E-因子 (Lancaster Ch. 2)",
      "替代溶剂：水与超临界 CO2 (Lancaster Ch. 5)",
      "替代催化剂与反应介质 (Lancaster Ch. 6)",
      "可再生原料与生物质转化 (Lancaster Ch. 7)",
      "绿色合成路线设计 (Lancaster Ch. 8)",
      "设计可降解化学品 (Anastas & Warner Ch. 5)",
      "绿色能源与工业应用 (Lancaster Ch. 9)"
    ],
  },
  'advanced/supramolecular-chemistry': {
    title: "超分子化学",
    books: [
          "Jonathan W. Steed and Jerry L. Atwood, \"Supramolecular Chemistry\" (2nd ed., Wiley, 2009)",
          "Jean-Marie Lehn, \"Supramolecular Chemistry: Concepts and Perspectives\" (VCH, 1995)",
          "Hans-Jörg Schneider and Anatoly Yatsimirsky, \"Principles and Methods in Supramolecular Chemistry\" (Wiley, 2000)"
    ],
    chapters: [
      "分子识别与受体设计 (Steed & Atwood Ch. 1-2)",
      "氢键、π-π 与疏水相互作用 (Steed & Atwood Ch. 1)",
      "冠醚与穴状配体 (Steed & Atwood Ch. 3)",
      "阴离子识别与结合（阴离子受体设计） (Steed & Atwood Ch. 4)",
      "环糊精与包合物 (Steed & Atwood Ch. 6)",
      "分子自组装与自组织 (Steed & Atwood Ch. 10)",
      "超分子聚合物 (Steed & Atwood Ch. 14)",
      "机械互锁分子：索烃与轮烷 (Steed & Atwood Ch. 11)",
      "晶体工程与共晶 (Steed & Atwood Ch. 8)",
      "分子机器与分子器件（分子开关、分子马达） (Steed & Atwood Ch. 11)",
      "超分子催化与酶模型 (Lehn Ch. 9)"
    ],
  },
  'humanities/history-of-chemistry': {
    title: "化学史",
    books: [
          "William H. Brock, \"The History of Chemistry\" (2nd ed., Norton, 1992)",
          "J.R. Partington, \"A History of Chemistry\" (4 vols., Macmillan, 1961-1970)",
          "Aaron J. Ihde, \"The Development of Modern Chemistry\" (Harper, 1964)"
    ],
    chapters: [
      "炼金术与早期化学工艺 (Brock Ch. 1-3)",
      "医疗化学与帕拉塞尔苏斯（十六世纪 iatrochemistry） (Brock Ch. 4-5)",
      "燃素说与拉瓦锡革命 (Brock Ch. 7-8)",
      "原子论与分子学说 (Brock Ch. 10)",
      "电化学史：伏打、戴维与法拉第 (Brock Ch. 11)",
      "元素周期表的发现 (Brock Ch. 12)",
      "有机化学的兴起 (Brock Ch. 13-14)",
      "物理化学的建立 (Brock Ch. 15)",
      "20 世纪化学键理论 (Brock Ch. 16)",
      "放射化学与核化学 (Brock Ch. 17)",
      "现代仪器分析与化学工业 (Brock Ch. 18)"
    ],
  },
  'advanced/single-molecule-chemistry': {
    title: "单分子化学",
    books: [
          "Cees Dekker, Nynke Dekker, \"Single Molecule Analysis: Methods and Protocols\" (Methods in Molecular Biology, 2018)",
          "A. Prakelt, \"Single Molecule Chemistry\" (2011)",
          "C. Bustamante, \"Single Molecule Studies of Nucleic Acids\" (Rev. Mod. Phys. 72, 2000)"
    ],
    chapters: [
      "单分子荧光显微技术 (Dekker Ch. 2-3)",
      "光镊与力学谱 (Dekker Ch. 5)",
      "AFM 单分子力谱 (Dekker Ch. 6)",
      "单分子酶动力学 (Dekker Ch. 7)",
      "荧光共振能量转移 FRET (Dekker Ch. 4)",
      "单分子电导与 STM 断结 (Dekker Ch. 12)",
      "超分辨荧光定位：STORM/PALM (Dekker)",
      "单分子拉曼：TERS 与单分子 SERS (Dekker)",
      "纳米孔单分子检测与 DNA 测序 (Dekker)",
      "单分子电化学（SECM） (Dekker)",
      "单分子化学动力学 (Dekker Ch. 8)",
      "单分子生物大分子与折叠 (Bustamante surveys)"
    ],
  },
  'advanced/mechanochemistry': {
    title: "机械化学",
    books: [
          "Adam A.L. Michalchuk, Ivan A. Tumanov, Elena V. Boldyreva (eds.), \"Advances in Mechanochemistry: From Molecular Aggregates to Covalent Bond Breaking\" (De Gruyter, 2023)",
          "Tomasz Rojek, \"Mechanochemistry: Fundamentals and Applications in Synthesis\" (2016)",
          "Sven Mienert and Bernhard V.K.J. Schmidt (eds.), \"Modern Mechanochemistry\" (RSC, 2023)"
    ],
    chapters: [
      "机械化学基础与历史 (Michalchuk Ch. 1)",
      "球磨与机械研磨设备 (Michalchuk Ch. 2)",
      "共价键机械断裂 (Michalchuk Ch. 3)",
      "机械诱导相变 (Michalchuk Ch. 4)",
      "力化学理论模型（Bell-Evans、力修正 Arrhenius、单分子力谱测键力） (Michalchuk Ch. 5)",
      "力响应材料与机械载体（力致变色、自修复、应力传感） (Michalchuk Ch. 5)",
      "机械催化（mechanocatalysis） (Modern Mechanochemistry Ch. 7)",
      "药物共晶与制药机械化学（milling 制备共晶/无定形） (Modern Mechanochemistry Ch. 8)",
      "无溶剂有机合成 (Modern Mechanochemistry Ch. 6)",
      "机械合金化 (Michalchuk Ch. 6)",
      "机械化学合成 MOF 与共价有机框架 (Modern Mechanochemistry Ch. 9)",
      "工业应用与放大 (Michalchuk Ch. 8)"
    ],
  },
  'cs/assembly-language': {
    title: "汇编语言",
    books: [
          "Bryant & O'Hallaron, \"Computer Systems: A Programmer's Perspective\" (3rd, 2015)",
          "Patterson & Hennessy, \"Computer Organization and Design RISC-V Edition\" (2nd, 2020)",
          "Randall Hyde, \"The Art of Assembly Language\" (2nd, 2010)"
    ],
    chapters: [
      "机器级表示与寄存器 (CS:APP §3.4)",
      "数据寻址模式 (CS:APP §3.4.2)",
      "算术与逻辑指令 (CS:APP §3.5)",
      "过程调用与栈帧 (CS:APP §3.7)",
      "x86-64 调用约定与 ABI（System V AMD64 / Windows x64） (CS:APP §3.7)",
      "控制流与跳转 (CS:APP §3.6)",
      "x86-64 指令编码 (CS:APP §3.2)",
      "汇编与 C 互操作 (CS:APP §3.11)",
      "浮点与 SIMD 指令（SSE/AVX/NEON） (CS:APP §3.11)",
      "中断与系统调用机制 (Patterson §5.9)",
      "汇编程序设计专题：数组/字符串/宏 (Hyde Ch.8-12)"
    ],
  },
  'cs/concurrent-parallel-programming': {
    title: "并发与并行编程",
    books: [
          "Herlihy & Shavit, \"The Art of Multiprocessor Programming\" (2nd, 2020)",
          "Bryant & O'Hallaron, \"Computer Systems: A Programmer's Perspective\" (3rd, 2015)",
          "Tanenbaum & Van Steen, \"Distributed Systems\" (3rd, 2017)"
    ],
    chapters: [
      "线程与并发执行 (CS:APP §12.1)",
      "共享变量与互斥 (CS:APP §12.5)",
      "信号量与生产者-消费者 (CS:APP §12.5.5)",
      "并发对象与同步原语相对能力 (Herlihy §3-6)",
      "锁的实现与 CAS (Herlihy §7)",
      "条件变量与读者-写者 (Herlihy §8)",
      "并发数据结构：链表/队列/栈 (Herlihy §9-10)",
      "并发哈希表与跳表 (Herlihy §11-12)",
      "死锁检测与预防 (CS:APP §12.7)",
      "并行算法与工作窃取 (Herlihy §15)",
      "同步屏障 Barrier (Herlihy §16)",
      "事务内存 TM (Herlihy §17)",
      "内存一致性与顺序 (Herlihy §3)"
    ],
  },
  'cs/reverse-engineering': {
    title: "逆向工程与二进制分析",
    books: [
          "Bruce Dang et al., \"Practical Binary Analysis\" (2019)",
          "Michael Sikorski & Andrew Honig, \"Practical Malware Analysis\" (2012)",
          "Eldad Eilam, \"Reversing: Secrets of Reverse Engineering\" (2005)"
    ],
    chapters: [
      "静态分析基础 (Sikorski Ch.3)",
      "动态分析与沙箱 (Sikorski Ch.9)",
      "调试器实战与动态调试（IDA/OllyDbg/gdb） (Eilam Part 2-3)",
      "反汇编与控制流恢复 (Dang Ch.2)",
      "ELF/PE 文件格式解析 (Dang Ch.1)",
      "反调试与反虚拟机技术 (Sikorski Ch.16)",
      "加壳与脱壳 / 混淆与反混淆 (Eilam Part 5)",
      "漏洞挖掘与模糊测试 (Dang Ch.6)",
      "恶意代码行为分析（沙箱/网络） (Sikorski Ch.12)",
      "恶意代码代码分析（IDA 逆向） (Sikorski Ch.12)",
      "API Hook 与钩子注入 (Eilam Part 5)",
      "固件与内核逆向 (Dang Ch.9)",
      "二进制插桩与符号执行 (Dang Ch.7)"
    ],
  },
  'cs/programming-language-theory': {
    title: "程序设计语言理论",
    books: [
          "Benjamin C. Pierce, \"Types and Programming Languages\" (2002)",
          "Michael L. Scott, \"Programming Language Pragmatics\" (4th, 2016)",
          "Robert W. Sebesta, \"Concepts of Programming Languages\" (11th, 2016)"
    ],
    chapters: [
      "λ 演算与求值 (Pierce Ch.5)",
      "类型系统与类型规则 (Pierce Ch.9)",
      "操作语义与指称语义 (Pierce Ch.3)",
      "参数传递与作用域 (Scott §3.4)",
      "子类型与多态 (Pierce Ch.15)",
      "对象与封装 (Pierce Ch.18)",
      "递归类型 (Pierce §20)",
      "类型推导 (Pierce §22)",
      "System F 参数化多态 (Pierce §23)",
      "运行时系统与存储管理 (Scott Ch.7/12)",
      "函数式语言范式 (Sebesta Ch.15)",
      "逻辑编程范式 Prolog (Scott §11)",
      "并发语言构造 (Scott §12.4)"
    ],
  },
  'cs/formal-methods': {
    title: "形式化方法",
    books: [
          "Michael Huth & Mark Ryan, \"Logic in Computer Science\" (2nd, 2004)",
          "Edmund Clarke et al., \"Model Checking\" (1999)",
          "Tobias Nipkow et al., \"Concrete Semantics with Isabelle/HOL\" (2014)"
    ],
    chapters: [
      "命题逻辑与谓词逻辑 (Huth §1-2)",
      "时序逻辑 LTL/CTL (Huth §3)",
      "模型检测与状态空间 (Clarke §2)",
      "符号模型检测与 BDD (Clarke §5-6)",
      "定理证明与 Coq/Isabelle (Nipkow Ch.5)",
      "程序验证与霍尔逻辑 (Nipkow Ch.7)",
      "抽象解释与不变量 (Clarke §13)",
      "实时与混成系统验证 (Clarke §16-17)",
      "SAT/SMT 求解（CDCL/DPLL(T)） (Kroening & Strichman, 书目外)",
      "精化演算与程序推导 (Morgan, 书目外)",
      "进程代数（CSP/CCS/π-演算） (Hoare Ch.4, 书目外)",
      "组合验证与接口理论 (Clarke §12)",
      "TLA+ / Z 规格语言 (Lamport/Spivey, 书目外)",
      "概率模型检测（PRISM） (书目外)"
    ],
  },
  'cs/storage-file-systems': {
    title: "存储与文件系统",
    books: [
          "Silberschatz, Galvin & Gagne, \"Operating System Concepts\" (10th, 2018)",
          "Andrew S. Tanenbaum & Herbert Bos, \"Modern Operating Systems\" (4th, 2014)",
          "Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau, \"Operating Systems: Three Easy Pieces\" (2018)"
    ],
    chapters: [
      "文件系统接口与目录 (Silberschatz §13)",
      "文件系统实现与 inode (Silberschatz §14)",
      "日志结构与日志文件系统 (OSTEP §40)",
      "崩溃一致性与日志文件系统 journaling (OSTEP §42)",
      "数据完整性保护（校验和/静默损坏） (OSTEP §45)",
      "磁盘调度与 RAID (Silberschatz §12)",
      "虚拟文件系统 VFS (Tanenbaum §4.5)",
      "分布式文件系统 NFS (Tanenbaum §10.4)",
      "块存储与闪存 SSD (OSTEP §44)",
      "闪存与 FTL (Tanenbaum §4.9)",
      "NVMe 与新型存储介质 (Silberschatz §12)",
      "分布式文件系统（GFS/HDFS/Ceph） (Tanenbaum §11)",
      "一致性模型（CAP/线性一致性） (Tanenbaum §7)",
      "对象存储（S3/Ceph RGW） (Tanenbaum §12)"
    ],
  },
  'cs/virtualization': {
    title: "虚拟化技术",
    books: [
          "Jim Smith & Ravi Nair, \"Virtual Machines: Versatile Platforms for Systems and Processes\" (2005)",
          "Andrew S. Tanenbaum & Herbert Bos, \"Modern Operating Systems\" (4th, 2014)",
          "Bryant & O'Hallaron, \"Computer Systems: A Programmer's Perspective\" (3rd, 2015)"
    ],
    chapters: [
      "虚拟化导论与分类 (Smith §1)",
      "进程虚拟机与系统虚拟机 (Smith §2-3)",
      "二进制翻译与动态翻译 (Smith Ch.2)",
      "高级语言虚拟机（JVM/.NET CLR） (Smith Ch.5-6)",
      "Hypervisor 类型与架构 (Tanenbaum §7.7)",
      "CPU 虚拟化与陷入模拟 (Smith §3.3)",
      "内存虚拟化与影子页表 (Smith §3.4)",
      "半虚拟化与硬件内存虚拟化 EPT/NPT (Smith Ch.8)",
      "I/O 虚拟化与设备模拟 (Smith §3.5)",
      "I/O 虚拟化 SR-IOV / VT-d 直通 (Tanenbaum §7.7)",
      "硬件辅助虚拟化 Intel VT-x (Tanenbaum §7.7.2)",
      "容器与操作系统级虚拟化 (Tanenbaum §7.8)"
    ],
  },
  'cs/computer-systems-integrated': {
    title: "计算机系统综合（CS:APP 视角）",
    books: [
          "Bryant & O'Hallaron, \"Computer Systems: A Programmer's Perspective\" (3rd, 2015)"
    ],
    chapters: [
      "信息表示与位运算 (CS:APP §2)",
      "程序的机器级表示 (CS:APP §3)",
      "处理器体系结构 (CS:APP §4)",
      "优化程序性能 (CS:APP §5)",
      "存储器层次结构 (CS:APP §6)",
      "链接与加载 (CS:APP §7)",
      "异常控制流 (CS:APP §8)",
      "系统级 I/O 与网络编程 (CS:APP §10-11)",
      "并发编程 (CS:APP §12)",
      "虚拟内存与地址翻译 (CS:APP §9)",
      "系统级 I/O 深入 (CS:APP §10)"
    ],
  },
  'cs/system-security': {
    title: "系统安全",
    books: [
          "Wenliang Du, \"Computer & Internet Security: A Hands-On Approach\" (2nd, 2019)",
          "William Stallings & Lawrie Brown, \"Computer Security: Principles and Practice\" (4th, 2018)",
          "Dieter Gollmann, \"Computer Security\" (3rd, 2011)"
    ],
    chapters: [
      "安全模型与访问控制 (Stallings §4)",
      "身份认证机制（口令/生物识别/MFA） (Stallings §3)",
      "认证与授权（Kerberos/OAuth/SAML） (Stallings §3-4)",
      "密码学基础与 PKI (Stallings §2)",
      "恶意代码与 Rootkit (Stallings §6)",
      "拒绝服务 DoS 与防护 (Stallings §7)",
      "入侵检测与防御 IDS/IPS (Stallings §8)",
      "网络安全与防火墙 (Stallings §9)",
      "缓冲区溢出与栈破坏 (Stallings §10)",
      "内存保护机制（ASLR/DEP/CFI） (Stallings §10)",
      "模糊测试（Fuzzing） (Stallings §11)",
      "Web 安全（SQLi/XSS/CSRF） (Du §10-12)",
      "漏洞利用与缓解技术 (Du §4-5)",
      "操作系统安全加固 (Gollmann §4)",
      "可信计算与硬件安全 (Gollmann §12)"
    ],
  },
  'cs/embedded-systems': {
    title: "嵌入式系统",
    books: [
          "Marilyn Wolf, \"Computers as Components: Principles of Embedded Computing System Design\" (4th, 2017)",
          "Steve Heath, \"Embedded Systems Design\" (2nd, 2002)",
          "Jack Ganssle, \"Embedded Systems: World Class Designs\" (2008)"
    ],
    chapters: [
      "嵌入式处理器与架构 (Wolf §2)",
      "嵌入式存储与外设 (Wolf §4)",
      "实时操作系统与调度 (Wolf §6)",
      "实时调度理论：RM/EDF 与可调度性分析 (Wolf §6)",
      "嵌入式 Linux 与驱动 (Heath Ch.7)",
      "功耗管理与低功耗设计 (Wolf §3-4)",
      "嵌入式通信总线 (Wolf §4)",
      "嵌入式程序设计：编译/性能与功耗分析/验证 (Wolf §5)",
      "硬件/软件协同设计 (Wolf §7)",
      "嵌入式多处理器与 SoC（AMP/SMP） (Wolf §10)",
      "嵌入式系统验证 (Ganssle Ch.10)",
      "应用领域：汽车/航空航天/IoT (Wolf §8-9)"
    ],
  },
  'cs/internet-of-things': {
    title: "物联网",
    books: [
          "Hossam Hassanein & Atif Alamri, \"Internet of Things: A Challenge and Opportunity\" (2017)",
          "Ovidiu Vermesan & Peter Friess, \"Internet of Things: Converging Technologies for Smart Environments\" (2013)",
          "Baher Zuhair, \"Internet of Things: Architecture, Protocols and Use Cases\" (2014)"
    ],
    chapters: [
      "IoT 体系架构与参考模型 (Vermesan Ch.2)",
      "传感器与感知层 (Hassanein §3)",
      "无线通信协议 ZigBee/LoRa/NB-IoT (Vermesan Ch.3)",
      "应用层协议 MQTT/CoAP/HTTP (Bahga & Madisetti, 书目外)",
      "IoT 操作系统与低功耗 MCU 平台（RIOT/Contiki/FreeRTOS） (书目外)",
      "RFID/NFC 与标识解析 (书目外)",
      "边缘智能与雾计算 (Hassanein §6)",
      "IoT 数据管理与云平台 (Zuhair Ch.5)",
      "物联网安全与隐私 (Vermesan Ch.6)",
      "智能城市与工业 IoT 应用 (Hassanein §8)",
      "IoT 互操作性与标准化 (Zuhair Ch.7)"
    ],
  },
  'cs/container-cloud-native': {
    title: "容器与云原生",
    books: [
          "Brendan Burns, Joe Beda & Kelsey Hightower, \"Kubernetes: Up & Running\" (3rd, 2022)",
          "Brendan Burns, \"Designing Distributed Systems\" (2018)",
          "Lee Atchison, \"Architecting for the Cloud\" (2020)"
    ],
    chapters: [
      "容器原理与 namespaces/cgroups (Burns Ch.2)",
      "Docker 镜像与运行时 (Burns Ch.2)",
      "Kubernetes 集群架构与控制平面（etcd/API Server/Scheduler） (Burns Ch.3)",
      "Kubernetes 对象模型 (Burns Ch.6)",
      "Pod 与服务发现 (Burns Ch.7)",
      "控制器与声明式 API (Burns Ch.9)",
      "弹性伸缩 HPA/VPA (Burns Ch.10)",
      "服务网格 Istio (Atchison §8)",
      "云原生 12 因素应用 (Atchison §3)",
      "分布式系统设计模式 (Burns \"Designing\" Ch.2)",
      "持久化存储与 CSI (Burns Ch.16)",
      "网络模型（CNI/Service Mesh） (Burns Ch.15)",
      "Ingress 与流量管理/负载均衡 (Burns Ch.8)",
      "安全（RBAC/NetworkPolicy/Secrets） (Burns Ch.14,19)",
      "Helm 与配置管理（ConfigMap/Secret） (Burns Ch.13)",
      "可观测性（Prometheus/OpenTelemetry） (书目外)"
    ],
  },
  'cs/microservices': {
    title: "微服务架构",
    books: [
          "Sam Newman, \"Building Microservices\" (2nd, 2021)",
          "Chris Richardson, \"Microservices Patterns\" (2018)",
          "Martin Fowler & James Lewis, \"Microservices: A Definition of This New Architectural Term\" (2014)"
    ],
    chapters: [
      "微服务设计原则 (Newman Ch.1)",
      "服务拆分与边界 (Newman Ch.2)",
      "API 网关模式 (Richardson Ch.8)",
      "服务发现与注册 (Richardson Ch.4)",
      "Saga 分布式事务模式 (Richardson Ch.4)",
      "事件驱动架构与 CQRS (Richardson Ch.7)",
      "微服务测试策略 (Newman Ch.7)",
      "微服务安全（认证/授权/边界） (Newman Ch.9)",
      "分布式追踪与可观测性 (Newman Ch.8)",
      "CAP 与最终一致性 (Newman Ch.11)",
      "微服务部署与监控 (Newman Ch.12)",
      "弹性模式（熔断/舱壁/重试） (Newman §11)",
      "每服务数据库与数据一致性 (Newman §4)",
      "契约测试与 API 版本管理 (Newman §8)"
    ],
  },
  'cs/big-data-systems': {
    title: "大数据系统",
    books: [
          "Tom White, \"Hadoop: The Definitive Guide\" (4th, 2015)",
          "Bill Chambers & Matei Zaharia, \"Spark: The Definitive Guide\" (2018)",
          "Nathan Marz & James Warren, \"Big Data: Principles and Best Practices of Scalable Realtime Data Systems\" (2015)"
    ],
    chapters: [
      "MapReduce 计算模型 (White Ch.2)",
      "HDFS 分布式存储 (White Ch.3)",
      "YARN 资源调度 (White Ch.4)",
      "序列化与列式存储（Avro/Parquet/ORC） (White §5,12-13)",
      "Spark RDD 与 DAG (Zaharia Ch.2)",
      "Spark SQL 与 DataFrame (Zaharia Ch.9)",
      "流处理与 Structured Streaming (Zaharia Ch.20)",
      "Lambda 架构与批流一体 (Marz Ch.1)",
      "数据仓库与 SQL-on-Hadoop（Hive） (White Ch.17)",
      "消息队列（Kafka） (书目外)",
      "分布式协调 ZooKeeper (White Ch.21)",
      "流处理（Flink） (书目外)",
      "NoSQL（Cassandra/MongoDB/HBase） (White Ch.20)"
    ],
  },
  'cs/devops-sre': {
    title: "DevOps 与 SRE",
    books: [
          "Betsy Beyer et al., \"Site Reliability Engineering: How Google Runs Production Systems\" (2016)",
          "Betsy Beyer et al., \"The Site Reliability Workbook\" (2018)",
          "Gene Kim et al., \"The DevOps Handbook\" (2nd, 2021)"
    ],
    chapters: [
      "SRE 原则与角色 (SRE Book Ch.1)",
      "SLI/SLO/SLA 指标体系 (SRE Book Ch.2)",
      "风险接纳与错误预算 (SRE Book Ch.3-4)",
      "消除工作量与 Toil (SRE Book Ch.5)",
      "自动化与简单性 (SRE Book Ch.7/9)",
      "事件管理与事后复盘 (SRE Book Ch.12)",
      "告警与 On-Call 工程 (SRE Book Ch.10-11)",
      "容量规划与负载测试 (SRE Book Ch.18)",
      "变更管理与发布工程 (SRE Book Ch.8)",
      "DevOps 三步工作法 (DevOps Handbook §1)",
      "混沌工程与弹性测试 (SRE Workbook Ch.9)",
      "CI/CD 流水线（Jenkins/GitLab/ArgoCD） (Beyer §16)",
      "基础设施即代码（Terraform/Ansible） (Beyer §14)",
      "监控与可观测性（Prometheus/Grafana/OpenTelemetry） (Beyer §6)"
    ],
  },
  'cs/edge-computing': {
    title: "边缘计算",
    books: [
          "Weisong Shi & Schahram Dustdar, \"Edge Computing\" (2020)",
          "Abdelrahman Osama et al., \"Edge Computing: Models, Technologies, and Applications\" (2020)",
          "Yang Yang et al., \"Edge Computing: A Primer\" (2019)"
    ],
    chapters: [
      "边缘计算定义与架构 (Shi Ch.1)",
      "移动边缘计算 MEC (Shi Ch.4)",
      "5G 与边缘协同（MEC 网络切片） (书目外)",
      "边缘 AI 与模型部署 (Shi Ch.7)",
      "边缘推理与模型轻量化（TensorRT/ONNX/量化/剪枝） (书目外)",
      "计算卸载与任务调度 (Osama §3)",
      "边缘存储与缓存 (Shi Ch.5)",
      "边缘安全与隐私 (Yang Ch.6)",
      "边缘计算平台与编排（KubeEdge/EdgeX/OpenYurt） (书目外)",
      "车联网与工业边缘应用 (Shi Ch.8)",
      "云-边-端协同 (Shi Ch.9)"
    ],
  },
  'cs/digital-twin': {
    title: "数字孪生",
    books: [
          "Fei Tao, Meng Zhang & A. Y. C. Nee, \"Digital Twin Driven Smart Manufacturing\" (2019)",
          "Wolfgang Kritzinger et al., \"Digital Twin in Manufacturing: A Categorical Literature Review\" (2018)",
          "Antariksh Dutta et al., \"Digital Twin: Enabling Technologies, Challenges and Open Research\" (2020)"
    ],
    chapters: [
      "数字孪生概念与建模 (Tao Ch.1)",
      "数字孪生使能技术栈（IoT/5G/AI/仿真建模） (Tao Ch.2)",
      "虚实映射与同步机制 (Tao Ch.3)",
      "数字孪生驱动的智能制造 (Tao Ch.4)",
      "实时数据采集与互联 (Tao Ch.5)",
      "仿真验证与预测维护 (Tao Ch.7)",
      "工业 4.0 与数字孪生应用 (Tao Ch.8)",
      "跨领域应用：城市/能源/交通数字孪生 (书目外)",
      "数字孪生平台架构 (Kritzinger §4)",
      "数字孪生标准与成熟度评估（Kritzinger 分类体系延伸） (Kritzinger §2)",
      "开放挑战与未来方向 (Dutta §5)"
    ],
  },
  'advanced/statistical-learning': {
    title: "统计学习方法",
    books: [
          "Trevor Hastie, Robert Tibshirani & Jerome Friedman, \"The Elements of Statistical Learning\" (2nd, 2009)",
          "Gareth James et al., \"An Introduction to Statistical Learning\" (2nd, 2021)",
          "Christopher M. Bishop, \"Pattern Recognition and Machine Learning\" (2006)"
    ],
    chapters: [
      "线性回归与最小二乘 (ESL §3)",
      "线性分类与逻辑回归 (ESL §4)",
      "线性判别分析（LDA）与感知机 (ESL §4.3)",
      "正则化与 Lasso/Ridge (ESL §3.4)",
      "基展开与样条 (ESL §5)",
      "高斯过程 (ESL §5.8)",
      "核方法与 SVM (ESL §12)",
      "决策树与随机森林 (ESL §9)",
      "神经网络基础 (ESL §11)",
      "集成学习与 Boosting (ESL §10)",
      "朴素贝叶斯与贝叶斯网络 (ESL §6.6.3, §17.1)",
      "EM 算法与高斯混合模型 (ESL §8.5)",
      "主成分分析（PCA）与降维 (ESL §14.5)",
      "聚类方法（K-means/层次聚类） (ESL §14.3)",
      "隐马尔可夫模型（HMM） (ESL §17.2)",
      "图模型与条件随机场 CRF (ESL §17)",
      "高维问题与多重比较 (ESL §18)"
    ],
  },
  'advanced/optimization-algorithms': {
    title: "优化算法（梯度下降族）",
    books: [
          "Jorge Nocedal & Stephen J. Wright, \"Numerical Optimization\" (2nd, 2006)",
          "Stephen Boyd & Lieven Vandenberghe, \"Convex Optimization\" (2004)",
          "Dimitri P. Bertsekas, \"Nonlinear Programming\" (3rd, 2016)"
    ],
    chapters: [
      "凸优化基础 (Boyd §4)",
      "梯度下降与线搜索 (Nocedal §3)",
      "牛顿法与拟牛顿 (Nocedal §6)",
      "信赖域方法 (Nocedal §4)",
      "共轭梯度法 (Nocedal §5)",
      "非线性最小二乘与 Levenberg-Marquardt (Nocedal §10.3)",
      "随机梯度下降与方差减小 (Bertsekas §2)",
      "Adam 与自适应优化器 (Goodfellow §8.5)",
      "对偶理论与 KKT (Boyd §5)",
      "次梯度与近端方法 (Boyd §5)",
      "罚函数与增广拉格朗日方法 (Nocedal §17)",
      "序列二次规划 SQP (Nocedal §18)",
      "内点法 (Boyd §11)"
    ],
  },
  'advanced/representation-learning': {
    title: "表示学习",
    books: [
          "Ian Goodfellow, Yoshua Bengio & Aaron Courville, \"Deep Learning\" (2016)",
          "Yoshua Bengio, Aaron Courville & Pascal Vincent, \"Representation Learning: A Review and New Perspectives\" (IEEE TPAMI 2013)",
          "Goodfellow & Bengio, \"Deep Learning\" Ch.15 (2016)"
    ],
    chapters: [
      "表示学习导论 (Goodfellow §15)",
      "分布式表示与因子解耦基础 (Bengio Review §2-3)",
      "自编码器与稀疏编码 (Goodfellow §14)",
      "变分自编码器 VAE (Goodfellow §20.10)",
      "解纠缠表示与 β-VAE (Higgins et al., β-VAE 2017)",
      "对比学习与 InfoNCE (van den Oord et al., CPC 2018)",
      "流形学习假设 (Goodfellow §5.11)",
      "生成对抗表示 (Goodfellow §20.4)",
      "表征评估与线性探测 (Bengio Review §7)",
      "深度聚类表示（DeepCluster） (Caron et al., 2018)"
    ],
  },
  'advanced/self-supervised-learning': {
    title: "自监督学习",
    books: [
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Liu et al., \"Self-Supervised Learning: Generative or Contrastive\" (IEEE TNNLS 2023)",
          "Liu Jing & Tian, \"Self-Supervised Visual Feature Learning with Deep Neural Networks: A Survey\" (IEEE TPAMI 2021)"
    ],
    chapters: [
      "自监督预训练任务 (Liu et al., TNNLS 2023 §2)",
      "对比学习框架 SimCLR/MoCo (Liu et al., TNNLS 2023 §3)",
      "对比损失与 InfoNCE (Liu et al., TNNLS 2023 §3.2)",
      "掩码图像建模 MAE (Liu et al., TNNLS 2023 §4)",
      "BYOL 与负样本免方法 (Jing & Tian, TPAMI 2021 §5)",
      "自蒸馏机制 BYOL/SimSiam 原理细化 (Grill et al., BYOL 2020; Chen & He, SimSiam 2021)",
      "生成式自监督主线（GPT 自回归/掩码语言建模） (Radford et al., GPT 2018; Devlin et al., BERT 2018)",
      "语言自监督 MLM (Devlin et al., BERT 2018)",
      "视觉自监督评估协议 (Jing & Tian, TPAMI 2021 §7)",
      "多模态自监督 CLIP (Liu et al., TNNLS 2023 §6)"
    ],
  },
  'advanced/transfer-meta-learning': {
    title: "迁移学习与元学习",
    books: [
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Timothy Hospedales et al., \"Meta-Learning in Neural Networks: A Survey\" (IEEE TPAMI 2022)",
          "Sinno Jialin Pan & Qiang Yang, \"A Survey on Transfer Learning\" (IEEE TKDE 2010)"
    ],
    chapters: [
      "迁移学习分类与定义 (Pan & Yang, Survey 2010 §2)",
      "域适应与域对抗 (Pan & Yang, Survey 2010 §4)",
      "微调与特征迁移 (Goodfellow §5.4)",
      "元学习框架 (Hospedales et al., TPAMI 2022 §3)",
      "基于度量的 ProtoNet/MatchingNet (Hospedales et al., TPAMI 2022 §4)",
      "基于优化的 MAML (Hospedales et al., TPAMI 2022 §5)",
      "基于模型的元学习 (Hospedales et al., TPAMI 2022 §6)",
      "少样本学习应用 (Hospedales et al., TPAMI 2022 §8)",
      "零样本学习与开放世界识别 (Hospedales et al., TPAMI 2022 §8)",
      "域泛化 Domain Generalization (Wang et al., DG Survey 2022)"
    ],
  },
  'advanced/federated-learning': {
    title: "联邦学习",
    books: [
          "Qiang Yang, Yang Liu & Tianjian Chen, \"Federated Learning\" (Synthesis Lectures on AI and ML 2021)",
          "Brendan McMahan et al., \"Communication-Efficient Learning of Deep Networks from Decentralized Data\" (AISTATS 2017)",
          "Peter Kairouz et al., \"Advances and Open Problems in Federated Learning\" (Foundations and Trends in ML 2021)"
    ],
    chapters: [
      "联邦学习框架与分类 (Yang et al., Ch.1)",
      "联邦平均 FedAvg (McMahan et al., 2017 §3)",
      "异构数据与非 IID (Kairouz et al., 2021 §2)",
      "联邦优化算法 FedProx/SCAFFOLD (Kairouz et al., 2021 §3)",
      "通信效率与梯度压缩/量化 (Kairouz et al., 2021 §3)",
      "差分隐私保护 (Kairouz et al., 2021 §4)",
      "安全聚合协议 (Kairouz et al., 2021 §5)",
      "客户端选择与激励 (Yang et al., Ch.4)",
      "纵向联邦与联邦迁移 (Yang et al., Ch.5)",
      "联邦学习的公平性 (Kairouz et al., 2021 §6)",
      "联邦学习系统部署 (Kairouz et al., 2021 §8)"
    ],
  },
  'advanced/causal-inference': {
    title: "因果推断与因果学习",
    books: [
          "Judea Pearl, \"Causality: Models, Reasoning, and Inference\" (2nd, 2009)",
          "Miguel A. Hernán & James M. Robins, \"Causal Inference: What If\" (2020)",
          "Jonas Peters, Dominik Janzing & Bernhard Schölkopf, \"Elements of Causal Inference\" (2017)"
    ],
    chapters: [
      "因果图与有向无环图 (Pearl, Causality §1)",
      "do 演算与干预 (Pearl, Causality §3)",
      "潜在结果框架 (Hernán & Robins §1)",
      "混杂控制与倾向评分 (Hernán & Robins §12)",
      "工具变量与 IV 估计 (Hernán & Robins §16)",
      "中介分析与直接/间接效应 (Pearl, Causality §4; Hernán & Robins Ch.6)",
      "反事实推理 (Pearl, Causality §7)",
      "结构因果模型 SCM (Pearl, Causality §2)",
      "因果发现算法 (Peters et al., Ch.4)",
      "因果强化学习 (Zhang & Bareinboim, 2016)"
    ],
  },
  'advanced/model-evaluation': {
    title: "模型评估与选择",
    books: [
          "Trevor Hastie et al., \"The Elements of Statistical Learning\" (2nd, 2009)",
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Kevin P. Murphy, \"Machine Learning: A Probabilistic Perspective\" (2012)"
    ],
    chapters: [
      "交叉验证与重抽样 (ESL §7.10)",
      "偏差-方差分解 (ESL §7.3)",
      "模型选择准则 AIC/BIC (Murphy §5.3)",
      "统计检验与置信区间 (Murphy §6)",
      "ROC 与 AUC 评估 (ESL §4.4.5; Murphy §4.6)",
      "代价敏感学习与混淆矩阵 (ESL §9.3)",
      "超参数调优方法 (Goodfellow §5.3)",
      "学习曲线与容量分析 (Goodfellow §5.2)",
      "误差分析与消融实验（可复现性） (Goodfellow §11)",
      "数据泄漏与评估陷阱 (Kaufmann et al., Leakage 2012)"
    ],
  },
  'advanced/decision-transformer': {
    title: "Decision Transformer（序列建模 RL）",
    books: [
          "Richard S. Sutton & Andrew G. Barto, \"Reinforcement Learning: An Introduction\" (2nd, 2018)",
          "Lili Chen et al., \"Decision Transformer: Reinforcement Learning via Sequence Modeling\" (NeurIPS 2021)",
          "Sergey Levine et al., \"Offline Reinforcement Learning: Tutorial and Review\" (2020)"
    ],
    chapters: [
      "MDP 与序列建模对比 (Sutton & Barto §3)",
      "Return-to-go 表示 (Chen et al., Decision Transformer §3)",
      "因果自注意力 Transformer (Chen et al., Decision Transformer §3)",
      "离线 RL 与数据分布 (Levine et al., Offline RL Tutorial §3)",
      "轨迹采样策略 (Chen et al., Decision Transformer §4)",
      "与 Q-learning 对比 (Chen et al., Decision Transformer §5)",
      "传统离线 RL 基线 BCQ/CQL/IQL (Fujimoto et al., 2019; Kumar et al., 2020; Kostrikov et al., 2022)",
      "Return-conditioned 监督学习范式（行为克隆视角） (Emmons et al., RvS 2022)",
      "Trajectory Transformer 与其他序列建模 RL (Janner et al., Trajectory Transformer 2021)",
      "Online Decision Transformer 扩展 (Zheng et al., 2022)",
      "序列建模 RL 局限性 (Chen et al., Decision Transformer §6)"
    ],
  },
  'advanced/nlp-syntax-semantics': {
    title: "NLP · 句法与语义分析",
    books: [
          "Daniel Jurafsky & James H. Martin, \"Speech and Language Processing\" (3rd, draft 2024)",
          "Christopher D. Manning & Hinrich Schütze, \"Foundations of Statistical Natural Language Processing\" (1999)",
          "Jacob Eisenstein, \"Introduction to Natural Language Processing\" (2019)"
    ],
    chapters: [
      "词法分析与分词 (Jurafsky §2)",
      "N-gram 语言模型 (Jurafsky §3)",
      "隐马尔可夫与词性标注 (Manning & Schütze §10)",
      "上下文无关文法与 CFG (Jurafsky §13)",
      "成分句法分析（CKY 算法） (Jurafsky §13)",
      "概率上下文无关文法 PCFG 与概率句法分析 (Jurafsky §14)",
      "依存句法分析 (Jurafsky §18)",
      "语义角色标注 (Jurafsky §22)",
      "词义消歧 WSD (Manning & Schütze §7)",
      "词汇语义与词汇资源（WordNet/FrameNet） (Manning & Schütze §19)",
      "分布语义与词嵌入 (Eisenstein §14)",
      "命名实体识别（NER） (Jurafsky §8)",
      "共指消解 (Jurafsky §21)"
    ],
  },
  'advanced/machine-translation': {
    title: "NLP · 机器翻译",
    books: [
          "Daniel Jurafsky & James H. Martin, \"Speech and Language Processing\" (3rd, draft 2024)",
          "Philipp Koehn, \"Neural Machine Translation\" (2nd, 2020)",
          "Philipp Koehn, \"Statistical Machine Translation\" (2010)"
    ],
    chapters: [
      "统计机器翻译与 IBM 模型 (Koehn SMT Ch.4)",
      "词对齐模型 (Koehn SMT Ch.5)",
      "短语翻译模型 (Koehn SMT Ch.6)",
      "神经机器翻译 Encoder-Decoder (Koehn NMT Ch.5)",
      "注意力机制 (Bahdanau et al., 2015; Vaswani et al., 2017)",
      "Transformer 完整结构（自注意力/多头/位置编码） (Vaswani et al., 2017)",
      "子词 BPE 与词汇表 (Koehn NMT Ch.7)",
      "解码策略与束搜索 (Koehn NMT Ch.6)",
      "BLEU 与翻译评估 (Papineni et al., BLEU 2002; Koehn NMT Ch.9)",
      "反向翻译与数据增强 (Koehn NMT Ch.9)",
      "多语言机器翻译与零资源 MT (Liu et al., mBART 2020; Xue et al., mT5 2021)"
    ],
  },
  'advanced/dialogue-systems': {
    title: "NLP · 对话系统",
    books: [
          "Daniel Jurafsky & James H. Martin, \"Speech and Language Processing\" (3rd, draft 2024)",
          "Jianfeng Gao et al., \"Neural Approaches to Conversational AI\" (Foundations and Trends 2019)",
          "Gokhan Tur & Renato De Mori, \"Spoken Dialog Systems\" (2011)"
    ],
    chapters: [
      "对话系统分类 (Jurafsky §24)",
      "意图识别与槽位填充 (Gao et al., 2019 §2)",
      "对话状态跟踪 (Tur & De Mori Ch.5)",
      "对话策略学习 (Tur & De Mori Ch.6)",
      "对话策略的强化学习训练 (Williams & Young, 2007)",
      "自然语言生成 NLG (Jurafsky §24.2)",
      "检索式问答系统 (Jurafsky §23)",
      "生成式闲聊模型 (Gao et al., 2019 §4)",
      "端到端神经对话系统 (Gao et al., 2019 §5)",
      "对话系统评测（自动指标/人工评估） (Jurafsky §24.5)"
    ],
  },
  'advanced/detection-segmentation': {
    title: "CV · 目标检测与图像分割",
    books: [
          "Richard Szeliski, \"Computer Vision: Algorithms and Applications\" (2nd, 2022)",
          "David A. Forsyth & Jean Ponce, \"Computer Vision: A Modern Approach\" (3rd, 2012)",
          "Jian Yang et al., \"Deep Learning for Computer Vision\" (2021)"
    ],
    chapters: [
      "滑动窗口与候选区域 (Szeliski §6)",
      "HOG 特征与 DPM (Forsyth & Ponce §16)",
      "R-CNN 系列 (Girshick et al., R-CNN 2014; Szeliski §6)",
      "YOLO 单阶段检测 (Yang et al., DLCV §6)",
      "FCN 语义分割 (Long et al., FCN 2015; Szeliski §6)",
      "U-Net 编码解码结构 (Yang et al., DLCV §8)",
      "Mask R-CNN 实例分割 (He et al., Mask R-CNN 2017)",
      "全景分割 Panoptic (Yang et al., DLCV §9)",
      "无锚框检测器（CenterNet/FCOS） (Duan et al., CenterNet 2019; Tian et al., FCOS 2019)",
      "DETR 与 Transformer 检测 (Carion et al., DETR 2020)",
      "3D 检测与 BEV 感知 (Lang et al., PointPillars 2019; Huang et al., BEVFormer 2022)",
      "检测评估指标与损失（mAP/IoU/Focal Loss） (Lin et al., Focal Loss 2017)",
      "传统图像分割基础（图割/超像素/均值漂移） (Boykov & Jolly 2001; Achanta et al., SLIC 2012; Comaniciu & Meer 2002)"
    ],
  },
  'advanced/video-understanding': {
    title: "CV · 视频理解",
    books: [
          "Richard Szeliski, \"Computer Vision: Algorithms and Applications\" (2nd, 2022)",
          "Limin Wang et al., \"Video Understanding: A Survey\" (IJCV 2024)",
          "S. Ji et al., \"3D Convolutional Neural Networks for Human Action Recognition\" (IEEE TPAMI 2013)"
    ],
    chapters: [
      "视频时序建模 (Wang Survey §2)",
      "3D 卷积 C3D/I3D (Ji §3)",
      "双流网络与光流 (Wang Survey §3)",
      "视频动作识别时序网络 (Szeliski §11.6)",
      "时空注意力机制 (Wang Survey §5)",
      "视频目标分割 (Szeliski §11.7)",
      "视频密集预测与跟踪 (Wang Survey §6)",
      "视频-语言多模态理解 (Wang Survey §8)",
      "视频数据集与评测基准（Kinetics/UCF101） (Kay et al., Kinetics 2017; Soomro et al., UCF101 2012)",
      "视频超分与视频生成 (Vondrick et al., 2016; OpenAI Sora 2024)"
    ],
  },
  'advanced/asr-tts': {
    title: "语音 · ASR 与 TTS",
    books: [
          "Lawrence R. Rabiner & Ronald W. Schafer, \"Theory and Applications of Digital Speech Processing\" (2011)",
          "Xuedong Huang, Alex Acero & Hsiao-Wuen Hon, \"Spoken Language Processing\" (2001)",
          "Dong Yu & Li Deng, \"Automatic Speech Recognition: A Deep Learning Approach\" (2015)"
    ],
    chapters: [
      "语音前端与端点检测（VAD） (Rabiner §9)",
      "声学特征 MFCC/Filterbank (Rabiner §5)",
      "隐马尔可夫模型 HMM (Huang §8)",
      "CTC 损失与端到端 (Yu & Deng §7)",
      "声学模型 DNN/TDNN (Yu & Deng §6)",
      "语言模型与解码 (Huang §13)",
      "RNN-T（RNN Transducer） (Graves, RNN-T 2012)",
      "端到端 ASR Conformer (Gulati et al., Conformer 2020)",
      "大规模 ASR（Whisper） (Radford et al., Whisper 2022)",
      "Tacotron 端到端 TTS (Shen et al., Tacotron 2018)",
      "声码器 WaveNet/HiFi-GAN (Oord et al., WaveNet 2016; Kong et al., HiFi-GAN 2020)",
      "语音增强与分离 (Huang §15)",
      "识别评测指标（WER/CER 与对齐） (Yu & Deng §3)"
    ],
  },
  'advanced/knowledge-graph': {
    title: "知识图谱",
    books: [
          "Aidan Hogan et al., \"Knowledge Graphs\" (ACM Computing Surveys 2021)",
          "Shaoxiong Ji et al., \"A Survey on Knowledge Graphs: Representation, Acquisition, and Applications\" (IEEE TNNLS 2022)",
          "Mayank Kejriwal, \"Knowledge Graphs: A Practical Primer\" (2021)"
    ],
    chapters: [
      "知识图谱表示与 RDF/OWL (Hogan §2)",
      "实体识别与链接 (Ji §3)",
      "关系抽取与远程监督 (Ji §4)",
      "实体对齐与知识融合 (Ji §6)",
      "知识图谱嵌入 TransE/RotatE (Ji §5)",
      "链接预测评估指标（Hits@K/MRR） (Ji §5; Bordes et al., TransE 2013)",
      "知识推理与规则 (Hogan §7)",
      "知识问答 QA (Kejriwal Ch.6)",
      "本体与本体构建 (Hogan §8)",
      "知识图谱应用 (Ji §7)"
    ],
  },
  'advanced/time-series': {
    title: "时间序列分析",
    books: [
          "George E. P. Box et al., \"Time Series Analysis: Forecasting and Control\" (5th, 2015)",
          "James D. Hamilton, \"Time Series Analysis\" (1994)",
          "Rob J. Hyndman & George Athanasopoulos, \"Forecasting: Principles and Practice\" (3rd, 2021)"
    ],
    chapters: [
      "ARIMA 模型 (Box §3)",
      "季节性分解与 SARIMA (Box §9)",
      "平稳性与单位根检验 (Hamilton §15)",
      "协整与误差修正 (Hamilton §19)",
      "向量自回归 VAR 与多元时间序列 (Hamilton Ch.11)",
      "GARCH 波动率建模 (Hamilton §21)",
      "状态空间与卡尔曼滤波 (Hamilton Ch.13)",
      "指数平滑与 ETS (Hyndman §8)",
      "预测评估指标与预测区间（MASE/RMSE） (Hyndman §3, §5)",
      "深度学习时序 LSTM/Transformer (Hyndman §12)"
    ],
  },
  'advanced/llm-pretraining': {
    title: "大模型预训练",
    books: [
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Alec Radford et al., \"Language Models are Unsupervised Multitask Learners\" (GPT-2, 2019)",
          "Hugo Touvron et al., \"LLaMA: Open and Efficient Foundation Language Models\" (2023)"
    ],
    chapters: [
      "Transformer 架构 (Vaswani et al., 2017)",
      "自回归预训练目标 (Radford et al., GPT-2 2019 §2)",
      "Scaling Laws (Kaplan et al., 2020; Hoffmann et al., Chinchilla 2022)",
      "预训练数据构造 (Touvron et al., LLaMA §2.1)",
      "位置编码 RoPE (Touvron et al., LLaMA §2.2)",
      "RMSNorm 与 SwiGLU (Touvron et al., LLaMA §2.2)",
      "高效训练与并行 (Touvron et al., LLaMA §2.4)",
      "训练稳定性与损失尖峰（loss spike） (Chowdhery et al., PaLM 2022)",
      "评估基准与 zero-shot (Radford et al., GPT-2 2019 §3)",
      "分词算法（BPE/SentencePiece/Unigram） (Sennrich et al., BPE 2016)",
      "模型初始化与正则化（Xavier/He init） (Goodfellow §8.4)",
      "优化器选择与学习率调度（AdamW/Cosine） (Goodfellow §8.5)",
      "检查点保存与断点续训 (Megatron-LM §4)"
    ],
  },
  'advanced/llm-alignment': {
    title: "对齐技术（RLHF / DPO）",
    books: [
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Long Ouyang et al., \"Training Language Models to Follow Instructions with Human Feedback\" (InstructGPT, 2022)",
          "Rafael Rafailov et al., \"Direct Preference Optimization: Your Language Model is Secretly a Reward Model\" (NeurIPS 2023)"
    ],
    chapters: [
      "SFT 监督微调 (Ouyang et al., InstructGPT 2022 §3)",
      "RLHF 框架流程 (Ouyang et al., InstructGPT 2022 §2)",
      "偏好数据构建与标注 (Ouyang et al., InstructGPT 2022 §2)",
      "奖励模型训练 (Ouyang et al., InstructGPT 2022 §4)",
      "奖励黑客与过度优化（reward hacking） (Gao et al., Reward Model Overoptimization 2023)",
      "PPO 强化学习优化 (Ouyang et al., InstructGPT 2022 §5)",
      "DPO 直接偏好优化 (Rafailov et al., DPO 2023 §3)",
      "宪法 AI 与 RLAIF (Bai et al., Constitutional AI 2022)",
      "对齐税与平衡 (Ouyang et al., InstructGPT 2022 §6)",
      "安全对齐与红队 (Ouyang et al., InstructGPT 2022 §7)"
    ],
  },
  'advanced/prompt-engineering': {
    title: "提示工程",
    books: [
          "Daniel Jurafsky & James H. Martin, \"Speech and Language Processing\" (3rd, draft 2024)",
          "Jason Wei et al., \"Chain-of-Thought Prompting Elicits Reasoning in Large Language Models\" (NeurIPS 2022)",
          "Shunyu Yao et al., \"ReAct: Synergizing Reasoning and Acting in Language Models\" (ICLR 2023)"
    ],
    chapters: [
      "Zero-shot/Few-shot 提示 (Jurafsky §13.3)",
      "思维链 Chain-of-Thought (Wei et al., CoT 2022)",
      "思维树 Tree-of-Thought (Yao et al., ToT 2023)",
      "自洽性 Self-Consistency (Wang et al., 2022)",
      "复杂推理分解 Least-to-Most (Zhou et al., 2022)",
      "ReAct 推理与行动 (Yao et al., ReAct 2023)",
      "提示模板与角色扮演 (Jurafsky §13.5)",
      "提示优化与自动提示搜索（AutoPrompt/OPRO） (Shin et al., AutoPrompt 2020; Yang et al., OPRO 2023)",
      "结构化输出提示（JSON/XML 输出约束） (工程实践)",
      "提示安全与注入防御 (Jurafsky §13.7)"
    ],
  },
  'advanced/mixture-of-experts': {
    title: "MoE 混合专家架构",
    books: [
          "Dmitry Lepikhin et al., \"GShard: Scaling Giant Models with Conditional Computation\" (2020)",
          "William Fedus et al., \"Switch Transformers: Scaling to Trillion Parameter Models\" (JMLR 2022)",
          "Noam Shazeer et al., \"Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer\" (ICLR 2017)"
    ],
    chapters: [
      "稀疏门控 MoE 原理 (Shazeer et al., Sparsely-Gated MoE 2017)",
      "Top-k 路由机制 (Fedus et al., Switch Transformer 2022)",
      "负载均衡损失 (Lepikhin et al., GShard 2020)",
      "负载均衡与专家退化（dead experts）机制 (Dai et al., DeepSeek-MoE 2024)",
      "专家容量因子 (Fedus et al., Switch Transformer 2022)",
      "专家并行 Expert Parallelism (Lepikhin et al., GShard 2020)",
      "路由策略对比 (Shazeer et al., 2017)",
      "DeepSeek MoE 细粒度专家 (Dai et al., DeepSeek-MoE 2024)",
      "MoE 训练稳定性 (Fedus et al., Switch Transformer 2022)",
      "MoE 推理部署与专家卸载 (Rajbhandari et al., DeepSpeed-MoE 2022)"
    ],
  },
  'advanced/long-context': {
    title: "长上下文与注意力优化",
    books: [
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Tri Dao & Daniel Y. Fu et al., \"FlashAttention: Fast and Memory-Efficient Exact Attention\" (NeurIPS 2022)",
          "Iz Beltagy et al., \"Longformer: The Long-Document Transformer\" (2020)"
    ],
    chapters: [
      "注意力计算复杂度 (Vaswani et al., 2017 §3.2)",
      "稀疏注意力 Longformer (Beltagy et al., Longformer 2020)",
      "线性注意力机制 (Katharopoulos et al., Linear Transformers 2020)",
      "FlashAttention IO 优化 (Dao et al., FlashAttention 2022)",
      "分块注意力 Blockwise (Dao et al., FlashAttention 2022)",
      "滑动窗口注意力 (Beltagy et al., Longformer 2020)",
      "位置编码外推（ALiBi/RoPE/YaRN） (Press et al., ALiBi 2022; Su et al., RoPE 2021; Peng et al., YaRN 2023)",
      "GQA/MQA 与 KV Cache 优化 (Ainslie et al., GQA 2023; Shazeer, MQA 2019)",
      "长上下文评估 LongBench (Bai et al., LongBench 2023)"
    ],
  },
  'advanced/llm-evaluation': {
    title: "大模型评测",
    books: [
          "Percy Liang et al., \"Holistic Evaluation of Language Models\" (HELM, 2022)",
          "Dan Hendrycks et al., \"Measuring Massive Multitask Language Understanding\" (MMLU, ICLR 2021)",
          "Lianmin Zheng et al., \"Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena\" (NeurIPS 2023)"
    ],
    chapters: [
      "MMLU 多任务知识评测 (Hendrycks et al., MMLU 2021)",
      "HELM 全方位评估 (Liang et al., HELM 2022)",
      "BigBench 任务基准 (Srivastava et al., BigBench 2022)",
      "人类对齐评估 (Zheng et al., MT-Bench 2023)",
      "LLM-as-a-Judge (Zheng et al., 2023)",
      "代码评测 HumanEval (Chen et al., HumanEval 2021)",
      "数学评测 GSM8K (Cobbe et al., GSM8K 2021)",
      "幻觉评测（TruthfulQA/HaluEval） (Lin et al., TruthfulQA 2022)",
      "多语言基准（C-Eval/CMMLU） (Huang et al., C-Eval 2023)",
      "智能体/工具使用评测 (Yao et al., 2023)",
      "安全与偏见评估 (Liang et al., HELM 2022)",
      "基准污染与数据泄漏评测 (Sainz et al., Contamination 2023)",
      "开放域与长上下文评测（NEEDLE） (Li et al., NEEDLE 2024; Bai et al., LongBench 2023)"
    ],
  },
  'advanced/rag': {
    title: "RAG 与检索增强",
    books: [
          "Patrick Lewis et al., \"Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks\" (NeurIPS 2020)",
          "Vladimir Karpukhin et al., \"Dense Passage Retrieval for Open-Domain Question Answering\" (EMNLP 2020)",
          "Akari Asai et al., \"Bridging the Generalization Gap in Retrieval-Augmented Generation\" (2024)"
    ],
    chapters: [
      "RAG 框架与架构 (Lewis et al., RAG 2020)",
      "向量检索与 ANN (Karpukhin et al., DPR 2020)",
      "稠密检索 DPR (Karpukhin et al., DPR 2020)",
      "重排序 Cross-Encoder (Karpukhin et al., DPR 2020)",
      "RAG-Sequence 与 RAG-Token (Lewis et al., RAG 2020)",
      "检索增强多跳推理 (Asai et al., 2024)",
      "检索增强自反 Self-RAG (Asai et al., Self-RAG 2023)",
      "RAG 评估与基准 (Asai et al., 2024)",
      "文档分块策略（固定/语义/递归分块） (Gao et al., RAG Survey 2023)",
      "嵌入模型选择与评估 (Reimers & Gurevych, SBERT 2019)",
      "向量数据库（FAISS/Milvus/Pinecone） (Johnson et al., FAISS 2017)",
      "生成融合与 RAG-Fusion (Raudaschl, RAG-Fusion 2023)",
      "查询重写与扩展 (Ma et al., 2023)",
      "混合检索（BM25+稠密混合） (Robertson & Zaragoza, BM25 2009)"
    ],
  },
  'advanced/agent-orchestration': {
    title: "AI 智能体编排",
    books: [
          "Shunyu Yao et al., \"ReAct: Synergizing Reasoning and Acting in Language Models\" (ICLR 2023)",
          "Noah Shinn et al., \"Reflexion: Language Agents with Verbal Reinforcement Learning\" (NeurIPS 2023)",
          "Zhi Xi et al., \"The Rise and Potential of Large Language Model Based Agents: A Survey\" (2024)"
    ],
    chapters: [
      "智能体架构概览 (Xi et al., Survey 2024 §2)",
      "规划与任务分解 (Yao et al., ReAct 2023)",
      "工具学习与函数调用（Toolformer） (Schick et al., Toolformer 2023)",
      "工具调用与 API 使用 (Yao et al., ReAct 2023)",
      "记忆与反思机制 (Shinn et al., Reflexion 2023)",
      "多智能体协作 (Xi et al., Survey 2024 §4)",
      "环境交互与具身智能 (Xi et al., Survey 2024 §5)",
      "智能体评估基准 (Xi et al., Survey 2024 §6)",
      "AutoGPT/MetaGPT 框架 (Xi et al., Survey 2024 §7)",
      "智能体安全与提示注入防御 (Greshake et al., 2023)"
    ],
  },
  'advanced/decision-transformer-paper': {
    title: "Decision Transformer（论文解析）",
    books: [
          "Lili Chen et al., \"Decision Transformer: Reinforcement Learning via Sequence Modeling\" (NeurIPS 2021)",
          "Richard S. Sutton & Andrew G. Barto, \"Reinforcement Learning: An Introduction\" (2nd, 2018)",
          "Sergey Levine et al., \"Offline Reinforcement Learning: Tutorial and Review\" (2020)"
    ],
    chapters: [
      "RL 序列建模范式 (Chen §1)",
      "状态-动作-回报轨迹 (Chen §3)",
      "Return-to-go 条件输入 (Chen §3.1)",
      "因果自注意力 Transformer (Chen §3.2)",
      "数据集 D4RL 评估 (Chen §4)",
      "与 Offline Q-learning 对比 (Chen §5)",
      "序列建模的优势与局限 (Chen §6)",
      "扩展工作 Online DT/Boosting (Levine §5)"
    ],
  },
  'advanced/rwkv': {
    title: "RWKV（论文解析）",
    books: [
          "Bo Peng, \"RWKV: Reinventing RNNs for the Transformer Era\" (2023)",
          "Ian Goodfellow et al., \"Deep Learning\" (2016)",
          "Ashish Vaswani et al., \"Attention Is All You Need\" (NeurIPS 2017)"
    ],
    chapters: [
      "RNN 与 Transformer 对比 (Peng §2)",
      "线性注意力机制 (Peng §3)",
      "时间混合 Time Mixing (Peng §4.1)",
      "通道混合 Channel Mixing (Peng §4.2)",
      "WKV 算子与并行训练 (Peng §5)",
      "时间衰减与位置编码 (Peng §6)",
      "模型扩展与 scaling (Peng §7)",
      "与 Transformer 的实验对比 (Peng §8)"
    ],
  },
  'advanced/gshard': {
    title: "GShard（论文解析）",
    books: [
          "Dmitry Lepikhin et al., \"GShard: Scaling Giant Models with Conditional Computation\" (2020)",
          "William Fedus et al., \"Switch Transformers\" (JMLR 2022)",
          "Noam Shazeer et al., \"Outrageously Large Neural Networks\" (ICLR 2017)"
    ],
    chapters: [
      "条件计算动机 (Lepikhin §2)",
      "MoE 层结构 (Lepikhin §3)",
      "Top-2 路由机制 (Lepikhin §3.2)",
      "辅助负载均衡损失 (Lepikhin §3.3)",
      "专家并行 Expert Parallelism (Lepikhin §4)",
      "XLA SPMD 编译 (Lepikhin §5)",
      "万亿参数模型训练 (Lepikhin §6)",
      "多语言翻译应用 (Lepikhin §7)"
    ],
  },
  'advanced/dit-sora': {
    title: "DiT / Sora（论文解析）",
    books: [
          "William Peebles & Saining Xie, \"Scalable Diffusion Models with Transformers\" (DiT, ICCV 2023)",
          "OpenAI, \"Video Generation Models as World Simulators\" (Sora, 2024)",
          "Jonathan Ho et al., \"Denoising Diffusion Probabilistic Models\" (DDPM, NeurIPS 2020)"
    ],
    chapters: [
      "扩散模型基础 DDPM (Ho §2)",
      "DiT 架构与 Patch 化 (Peebles §3)",
      "条件控制 AdaLN-Zero (Peebles §4)",
      "Scaling 与性能分析 (Peebles §5)",
      "Sora 视频生成框架 (OpenAI §2)",
      "时空 Patch 表示 (OpenAI §3)",
      "扩散 Transformer 视频 scaling (OpenAI §4)",
      "世界模拟器与世界模型 (OpenAI §5)"
    ],
  },
  'advanced/ai-chip': {
    title: "AI 芯片",
    books: [
          "John L. Hennessy & David A. Patterson, \"Computer Architecture: A Quantitative Approach\" (6th, 2017)",
          "Vivienne Sze et al., \"Efficient Processing of Deep Neural Networks\" (Synthesis Lectures 2020)",
          "Karl G. H. J. et al., \"In-Memory Computing\" (2019)"
    ],
    chapters: [
      "GPU 架构与 Tensor Core (Hennessy & Patterson §4)",
      "TPU 脉动阵列 (Jouppi et al., TPU 2017; Hennessy & Patterson §7)",
      "AI 加速器数据流 (Sze et al., 2020 §3)",
      "硬件加速器设计与编译器映射 (Sze et al., 2020 §6)",
      "量化与稀疏计算 (Sze et al., 2020 §5)",
      "内存层次与带宽 (Sze et al., 2020 §4)",
      "近存计算与存内计算 (Karl et al., 2019 §3)",
      "能效评估与 Roofline (Hennessy & Patterson §4.6)",
      "AI 芯片评测基准（MLPerf） (Reddi et al., MLPerf 2020)",
      "代表性 AI 加速器案例（昇腾/寒武纪/Groq/Habana） (厂商技术白皮书; Jouppi et al., TPU 2017)",
      "边缘 AI 芯片设计 (Sze et al., 2020 §7)"
    ],
  },
  'advanced/neuromorphic-computing': {
    title: "神经形态计算",
    books: [
          "Giacomo Indiveri & Shih-Chii Liu, \"Memory and Current Conductor: Neuromorphic Engineering\" (2015)",
          "Mike Davies et al., \"Loihi: A Neuromorphic Manycore Processor with On-Chip Learning\" (IEEE Micro 2018)",
          "Catherine D. Schuman et al., \"A Survey of Neuromorphic Computing and Neural Networks in Hardware\" (2017)"
    ],
    chapters: [
      "脉冲神经网络 SNN (Indiveri & Liu §2)",
      "LIF 神经元模型 (Indiveri & Liu §3)",
      "STDP 在线学习规则 (Indiveri & Liu §5)",
      "SNN 监督训练：替代梯度与时间反向传播 (Neftci et al., Surrogate Gradient 2019)",
      "事件驱动计算 (Davies et al., Loihi 2018)",
      "事件相机与神经形态感知 (Gallego et al., Event-based Vision 2020)",
      "神经形态硬件架构 (Schuman et al., 2017 §3)",
      "Loihi 架构与可编程学习 (Davies et al., Loihi 2018)",
      "TrueNorth 与 SpiNNaker (Schuman et al., 2017 §4)",
      "神经形态应用与基准 (Schuman et al., 2017 §5)"
    ],
  },
  'advanced/brain-inspired-intelligence': {
    title: "脑科学类脑智能",
    books: [
          "Kandel et al., \"Principles of Neural Science\" (6th, 2021)",
          "Dayan & Abbott, \"Theoretical Neuroscience\" (2001)"
    ],
    chapters: [
      "神经元与神经回路 (Kandel §2)",
      "突触传递与突触可塑性 (Kandel §11-16, §52-54)",
      "感知系统与信息处理 (Kandel §22)",
      "学习与记忆的神经基础 (Kandel §52-54)",
      "脑认知与意识 (Kandel §56)",
      "神经编码与放电模型 (Dayan & Abbott §1-2)",
      "神经解码与信息论方法 (Dayan & Abbott §3-4)",
      "计算神经科学与类脑算法 (Dayan & Abbott §7)",
      "人工神经网络与类脑计算 (Dayan & Abbott §8)",
      "大规模脑模拟与类脑计算平台 (Eliasmith et al., Spaun 2012)"
    ],
  },
  'frontier/artificial-life': {
    title: "人工生命",
    books: [
          "Christopher G. Langton, \"Artificial Life\" (Santa Fe Institute Studies, 1989)",
          "Christoph Adami, \"Introduction to Artificial Life\" (1998)",
          "Margaret A. Boden, \"The Philosophy of Artificial Life\" (1996)"
    ],
    chapters: [
      "人工生命定义与方法论 (Langton §1)",
      "生命的形式化与边界 (Boden §2)",
      "元胞自动机与生命游戏 (Adami §3)",
      "人工化学与自复制 (Adami §4)",
      "生命起源与最小细胞：RNA 世界与人工细胞 (Adami §2)",
      "演化与数字生命 Tierra (Adami §5)",
      "Avida 与开放进化 (Adami §5-6)",
      "复杂系统与涌现 (Langton §4)",
      "群体动力学 (Adami §7)",
      "形态发生与发育模型 (Adami §8 / Langton 论文集)",
      "合成生物学交叉 (Boden §5)"
    ],
  },
  'frontier/brain-computer-interface': {
    title: "脑机接口",
    books: [
          "Jonathan R. Wolpaw & Elizabeth Winter Wolpaw, \"Brain-Computer Interfaces: Principles and Practice\" (2012)",
          "Miguel A. L. Nicolelis, \"Brain-Machine Interfaces: From Basic Science to Clinical Applications\" (2016)",
          "Rajesh P. N. Rao, \"Brain-Computer Interfacing: An Introduction\" (2013)"
    ],
    chapters: [
      "BCI 原理与分类 (Wolpaw §1)",
      "脑信号采集：EEG 与多模态技术对比 (Rao §2 / Wolpaw Part II)",
      "皮层脑电 ECoG 与侵入式 (Nicolelis §3)",
      "特征提取与信号处理 (Wolpaw §4)",
      "解码算法与机器学习 (Rao §5)",
      "运动想象与神经反馈 (Wolpaw §5)",
      "诱发型 BCI：P300 拼写器与 SSVEP (Rao §5 / Wolpaw Part V)",
      "感觉恢复与神经假体 (Nicolelis §6)",
      "BCI 应用与神经康复 (Wolpaw §11)",
      "BCI 伦理与安全 (Wolpaw §12)"
    ],
  },
  'frontier/swarm-intelligence': {
    title: "群体智能",
    books: [
          "James Kennedy & Russell C. Eberhart, \"Swarm Intelligence\" (Morgan Kaufmann, 2001)",
          "Eric Bonabeau, Marco Dorigo & Guy Theraulaz, \"Swarm Intelligence: From Natural to Artificial Systems\" (1999)",
          "Xin-She Yang, \"Nature-Inspired Metaheuristic Algorithms\" (2nd, 2010)"
    ],
    chapters: [
      "群体智能原理 (Bonabeau §1)",
      "涌现与自组织 (Kennedy §2)",
      "蚁群优化算法 ACO (Bonabeau §2)",
      "蚁群聚类与网络路由 (Bonabeau §3-4)",
      "粒子群优化 PSO (Kennedy §3)",
      "蜂群算法 ABC (Yang §6)",
      "萤火虫、布谷鸟与蝙蝠等新元启发式 (Yang §3-8)",
      "群体机器人系统 (Bonabeau §7)",
      "群体分工与协同 (Kennedy §5)",
      "集体决策与群体协同机制：quorum sensing (Bonabeau §5-6)",
      "群体智能应用 (Yang §9)"
    ],
  },
  'frontier/explainable-ai': {
    title: "可解释 AI",
    books: [
          "Amina Adadi & Mohammed Berrada, \"Peeking Inside the Black-Box: A Survey on Explainable AI\" (IEEE Access 2018)",
          "Alejandro Barredo Arrieta et al., \"Explainable Artificial Intelligence (XAI): Concepts, Taxonomies, Opportunities and Challenges\" (Information Fusion 2020)",
          "Christoph Molnar, \"Interpretable Machine Learning\" (2nd, 2022)"
    ],
    chapters: [
      "可解释性概念与分类 (Arrieta §2)",
      "内在可解释模型 (Molnar §4)",
      "事后解释方法（模型无关）(Molnar §5)",
      "全局模型无关事后解释：PDP/ICE/ALE 与特征重要性 (Molnar §5)",
      "LIME 局部解释 (Arrieta §3.2)",
      "SHAP 与博弈论归因 (Molnar §5)",
      "基于示例的解释：原型/批评样本/锚点/对抗样本 (Molnar §6)",
      "反事实解释 (Molnar §6)",
      "深度学习可解释性 (Arrieta §4)",
      "深度模型可视化归因：saliency/Grad-CAM/注意力 (Arrieta §4)",
      "XAI 评估与人类研究 (Adadi §4)"
    ],
  },
  'frontier/quantum-information': {
    title: "量子信息",
    books: [
          "Michael A. Nielsen & Isaac L. Chuang, \"Quantum Computation and Quantum Information\" (10th Anniversary, 2010)",
          "Mark M. Wilde, \"Quantum Information Theory\" (2nd, 2017)",
          "John Preskill, \"Lecture Notes on Quantum Computation\" (Caltech, 2018)"
    ],
    chapters: [
      "量子比特与态空间 (Nielsen §2)",
      "量子纠缠与 Bell 态 (Nielsen §2)",
      "量子门与量子电路 (Nielsen §4)",
      "量子算法 Shor/Grover (Nielsen §5-6)",
      "量子测量与不可克隆 (Wilde §3)",
      "量子熵与量子香农理论：von Neumann 熵与 coherent information (Wilde §11 / Preskill Part II)",
      "量子信道与容量：Holevo 界与量子信道容量 (Wilde §13 / Preskill Part II)",
      "量子计算的物理实现：离子阱/超导/光量子/拓扑 (Nielsen §7)",
      "量子噪声、退相干与量子信道 (Nielsen §8)",
      "量子纠错码 (Nielsen §10)",
      "纠缠度量与量子资源理论 (Nielsen §12)",
      "量子密钥分发 QKD (Preskill §4)",
      "量子隐形传态与超密编码 (Nielsen §2.3)"
    ],
  },
  'frontier/sustainability-science': {
    title: "可持续发展科学",
    books: [
          "William C. Clark, \"Sustainability Science: A Room of Its Own\" (PNAS 2007)",
          "Robert W. Kates et al., \"Sustainability Science\" (Science 2001)",
          "Frank Biermann, \"Earth System Governance: World Politics in the Anthropocene\" (2014)"
    ],
    chapters: [
      "可持续发展定义与演进 (Kates 主题对标)",
      "地球系统与人类世 (Clark 主题对标)",
      "行星边界与地球承载力 (Biermann 主题对标)",
      "社会-生态系统耦合 (Biermann §4)",
      "韧性与阈值 (Kates 主题对标)",
      "食物-能源-水纽带与资源安全 (Biermann 主题对标)",
      "可持续性转型 (Biermann §6)",
      "可持续指标 SDGs (Clark 主题对标)",
      "可持续消费与循环经济 (Kates 主题对标)",
      "地球系统治理 (Biermann §5)",
      "代际正义与环境伦理 (Kates 主题对标)",
      "可持续性科学与政策 (Kates 主题对标)"
    ],
  },
  'frontier/tech-ethics': {
    title: "科技伦理",
    books: [
          "Luciano Floridi, \"The Ethics of Information\" (Oxford, 2013)",
          "Shannon Vallor, \"Technology and the Virtues: A Philosophical Guide to a Future Worth Wanting\" (2016)",
          "J. van den Hoven et al., \"Handbook of Ethics, Values, and Technological Design\" (2015)"
    ],
    chapters: [
      "信息伦理框架 (Floridi §1)",
      "数据伦理与隐私 (van den Hoven §3)",
      "AI 伦理原则 (Vallor §6)",
      "算法公平与偏见 (van den Hoven §5)",
      "价值敏感设计（VSD）与伦理设计 (van den Hoven Part I)",
      "技术责任与问责 (Floridi §5)",
      "技术美德伦理 (Vallor §3)",
      "自动化伦理与自主性 (Vallor §7)",
      "新兴技术伦理个案：自主武器/基因编辑/神经增强 (van den Hoven Part IV)",
      "数字鸿沟与技术包容性 (van den Hoven §18)",
      "新兴技术治理 (van den Hoven §10)"
    ],
  },
  'frontier/space-exploration': {
    title: "太空探索与空间资源",
    books: [
          "Carl Sagan, \"Cosmos\" (Ballantine, 1980)",
          "National Research Council, \"Pathways to Exploration: Rationales and Approaches for a U.S. Program of Human Space Exploration\" (2014)",
          "John M. Logsdon, \"Exploring the Unknown: Selected Documents in the History of the U.S. Civil Space Program\" (2017)"
    ],
    chapters: [
      "太空探索历史与动机 (Logsdon §1)",
      "空间环境与轨道力学 (NRC §3)",
      "深空探测与行星科学 (Sagan §10)",
      "载人航天与生命保障 (NRC §4)",
      "空间资源与小行星采矿 (NRC §5)",
      "ISRU 原位资源利用与月球/阿尔忒弥斯基地 (NRC 主题对标)",
      "空间站与近地轨道利用 (Logsdon §6)",
      "火星与外太阳系探测 (Sagan §5)",
      "商业航天与发射体系 (NRC 主题对标)",
      "空间法与外空资源治理 (Logsdon 主题对标)",
      "空间碎片与轨道环境可持续性 (NRC 主题对标)",
      "空间政策与国际合作 (NRC §6)"
    ],
  },
  'frontier/climate-engineering': {
    title: "气候工程",
    books: [
          "National Research Council, \"Climate Intervention: Reflecting Sunlight to Cool Earth\" (2015)",
          "National Research Council, \"Climate Intervention: Carbon Dioxide Removal and Reliable Sequestration\" (2015)",
          "David Keith, \"A Case for Climate Engineering\" (2013)"
    ],
    chapters: [
      "气候工程分类与动机 (NRC Reflecting §1)",
      "太阳辐射管理 SRM (NRC Reflecting §3)",
      "平流层气溶胶注入 (Keith §3)",
      "海洋云层增亮 (NRC Reflecting §3)",
      "卷云薄化 (NRC Reflecting §3)",
      "碳移除 CDR 技术 (NRC CDR §2)",
      "陆基碳移除：造林/BECCS/土壤固碳/增强风化 (NRC CDR Ch3 子节)",
      "直接空气捕集 DAC (NRC CDR Ch3 子节)",
      "海洋施肥与碱化 (NRC CDR Ch3 子节)",
      "地球工程风险评估：终止冲击、道德风险 (Keith §5-6)",
      "气候工程伦理与治理 (Keith §6)"
    ],
  },
  'frontier/global-governance': {
    title: "全球治理",
    books: [
          "James N. Rosenau & Ernst-Otto Czempiel, \"Governance Without Government: Order and Change in World Politics\" (1992)",
          "Frank Biermann, \"Earth System Governance: World Politics in the Anthropocene\" (2014)",
          "Thomas G. Weiss & Rorden Wilkinson, \"International Organization and Global Governance\" (2nd, 2018)"
    ],
    chapters: [
      "全球治理概念与兴起 (Rosenau §1)",
      "全球化与跨国议题 (Weiss §3)",
      "国际机制与制度 (Weiss §5)",
      "联合国体系与机构运作 (Weiss Part IV)",
      "全球公共物品 (Biermann §3)",
      "多层治理与网络 (Rosenau §4)",
      "地球系统治理 (Biermann §4)",
      "非国家行为体参与 (Weiss §8)",
      "全球安全治理：冲突、维和、军控与核不扩散 (Weiss Part VI)",
      "全球经济与金融治理：IMF/世界银行/WTO、贸易与金融规制 (Weiss §42-43)",
      "全球环境与气候治理：巴黎协定与多边环境机制 (Weiss Part VII)",
      "全球卫生与人权治理：WHO、全球公共卫生危机与国际人权机制 (Weiss Part VI-VII)",
      "全球治理改革 (Biermann §8)"
    ],
  },
  'engineering/robotics-engineering': {
    title: "机器人工程",
    books: [
          "Spong, Hutchinson, Vidyasagar, \"Robot Modeling and Control\" (2nd ed., 2020)",
          "Craig, \"Introduction to Robotics: Mechanics and Control\" (4th ed., 2017)"
    ],
    chapters: [
      "机器人空间描述与变换 (Spong §2)",
      "正向与逆向运动学 (Spong §3-4)",
      "速度与静力学雅可比 (Spong §5)",
      "机器人动力学 (Spong §7)",
      "轨迹规划 (Spong §8)",
      "独立关节控制 (Spong §9)",
      "非线性控制（计算力矩/李雅普诺夫/自适应鲁棒） (Spong §11)",
      "力控制与视觉伺服 (Craig §6-9)",
      "移动机器人与运动规划 (Spong §10)",
      "机械结构设计 (Craig §8)",
      "机器人编程与离线编程 (Craig §12-13)"
    ],
  },
  'engineering/intelligent-manufacturing': {
    title: "智能制造",
    books: [
          "Groover, \"Automation, Production Systems, and Computer-Integrated Manufacturing\" (5th ed., 2020)",
          "Tay, Jon, Chan, \"Smart Manufacturing: Concepts and Applications\" (1st ed., 2021)"
    ],
    chapters: [
      "制造自动化概述 (Groover §1)",
      "自动化与控制系统基础（PLC/传感器/执行器） (Groover §3-5)",
      "数控技术与编程 (Groover §6)",
      "工业机器人应用 (Groover §7)",
      "物料搬运与自动识别 (Groover §10)",
      "柔性制造系统 (Groover §16)",
      "计算机集成制造系统 (Groover §23)",
      "质量控制系统（SPC/在线检验/六西格玛） (Groover §20-21)",
      "工艺规划与生产计划控制 (Groover §22-24)",
      "制造执行系统与工业物联网 (Tay §4)",
      "数字孪生与智能工厂 (Tay §7)"
    ],
  },
  'engineering/additive-manufacturing': {
    title: "增材制造（3D 打印）",
    books: [
          "Gibson, Rosen, Stucker, Khorasani, \"Additive Manufacturing Technologies\" (3rd ed., 2021)",
          "Chua, Leong, Lim, \"3D Printing and Additive Manufacturing: Principles and Applications\" (5th ed., 2017)"
    ],
    chapters: [
      "增材制造原理与流程 (Gibson §1-2)",
      "增材制造数据模型与软件（STL/AMF/切片/工艺链） (Gibson §3)",
      "光固化成型工艺 (Gibson §4)",
      "选择性激光烧结与熔化 (Gibson §5)",
      "熔融沉积成型 (Gibson §6)",
      "材料喷射工艺 (Gibson §7)",
      "三维打印与黏结剂喷射 (Gibson §8)",
      "薄片叠层与间接制造 (Gibson §9)",
      "定向能量沉积（DED）与激光熔覆 (Gibson §10)",
      "直接金属激光烧结 (Gibson §5)",
      "面向增材制造的设计（DfAM）与拓扑优化 (Gibson §19)",
      "增材制造后处理与质量 (Chua §11)",
      "增材制造设计与生物医学应用 (Gibson §12)"
    ],
  },
  'engineering/engineering-mechanics': {
    title: "工程力学（理论/材料/结构力学）",
    books: [
          "Hibbeler, \"Mechanics of Materials\" (11th ed., 2019)",
          "Hibbeler, \"Engineering Mechanics: Statics\" (14th ed., 2019)"
    ],
    chapters: [
      "静力学基础（力系简化与平衡/平面任意力系/摩擦/形心与惯性矩） (Hibbeler Statics §1-2, §8-10)",
      "质点与刚体静力学 (Hibbeler Statics §3-5)",
      "应力与应变 (Hibbeler Mechanics §1-2)",
      "轴向载荷与扭转 (Hibbeler Mechanics §3-5)",
      "梁的弯曲 (Hibbeler Mechanics §6)",
      "横向剪切应力 (Hibbeler MechMat §7)",
      "组合载荷下的应力 (Hibbeler Mechanics §8)",
      "Mohr 应力圆与应变转换 (Hibbeler MechMat §9)",
      "失效理论（von Mises/Tresca/Mohr） (Hibbeler MechMat §10)",
      "梁的挠度与变形 (Hibbeler MechMat §12)",
      "压杆稳定 (Hibbeler Mechanics §13)",
      "能量方法 (Hibbeler MechMat §14)",
      "运动学与动力学（速度/加速度分析） (Hibbeler §Dynamics Ch.12-16)",
      "功-能原理与冲量-动量 (Hibbeler §Dynamics Ch.17-19)",
      "机械振动（自由/受迫振动） (Hibbeler §Dynamics Ch.22)"
    ],
  },
  'engineering/machine-design': {
    title: "机械设计与机械原理",
    books: [
          "Shigley, Mischke, Brown, \"Mechanical Engineering Design\" (11th ed., 2020)",
          "Norton, \"Machine Design: An Integrated Approach\" (6th ed., 2020)"
    ],
    chapters: [
      "材料强度与失效理论 (Shigley §5-6)",
      "载荷与应力分析基础、挠度与刚度 (Shigley §3-4)",
      "疲劳设计 (Shigley §6)",
      "螺纹联接与紧固件 (Shigley §8)",
      "焊接与永久联接 (Shigley §9)",
      "轴的设计 (Shigley §12)",
      "齿轮传动设计 (Shigley §13-14)",
      "机构结构分析与自由度 (Norton Mechanism §2)",
      "机构运动学（连杆/凸轮/齿轮系） (Norton §Mechanism Ch.2-4)",
      "凸轮机构设计 (Norton Mechanism §8)",
      "轮系传动比计算 (Norton Mechanism §9)",
      "机械平衡与飞轮调速 (Norton Mechanism §12-13)",
      "间歇运动机构（槽轮/棘轮/不完全齿轮机构） (Norton Mechanism §2)",
      "离合器与制动器 (Norton §16)",
      "轴承设计（滚动/滑动接触） (Norton §Ch.10)",
      "弹簧设计与带链传动 (Norton §Ch.13-14)",
      "摩擦学与润滑 (Norton §Ch.11)"
    ],
  },
  'engineering/precision-machining': {
    title: "精密与超精密加工",
    books: [
          "Kalpakjian, Schmid, \"Manufacturing Engineering and Technology\" (8th ed., 2020)",
          "Whitehouse, \"Surfaces and their Measurement\" (1st ed., 2002)"
    ],
    chapters: [
      "精密加工基础 (Kalpakjian §25)",
      "金刚石车削技术 (Kalpakjian §25)",
      "超精密加工刀具与材料 (Kalpakjian §25)",
      "精密磨削工艺 (Kalpakjian §26)",
      "抛光与化学机械抛光 (Kalpakjian §27)",
      "加工误差来源与精度分析理论 (Kalpakjian §25)",
      "加工环境控制（恒温/洁净/隔振） (Kalpakjian §27)",
      "微细加工与特种加工（微EDM/激光微加工/电化学微加工） (Kalpakjian §28)",
      "表面粗糙度与测量 (Whitehouse §3-4)",
      "纳米尺度加工 (Kalpakjian §29)",
      "超精密机床结构与静压轴承 (Kalpakjian §27)",
      "在线测量与误差补偿技术 (Whitehouse §6)"
    ],
  },
  'engineering/smart-grid': {
    title: "智能电网",
    books: [
          "Borlase, \"Smart Grids: Infrastructure, Technology, and Solutions\" (2nd ed., 2017)",
          "Momoh, \"Smart Grid: Fundamentals of Design and Analysis\" (2nd ed., 2018)"
    ],
    chapters: [
      "智能电网架构与标准 (Borlase §1-2)",
      "智能电网通信网络与信息安全 (Borlase §3)",
      "高级量测体系AMI (Borlase §4)",
      "配电自动化系统 (Borlase §6)",
      "智能变电站与 IEC 61850 (Borlase §7)",
      "分布式能源接入 (Borlase §8)",
      "储能系统并网应用 (Momoh §9)",
      "需求响应与负荷管理 (Borlase §9)",
      "电网自愈与可靠性 (Momoh §7)",
      "电能质量与谐波治理 (Momoh §6)",
      "广域测量系统（WAMS/PMU） (Momoh §10)",
      "微电网与分布式能源 (Momoh §8)",
      "EV/V2G 集成 (Momoh §11)"
    ],
  },
  'engineering/electric-machines': {
    title: "电机学",
    books: [
          "Fitzgerald, Kingsley, Umans, \"Electric Machinery\" (7th ed., 2014)",
          "Chapman, \"Electric Machinery Fundamentals\" (6th ed., 2016)"
    ],
    chapters: [
      "磁路与磁性材料 (Fitzgerald §1)",
      "变压器原理 (Fitzgerald §2)",
      "机电能量转换原理 (Fitzgerald §3)",
      "旋转电机概述与交流绕组磁场 (Fitzgerald §4)",
      "同步电机 (Fitzgerald §5)",
      "感应电机 (Fitzgerald §6)",
      "直流电机 (Fitzgerald §7)",
      "变磁阻电机与步进电机 (Fitzgerald §8)",
      "单相与两相电机 (Fitzgerald §9)",
      "电机调速与转速转矩控制 (Fitzgerald §10)",
      "永磁与特种电机 (Chapman §9)",
      "电力电子驱动与变频控制 (Chapman §10)"
    ],
  },
  'engineering/power-systems': {
    title: "电力系统分析",
    books: [
          "Grainger, Stevenson, \"Power System Analysis\" (1st ed., 1994)",
          "Bergen, Vittal, \"Power Systems Analysis\" (2nd ed., 2000)"
    ],
    chapters: [
      "电力网络模型与参数 (Grainger §3-4)",
      "潮流计算 (Grainger §6-9)",
      "对称故障分析 (Grainger §11)",
      "不对称故障分析 (Grainger §12)",
      "电力系统稳定性 (Bergen §12-13)",
      "经济调度与最优潮流 (Bergen §14)",
      "继电保护原理（继电器/断路器） (Grainger §14-15)",
      "状态估计 (Grainger §16)",
      "AGC 与 AVR 控制 (Grainger §12)",
      "HVDC 输电 (Grainger §17)"
    ],
  },
  'engineering/power-electronics': {
    title: "电力电子与电力传动",
    books: [
          "Mohan, Undeland, Robbins, \"Power Electronics: Converters, Applications, and Design\" (3rd ed., 2003)",
          "Rashid, \"Power Electronics: Circuits, Devices & Applications\" (4th ed., 2013)"
    ],
    chapters: [
      "电力电子器件 (Mohan §4-6)",
      "整流电路与有源逆变 (Mohan §7)",
      "交流-交流变换（周波变换器/矩阵变换器） (Mohan §6)",
      "直流-直流变换器 (Mohan §8)",
      "电压源逆变器与PWM技术 (Mohan §9)",
      "谐振变换器 (Rashid §8)",
      "多电平变换器 (Mohan §11)",
      "功率因数校正（PFC） (Mohan §8)",
      "门极驱动与保护电路（含缓冲/保护） (Mohan §27-28)",
      "开关电源与不间断电源（UPS）应用 (Mohan §10-11)",
      "交流传动控制 (Mohan §11)",
      "HVDC 与柔性交流输电（FACTS） (Mohan §12)"
    ],
  },
  'engineering/signals-systems': {
    title: "信号与系统",
    books: [
          "Oppenheim, Willsky, Nawab, \"Signals and Systems\" (2nd ed., 1997)",
          "Haykin, Van Veen, \"Signals and Systems\" (2nd ed., 2003)"
    ],
    chapters: [
      "连续时间信号与系统 (Oppenheim §1-2)",
      "线性时不变系统与卷积 (Oppenheim §2-3)",
      "连续时间傅里叶分析 (Oppenheim §4-5)",
      "频域系统特性与滤波（Bode/谐振/滤波器整形） (Oppenheim §6)",
      "采样定理 (Oppenheim §7)",
      "通信系统与调制 (Oppenheim §8)",
      "拉普拉斯变换 (Oppenheim §9)",
      "离散时间信号与Z变换 (Oppenheim §10)",
      "线性反馈系统 (Oppenheim §11)"
    ],
  },
  'engineering/nano-engineering': {
    title: "纳米工程",
    books: [
          "Rogers, Adams, Pennathur, \"Nanotechnology: Understanding Small Systems\" (3rd ed., 2014)",
          "Poole, Owens, \"Introduction to Nanotechnology\" (1st ed., 2003)"
    ],
    chapters: [
      "纳米尺度物理效应 (Rogers §1-2)",
      "纳米材料合成方法 (Rogers §5)",
      "纳米制造技术 (Rogers §8)",
      "纳米表征技术 (Rogers §6)",
      "纳米电子器件 (Rogers §7)",
      "纳米力学与原子尺度测量操作（SPM 工作模式） (Rogers §6)",
      "纳米传感器与纳机电系统（NEMS） (Rogers §9)",
      "纳米材料在能源与催化中的应用 (Rogers §10)",
      "纳米生物医学应用 (Poole §7)",
      "自组装与分子机器 (Rogers §5)",
      "纳米环境安全与毒性 (Poole §10)"
    ],
  },
  'engineering/chemical-thermodynamics': {
    title: "化工热力学",
    books: [
          "Smith, Van Ness, Abbott, Swihart, \"Introduction to Chemical Engineering Thermodynamics\" (9th ed., 2022)",
          "Prausnitz, Lichtenthaler, Azevedo, \"Molecular Thermodynamics of Fluid-Phase Equilibria\" (3rd ed., 1999)"
    ],
    chapters: [
      "热力学基本概念与定律 (Smith §1-3)",
      "流体的容积性质与状态方程 (Smith §3)",
      "化学反应热效应与燃烧热 (Smith §4)",
      "热力学性质计算 (Smith §6)",
      "动力循环（朗肯/布雷顿/联合循环） (Smith §8)",
      "制冷与液化循环 (Smith §9)",
      "相平衡与活度系数 (Smith §10-12)",
      "溶液热力学与超额性质 (Smith §11)",
      "化学反应平衡 (Smith §13)",
      "过程热力学分析 (Smith §15)",
      "分子热力学与状态方程 (Prausnitz §4-5)"
    ],
  },
  'engineering/transport-phenomena': {
    title: "传递过程（动量/热量/质量传递）",
    books: [
          "Bird, Stewart, Lightfoot, \"Transport Phenomena\" (2nd ed., 2007)",
          "Welty, Rorrer, Foster, \"Fundamentals of Momentum, Heat, and Mass Transfer\" (6th ed., 2014)"
    ],
    chapters: [
      "动量传递与黏性流动 (Bird §1-3)",
      "层流与湍流 (Bird §5)",
      "能量传递与导热 (Bird §9-10)",
      "对流传热 (Bird §11-14)",
      "质量传递与扩散 (Bird §16-17)",
      "对流传质、传质系数与界面传质 (Bird §21)",
      "动量/热量/质量传递类比与类比律 (Bird §20)",
      "多组分传递与耦合 (Bird §22-24)",
      "辐射传热 (Bird §14-15)",
      "边界层理论 (Bird §4)",
      "非牛顿流体与流变学 (Bird §8)"
    ],
  },
  'engineering/separation-processes': {
    title: "分离工程",
    books: [
          "Seader, Henley, Roper, \"Separation Process Principles\" (4th ed., 2016)",
          "Geankoplis, \"Transport Processes and Separation Process Principles\" (5th ed., 2018)"
    ],
    chapters: [
      "分离过程基础与平衡级 (Seader §1-2)",
      "传质与扩散基础（对流传质/传质系数） (Seader §3)",
      "汽-液平衡与单级闪蒸计算 (Seader §4)",
      "精馏过程 (Seader §5-7)",
      "多组分精馏的近似与严格计算 (Seader §9-10)",
      "吸收与汽提 (Seader §6)",
      "液-液萃取 (Seader §8)",
      "膜分离过程 (Seader §11)",
      "吸附与离子交换 (Seader §15)",
      "结晶过程与设计 (Seader §17)",
      "干燥过程与设备 (Geankoplis §9)"
    ],
  },
  'engineering/chemical-reaction-engineering': {
    title: "反应工程",
    books: [
          "Fogler, \"Elements of Chemical Reaction Engineering\" (5th ed., 2016)",
          "Levenspiel, \"Chemical Reaction Engineering\" (3rd ed., 1999)"
    ],
    chapters: [
      "反应速率与化学计量 (Fogler §2-3)",
      "反应速率数据收集与分析 (Fogler §5)",
      "理想反应器设计 (Fogler §4-5)",
      "多重反应（平行/串联）与选择性优化 (Fogler §6)",
      "反应机理与速率方程推导 (Fogler §7)",
      "非等温反应器设计与能量平衡 (Fogler §9)",
      "催化反应与催化剂 (Fogler §10)",
      "多相催化反应器 (Fogler §12)",
      "非理想反应器与停留时间分布 (Fogler §13-14)",
      "停留时间分布（RTD）建模 (Fogler §13-14)",
      "聚合反应工程 (Levenspiel §9)",
      "生化反应器与酶动力学 (Fogler §12)"
    ],
  },
  'engineering/structural-engineering': {
    title: "结构工程",
    books: [
          "Hibbeler, \"Structural Analysis\" (10th ed., 2018)",
          "Nilson, Darwin, Dolan, \"Design of Concrete Structures\" (16th ed., 2021)"
    ],
    chapters: [
      "结构静力分析与桁架 (Hibbeler §1-4)",
      "影响线 (Hibbeler §7)",
      "矩阵位移法 (Hibbeler §13-14)",
      "结构动力学基础（单/多自由度振动） (Chopra §3-4)",
      "有限元分析 (Hutton §1-3)",
      "混凝土梁的弯曲设计 (Nilson §2-3)",
      "混凝土梁抗剪设计 (Nilson §4)",
      "混凝土结构使用性与耐久性 (Nilson §6)",
      "混凝土柱设计 (Nilson §8)",
      "钢筋混凝土板 (Nilson §12)",
      "预应力混凝土设计 (Nilson §12)",
      "钢结构设计（梁/柱/连接） (Segui §3-6)",
      "屈曲与稳定性分析 (Hibbeler MechMat §13)",
      "抗震设计原理与反应谱 (Chopra §6)"
    ],
  },
  'engineering/geotechnical-engineering': {
    title: "岩土工程",
    books: [
          "Das, \"Principles of Geotechnical Engineering\" (10th ed., 2021)",
          "Das, \"Principles of Foundation Engineering\" (9th ed., 2019)"
    ],
    chapters: [
      "土的物理性质与分类 (Das §3-4)",
      "土的压实 (Das §5)",
      "土的渗透性与渗流（流网/达西定律） (Das §6)",
      "土中应力与有效应力原理 (Das §7)",
      "土的压缩性与固结 (Das §8-9)",
      "土的抗剪强度 (Das §10-11)",
      "土动力学与地基抗震（液化/动强度） (Das Foundation §12)",
      "浅基础承载力 (Das Foundation §3)",
      "深基础桩基 (Das Foundation §11)",
      "边坡稳定性分析 (Das §11-12)",
      "挡土墙与土压力 (Das §13)",
      "原位测试（SPT/CPT/十字板） (Das §7)"
    ],
  },
  'engineering/bridge-tunnel-engineering': {
    title: "桥梁与隧道工程",
    books: [
          "Troitsky, \"Planning and Design of Bridges\" (1st ed., 1994)",
          "Hoek, \"Practical Rock Engineering\" (1st ed., 2007)"
    ],
    chapters: [
      "桥梁类型与规划 (Troitsky §1-2)",
      "桥梁荷载与作用 (Troitsky §3)",
      "桥梁结构设计 (Troitsky §5-6)",
      "桥梁下部结构（桥墩/桥台/基础）设计 (Troitsky §8)",
      "桥梁施工方法与架设 (Troitsky §7)",
      "桥梁耐久性与养护管理 (Troitsky §9)",
      "隧道工程地质与围岩分级 (关宝树《隧道工程》 §2)",
      "隧道围岩力学 (Hoek §2-4)",
      "隧道支护与衬砌设计 (关宝树《隧道工程》 §3)",
      "隧道施工方法（钻爆/矿山法） (关宝树《隧道工程》 §4)",
      "隧道盾构/TBM 施工方法 (关宝树《隧道工程》 §4)",
      "隧道监控量测与信息化施工 (关宝树《隧道工程》 §7)",
      "隧道通风与防灾救援 (关宝树《隧道工程》 §5-6)"
    ],
  },
  'engineering/renewable-energy': {
    title: "新能源工程",
    books: [
          "Tester, Drake, Driscoll, Golay, Peters, \"Sustainable Energy: Choosing Among Options\" (2nd ed., 2012)",
          "Sørensen, \"Renewable Energy: Physics, Engineering, Environmental Impacts, Economics & Planning\" (5th ed., 2017)"
    ],
    chapters: [
      "能源系统与可持续性 (Tester §1-2)",
      "终端能源利用与能效提升 (Tester §4)",
      "能源经济学与能源政策（成本/外部性/补贴） (Tester §6)",
      "太阳能光伏与热利用 (Tester §7-8)",
      "风能系统 (Tester §9)",
      "生物质能与生物燃料 (Tester §10)",
      "地热能 (Tester §11)",
      "海洋能与水电 (Sørensen §4)",
      "分布式能源与微电网 (Tester §13)",
      "储能与能源系统集成（电池/氢） (Tester §14)",
      "氢能与燃料电池 (Sørensen §5)"
    ],
  },
  'engineering/energy-storage': {
    title: "储能技术",
    books: [
          "Huggins, \"Energy Storage: Fundamentals, Materials and Applications\" (2nd ed., 2016)",
          "Doughty, Butler, \"Battery Technologies\" (1st ed., 2018)"
    ],
    chapters: [
      "储能原理与分类 (Huggins §1-2)",
      "电化学储能基础 (Huggins §6)",
      "锂离子电池技术 (Huggins §7)",
      "液流电池（钒/锌溴）技术 (Huggins §8)",
      "钠硫/钠离子/固态电池等新型电化学储能 (Huggins §10)",
      "超级电容器 (Huggins §9)",
      "氢储能与燃料电池储能系统 (Huggins §14)",
      "飞轮与机械储能 (Huggins §12)",
      "压缩空气与抽水蓄能 (Huggins §13)",
      "相变热储能技术 (Huggins §11)",
      "储能系统集成与应用 (Doughty §8)"
    ],
  },
  'engineering/hydrogen-energy': {
    title: "氢能工程",
    books: [
          "O'Hayer, Cha, Colella, Prinz, \"Fuel Cell Fundamentals\" (3rd ed., 2016)",
          "Gupta, \"Hydrogen Fuel: Production, Transport, and Storage\" (1st ed., 2008)"
    ],
    chapters: [
      "燃料电池热力学基础 (O'Hayre §2-3)",
      "燃料电池电化学原理 (O'Hayre §4)",
      "燃料电池系统建模与表征（极化曲线/阻抗谱） (O'Hayre §6-7)",
      "质子交换膜燃料电池 (O'Hayre §8)",
      "碱性燃料电池 (O'Hayre §8.4)",
      "直接甲醇燃料电池 (O'Hayre §8.7)",
      "固体氧化物与熔融碳酸盐燃料电池 (O'Hayre §9)",
      "燃料重整与燃料处理（天然气重整/水煤气变换/净化） (O'Hayre §11)",
      "氢气制备方法 (Gupta §1-3)",
      "氢气储存与运输 (Gupta §6-8)",
      "氢能系统集成 (Gupta §12)",
      "氢安全与规范标准 (Gupta §14)"
    ],
  },
  'engineering/internal-combustion-engine': {
    title: "内燃机",
    books: [
          "Heywood, \"Internal Combustion Engine Fundamentals\" (2nd ed., 2018)",
          "Ferguson, Kirkpatrick, \"Internal Combustion Engines: Applied Thermosciences\" (3rd ed., 2015)"
    ],
    chapters: [
      "内燃机分类与循环 (Heywood §1-2)",
      "发动机性能与循环分析 (Heywood §3)",
      "换气过程与进排气系统设计 (Heywood §5)",
      "缸内流动与混合气形成 (Heywood §4)",
      "汽油机燃烧过程 (Heywood §4-9)",
      "柴油机燃烧过程 (Heywood §7)",
      "燃油喷射与雾化 (Heywood §7)",
      "排放污染物与控制 (Heywood §11)",
      "发动机热传递与冷却系统 (Heywood §9)",
      "摩擦与润滑 (Heywood §10)",
      "增压与进气系统 (Ferguson §5)",
      "替代燃料与混合动力总成 (Heywood §13)"
    ],
  },
  'engineering/refrigeration-air-conditioning': {
    title: "制冷与空调",
    books: [
          "Stoecker, Jones, \"Refrigeration and Air Conditioning\" (2nd ed., 1982)",
          "ASHRAE, \"ASHRAE Handbook: Fundamentals\" (2021 ed., 2021)"
    ],
    chapters: [
      "制冷热力学基础 (Stoecker §2)",
      "湿空气性质与焓湿图（psychrometrics） (Stoecker §3)",
      "蒸气压缩制冷循环 (Stoecker §4-5)",
      "制冷剂性质与选择 (ASHRAE §29)",
      "制冷系统部件设计 (Stoecker §7)",
      "复叠与低温制冷系统 (Stoecker §13)",
      "吸收式制冷系统 (Stoecker §15)",
      "空气调节原理 (Stoecker §16-17)",
      "空调冷热负荷计算 (ASHRAE §18)",
      "通风与室内空气品质 (ASHRAE §12)",
      "制冷空调自动控制 (Stoecker §18)"
    ],
  },
  'engineering/water-pollution-control': {
    title: "水污染控制与治理",
    books: [
          "Metcalf & Eddy, \"Wastewater Engineering: Treatment and Resource Recovery\" (6th ed., 2013)",
          "Davis, Masten, \"Principles of Environmental Engineering and Science\" (4th ed., 2017)"
    ],
    chapters: [
      "水质指标与标准 (Metcalf §1-2)",
      "物理处理工艺 (Metcalf §5)",
      "化学处理工艺（混凝/絮凝/化学沉淀/氧化还原） (Metcalf §6)",
      "生物处理原理 (Metcalf §8-9)",
      "生物脱氮除磷（BNR） (Metcalf&Eddy §8)",
      "活性污泥法 (Metcalf §10-11)",
      "膜生物反应器（MBR） (Metcalf&Eddy §10)",
      "脱盐与膜法深度处理（RO/纳滤） (Metcalf §11)",
      "生物膜法与厌氧处理 (Metcalf §13)",
      "深度处理与消毒 (Metcalf §15)",
      "工业废水处理专题 (Metcalf §12)",
      "污泥处理与处置（浓缩/脱水/消化） (Metcalf&Eddy §14-15)",
      "污水再生回用 (Metcalf&Eddy §13)"
    ],
  },
  'engineering/air-pollution-control': {
    title: "大气污染控制工程",
    books: [
          "Cooper, Alley, \"Air Pollution Control: A Design Approach\" (4th ed., 2010)",
          "Davis, Masten, \"Principles of Environmental Engineering and Science\" (4th ed., 2017)"
    ],
    chapters: [
      "大气污染物来源与分类 (Cooper §1-2)",
      "颗粒物控制原理 (Cooper §5)",
      "旋风与静电除尘器 (Cooper §6-7)",
      "湿式洗涤塔（文丘里/填料塔） (Cooper §8)",
      "袋式除尘（过滤式除尘器） (Cooper §9)",
      "硫氧化物控制技术 (Cooper §11)",
      "氮氧化物控制技术 (Cooper §12)",
      "挥发性有机物控制 (Cooper §13)",
      "废气生物处理与等离子体净化技术 (Davis §12)",
      "大气扩散模型（高斯烟羽） (Cooper§Alley §4)",
      "重金属与汞污染控制 (Cooper§Alley §9)",
      "室内空气质量 (Cooper§Alley §11)",
      "CEMS 连续排放监测 (Cooper§Alley §12)"
    ],
  },
  'engineering/solid-waste-management': {
    title: "固体废物处理与资源化",
    books: [
          "Tchobanoglous, Theisen, Vigil, \"Integrated Solid Waste Management: Engineering Principles and Management Issues\" (1st ed., 1993)",
          "Vesilind, Worrell, Reinhart, \"Solid Waste Engineering\" (3rd ed., 2016)"
    ],
    chapters: [
      "固体废物来源与特性 (Tchobanoglous §1-3)",
      "收集与运输系统 (Tchobanoglous §4-7)",
      "垃圾转运站与物流系统设计 (Tchobanoglous §5-7)",
      "资源回收与循环利用 (Tchobanoglous §8-9)",
      "卫生填埋设计 (Tchobanoglous §11)",
      "填埋场渗滤液与填埋气体处理 (Tchobanoglous §11)",
      "焚烧与热解处理 (Tchobanoglous §12-13)",
      "堆肥工艺设计（生物转化细分） (Tchobanoglous §14)",
      "生物转化处理 (Vesilind §9)",
      "危险废物处理与处置 (Tchobanoglous §15)",
      "电子废物回收处理 (Vesilind §10)"
    ],
  },
  'engineering/soil-remediation': {
    title: "土壤污染与修复",
    books: [
          "Sharma, Reddy, \"Geoenvironmental Engineering: Site Remediation, Waste Containment, and Emerging Waste Management Technologies\" (1st ed., 2004)",
          "Hester, Harrison, \"Soil Remediation and Rehabilitation\" (1st ed., 2014)"
    ],
    chapters: [
      "土壤污染物类型与来源 (Sharma §5)",
      "污染物运移与模拟（对流-弥散方程） (Sharma §6)",
      "土壤污染调查与监测 (Sharma §7)",
      "土壤物理修复技术 (Sharma §8)",
      "土壤化学修复与稳定化 (Sharma §9)",
      "土壤生物修复 (Sharma §10)",
      "地下水污染与修复 (Sharma §11)",
      "废物隔离与防渗屏障系统（衬垫/封盖） (Sharma §12)",
      "原位热脱附与电动修复技术 (Sharma §13)",
      "污染场地风险评估 (Hester §3)",
      "原位化学氧化与还原修复 (Hester §5)"
    ],
  },
  'engineering/ecological-restoration': {
    title: "生态修复工程",
    books: [
          "Clewell, Aronson, \"Ecological Restoration: Principles, Values, and Structure of an Emerging Profession\" (2nd ed., 2013)",
          "Hobbs, Walker, \"Land Restoration: Reclaiming Landscapes for a Sustainable Future\" (1st ed., 2013)"
    ],
    chapters: [
      "生态恢复理论与目标 (Clewell §1-3)",
      "生态修复规划与设计方法（目标设定/参考生态系统） (Clewell §4)",
      "植被恢复与重建 (Clewell §5)",
      "恢复生态学的种群与群落生态学基础（演替/干扰） (Clewell §6)",
      "乡土物种与种源/种子库技术 (Hobbs §7)",
      "土壤生态修复 (Hobbs §8)",
      "湿地生态系统恢复 (Hobbs §12)",
      "河流与水生生态系统恢复 (Hobbs §10)",
      "矿区生态重建 (Hobbs §15)",
      "城市与受损土地生态修复 (Hobbs §18)",
      "恢复效果监测与评价 (Clewell §7)"
    ],
  },
  'engineering/environmental-planning-management': {
    title: "环境规划与管理",
    books: [
          "Goodstein, Polasky, \"Economics and the Environment\" (8th ed., 2019)",
          "Clements, \"The Encyclopedia of Environmental Management\" (1st ed., 2013)"
    ],
    chapters: [
      "环境规划原理与方法 (叶文虎《环境管理学》 §4)",
      "环境功能区划与环境容量/总量控制 (叶文虎《环境管理学》 §5)",
      "环境影响评价技术方法（预测/评价）与公众参与 (叶文虎《环境管理学》 §6)",
      "排污许可与排污权交易/环境税等政策工具 (叶文虎《环境管理学》 §7)",
      "环境法规与政策工具 (叶文虎《环境管理学》 §3)",
      "环境管理体系ISO14001 (Barrow §4)",
      "环境监测与质量评价 (叶文虎《环境管理学》 §8)",
      "环境风险评价与管理 (Barrow §7)",
      "生态补偿与绿色金融 (叶文虎《环境管理学》 §10)",
      "可持续发展管理 (Barrow §12)",
      "企业环境责任与生命周期管理 (Barrow §8)"
    ],
  },
  'social/environmental-policy-governance': {
    title: "环境政策与治理",
    books: [
          "Tietenberg, Lewis, \"Environmental and Natural Resource Economics\" (12th ed., 2021)",
          "Kraft, \"Environmental Policy and Politics\" (7th ed., 2018)"
    ],
    chapters: [
      "环境政策理论基础 (Tietenberg §1-2)",
      "动态效率与可持续发展 (Tietenberg §5, §20)",
      "环境价值评估与成本—收益分析 (Tietenberg §3-4)",
      "可耗竭资源与能源转型经济 (Tietenberg §6)",
      "生态系统服务、公地资源（渔业）、森林、土地等自然资源经济 (Tietenberg §16-19)",
      "环境政策工具与设计 (Tietenberg §7)",
      "污染控制经济学总览与空气/水/有毒物质污染控制 (Tietenberg §7-10)",
      "排污权交易制度 (Tietenberg §13)",
      "环境法规与监管 (Kraft §3-4)",
      "多层级环境治理体系 (Kraft §6)",
      "空气、水与固废污染立法分述 (Kraft §5-7)",
      "能源政策 (Kraft §5-7)",
      "国际环境治理 (Kraft §8)",
      "全球气候治理与碳中和政策 (Tietenberg §14)",
      "环境正义与公平 (Kraft §11)"
    ],
  },
  'engineering/circular-economy-carbon-neutrality': {
    title: "循环经济与碳中和",
    books: [
          "Lyle, \"Regenerative Design for Sustainable Development\" (1st ed., 1994)",
          "MacDowell, \"Carbon Capture and Storage\" (2nd ed., 2020)"
    ],
    chapters: [
      "循环经济原理与模式 (《循环经济概论》 §1-3)",
      "物质流分析（MFA）与循环经济评价指标 (《循环经济概论》 §4)",
      "再制造与产品再生设计 (《循环经济概论》 §6)",
      "工业共生与资源循环 (《循环经济概论》 §7)",
      "生命周期评价方法 (《循环经济概论》 §5)",
      "碳核算标准（ISO 14064/GHG Protocol） (Rackley §2)",
      "碳足迹核算与碳计量 (Rackley §3)",
      "碳市场与碳定价（碳交易/碳税）机制 (Rackley §11)",
      "碳捕集与封存技术 (Rackley §5-7)",
      "碳捕集利用与封存（CCUS） (Rackley §8)",
      "BECCS 与直接空气捕集（DAC）技术 (Rackley §14)",
      "负排放技术与碳汇 (Rackley §13)",
      "零碳能源系统 (《循环经济概论》 §9)"
    ],
  },
  'engineering/biomimetic-engineering': {
    title: "仿生工程",
    books: [
          "Vincent, \"Structural Biomaterials\" (3rd ed., 2012)",
          "Bar-Cohen, \"Biomimetics: Nature-Based Innovation\" (1st ed., 2011)"
    ],
    chapters: [
      "仿生学原理与方法 (Bar-Cohen §1-2)",
      "仿生材料设计 (Vincent §3-5)",
      "仿生结构与力学 (Vincent §7-8)",
      "仿生传感与执行 (Bar-Cohen §5)",
      "仿生表面与界面 (Bar-Cohen §6)",
      "仿生能源与仿生绿色化学（光合作用模拟/人工光合） (Bar-Cohen §7)",
      "仿生机器人与运动 (Bar-Cohen §8)",
      "仿生运动机理（生物运动学/动力学仿生） (Bar-Cohen §9)",
      "仿生功能系统与自修复材料 (Vincent §10)",
      "仿生医学器件与应用 (Bar-Cohen §10)",
      "生物启发计算与群体智能简述 (Bar-Cohen §12)"
    ],
  },
  'engineering/fermentation-engineering': {
    title: "发酵工程",
    books: [
          "Stanbury, Whitaker, Hall, \"Principles of Fermentation Technology\" (3rd ed., 2016)",
          "Shuler, Kargi, \"Bioprocess Engineering: Basic Concepts\" (3rd ed., 2017)"
    ],
    chapters: [
      "微生物生长动力学 (Shuler §6-7)",
      "发酵动力学模型（Monod/产物抑制） (Shuler §6-7)",
      "发酵培养基设计 (Stanbury §3)",
      "灭菌与无菌技术 (Stanbury §5)",
      "发酵罐设计与传质 (Stanbury §7-8)",
      "发酵过程控制 (Stanbury §9)",
      "下游处理与产物回收 (Stanbury §10)",
      "发酵产物分离纯化与制剂 (Stanbury §11)",
      "分批补料与连续发酵操作策略 (Shuler §8)",
      "动植物细胞培养与生物反应器 (Shuler §9)",
      "发酵放大与规模效应 (Stanbury §13)",
      "工业发酵案例（抗生素/氨基酸/疫苗） (Stanbury §14)"
    ],
  },
  'foundations/mineralogy': {
    title: "矿物学",
    books: [
          "Klein, Philpotts, \"Earth Materials: Introduction to Mineralogy and Petrology\" (2nd ed., 2017)",
          "Nesse, \"Introduction to Mineralogy\" (3rd ed., 2017)"
    ],
    chapters: [
      "矿物晶体化学 (Nesse §2-3)",
      "矿物形态与对称 (Nesse §4-5)",
      "矿物物理性质 (Nesse §6)",
      "硅酸盐矿物分类 (Klein §6)",
      "矿物光性与鉴定 (Nesse §7)",
      "矿物成因与共生 (Klein §8)",
      "X 射线衍射（XRD）与晶体结构测定 (Klein §6)",
      "非硅酸盐系统矿物学（氧化物/硫化物/碳酸盐） (Klein §14-16)",
      "晶体学基础（Miller 指数/点群/空间群） (Klein §5)"
    ],
  },
  'foundations/petrology': {
    title: "岩石学",
    books: [
          "Winter, \"Principles of Igneous and Metamorphic Petrology\" (2nd ed., 2013)",
          "Boggs, \"Principles of Sedimentology and Stratigraphy\" (5th ed., 2012)"
    ],
    chapters: [
      "岩浆成因与演化 (Winter §10-11)",
      "火成岩分类与岩相学 (Winter §2-4)",
      "相图与相律：热力学基础与一元/二元/三元系统相图 (Winter §5-7)",
      "变质作用与变质相 (Winter §21-25)",
      "变质岩分类与结构 (Winter §22-23)",
      "变质反应与变质热力学、P-T 轨迹与温压计 (Winter §26-27)",
      "沉积岩形成与成岩作用 (Boggs §1-2)",
      "沉积岩的分类 (Boggs §4-6)",
      "沉积构造与沉积环境、层序地层学 (Boggs 后半部分)",
      "岩石构造与组构 (Winter §3-4, 23)",
      "火成岩地球化学与同位素 (Winter §8-9)",
      "岩石大地构造与构造环境 (Winter §13-20)"
    ],
  },
  'foundations/paleontology': {
    title: "古生物学",
    books: [
          "Prothero, \"Bringing Fossils to Life: An Introduction to Paleobiology\" (3rd ed., 2013)",
          "Clarkson, \"Invertebrate Palaeontology and Evolution\" (4th ed., 1998)"
    ],
    chapters: [
      "化石保存与埋藏学 (Prothero §1-2)",
      "系统学与分类学基础：物种概念与支序分类 (Prothero §3-4)",
      "微体古生物与化石记录 (Prothero §12)",
      "无脊椎动物演化（按门类展开） (Clarkson §2-5)",
      "脊椎动物演化 (Prothero §18)",
      "古植物与孢粉学 (Prothero §20)",
      "演化模式与灭绝事件 (Prothero §5-6)",
      "古生态学与群落分析 (Prothero §8)",
      "古生物地理学 (Prothero §9)",
      "生物地层学 (Prothero §10)"
    ],
  },
  'foundations/geochemistry': {
    title: "地球化学",
    books: [
          "White, \"Geochemistry: An Introduction\" (2nd ed., 2013)",
          "Faure, \"Principles of Isotope Geology\" (2nd ed., 1986)"
    ],
    chapters: [
      "地球化学组成与元素丰度 (White §1-2)",
      "同位素地球化学 (Faure §1-3)",
      "稳定同位素地球化学应用：H、O、C、S、N 分馏与示踪 (White §9)",
      "微量元素地球化学 (White §7)",
      "地幔与地壳地球化学、岩浆过程示踪 (White §11)",
      "有机地球化学 (White §12)",
      "生物地球化学与全球元素循环 (White §12 之碳循环与气候；Faure §24-25)",
      "水-岩相互作用 (White §6)",
      "放射性同位素与年代学 (Faure §11-12)",
      "地球化学热力学与相平衡 (White §2-4)"
    ],
  },
  'foundations/geophysics': {
    title: "地球物理学",
    books: [
          "Fowler, \"The Solid Earth: An Introduction to Global Geophysics\" (2nd ed., 2005)",
          "Lowrie, \"Fundamentals of Geophysics\" (2nd ed., 2007)"
    ],
    chapters: [
      "地球形状与重力学 (Lowrie §2)",
      "地磁场与古地磁 (Lowrie §5)",
      "地震学基础 (Fowler §4)",
      "地震机制与灾害评估：震级、烈度与地震预报 (Lowrie §3.5；Fowler §4)",
      "岩石物理性质：密度、磁化率与弹性波速 (Lowrie §2-3, 5.3)",
      "地热学与地球内部温度 (Fowler §7)",
      "地球内部结构与板块构造 (Fowler §2)",
      "地球物理探测方法 (Lowrie §2-5)",
      "地电学与电磁方法 (Lowrie §4.3)",
      "板块运动与地幔动力学 (Fowler §8)",
      "地球的年龄与地质年代学：放射性测年与地质年代表 (Lowrie §4.1)"
    ],
  },
  'intermediate/sedimentology-stratigraphy': {
    title: "沉积学与地层学",
    books: [
          "Boggs, \"Principles of Sedimentology and Stratigraphy\" (5th ed., 2012)",
          "Catuneanu, \"Principles of Sequence Stratigraphy\" (1st ed., 2006)"
    ],
    chapters: [
      "风化作用与土壤、碎屑物搬运与沉积作用 (Boggs §1-2)",
      "沉积结构与沉积构造 (Boggs §3-4)",
      "硅质碎屑岩与沉积岩分类 (Boggs §5)",
      "碳酸盐沉积学 (Boggs §6)",
      "成岩作用 (Boggs §5-7)",
      "大陆沉积环境 (Boggs §8)",
      "滨岸与边缘海沉积环境 (Boggs §9)",
      "硅质碎屑海相沉积环境 (Boggs §10)",
      "碳酸盐与蒸发岩沉积环境 (Boggs §11)",
      "岩石地层学与地层对比 (Boggs §12)",
      "地震地层学与磁性地层学 (Boggs §13)",
      "生物地层学与化石分带 (Boggs §14)",
      "年代地层学与地质时间/同位素定年 (Boggs §15)",
      "沉积盆地分析 (Boggs §16)",
      "层序地层学 (Catuneanu §3-5)",
      "层序地层学模型 (Catuneanu §6-8)"
    ],
  },
  'intermediate/structural-geology': {
    title: "构造地质学",
    books: [
          "Fossen, \"Structural Geology\" (2nd ed., 2016)",
          "Davis, Reynolds, Kluth, \"Structural Geology of Rocks and Regions\" (3rd ed., 2011)"
    ],
    chapters: [
      "应力与应变分析 (Fossen §2-4)",
      "显微构造与变形机制 (Fossen §11)",
      "褶皱构造与机制 (Fossen §12)",
      "断层与剪切带 (Fossen §9)",
      "节理与裂隙分析 (Fossen §8)",
      "韧性剪切带与糜棱岩 (Fossen §16)",
      "伸展构造与裂谷、走滑构造体系 (Fossen §18-19)",
      "挤压构造与逆冲推覆、断层相关褶皱 (Fossen §17, Davis 逆冲构造)",
      "构造分析方法 (Davis §10)",
      "区域构造与造山带 (Davis §20)",
      "盐构造与底辟作用 (Fossen §20, Davis §18)"
    ],
  },
  'intermediate/seismology': {
    title: "地震学",
    books: [
          "Stein, Wysession, \"An Introduction to Seismology, Earthquakes, and Earth Structure\" (1st ed., 2003)",
          "Shearer, \"Introduction to Seismology\" (3rd ed., 2019)"
    ],
    chapters: [
      "弹性理论与地震波 (Shearer §2-3)",
      "地震波传播与射线理论 (Stein §2-3)",
      "震源机制与震源参数 (Stein §4-5)",
      "面波与地球自由振荡 (Stein §5, Shearer §8)",
      "地震观测与地震仪 (Shearer §11)",
      "地震定位方法 (Shearer §4-5)",
      "地球内部结构与地震学成像 (Stein §8-9, Shearer §4-5)",
      "震源破裂过程与震源谱/矩张量反演 (Stein §8, Shearer §9)",
      "地震层析成像 (Stein §7, Shearer §5)",
      "地震危险性与预测 (Stein §10, Shearer §10)",
      "强震动地震学与工程地震 (Stein §10)"
    ],
  },
  'intermediate/volcanology': {
    title: "火山学",
    books: [
          "Francis, Oppenheimer, \"Volcanoes\" (2nd ed., 2004)",
          "Sigurdsson, \"The Encyclopedia of Volcanoes\" (2nd ed., 2015)"
    ],
    chapters: [
      "岩浆的物理性质 (Sigurdsson §3)",
      "火山喷发机制与类型 (Francis §5-6)",
      "火山类型与火山地形、破火山口 (Sigurdsson §12-14)",
      "火山气体与喷发柱动力学 (Sigurdsson §32, Francis §5)",
      "火山喷发产物 (Sigurdsson §15-17)",
      "火山碎屑沉积与相模式 (Sigurdsson §25)",
      "火山口/地热系统与火山-水文交互 (Sigurdsson §48-50)",
      "火山与板块构造 (Sigurdsson §2)",
      "火山监测与预警 (Francis §7-8)",
      "火山灾害评估 (Sigurdsson §60)",
      "火山喷发与气候影响 (Sigurdsson §65)"
    ],
  },
  'intermediate/meteorology': {
    title: "气象学",
    books: [
          "Wallace, Hobbs, \"Atmospheric Science: An Introductory Survey\" (2nd ed., 2006)",
          "Ahrens, Henson, \"Meteorology Today: An Introduction to Weather, Climate, and the Environment\" (12th ed., 2019)"
    ],
    chapters: [
      "大气组成与结构 (Wallace §1-2)",
      "大气热力学与辐射 (Wallace §3-4)",
      "云与降水物理 (Wallace §6)",
      "大气化学与大气污染 (Wallace §5, Ahrens §19)",
      "大气环流与一般环流 (Wallace §7)",
      "气团与锋面 (Ahrens §11)",
      "中纬度气旋与天气系统 (Wallace §8, Ahrens §12)",
      "热带气旋与飓风 (Ahrens §16)",
      "中尺度对流系统与龙卷 (Ahrens §14-15)",
      "大气边界层 (Wallace §9)",
      "气象观测、预报与雷达卫星遥感 (Ahrens §13)"
    ],
  },
  'intermediate/climatology': {
    title: "气候学",
    books: [
          "Hartmann, \"Global Physical Climatology\" (2nd ed., 2016)",
          "Ruddiman, \"Earth's Climate: Past and Future\" (3rd ed., 2013)"
    ],
    chapters: [
      "气候系统组成 (Hartmann §1-2)",
      "能量平衡与辐射传输 (Hartmann §3-4)",
      "水循环与水文气候 (Hartmann §5)",
      "大气环流与气候 (Hartmann §6)",
      "海洋环流与气候 (Hartmann §7)",
      "海洋-大气耦合与 ENSO (Hartmann §8)",
      "气候敏感性与反馈机制 (Hartmann §10)",
      "全球气候模式与气候模拟 (Hartmann §11)",
      "自然与人为气候变化 (Hartmann §12-13)",
      "古气候与气候历史 (Ruddiman §8-12)",
      "气候变率与年际-年代际振荡 (Hartmann §8, Ruddiman §11)"
    ],
  },
  'intermediate/atmospheric-dynamics': {
    title: "大气动力学",
    books: [
          "Holton, Hakim, \"An Introduction to Dynamic Meteorology\" (5th ed., 2013)",
          "Vallis, \"Atmospheric and Oceanic Fluid Dynamics\" (2nd ed., 2017)"
    ],
    chapters: [
      "大气运动方程组 (Holton §1-2)",
      "地转风与平衡流、热成风 (Holton §3)",
      "环流定理、涡度与大气环流 (Holton §4, §10)",
      "大气波动与罗斯贝波 (Holton §5)",
      "准地转分析与天气尺度动力学 (Holton §6-7)",
      "斜压不稳定性与气旋发展 (Holton §7)",
      "行星边界层动力学与埃克曼抽吸 (Holton §8)",
      "中层大气动力学与平流层 (Holton §12)",
      "热带动力学与赤道波 (Holton §11)",
      "数值天气预报与模式 (Holton §13)"
    ],
  },
  'intermediate/physical-oceanography': {
    title: "物理海洋学",
    books: [
          "Knauss, Garfield, \"Introduction to Physical Oceanography\" (3rd ed., 2017)",
          "Talley, Pickard, Emery, Swift, \"Descriptive Physical Oceanography: An Introduction\" (6th ed., 2011)"
    ],
    chapters: [
      "海水物理性质与状态方程 (Knauss §2, Talley §3)",
      "大洋水团与温盐结构、TS 图解 (Talley §4, Knauss §2)",
      "海洋热量与水量平衡 (Talley §5, Knauss §3-4)",
      "海洋-大气相互作用 (Knauss §3, Talley §5)",
      "地转流与海洋动力学方程、海流测量 (Knauss §5-6, Talley §6-7)",
      "风驱大洋环流与埃克曼输运 (Knauss §6)",
      "主要洋流系统与区域大洋环流（大西洋/太平洋/印度洋/北冰洋）(Knauss §7, Talley §9-13)",
      "温盐环流与深层水形成（NADW/AABW）(Knauss §8, Talley §14)",
      "中尺度涡旋与内波 (Knauss §8)",
      "风浪与涌浪、海浪谱 (Knauss §9)",
      "潮汐与潮流 (Knauss §10)",
      "海岸海洋学与河口动力学 (Knauss §11)"
    ],
  },
  'intermediate/marine-chemistry': {
    title: "海洋化学",
    books: [
          "Millero, \"Chemical Oceanography\" (4th ed., 2013)",
          "Libes, \"An Introduction to Marine Biogeochemistry\" (2nd ed., 2009)"
    ],
    chapters: [
      "海水化学组成与盐度 (Millero §1-2)",
      "微量元素与痕量金属 (Millero §3, §8)",
      "海洋碳循环与碳酸盐体系 (Libes §10-11)",
      "海洋酸化 (Millero §7, Libes §10-11)",
      "海洋营养盐循环 (Libes §14)",
      "海-气气体交换与溶解气体 (Millero §5-6)",
      "海洋氧化还原过程与沉积物早期成岩 (Libes §17-18)",
      "海洋同位素化学 (Libes §16)",
      "海洋有机地球化学 (Libes §20)",
      "化学示踪与古海洋学 (Libes §15)",
      "海洋污染化学 (Millero §10)"
    ],
  },
  'intermediate/marine-biology': {
    title: "海洋生物学",
    books: [
          "Levinton, \"Marine Biology: Function, Biodiversity, Ecology\" (5th ed., 2021)",
          "Castro, Huber, \"Marine Biology\" (11th ed., 2018)"
    ],
    chapters: [
      "海洋生物分类与多样性 (Castro §3-5)",
      "浮游生物生态 (Levinton §3-4)",
      "生殖策略、生活史与幼体生态 (Levinton §8)",
      "底栖生物群落 (Levinton §5-6)",
      "潮间带与河口生态系统（盐沼、红树林）(Castro §11)",
      "海洋鱼类与脊椎动物 (Castro §7-9)",
      "深海生物与热液生态 (Levinton §7)",
      "珊瑚礁生态系统 (Levinton §10)",
      "海洋食物网与能流 (Levinton §9)",
      "渔业资源与海洋管理、人类活动影响 (Castro §18)",
      "海洋生态系统与保护 (Levinton §11)"
    ],
  },
  'intermediate/space-weather': {
    title: "空间天气学",
    books: [
          "Bothmer, Vourlidas, \"Space Weather: Physics and Effects\" (1st ed., 2021)",
          "Hanslmeier, \"Space Weather: A Multidisciplinary Approach\" (1st ed., 2017)"
    ],
    chapters: [
      "太阳活动与日冕物质抛射 (Bothmer §2-3)",
      "太阳风与行星际介质 (Bothmer §4)",
      "太阳高能粒子事件（SEP）与质子事件 (Bothmer §6-7)",
      "磁层物理与磁暴 (Bothmer §5)",
      "地球辐射带与高能粒子 (Bothmer §6)",
      "电离层扰动与影响 (Hanslmeier §6)",
      "极光、高层大气与电离层-热层耦合 (Hanslmeier §5)",
      "地磁感应电流（GIC）与地面基础设施影响 (Bothmer §9)",
      "空间天气对技术系统的影响 (Bothmer §9)",
      "空间天气监测与预报 (Hanslmeier §9)"
    ],
  },
  'intermediate/geodesy': {
    title: "大地测量学",
    books: [
          "Hofmann-Wellenhof, Moritz, \"Physical Geodesy\" (2nd ed., 2006)",
          "Torge, Müller, \"Geodesy\" (4th ed., 2012)"
    ],
    chapters: [
      "大地测量基础与坐标系统 (Torge §1-2)",
      "参考系与时间系统 (Torge §2)",
      "地球形状与重力场 (Hofmann-Wellenhof §2-3)",
      "地球重力场模型与球谐展开 (Hofmann-Wellenhof §2, Torge §3)",
      "重力测量与重力位理论 (Hofmann-Wellenhof §4-6)",
      "大地水准面确定 (Hofmann-Wellenhof §7)",
      "高程系统与水准测量（正高/正常高/大地高）(Torge §3-4)",
      "几何大地测量与三角/导线测量 (Torge §5)",
      "大地测量网与测量平差 (Torge §7)",
      "卫星大地测量与定位方法 (Torge §6)",
      "地壳形变监测与地球动力学 (Torge §8)"
    ],
  },
  'intermediate/gnss-positioning': {
    title: "GNSS 定位与导航",
    books: [
          "Hofmann-Wellenhof, Lichtenegger, Wasle, \"GNSS: Global Navigation Satellite Systems\" (2nd ed., 2008)",
          "Misra, Enge, \"Global Positioning System: Signals, Measurements, and Performance\" (2nd ed., 2011)"
    ],
    chapters: [
      "GNSS 系统组成与星座 (Hofmann-Wellenhof §4-8, Misra §2-3)",
      "卫星轨道与星历、卫星钟差 (Misra §4, Hofmann-Wellenhof §3)",
      "信号结构与测距码 (Misra §9, Hofmann-Wellenhof §5)",
      "观测量与定位原理 (Misra §5-6)",
      "误差源与修正模型 (Misra §5)",
      "载波相位观测与整周模糊度解算 (Misra §7)",
      "差分 GPS 与 RTK 定位 (Hofmann-Wellenhof §9, Misra §5)",
      "精密单点定位技术 (Hofmann-Wellenhof §10)",
      "精度几何因子（DOP）与定位精度评估 (Misra §6)",
      "多星座与多频定位 (Hofmann-Wellenhof §11, Misra §3)",
      "GNSS 应用与组合导航 (Hofmann-Wellenhof §12)"
    ],
  },
  'intermediate/gis': {
    title: "地理信息系统（GIS）",
    books: [
          "Longley, Goodchild, Maguire, Rhind, \"Geographic Information Science and Systems\" (4th ed., 2015)",
          "Bolstad, \"GIS Fundamentals: A First Text on Geographic Information Systems\" (6th ed., 2019)"
    ],
    chapters: [
      "GIS基础与空间数据模型 (Longley §3-5)",
      "矢量与栅格数据结构 (Bolstad §4-5)",
      "坐标系统与地图投影 (Bolstad §4)",
      "地形分析与数字高程模型（DEM）(Bolstad §12)",
      "空间分析方法 (Longley §14-15)",
      "空间数据库设计 (Longley §9)",
      "地理可视化与制图 (Bolstad §7)",
      "空间数据质量与不确定性 (Bolstad §9)",
      "遥感与 GIS 集成 (Bolstad §13)",
      "空间统计 (Bolstad §14)",
      "GIS应用与项目管理 (Longley §16-17)"
    ],
  },
  'advanced/deep-space-exploration': {
    title: "深空探测",
    books: [
          "Brown, \"Elements of Spacecraft Design\" (1st ed., 2002)",
          "Wertz, Everett, Puschell, \"Space Mission Engineering: The New SMAD\" (1st ed., 2011)"
    ],
    chapters: [
      "深空探测任务案例与科学目标总览 (Brown §1 / SMAD §1)",
      "任务分析与系统工程（任务生命周期） (SMAD §1-6)",
      "轨道力学与星际轨道设计 (Brown §3)",
      "空间环境与防护（空间辐射与微流星体） (Brown §2 / SMAD §7)",
      "航天器结构与机构 (Brown §10)",
      "推进系统设计 (Brown §4)",
      "深空通信与测控 (Wertz §19)",
      "电源系统设计 (Brown §6)",
      "行星着陆与表面探测 (Wertz §21)",
      "热控系统设计 (Brown §7)",
      "自主导航与制导 (Wertz §15)",
      "姿态确定与控制 ADCS (Brown §5 / SMAD §19)",
      "星载数据管理与星上计算机 C&DH (Brown §8 / SMAD §20)"
    ],
  },
  'advanced/climate-change-science': {
    title: "气候变化科学",
    books: [
          "Ruddiman, \"Earth's Climate: Past and Future\" (3rd ed., 2013)",
          "IPCC, \"Climate Change 2021: The Physical Science Basis\" (6th Assessment, 2021)"
    ],
    chapters: [
      "气候变化历史与古气候 (Ruddiman §8-12)",
      "构造尺度气候演化（深时气候） (Ruddiman §3-7)",
      "千年尺度与突变气候（D-O 振荡、海因里希事件） (Ruddiman §15)",
      "观测到的气候系统变化 (IPCC §2)",
      "温室效应与辐射强迫 (IPCC §7)",
      "气候系统反馈机制 (IPCC §7)",
      "气溶胶与短寿命气候强迫因子 (IPCC §6)",
      "气候模型与情景预测 (IPCC §1-4)",
      "水循环变化 (IPCC §8)",
      "碳循环与生物地球化学 (IPCC §5)",
      "气候变化的检测与归因 (IPCC §3)",
      "海洋、冰冻圈与海平面上升 (IPCC §9)",
      "区域气候变化与气候连接 (IPCC §10)",
      "极端事件与气候影响 (IPCC §11)",
      "区域气候风险评估 (IPCC §12)"
    ],
  },
  'advanced/planetary-geology': {
    title: "行星地质学",
    books: [
          "Melosh, \"Planetary Surface Processes\" (1st ed., 2011)",
          "Carr, \"The Surface of Mars\" (2nd ed., 2006)"
    ],
    chapters: [
      "行星形成与早期演化 (Melosh §1-2)",
      "行星内部结构 (Melosh §3)",
      "行星构造过程（张性与压性构造） (Melosh §4)",
      "撞击坑形成与作用 (Melosh §6)",
      "陨石坑定年与行星地质年代学 (Melosh §6 / Carr §3)",
      "风化与风化层（风化与风化壳） (Melosh §7)",
      "坡地与物质运移 (Melosh §8)",
      "风成过程与地貌 (Melosh §9)",
      "水成过程与河谷地貌 (Melosh §10)",
      "火星表面地质 (Carr §3-5)",
      "行星大气与气候演化 (Carr §13)",
      "冰卫星与海洋世界 (Melosh §11)",
      "行星火山活动 (Melosh §5 / Carr §4)",
      "月球地质学（月海与高地） (Greeley §4)",
      "金星与水星地质及类地行星对比 (Greeley §5-6)",
      "小行星与彗星地质 (Greeley §8)"
    ],
  },
  'advanced/space-physics': {
    title: "空间物理学",
    books: [
          "Kivelson, Russell, \"Introduction to Space Physics\" (1st ed., 1995)",
          "Gurnett, Bhattacharjee, \"Introduction to Plasma Physics: With Space, Laboratory and Astrophysical Applications\" (2nd ed., 2017)"
    ],
    chapters: [
      "空间等离子体基础 (Gurnett §1-3)",
      "太阳大气与日球层 (Kivelson §3)",
      "太阳爆发活动（耀斑、CME 与太阳高能粒子） (Kivelson §3)",
      "行星际介质与太阳风 (Kivelson §4)",
      "无碰撞激波（弓激波与日球层激波） (Kivelson §5)",
      "地球磁层物理（磁层顶、磁尾与磁重联） (Kivelson §9-10)",
      "磁层对流与等离子体输运 (Kivelson §10)",
      "电离层物理 (Kivelson §7)",
      "极光与磁层-电离层耦合 (Kivelson §14)",
      "空间等离子体波动 (Gurnett §8-9 / Kivelson §12)",
      "磁重联与爆发过程 (Kivelson §9)",
      "地球辐射带动力学 (Kivelson §10, §13)",
      "磁暴与磁层亚暴（磁层动力学） (Kivelson §13)",
      "空间天气（日地耦合与空间环境应用） (Kivelson §13-14)",
      "比较行星磁层（水星、木星、土星、金星与彗星） (Kivelson §15)"
    ],
  },
  'intermediate/biochemistry': {
    title: "生物化学",
    books: [
          "Nelson & Cox, \"Lehninger Principles of Biochemistry\" (8th, 2021)",
          "Berg, Tymoczko & Stryer, \"Biochemistry\" (9th, 2019)"
    ],
    chapters: [
      "生物化学基础 (Lehninger §1)",
      "氨基酸、肽与蛋白质 (Lehninger §3)",
      "蛋白质的三维结构 (Lehninger §4)",
      "蛋白质功能：结合与变构（血红蛋白）(Lehninger §5)",
      "酶催化机制 (Lehninger §6)",
      "糖类与糖生物学 (Lehninger §7)",
      "核苷酸与核酸 (Lehninger §8)",
      "DNA 技术与基因工程方法 (Lehninger §9)",
      "脂质：结构与功能 (Lehninger §10)",
      "生物膜与物质转运 (Lehninger §11)",
      "信号转导 (Lehninger §12)",
      "维生素与辅酶 (Lehninger §13)",
      "糖酵解与糖异生 (Lehninger §14)",
      "代谢调节原理 (Lehninger §15)",
      "三羧酸循环 (Lehninger §16)",
      "氧化磷酸化与光合磷酸化 (Lehninger §19)",
      "脂质代谢 (Lehninger §17, §21)",
      "氨基酸代谢与尿素循环 (Lehninger §18)",
      "哺乳动物激素调节与代谢整合 (Lehninger §23)",
      "核苷酸代谢 (Lehninger §22)",
      "DNA复制与修复 (Lehninger §25)",
      "转录与 RNA 加工 (Lehninger §26)",
      "翻译与蛋白质合成 (Lehninger §27)",
      "基因表达调控 (Lehninger §28)"
    ],
  },
  'intermediate/taxonomy-systematics': {
    title: "生物分类学与系统学",
    books: [
          "Schuh & Brower, \"Biological Systematics\" (3rd, 2020)",
          "Mayr & Ashlock, \"Principles of Systematic Zoology\" (1991)"
    ],
    chapters: [
      "系统学历史与基本概念 (Schuh §1)",
      "系统学与科学哲学 (Schuh §2)",
      "物种概念与界定 (Schuh §7, Mayr §2)",
      "特征与同源性 (Schuh §3-4)",
      "支序分析原理 (Schuh §5)",
      "分支图构建与检验 (Schuh §5-6)",
      "系统发育与分类体系 (Schuh §8)",
      "分子系统学方法 (Schuh §11)",
      "分子钟与时间树 (Schuh §11)",
      "历史生物地理学与宿主-寄生虫协同进化 (Schuh §9)",
      "基于系统发育的生态与适应推断 (Schuh §10)",
      "生物多样性与保护 (Schuh §12)",
      "生物命名法与法规 (Mayr §5)"
    ],
  },
  'intermediate/comparative-anatomy': {
    title: "生物解剖与比较解剖",
    books: [
          "Kardong, \"Vertebrates: Comparative Anatomy, Function, Evolution\" (8th, 2018)",
          "Hildebrand & Goslow, \"Analysis of Vertebrate Structure\" (6th, 2001)"
    ],
    chapters: [
      "脊椎动物起源与演化 (Kardong §2-3)",
      "组织与基本结构层次 (Kardong §4)",
      "脊椎动物胚胎学 (Kardong §5)",
      "皮肤及其衍生物 (Kardong §6)",
      "骨骼系统：颅骨与中轴骨骼 (Kardong §7-8)",
      "附肢骨骼与关节 (Kardong §9)",
      "肌肉系统 (Kardong §10)",
      "消化与呼吸系统 (Kardong §11, §13)",
      "循环系统 (Kardong §12)",
      "泌尿生殖系统 (Kardong §14)",
      "内分泌系统 (Kardong §15)",
      "神经系统与感觉器官 (Kardong §16-17)",
      "运动与功能形态学（游泳/飞行/陆地运动）(Kardong §18)"
    ],
  },
  'intermediate/developmental-biology': {
    title: "发育生物学",
    books: [
          "Gilbert, \"Developmental Biology\" (12th, 2019)",
          "Wolpert & Tickle, \"Principles of Development\" (5th, 2015)"
    ],
    chapters: [
      "发育的模式与原理 (Gilbert §1-3)",
      "受精与早期卵裂 (Gilbert §7)",
      "果蝇体轴建立的遗传学 (Gilbert §9)",
      "囊胚与原肠胚形成 (Gilbert §10-12)",
      "神经胚形成与体轴建立 (Gilbert §13)",
      "神经嵴细胞与轴突导向 (Gilbert §15)",
      "同源异型框基因与体节发育 (Gilbert §17)",
      "中胚层与内胚层分化 (Gilbert §17-18, §20)",
      "器官发生与四肢发育 (Gilbert §19-20)",
      "变态、退化与衰老 (Gilbert §21, §23)",
      "干细胞与再生 (Gilbert §5, §22)",
      "环境对发育的调节 (Gilbert §24)"
    ],
  },
  'intermediate/structural-biology': {
    title: "结构生物学",
    books: [
          "Branden & Tooze, \"Introduction to Protein Structure\" (2nd, 1999)",
          "Liljas et al., \"Textbook of Structural Biology\" (2009)"
    ],
    chapters: [
      "蛋白质结构层次 (Branden §1-2)",
      "氨基酸与一级结构 (Branden §1)",
      "α螺旋与β折叠 (Branden §3)",
      "蛋白质折叠与三级结构 (Branden §4)",
      "球状蛋白结构与功能 (Branden §5-6)",
      "膜蛋白结构 (Branden §12)",
      "X射线晶体学原理 (Liljas §2)",
      "核磁共振与冷冻电镜 (Liljas §3-4)",
      "DNA 识别蛋白的结构基础 (Branden §8-10)",
      "RNA 结构与蛋白质-RNA 识别 (Liljas §5)",
      "蛋白质-配体相互作用与酶的结构基础 (Branden §11, Liljas §8)",
      "结构生物学中的信号转导蛋白 (Branden §13, Liljas §14)",
      "核酸结构（DNA/RNA）(Branden §7, Liljas §5)",
      "蛋白质-蛋白质复合物 (Liljas §3)",
      "蛋白质结构预测（AlphaFold/CASP）(Branden §17)"
    ],
  },
  'intermediate/immunology': {
    title: "免疫学",
    books: [
          "Janeway et al., \"Immunobiology\" (10th, 2022)",
          "Abbas, Lichtman & Pillai, \"Cellular and Molecular Immunology\" (10th, 2021)"
    ],
    chapters: [
      "免疫系统要素与概念 (Janeway §1)",
      "固有免疫 (Janeway §2-3)",
      "抗原识别受体 (Janeway §4)",
      "淋巴细胞抗原受体的生成：V(D)J 重组 (Janeway §5)",
      "抗原加工与呈递（MHC）(Janeway §6)",
      "免疫受体信号转导 (Janeway §7)",
      "T细胞发育与活化 (Janeway §8-9)",
      "B细胞发育与抗体应答 (Janeway §8, §10)",
      "体液与细胞免疫效应机制 (Janeway §9-10)",
      "免疫耐受与自身免疫 (Janeway §15)",
      "过敏与超敏反应 (Janeway §14)",
      "补体系统与激活途径 (Janeway §2)",
      "移植免疫与组织相容性 (Janeway §15)",
      "肿瘤免疫与免疫治疗 (Janeway §16)",
      "免疫缺陷病 (Janeway §13)",
      "黏膜免疫 (Janeway §12)"
    ],
  },
  'intermediate/virology': {
    title: "病毒学",
    books: [
          "Flint et al., \"Principles of Virology\" (5th, 2020)",
          "Knipe & Howley, \"Fields Virology\" (7th, 2021)"
    ],
    chapters: [
      "病毒学基础与分类 (Flint Vol 1 §1)",
      "病毒颗粒结构与组装 (Flint Vol 1 §4, §13)",
      "病毒进入（吸附-穿膜-脱壳）(Flint Vol 1 §5)",
      "病毒基因组复制 (Flint Vol 1 §6-10)",
      "病毒基因表达调控 (Flint Vol 1 §7-8, §11)",
      "病毒致病机制 (Flint Vol 2 §5)",
      "病毒持续性感染与潜伏 (Flint Vol 2 §5)",
      "宿主免疫与抗病毒应答 (Flint Vol 2 §3-4)",
      "病毒与肿瘤 (Flint Vol 2 §6)",
      "病毒疫苗与抗病毒药物 (Flint Vol 2 §7-8)",
      "新发病毒与病毒进化 (Flint Vol 2 §10-11)",
      "病毒载体与诊断技术 (Flint Vol 2 §9)",
      "HIV 与艾滋病 (Fields Vol 3 · 逆转录病毒科)",
      "流感病毒 (Fields Vol 1 · 正黏病毒科)",
      "疱疹病毒 (Fields Vol 2 · 疱疹病毒科)",
      "肝炎病毒 (Fields Vol 1-2 · 黄病毒科/嗜肝DNA病毒科)"
    ],
  },
  'intermediate/behavioral-ecology': {
    title: "行为生态学",
    books: [
          "Davies, Krebs & West, \"An Introduction to Behavioural Ecology\" (4th, 2012)",
          "Alcock, \"Animal Behavior\" (11th, 2013)"
    ],
    chapters: [
      "自然选择与行为适应 (Davies §1)",
      "行为生态学研究方法：假设检验 (Davies §2)",
      "觅食行为与最优化 (Davies §3)",
      "捕食者与猎物策略 (Davies §4)",
      "资源竞争 (Davies §5)",
      "打斗与评估 (Davies §5)",
      "群居生活 (Davies §6)",
      "配偶选择与性选择 (Davies §7)",
      "交配系统 (Davies §9)",
      "亲本照顾与繁殖策略 (Davies §8)",
      "社会行为与利他主义 (Davies §11)",
      "通讯行为 (Davies §14)"
    ],
  },
  'intermediate/plant-physiology': {
    title: "植物生理学",
    books: [
          "Taiz, Zeiger et al., \"Plant Physiology and Development\" (6th, 2014)",
          "Hopkins & Hüner, \"Introduction to Plant Physiology\" (4th, 2008)"
    ],
    chapters: [
      "植物细胞结构与功能 (Taiz §1)",
      "水分与转运 (Taiz §3-4)",
      "矿质营养 (Taiz §5)",
      "溶质转运（共质体/质外体途径）(Taiz §6)",
      "光合作用：光反应 (Taiz §7)",
      "光合作用：碳同化 (Taiz §8)",
      "韧皮部运输 (Taiz §11)",
      "植物呼吸作用 (Taiz §12)",
      "矿质养分同化（氮/硫/磷同化）(Taiz §13)",
      "细胞壁：结构与扩展 (Taiz §14)",
      "细胞信号转导机制 (Taiz §15)",
      "植物激素与信号 (Taiz §15)",
      "植物生长发育与光形态建成 (Taiz §16, §19)",
      "蓝光应答与隐花色素 (Taiz §16)",
      "次生代谢物 (Taiz §23)",
      "逆境生理（干旱/盐/低温）(Taiz §24)",
      "开花与生殖生理 (Taiz §20-21)"
    ],
  },
  'intermediate/animal-physiology': {
    title: "动物生理学",
    books: [
          "Hill, Wyse & Anderson, \"Animal Physiology\" (4th, 2016)",
          "Schmidt-Nielsen, \"Animal Physiology: Adaptation and Environment\" (5th, 1997)"
    ],
    chapters: [
      "稳态与生理调节 (Hill §1)",
      "温度生理与体温调节 (Hill §10-11)",
      "细胞膜转运与细胞生理基础 (Hill §2, §5)",
      "神经元与神经生理 (Hill §12)",
      "突触传递与神经整合 (Hill §13, §15)",
      "感觉系统 (Hill §14)",
      "肌肉与运动 (Hill §19-20)",
      "呼吸生理 (Hill §23)",
      "循环与血液 (Hill §24-25)",
      "渗透调节与排泄 (Hill §27-29)",
      "消化与能量代谢 (Hill §6-7)",
      "内分泌生理 (Hill §16)",
      "生殖生理 (Hill §17)",
      "免疫生理 (Hill §2)"
    ],
  },
  'intermediate/conservation-biology': {
    title: "保护生物学",
    books: [
          "Primack, \"Essentials of Conservation Biology\" (6th, 2014)",
          "Sodhi & Ehrlich, \"Conservation Biology for All\" (2010)"
    ],
    chapters: [
      "保护生物学原理 (Primack §1)",
      "生物多样性价值 (Primack §4-6)",
      "物种灭绝机制 (Primack §7)",
      "生境破坏与破碎化 (Primack §9)",
      "外来入侵种 (Primack §10)",
      "过度开发、污染与资源利用威胁 (Primack §10)",
      "种群与生活史分析 (Primack §11-12)",
      "种群生存力分析（PVA）(Primack §12)",
      "保护遗传与小种群遗传学 (Primack §11)",
      "迁地保护与物种回归/人工繁育 (Primack §13-14)",
      "保护地与保护区设计 (Primack §15-16)",
      "恢复生态学 (Primack §19)",
      "气候变化与生物多样性 (Primack §9)",
      "生态系统服务与可持续利用 (Primack §20-21)",
      "保护政策与立法 (Primack §20-21)"
    ],
  },
  'intermediate/enzymology': {
    title: "酶学",
    books: [
          "Cornish-Bowden, \"Fundamentals of Enzyme Kinetics\" (4th, 2012)",
          "Palmer & Bonner, \"Enzymes: Biochemistry, Biotechnology, Clinical Chemistry\" (2nd, 2007)"
    ],
    chapters: [
      "酶催化基础 (Palmer §1)",
      "米氏动力学与抑制 (Cornish-Bowden §2, §6-7)",
      "稳态速率方程推导与动力学数据处理 (Cornish-Bowden §4-5)",
      "多底物反应动力学（有序/随机/乒乓）(Cornish-Bowden §8)",
      "pH 与温度对酶活的影响 (Cornish-Bowden §10-11)",
      "别构酶与协同性 (Cornish-Bowden §12)",
      "酶反应机制 (Palmer §11)",
      "辅酶与辅因子 (Palmer §11)",
      "酶活性调节 (Palmer §12-14)",
      "酶工程与定向进化 (Palmer §20)",
      "酶固定化与生物传感器 (Palmer §20)",
      "酶的应用与临床 (Palmer §19)"
    ],
  },
  'intermediate/glycobiology': {
    title: "糖生物学",
    books: [
          "Varki et al., \"Essentials of Glycobiology\" (3rd, 2017)",
          "Taylor & Drickamer, \"Introduction to Glycobiology\" (3rd, 2011)"
    ],
    chapters: [
      "糖类结构与命名 (Varki §2)",
      "糖苷键与寡糖合成 (Varki §3)",
      "N-连接与O-连接糖基化 (Varki §9-10)",
      "糖蛋白与蛋白聚糖 (Varki §14, §17)",
      "糖基转移酶 (Varki §6)",
      "糖苷酶与聚糖降解 (Varki §6, §44)",
      "糖脂与鞘糖脂 (Varki §11)",
      "GPI 锚定 (Varki §12)",
      "唾液酸与 Siglec (Varki §15, §35)",
      "凝集素与糖识别 (Varki §28-29)",
      "糖基化与疾病 (Varki §44-46)",
      "先天性糖基化障碍（CDG）(Varki §45)",
      "糖组学技术 (Varki §50-51)"
    ],
  },
  'intermediate/bioimaging': {
    title: "生物成像技术",
    books: [
          "Murphy & Davidson, \"Fundamentals of Light Microscopy and Electronic Imaging\" (2nd, 2013)",
          "Pawley, \"Handbook of Biological Confocal Microscopy\" (3rd, 2006)"
    ],
    chapters: [
      "显微镜光学基础 (Murphy §1)",
      "明场与相差显微术 (Murphy §7)",
      "荧光显微术原理 (Murphy §11)",
      "荧光探针与染料（荧光蛋白/有机荧光团）(Murphy §11-12)",
      "共聚焦与多光子显微术 (Murphy §13-14)",
      "光片显微术与组织透明化（Light-sheet）(Pawley §37)",
      "FRET/FLIM 与单分子成像 (Pawley §27, §45)",
      "超分辨显微技术 (Murphy §15)",
      "电子显微术基础（Bozzola & Russell《Electron Microscopy》）",
      "探测器与数字相机（CCD/CMOS/PMT）(Murphy §17)",
      "活细胞成像 (Pawley §19)",
      "图像处理与分析 (Murphy §17-18)"
    ],
  },
  'advanced/evolutionary-biology': {
    title: "进化生物学（深化）",
    books: [
          "Futuyma & Kirkpatrick, \"Evolution\" (4th, 2017)",
          "Freeman & Herron, \"Evolutionary Analysis\" (5th, 2015)"
    ],
    chapters: [
      "进化思想与证据 (Futuyma §1-2)",
      "自然选择与适应 (Futuyma §3)",
      "遗传变异 (Futuyma §4)",
      "种群遗传理论 (Futuyma §5)",
      "表型进化 (Futuyma §6)",
      "遗传漂变 (Futuyma §7)",
      "进化中的空间 (Futuyma §8)",
      "物种形成 (Futuyma §9)",
      "性选择与性的进化 (Futuyma §10)",
      "合作与冲突 (Futuyma §12)",
      "种间相互作用与协同进化 (Futuyma §13)",
      "分子进化与基因组进化 (Futuyma §14)",
      "进化发育生物学 (Futuyma §15)",
      "系统发育与比较方法 (Futuyma §16)",
      "生命历史 (Futuyma §17)",
      "生物多样性演化与宏进化 (Futuyma §19-20)",
      "人类进化 (Futuyma §21)",
      "进化与社会 (Futuyma §22)"
    ],
  },
  'advanced/systems-biology': {
    title: "系统生物学",
    books: [
          "Alon, \"An Introduction to Systems Biology: Design Principles of Biological Circuits\" (2nd, 2019)",
          "Klipp et al., \"Systems Biology: A Textbook\" (2nd, 2016)"
    ],
    chapters: [
      "系统生物学导论 (Klipp §1)",
      "高通量实验技术 (Klipp §2)",
      "转录网络基本概念与基序 (Alon §1,3)",
      "自动调控 (Alon §2)",
      "时间程序与转录网络全局结构 (Alon §4)",
      "正反馈、双稳态与记忆 (Alon §5)",
      "生物振荡器与负反馈 (Alon §6)",
      "鲁棒性 (Alon §7)",
      "动力学校读与构象校读 (Alon §8)",
      "趋化性鲁棒信号与双功能组件 (Alon §9)",
      "倍数变化检测与动态补偿 (Alon §10-11)",
      "发育图式形成的鲁棒性 (Alon §12)",
      "最优基因线路设计与多目标最优性 (Alon §13-14)",
      "模块性 (Alon §15)",
      "建模方法学：模型构建、参数估计与稳定性分析 (Klipp §4-6)",
      "代谢网络建模 (Klipp §7)",
      "信号转导通路建模 (Klipp §9)",
      "基因调控网络推断 (Klipp §11)",
      "系统生物学与医学 (Klipp §12)"
    ],
  },
  'advanced/single-cell-spatial-omics': {
    title: "单细胞测序与空间组学",
    books: [
          "Kolodziejczyk et al., \"Single-Cell RNA Sequencing: Methods and Protocols\" (2019)",
          "Tamir et al., \"Single-Cell Omics\" (2018)"
    ],
    chapters: [
      "单细胞测序原理与技术 (Kolodziejczyk §1)",
      "微流控与样品制备 (Kolodziejczyk §3)",
      "单细胞全基因组测序与拷贝数变异（scWGS/CNV） (Tamir §6)",
      "单细胞表观组学（scATAC-seq、DNA甲基化与染色质可及性） (Tamir §9)",
      "scRNA-seq数据分析流程 (Kolodziejczyk §7)",
      "细胞类型鉴定与聚类 (Kolodziejczyk §8)",
      "细胞通讯推断与配受体分析（CellChat/CellPhoneDB） (Tamir §11)",
      "细胞轨迹与发育推断 (Tamir §4)",
      "空间转录组学方法 (Tamir §12)",
      "空间多组学与空间蛋白质组学（CODEX/MIBI） (Tamir §13)",
      "多组学单细胞整合 (Tamir §15)",
      "人类细胞图谱与跨平台整合（HCA、批次校正） (Tamir §16)",
      "单细胞免疫组库（scTCR/scBCR） (Tamir §10)",
      "疾病与临床应用 (Tamir §18)"
    ],
  },
  'advanced/gene-editing': {
    title: "基因编辑（CRISPR）",
    books: [
          "Doudna & Mali, \"CRISPR-Cas: A Laboratory Manual\" (2016)",
          "Bollmann & Venselaar, \"CRISPR-Cas: Methods and Protocols\" (2021)"
    ],
    chapters: [
      "CRISPR-Cas系统发现与机制 (Doudna §1)",
      "Cas9结构与功能 (Doudna §2)",
      "Cas12a/Cas13等新型系统与RNA编辑 (Doudna §3)",
      "sgRNA设计与靶点选择 (Doudna §4)",
      "基因敲除与敲入技术 (Doudna §5)",
      "CRISPRi/CRISPRa转录调控工具 (Bollmann §6)",
      "脱靶检测与特异性评估（GUIDE-seq/CIRCLE-seq/DISCOVER-seq） (Bollmann §7)",
      "碱基编辑与先导编辑 (Bollmann §8)",
      "表观基因组编辑（CRISPR-off与融合编辑器） (Bollmann §9)",
      "CRISPR筛选 (Bollmann §10)",
      "体内递送与细胞递送技术（LNP、AAV、电转、RNP） (Bollmann §12)",
      "基于CRISPR的分子诊断（SHERLOCK/DETECTR） (Bollmann §13)",
      "CRISPR在疾病模型中的应用 (Bollmann §14)",
      "基因治疗与伦理 (Bollmann §18)"
    ],
  },
  'advanced/directed-evolution': {
    title: "定向进化与蛋白质设计",
    books: [
          "Arnold & Georgiou, \"Directed Evolution Library Creation: Methods and Protocols\" (2003)",
          "Park, \"Protein Engineering\" (2020)"
    ],
    chapters: [
      "定向进化原理 (Arnold §1)",
      "随机突变与易错PCR (Arnold §2)",
      "DNA重组与嵌合酶 (Arnold §5)",
      "高通量筛选与选择策略 (Arnold §10)",
      "文库构建与筛选的定量设计（库质量与FACS） (Arnold §11)",
      "连续定向进化（PACE/PANCE） (Arnold §12)",
      "展示技术（噬菌体/酵母/核糖体展示） (Arnold §13)",
      "酶活性的定向改造 (Park §3)",
      "酶稳定性与底物特异性 (Park §4)",
      "理性设计与结构导向改造（Rosetta与稳定性预测） (Park §6)",
      "计算蛋白质设计 (Park §7)",
      "机器学习指导的定向进化与蛋白质工程 (Park §8)",
      "从头设计蛋白质 (Park §9)"
    ],
  },
  'life/anatomy': {
    title: "系统解剖学",
    books: [
          "Netter, \"Atlas of Human Anatomy\" (8th, 2022)",
          "Moore, Dalley & Agur, \"Clinically Oriented Anatomy\" (8th, 2017)"
    ],
    chapters: [
      "骨学与关节学 (Moore §2-6)",
      "骨骼肌系统 (Moore §2-7)",
      "头颈部解剖 (Netter §1)",
      "胸部与纵隔 (Netter §3)",
      "腹部与盆部 (Netter §4)",
      "上肢与下肢 (Netter §6)",
      "神经系统解剖 (Netter §7)",
      "心血管系统解剖 (Moore §4)",
      "感觉器解剖（视器与前庭蜗器） (Netter §7)",
      "泌尿与生殖系统解剖 (Netter §5)",
      "消化与呼吸内脏各论（内脏学） (Netter §4)",
      "淋巴系统 (Moore §4)",
      "周围神经与传导通路 (Netter §7)"
    ],
  },
  'life/histology-embryology': {
    title: "组织学与胚胎学",
    books: [
          "Junqueira, \"Basic Histology: Text and Atlas\" (15th, 2018)",
          "Moore, Persaud & Torchia, \"The Developing Human\" (11th, 2020)"
    ],
    chapters: [
      "细胞结构与细胞器 (Junqueira §2)",
      "上皮组织 (Junqueira §4)",
      "结缔组织与软骨 (Junqueira §5-7)",
      "肌肉与神经组织 (Junqueira §9-10)",
      "循环系统组织学 (Junqueira §11)",
      "血液与免疫系统组织学 (Junqueira §12-13)",
      "消化管与消化腺组织学 (Junqueira §15)",
      "呼吸系统组织学 (Junqueira §17)",
      "皮肤组织学 (Junqueira §18)",
      "泌尿系统组织学 (Junqueira §19)",
      "内分泌系统组织学 (Junqueira §20)",
      "生殖系统组织学 (Junqueira §21-22)",
      "受精与早期胚胎发育 (Moore §2)",
      "胚层分化与器官形成 (Moore §5)",
      "心血管系统发生 (Moore §13)",
      "消化与呼吸系统发生 (Moore §11)"
    ],
  },
  'life/pathology': {
    title: "病理学",
    books: [
          "Kumar, Abbas & Aster, \"Robbins & Cotran Pathologic Basis of Disease\" (10th, 2020)",
          "Kumar et al., \"Robbins Basic Pathology\" (10th, 2017)"
    ],
    chapters: [
      "细胞损伤、适应与死亡 (Robbins §2)",
      "炎症与修复 (Robbins §3)",
      "血流动力学障碍与休克 (Robbins §4)",
      "遗传性疾病病理 (Robbins §5)",
      "免疫性疾病病理 (Robbins §6)",
      "肿瘤总论 (Robbins §7)",
      "感染性疾病病理 (Robbins §8)",
      "心血管系统疾病 (Robbins §11)",
      "血液淋巴系统病理 (Robbins §13-14)",
      "呼吸系统疾病 (Robbins §15)",
      "头颈与胰腺病理 (Robbins §16)",
      "消化系统疾病 (Robbins §17)",
      "肝胆系统病理 (Robbins §18)",
      "肾脏疾病 (Robbins §20)",
      "生殖系统与乳腺病理 (Robbins §21-22)",
      "内分泌系统病理 (Robbins §24)",
      "皮肤病理 (Robbins §25)",
      "骨骼与软组织病理 (Robbins §26)",
      "神经系统病理 (Robbins §28)"
    ],
  },
  'life/pathophysiology': {
    title: "病理生理学",
    books: [
          "McCance & Huether, \"Pathophysiology: The Biologic Basis for Disease in Adults and Children\" (8th, 2018)",
          "Porth, \"Pathophysiology: Concepts of Altered Health States\" (9th, 2014)"
    ],
    chapters: [
      "细胞病理生理学 (McCance §2)",
      "水电解质与酸碱平衡 (McCance §5)",
      "炎症与免疫紊乱 (McCance §6)",
      "应激与缺血再灌注损伤 (McCance §6)",
      "发热与缺氧 (Porth §9)",
      "肿瘤生物学 (McCance §10)",
      "血液系统病理生理（贫血与出血） (McCance §20)",
      "心血管病理生理 (McCance §23)",
      "呼吸病理生理 (McCance §27)",
      "休克与DIC（弥散性血管内凝血） (Porth §29)",
      "肾脏病理生理 (McCance §31)",
      "内分泌系统病理生理（糖尿病等） (McCance §32)",
      "肝功能不全与消化系统病理生理 (McCance §33)",
      "神经病理生理 (McCance §36)"
    ],
  },
  'life/medical-imaging': {
    title: "医学影像学",
    books: [
          "Brant & Helms, \"Fundamentals of Diagnostic Radiology\" (4th, 2012)",
          "Webb, Brant & Major, \"Fundamentals of Body CT\" (4th, 2015)"
    ],
    chapters: [
      "X线成像原理 (Brant §1)",
      "CT成像原理与技术 (Webb §1)",
      "MRI成像基础 (Brant §2)",
      "超声成像原理与诊断 (Brant §3)",
      "胸部影像学 (Webb §2)",
      "心血管影像（CTA/MRA/超声心动） (Brant §5)",
      "腹部与盆腔CT (Webb §4)",
      "泌尿生殖系统影像 (Webb §5)",
      "骨骼系统影像 (Brant §8)",
      "神经影像学 (Brant §10)",
      "乳腺影像 (Brant §13)",
      "介入放射学 (Brant §12)"
    ],
  },
  'life/anesthesiology': {
    title: "麻醉学",
    books: [
          "Gropper et al., \"Miller's Anesthesia\" (9th, 2020)",
          "Butterworth, Mackey & Wasnick, \"Morgan & Mikhail's Clinical Anesthesiology\" (6th, 2018)"
    ],
    chapters: [
      "麻醉药理学基础 (Morgan §5)",
      "吸入与静脉麻醉药 (Morgan §7-8)",
      "术前评估与麻醉计划 (Morgan §2)",
      "气道管理与插管 (Morgan §19)",
      "全身麻醉的实施 (Morgan §25)",
      "局部麻醉与神经阻滞 (Morgan §23)",
      "围术期监测 (Morgan §6)",
      "并存疾病与高危患者麻醉（心/肺/肾/内分泌） (Morgan §33-37)",
      "小儿与产科麻醉 (Morgan §38-41)",
      "术后管理与PACU（术后恢复与镇痛） (Morgan §43)",
      "重症监护医学 (Morgan §44)",
      "麻醉并发症与处理 (Morgan §35)",
      "疼痛医学 (Morgan §42)"
    ],
  },
  'life/emergency-medicine': {
    title: "急诊医学",
    books: [
          "Tintinalli et al., \"Tintinalli's Emergency Medicine\" (9th, 2020)",
          "Marx et al., \"Rosen's Emergency Medicine\" (9th, 2018)"
    ],
    chapters: [
      "急诊医学基础与分诊 (Tintinalli §1)",
      "心肺复苏与急救 (Tintinalli §9)",
      "休克与复苏 (Tintinalli §11)",
      "创伤评估与处理 (Rosen §32)",
      "急性中毒 (Tintinalli §103)",
      "胸痛与急性冠脉综合征 (Tintinalli §52)",
      "急性腹痛 (Tintinalli §71)",
      "神经系统急症 (Tintinalli §157)",
      "环境急症（中暑/低温/淹溺/电击伤/高原病） (Tintinalli §91)",
      "儿科与妇产科急症 (Tintinalli §89)",
      "过敏反应与过敏性休克 (Tintinalli §34)",
      "急诊操作（穿刺/插管/超声） (Tintinalli §36)"
    ],
  },
  'life/geriatrics': {
    title: "老年医学",
    books: [
          "Hazzard et al., \"Hazzard's Geriatric Medicine and Gerontology\" (7th, 2017)",
          "Bourne, \"Geriatric Medicine\" (2019)"
    ],
    chapters: [
      "老年医学原理与老年综合征 (Hazzard §1)",
      "衰老生物学机制 (Hazzard §4)",
      "老年综合评估 (Hazzard §14)",
      "跌倒与步态障碍 (Hazzard §66)",
      "谵妄 (Hazzard §64)",
      "老年尿失禁与衰弱（frailty） (Hazzard §70)",
      "老年心血管疾病 (Hazzard §39)",
      "老年常见系统疾病（呼吸/内分泌/骨关节） (Hazzard §46)",
      "骨质疏松与骨折 (Hazzard §47)",
      "老年营养与压疮 (Hazzard §29)",
      "老年睡眠障碍 (Hazzard §58)",
      "老年痴呆与认知障碍 (Hazzard §63)",
      "多重用药与药物管理 (Hazzard §18)",
      "临终关怀与缓和医疗 (Hazzard §25)"
    ],
  },
  'life/dermatology': {
    title: "皮肤性病学",
    books: [
          "Kang et al., \"Fitzpatrick's Dermatology\" (9th, 2019)",
          "Habif, \"Clinical Dermatology: A Color Guide to Diagnosis and Therapy\" (7th, 2021)"
    ],
    chapters: [
      "皮肤结构与功能 (Fitzpatrick §1)",
      "皮肤病诊断方法 (Habif §1)",
      "湿疹与接触性皮炎 (Habif §3)",
      "荨麻疹与药疹（过敏反应） (Habif §6)",
      "银屑病 (Habif §8)",
      "痤疮与玫瑰痤疮 (Habif §5)",
      "皮肤感染 (Habif §9-10)",
      "性传播疾病 (Habif §11)",
      "色素性皮肤病（白癜风/黄褐斑） (Habif §14)",
      "毛发与甲病 (Habif §22-23)",
      "皮肤肿瘤 (Habif §20-21)",
      "自身免疫性大疱病 (Fitzpatrick §37)",
      "皮肤血管炎与皮肤淋巴瘤 (Fitzpatrick §48)"
    ],
  },
  'life/otolaryngology': {
    title: "耳鼻喉科学",
    books: [
          "Flint et al., \"Cummings Otolaryngology – Head and Neck Surgery\" (7th, 2021)",
          "Bailey & Johnson, \"Head and Neck Surgery – Otolaryngology\" (5th, 2013)"
    ],
    chapters: [
      "耳鼻喉解剖与生理 (Cummings §1)",
      "听觉与前庭系统 (Cummings §157)",
      "神经耳科学与听力重建（人工耳蜗） (Cummings §158)",
      "外耳与中耳疾病 (Cummings §160)",
      "眩晕与前庭康复 (Cummings §163)",
      "鼻与鼻窦疾病 (Cummings §42)",
      "咽喉与吞咽障碍 (Cummings §109)",
      "唾液腺疾病 (Cummings §96)",
      "小儿耳鼻喉 (Cummings §196)",
      "喉返神经麻痹与声带病变 (Cummings §111)",
      "头颈肿瘤 (Cummings §87)",
      "甲状腺与甲状旁腺外科 (Cummings §91)",
      "睡眠呼吸障碍 (Cummings §101)"
    ],
  },
  'life/ophthalmology': {
    title: "眼科学",
    books: [
          "Yanoff & Duker, \"Ophthalmology\" (5th, 2018)",
          "Riordan-Eva, \"Vaughan & Asbury's General Ophthalmology\" (19th, 2020)"
    ],
    chapters: [
      "眼解剖与生理 (Vaughan §1)",
      "眼科检查方法 (Yanoff §1.1)",
      "屈光不正与视光学（近视/远视/散光） (Vaughan §3)",
      "眼睑与泪器疾病 (Yanoff §1.8)",
      "角膜与外眼疾病 (Yanoff §4.1)",
      "青光眼 (Yanoff §10)",
      "白内障 (Yanoff §8)",
      "葡萄膜与眼眶疾病（葡萄膜炎/眼肿瘤） (Yanoff §9)",
      "玻璃体疾病 (Yanoff §7)",
      "视网膜疾病 (Yanoff §6.1)",
      "斜视与弱视 (Yanoff §11)",
      "神经眼科学 (Yanoff §12)",
      "眼外伤 (Vaughan §16)"
    ],
  },
  'life/sports-medicine': {
    title: "运动医学",
    books: [
          "DeLee, Drez & Miller, \"DeLee and Drez's Orthopaedic Sports Medicine\" (5th, 2020)",
          "Brukner & Khan, \"Clinical Sports Medicine\" (5th, 2017)"
    ],
    chapters: [
      "运动医学基础与生物力学 (DeLee §1)",
      "运动损伤机制 (DeLee §2)",
      "膝关节损伤 (DeLee §24)",
      "肩关节损伤 (DeLee §17)",
      "踝与足部损伤 (Brukner §27)",
      "髋/肘/腕关节损伤 (DeLee §19)",
      "脑震荡与头部损伤 (Brukner §24)",
      "运动中急性创伤 (Brukner §47)",
      "过度使用与慢性损伤 (Brukner §14)",
      "特殊人群运动医学（青少年/女性/老年） (Brukner §5)",
      "运动营养与运动心理 (Brukner §8)",
      "运动康复与训练 (Brukner §11)"
    ],
  },
  'life/forensic-medicine': {
    title: "法医学",
    books: [
          "Dolinak, Matshes & Lew, \"Forensic Pathology: Principles and Practice\" (2005)",
          "Spitz & Spitz, \"Spitz and Fisher's Medicolegal Investigation of Death\" (4th, 2006)"
    ],
    chapters: [
      "法医学概论与现场调查 (Spitz §1)",
      "死亡时间推断 (Spitz §2)",
      "机械性损伤 (Dolinak §4)",
      "枪伤与爆炸伤 (Spitz §9)",
      "性犯罪与虐待 (Spitz §10)",
      "窒息死亡 (Dolinak §6)",
      "溺死与电击烧死（热损伤） (Spitz §12)",
      "猝死与心血管死亡 (Dolinak §8)",
      "中毒法医学 (Dolinak §11)",
      "法医临床学与伤残鉴定（损伤程度） (Dolinak §12)",
      "法医精神病学 (Dolinak §13)",
      "法医物证与亲子鉴定（血痕/精斑） (Spitz §28)",
      "法医DNA分型 (Spitz §28)"
    ],
  },
  'life/cardiology': {
    title: "心血管内科",
    books: [
          "Braunwald et al., \"Braunwald's Heart Disease: A Textbook of Cardiovascular Medicine\" (12th, 2022)",
          "Guyton & Hall, \"Textbook of Medical Physiology\" (14th, 2020)"
    ],
    chapters: [
      "心血管系统解剖与生理 (Braunwald §1)",
      "心脏影像学诊断（超声/CMR/CTA） (Braunwald §13)",
      "动脉粥样硬化发病机制 (Braunwald §44)",
      "冠心病与急性冠脉综合征 (Braunwald §58)",
      "心力衰竭 (Braunwald §25)",
      "心律失常与电生理 (Braunwald §35)",
      "心脏瓣膜病 (Braunwald §66)",
      "心肌病与心肌炎 (Braunwald §79)",
      "高血压与肺动脉高压 (Braunwald §47)",
      "血脂异常与调脂治疗 (Braunwald §46)",
      "主动脉与外周血管疾病 (Braunwald §68)",
      "先天性心脏病 (Braunwald §14)",
      "心脏药理学（ACEI/ARB/β受体阻滞剂） (Braunwald §36)",
      "心包疾病 (Braunwald §71)",
      "感染性心内膜炎 (Braunwald §73)"
    ],
  },
  'life/pulmonology': {
    title: "呼吸内科",
    books: [
          "Broaddus et al., \"Murray & Nadel's Textbook of Respiratory Medicine\" (7th, 2022)",
          "West, \"Respiratory Physiology: The Essentials\" (11th, 2020)"
    ],
    chapters: [
      "呼吸系统结构与功能 (Murray §1)",
      "肺通气与气体交换 (West §3)",
      "氧与二氧化碳运输 (West §7)",
      "肺通气功能检测 (Murray §10)",
      "呼吸衰竭与机械通气 (Murray §22)",
      "哮喘 (Murray §57)",
      "慢性阻塞性肺疾病 (Murray §55)",
      "支气管扩张与气道清除 (Murray §50)",
      "肺部感染与肺炎 (Murray §33)",
      "肺栓塞与肺动脉高压 (Murray §70)",
      "肺血管炎与肺出血-肾炎综合征 (Murray §78)",
      "间质性肺疾病 (Murray §74)",
      "肺癌 (Murray §29)",
      "肺结节评估与肺移植 (Murray §85)",
      "胸膜疾病 (Murray §76)",
      "睡眠呼吸障碍（OSA） (Murray §96)"
    ],
  },
  'life/gastroenterology': {
    title: "消化内科",
    books: [
          "Feldman et al., \"Sleisenger and Fordtran's Gastrointestinal and Liver Disease\" (11th, 2020)",
          "Yamada, \"Textbook of Gastroenterology\" (6th, 2015)"
    ],
    chapters: [
      "消化系统生理 (Yamada §1)",
      "胃肠内镜诊断 (Sleisenger §10)",
      "胃食管反流病 (Sleisenger §14)",
      "消化性溃疡 (Sleisenger §17)",
      "胃炎与胃癌前病变 (Sleisenger §16)",
      "消化道出血（上下消化道） (Sleisenger §12)",
      "腹泻与吸收不良 (Sleisenger §20)",
      "炎症性肠病 (Sleisenger §22)",
      "肠梗阻与急腹症 (Sleisenger §53)",
      "胰腺疾病 (Sleisenger §32)",
      "胆道与胆囊疾病（胆囊炎/胆管炎/胆石症） (Sleisenger §90)",
      "病毒性肝炎 (Sleisenger §80)",
      "肝硬化与门脉高压 (Sleisenger §86)",
      "消化道肿瘤 (Sleisenger §52)",
      "功能性胃肠病 (Sleisenger §48)"
    ],
  },
  'life/endocrinology': {
    title: "内分泌科",
    books: [
          "Melmed et al., \"Williams Textbook of Endocrinology\" (14th, 2020)",
          "Guyton & Hall, \"Textbook of Medical Physiology\" (14th, 2020)"
    ],
    chapters: [
      "内分泌学原理与激素作用 (Williams §1)",
      "下丘脑与垂体疾病 (Williams §8)",
      "甲状腺疾病 (Williams §11)",
      "甲状腺结节与甲状腺癌 (Williams §12)",
      "肾上腺皮质与髓质疾病 (Williams §15)",
      "糖尿病与代谢综合征 (Williams §34)",
      "低血糖症 (Williams §34)",
      "肥胖症与能量代谢 (Williams §35)",
      "钙磷代谢与骨病 (Williams §27)",
      "男性与女性生殖内分泌 (Williams §19-22)",
      "性腺与性分化异常 (Williams §23)",
      "多发性内分泌腺瘤（MEN）综合征 (Williams §42)",
      "内分泌肿瘤综合征 (Williams §42)"
    ],
  },
  'life/nephrology': {
    title: "肾内科",
    books: [
          "Skorecki et al., \"Brenner & Rector's The Kidney\" (11th, 2020)",
          "Guyton & Hall, \"Textbook of Medical Physiology\" (14th, 2020)"
    ],
    chapters: [
      "肾脏结构与功能 (Brenner §1)",
      "肾小球滤过与肾血流动力学 (Brenner §3)",
      "酸碱平衡与电解质 (Brenner §15)",
      "肾小球疾病 (Brenner §31)",
      "高血压与肾脏 (Brenner §32)",
      "急性肾损伤 (Brenner §33)",
      "肾小管间质疾病 (Brenner §34)",
      "药物与造影剂肾病 (Brenner §34)",
      "血管性肾病（肾动脉狭窄） (Brenner §36)",
      "肾结石与梗阻性肾病 (Brenner §37)",
      "糖尿病肾病 (Brenner §38)",
      "尿路感染与肾盂肾炎 (Brenner §44)",
      "慢性肾脏病与透析 (Brenner §54)",
      "肾囊肿性疾病（多囊肾） (Brenner §58)",
      "遗传性肾病 (Brenner §58)",
      "肾移植 (Brenner §81)"
    ],
  },
  'life/hematology': {
    title: "血液内科",
    books: [
          "Hoffman et al., \"Hematology: Basic Principles and Practice\" (8th, 2022)",
          "Kaushansky et al., \"Williams Hematology\" (10th, 2020)"
    ],
    chapters: [
      "造血系统与血细胞生成 (Hoffman §1)",
      "贫血概论与缺铁性贫血 (Hoffman §34)",
      "溶血性贫血 (Hoffman §40)",
      "骨髓增生异常综合征（MDS） (Hoffman §60)",
      "白血病 (Hoffman §65)",
      "骨髓增殖性肿瘤（真红/血小板增多/骨髓纤维化） (Hoffman §67)",
      "淋巴瘤 (Hoffman §76)",
      "多发性骨髓瘤 (Hoffman §83)",
      "凝血因子缺乏（血友病） (Hoffman §112)",
      "出血与血栓性疾病 (Hoffman §111)",
      "血小板疾病（ITP/TTP） (Hoffman §115)",
      "输血医学 (Hoffman §119)",
      "造血干细胞移植 (Hoffman §92)"
    ],
  },
  'life/neurology': {
    title: "神经内科",
    books: [
          "Ropper, Samuels & Klein, \"Adams and Victor's Principles of Neurology\" (11th, 2019)",
          "Daroff et al., \"Bradley's Neurology in Clinical Practice\" (8th, 2021)"
    ],
    chapters: [
      "神经系统检查与定位诊断 (Adams §1)",
      "脑神经与脑干病变 (Adams §13)",
      "头痛与疼痛障碍（偏头痛/紧张型） (Adams §10)",
      "脊髓疾病（脊髓炎/脊髓压迫） (Adams §11)",
      "脑血管疾病 (Adams §34)",
      "癫痫与发作性疾病 (Adams §16)",
      "运动障碍疾病 (Adams §39)",
      "多发性硬化与脱髓鞘疾病 (Bradley §90)",
      "痴呆与认知障碍 (Adams §21)",
      "周围神经病 (Bradley §85)",
      "神经肌肉接头与肌肉疾病（重症肌无力/肌营养不良） (Bradley §96)",
      "中枢神经系统感染 (Bradley §75)"
    ],
  },
  'life/psychiatry': {
    title: "精神病学",
    books: [
          "Sadock, Sadock & Ruiz, \"Kaplan & Sadock's Synopsis of Psychiatry\" (12th, 2022)",
          "American Psychiatric Association, \"DSM-5-TR\" (2022)"
    ],
    chapters: [
      "精神障碍分类与诊断 (DSM-5 §1)",
      "精神分裂症 (Synopsis §7)",
      "心境障碍 (Synopsis §8)",
      "焦虑与应激相关障碍 (Synopsis §9)",
      "躯体症状障碍 (Synopsis §14)",
      "进食障碍与睡眠障碍 (Synopsis §20)",
      "物质相关障碍 (Synopsis §12)",
      "人格障碍 (Synopsis §22)",
      "精神药理学（抗精神病药/抗抑郁药/锂盐） (Synopsis §34)",
      "心理治疗 (Synopsis §30)",
      "自杀与危机干预 (Synopsis §29)",
      "儿童青少年精神障碍 (Synopsis §31)",
      "认知障碍 (Synopsis §21)"
    ],
  },
  'life/oncology': {
    title: "肿瘤学",
    books: [
          "DeVita, Lawrence & Rosenberg, \"Cancer: Principles & Practice of Oncology\" (12th, 2022)",
          "Hong et al., \"Holland-Frei Cancer Medicine\" (9th, 2017)"
    ],
    chapters: [
      "肿瘤生物学与发病机制 (DeVita §2)",
      "肿瘤流行病学与预防 (DeVita §4)",
      "肿瘤诊断与影像学 (DeVita §5)",
      "肿瘤分期与预后原则 (DeVita §6)",
      "遗传性肿瘤综合征与遗传咨询 (DeVita §13)",
      "化疗与靶向治疗 (DeVita §14)",
      "放射治疗 (DeVita §15)",
      "免疫治疗 (DeVita §16)",
      "抗肿瘤药物临床试验设计 (DeVita §19)",
      "肿瘤外科治疗 (DeVita §20)",
      "常见实体瘤诊治 (DeVita §49-65)",
      "肿瘤急症 (DeVita §60)",
      "血液肿瘤与姑息治疗 (DeVita §146)",
      "癌症疼痛与姑息治疗 (DeVita §146)"
    ],
  },
  'life/infectious-disease': {
    title: "感染病学",
    books: [
          "Bennett, Dolin & Blaser, \"Mandell, Douglas, and Bennett's Principles and Practice of Infectious Diseases\" (9th, 2020)"
    ],
    chapters: [
      "感染病学基本原理 (Mandell §1)",
      "抗生素作用机制与耐药 (Mandell §14)",
      "败血症与感染性休克 (Mandell §17)",
      "免疫功能低下宿主感染（HIV机会性感染） (Mandell §52)",
      "真菌感染（念珠菌/曲霉/隐球菌） (Mandell §55)",
      "局部感染综合征（脑膜炎/心内膜炎/骨髓炎/UTI） (Mandell §65)",
      "革兰阳性菌感染 (Mandell §31)",
      "革兰阴性菌感染 (Mandell §35)",
      "结核与非结核分枝杆菌 (Mandell §41)",
      "病毒性感染 (Mandell §76)",
      "性传播感染 (Mandell §75)",
      "寄生虫感染 (Mandell §88)",
      "医院感染与感染控制 (Mandell §90)"
    ],
  },
  'life/rheumatology': {
    title: "风湿免疫科",
    books: [
          "Firestein et al., \"Kelley and Firestein's Textbook of Rheumatology\" (11th, 2020)",
          "Klippel & Stone, \"Primer on the Rheumatic Diseases\" (13th, 2008)"
    ],
    chapters: [
      "风湿病免疫学基础 (Kelley §15)",
      "类风湿关节炎 (Kelley §65)",
      "系统性红斑狼疮 (Kelley §72)",
      "抗磷脂综合征与生物制剂治疗 (Kelley §73)",
      "脊柱关节炎 (Kelley §68)",
      "干燥综合征 (Kelley §76)",
      "系统性硬化症 (Kelley §74)",
      "血管炎 (Kelley §84)",
      "炎症性肌病（皮肌炎/多发性肌炎） (Kelley §80)",
      "骨关节炎 (Kelley §92)",
      "晶体性关节炎 (Kelley §90)",
      "儿童风湿病与妊娠期风湿病 (Kelley §26)"
    ],
  },
  'life/general-practice': {
    title: "全科医学",
    books: [
          "Rakel & Rakel, \"Textbook of Family Medicine\" (9th, 2016)",
          "Taylor, \"Family Medicine: Principles and Practice\" (8th, 2017)"
    ],
    chapters: [
      "全科医学原理与模式 (Rakel §1)",
      "以患者为中心的诊疗 (Rakel §3)",
      "循证临床决策 (Rakel §2)",
      "预防保健与健康促进 (Rakel §7)",
      "妇幼保健与筛查（女性与儿童保健） (Rakel §14)",
      "常见呼吸道感染 (Taylor §22)",
      "高血压与心血管管理 (Taylor §35)",
      "糖尿病管理 (Taylor §45)",
      "肌肉骨骼常见问题（背痛/关节痛） (Taylor §49)",
      "心理健康与基层干预 (Taylor §55)",
      "慢病共病管理与转诊、连续性照护 (Rakel §17)",
      "老年与缓和医疗 (Rakel §21)"
    ],
  },
  'life/nuclear-medicine': {
    title: "核医学",
    books: [
          "Mettler & Guiberteau, \"Essentials of Nuclear Medicine Imaging\" (6th, 2012)",
          "Saha, \"Fundamentals of Nuclear Pharmacy\" (7th, 2018)"
    ],
    chapters: [
      "放射性核素与物理基础 (Mettler §1)",
      "辐射剂量与防护 (Mettler §2)",
      "PET与SPECT成像 (Mettler §3)",
      "骨显像 (Mettler §7)",
      "心血管核医学 (Mettler §6)",
      "神经核医学（痴呆/癫痫定位） (Mettler §8)",
      "肾脏与肺核医学显像（肾图/VQ） (Mettler §9)",
      "肿瘤PET显像 (Mettler §10)",
      "炎症/感染与消化系统显像（白细胞显像） (Mettler §11)",
      "甲状腺与甲状旁腺显像 (Mettler §12)",
      "放射性药物治疗 (Saha §14)"
    ],
  },
  'humanities/medical-ethics': {
    title: "医学伦理学",
    books: [
          "Beauchamp & Childress, \"Principles of Biomedical Ethics\" (8th, 2019)",
          "Lo, \"Resolving Ethical Dilemmas: A Guide for Clinicians\" (6th, 2020)"
    ],
    chapters: [
      "医学伦理学基本原则 (Beauchamp §1)",
      "道德地位（moral status） (Beauchamp §3)",
      "自主原则与知情同意 (Beauchamp §4)",
      "不伤害与有利原则 (Beauchamp §5-6)",
      "公正原则与资源分配 (Beauchamp §7)",
      "医患关系与共同决策（专业—患者关系） (Beauchamp §8 / Lo §4)",
      "医疗保密与隐私 (Beauchamp §8 / Lo §保密)",
      "临终决策与安乐死 (Lo §17)",
      "人类受试者研究伦理 (Belmont Report §B)",
      "器官移植伦理 (Lo §20)",
      "公共卫生伦理：防疫、疫苗与分配 (Beauchamp §11)",
      "遗传与生殖伦理：基因检测与优生 (Beauchamp §9)"
    ],
  },
  'life/epidemiology': {
    title: "流行病学",
    books: [
          "Gordis, \"Epidemiology\" (6th, 2022)",
          "Rothman, \"Epidemiology: An Introduction\" (3rd, 2021)"
    ],
    chapters: [
      "流行病学概论与测量 (Gordis §1-2)",
      "疾病频率测量 (Gordis §2)",
      "描述性研究 (Gordis §5)",
      "病例对照与队列研究 (Gordis §6-7)",
      "实验性研究/RCT 设计 (Gordis §9)",
      "混杂与偏倚 (Rothman §10)",
      "偏倚类型细化（选择/信息偏倚） (Rothman §10)",
      "因果推断 (Rothman §12)",
      "暴发调查与疾病监测（哨点监测） (Gordis §13)",
      "传染病流行病学 (Gordis §14)",
      "筛检试验评价 (Gordis §11)",
      "临床流行病学 (Gordis §12)",
      "系统综述与Meta分析 (Rothman §15)",
      "分子流行病学 (Rothman §20)"
    ],
  },
  'life/health-statistics': {
    title: "卫生统计学",
    books: [
          "Rosner, \"Fundamentals of Biostatistics\" (8th, 2015)",
          "Daniel & Cross, \"Biostatistics: A Foundation for Analysis in the Health Sciences\" (11th, 2018)"
    ],
    chapters: [
      "统计学基础与描述 (Rosner §2)",
      "概率与概率分布 (Rosner §3)",
      "参数估计与置信区间 (Rosner §6)",
      "假设检验 (Rosner §7)",
      "方差分析 (Rosner §8)",
      "样本量与检验效能计算 (Rosner §8)",
      "非参数检验 (Rosner §9)",
      "分类数据分析 (Rosner §10)",
      "回归与相关 (Rosner §11)",
      "多元线性回归与Logistic回归 (Daniel §11)",
      "重复测量与纵向数据分析 (Rosner §12)",
      "统计软件应用（R/SAS/SPSS） (Daniel §13)",
      "生存分析 (Rosner §14)"
    ],
  },
  'life/nutrition-food-hygiene': {
    title: "营养与食品卫生学",
    books: [
          "Ross et al., \"Modern Nutrition in Health and Disease\" (11th, 2014)",
          "Belitz et al., \"Food Chemistry\" (5th, 2009)"
    ],
    chapters: [
      "营养学基础与能量代谢 (Ross §1-2)",
      "蛋白质与氨基酸营养 (Ross §3)",
      "脂类与碳水化合物 (Ross §5-6)",
      "维生素与矿物质 (Ross §11-15)",
      "营养状况评价 (Ross §32)",
      "特殊人群营养（孕妇/儿童/老年） (Ross §23)",
      "膳食指南与DRIs（膳食营养素参考摄入量） (Ross §97)",
      "临床营养支持（肠内肠外营养） (Ross §36)",
      "营养与慢性病 (Ross §114)",
      "食品卫生与食品安全 (Belitz §13)",
      "食品添加剂 (Belitz §12)",
      "食物中毒分类（细菌性/化学性/有毒动植物） (Belitz §13)",
      "食源性疾病 (Belitz §13)"
    ],
  },
  'life/toxicology': {
    title: "毒理学",
    books: [
          "Klaassen, \"Casarett & Doull's Toxicology: The Basic Science of Poisons\" (9th, 2019)",
          "Hayes, \"Principles and Methods of Toxicology\" (6th, 2014)"
    ],
    chapters: [
      "毒理学原理与剂量反应 (Casarett §1-2)",
      "毒物代谢动力学 (Casarett §5)",
      "毒物效应动力学 (Casarett §4)",
      "靶器官毒性：肝脏与肾脏 (Casarett §13-14)",
      "呼吸与皮肤毒性 (Casarett §15)",
      "神经系统与免疫毒性 (Casarett §12)",
      "生殖与发育毒理 (Casarett §10)",
      "遗传毒理与致癌 (Casarett §8-9)",
      "重金属/农药/有机溶剂/天然毒素毒理各论 (Casarett §21)",
      "临床毒理学与中毒救治 (Casarett §26)",
      "环境毒理与风险评估 (Casarett §25)",
      "毒理学试验方法 (Hayes §3)"
    ],
  },
  'advanced/regenerative-medicine': {
    title: "再生医学与干细胞",
    books: [
          "Atala, Lanza & Thomson, \"Principles of Regenerative Medicine\" (3rd, 2019)",
          "Lanza et al., \"Essentials of Stem Cell Biology\" (3rd, 2014)"
    ],
    chapters: [
      "再生医学原理与干细胞 (Atala §1)",
      "胚胎干细胞 (Essentials §2)",
      "成体干细胞 (Essentials §7)",
      "诱导多能干细胞 (Essentials §9)",
      "直接重编程与谱系转化 (Essentials §10)",
      "干细胞微环境（niche）与干细胞异质性 (Essentials §11)",
      "癌症干细胞与肿瘤再生 (Essentials §14)",
      "干细胞的衰老与再生能力下降 (Essentials §15)",
      "组织工程原理 (Atala §15)",
      "生物材料与支架 (Atala §17)",
      "类器官与3D生物打印 (Atala §25)",
      "器官再生与移植 (Atala §32)",
      "干细胞治疗产品与GMP规模化生产 (Atala §45)",
      "临床转化与伦理 (Atala §85)"
    ],
  },
  'advanced/translational-medicine': {
    title: "转化医学",
    books: [
          "Wehling, \"Principles of Translational Science in Medicine\" (2nd, 2015)",
          "Sectish & Prober, \"Translational Research and Clinical Practice\" (2018)"
    ],
    chapters: [
      "转化医学概论 (Wehling §1)",
      "从基础到临床的转化路径 (Wehling §2)",
      "生物标志物与诊断 (Wehling §5)",
      "药物研发与临床前研究 (Wehling §7)",
      "临床试验设计 (Sectish §9)",
      "精准医学与个体化治疗 (Wehling §12)",
      "健康大数据与真实世界研究 (Sectish §14)",
      "转化医学伦理与监管 (Wehling §19)"
    ],
  },
  'advanced/gene-therapy': {
    title: "基因治疗",
    books: [
          "Sibbald et al., \"Gene Therapy: Principles and Applications\" (2019)",
          "Mátrai, Chuah & VandenDriessche, \"Recent Advances in Gene Therapy\" (2018)"
    ],
    chapters: [
      "基因治疗原理与发展 (Sibbald §1)",
      "病毒载体：腺病毒与AAV (Sibbald §3)",
      "逆转录病毒与慢病毒载体 (Sibbald §5)",
      "非病毒载体递送 (Mátrai §6)",
      "衣壳工程与靶向递送 (Mátrai §7)",
      "基因沉默疗法（ASO、siRNA/shRNA与RNAi机制） (Sibbald §7)",
      "病毒载体生产放大与质量（AAV manufacturing） (Mátrai §8)",
      "基因编辑治疗 (Sibbald §9)",
      "单基因遗传病治疗 (Mátrai §10)",
      "肿瘤基因治疗 (Sibbald §12)",
      "免疫原性与预存抗体（anti-AAV immunity） (Mátrai §9)",
      "已上市基因治疗产品案例（Zolgensma、Luxturna、CAR-T） (Mátrai §11)",
      "插入致突变与基因毒性安全性评价 (Sibbald §14)",
      "免疫与伦理监管 (Sibbald §15)"
    ],
  },
  'life/agricultural-meteorology': {
    title: "农业气象学",
    books: [
          "Rosenberg, Blad & Verma, \"Microclimate: The Biological Environment\" (2nd, 1983)",
          "Hay & Walker, \"An Introduction to the Principles of Crop Physiology and Agricultural Meteorology\" (1989)"
    ],
    chapters: [
      "农业气象学基础 (Hay §1)",
      "辐射与作物光合 (Rosenberg §2)",
      "温度与作物生长 (Rosenberg §3)",
      "水分与蒸散 (Rosenberg §7)",
      "风与湍流输送 (Rosenberg §5)",
      "农业气象观测方法与物候学 (Hay §4)",
      "农业气候资源与区划 (Hay §10)",
      "农业气象灾害 (Hay §12)",
      "农业气象服务与预报应用 (Hay §15)",
      "气候变化与农业 (Hay §14)"
    ],
  },
  'life/crop-genetics-breeding': {
    title: "作物遗传育种",
    books: [
          "Acquaah, \"Principles of Plant Genetics and Breeding\" (2nd, 2012)",
          "Allard, \"Principles of Plant Breeding\" (2nd, 1999)"
    ],
    chapters: [
      "植物育种基础与遗传学原理 (Acquaah §1-3)",
      "群体遗传与数量遗传 (Acquaah §7)",
      "种质资源与遗传多样性 (Acquaah §11)",
      "杂交育种与杂种优势 (Acquaah §15)",
      "选择育种与系谱法 (Acquaah §16)",
      "回交与轮回选择 (Allard §14)",
      "突变育种与多倍体育种 (Acquaah §12)",
      "分子标记辅助选择 (Acquaah §22)",
      "GWAS 与全基因组关联分析 (Acquaah §21)",
      "全基因组选择（GS）与加速育种（DH/双单倍体） (Acquaah §23)",
      "抗逆育种（抗病/抗虫/抗旱） (Acquaah §18)",
      "品质育种与品种审定（种子法规） (Acquaah §19)",
      "转基因与基因编辑育种 (Acquaah §25)"
    ],
  },
  'life/plant-protection': {
    title: "植物保护（植物病理/农业昆虫/农药）",
    books: [
          "Agrios, \"Plant Pathology\" (6th, 2020)",
          "Pedigo & Rice, \"Entomology and Pest Management\" (6th, 2014)"
    ],
    chapters: [
      "植物病理学概论 (Agrios §1)",
      "植物病原生物 (Agrios §3)",
      "植物病害发生与流行 (Agrios §5)",
      "植物病害防治策略 (Agrios §9)",
      "昆虫形态与生理 (Pedigo §3-4)",
      "昆虫分类与种群动态 (Pedigo §6-7)",
      "害虫综合治理 (Pedigo §13)",
      "杂草防除与除草剂 (Pedigo §14)",
      "植物检疫与生物防治 (Pedigo §12)",
      "主要作物病害各论（小麦锈病/稻瘟病/病毒病/线虫病） (Agrios §11)",
      "杀虫剂与杀菌剂应用 (Pedigo §11)",
      "农药安全与环境 (Pedigo §15)"
    ],
  },
  'life/horticulture': {
    title: "园艺学（果树/蔬菜/观赏）",
    books: [
          "Hartmann et al., \"Hartmann and Kester's Plant Propagation: Principles and Practices\" (9th, 2018)",
          "Adams, Bamford & Early, \"Principles of Horticulture\" (3rd, 2008)"
    ],
    chapters: [
      "园艺植物分类与利用 (Adams §1)",
      "植物繁殖原理 (Hartmann §1)",
      "种子繁殖 (Hartmann §5)",
      "无性繁殖：扦插与嫁接 (Hartmann §9-11)",
      "园艺植物生长发育 (Adams §4)",
      "果树栽培各论（仁果/核果/浆果）与整形修剪 (Adams §11)",
      "蔬菜栽培各论 (Adams §12)",
      "观赏园艺与园林植物 (Adams §13)",
      "果园与蔬菜栽培管理 (Adams §10)",
      "设施园艺与气候调控 (Adams §15)",
      "园艺产品采后商品化处理 (Hartmann §16)",
      "采后生理与贮藏 (Adams §17)"
    ],
  },
  'life/seed-science': {
    title: "种子科学与工程",
    books: [
          "Bewley et al., \"Seeds: Physiology of Development, Germination and Dormancy\" (3rd, 2013)",
          "Copeland & McDonald, \"Principles of Seed Science and Technology\" (4th, 2001)"
    ],
    chapters: [
      "种子发育与结构 (Bewley §1-2)",
      "种子萌发生理 (Bewley §4)",
      "种子休眠机制 (Bewley §5)",
      "种子活力与寿命 (Copeland §8)",
      "种子生产与遗传纯度 (Copeland §10)",
      "种子加工与贮藏 (Copeland §13)",
      "种子检验与质量标准 (Copeland §15)",
      "种子病理与种传病害 (Copeland §16)",
      "种子法规与品种认证 (Copeland §18)",
      "种子生物技术 (Bewley §9)"
    ],
  },
  'life/protected-agriculture': {
    title: "设施农业",
    books: [
          "Hanan, \"Greenhouses: Advanced Technology for Protected Horticulture\" (1998)",
          "Resh, \"Hydroponic Food Production\" (7th, 2013)"
    ],
    chapters: [
      "设施农业概论与发展 (Hanan §1)",
      "温室结构与材料 (Hanan §3)",
      "温室光环境与覆盖 (Hanan §5)",
      "覆盖材料与节能新能源 (Hanan §6)",
      "温室温度与湿度调控 (Hanan §8)",
      "温室通风与CO2施肥 (Hanan §11)",
      "植物工厂与人工光环境 (Hanan §12)",
      "温室自动化与环境调控（计算机管理） (Hanan §13)",
      "设施作物栽培管理 (Resh §9)",
      "育苗设施与集约化育苗 (Resh §8)",
      "无土栽培原理 (Resh §1)",
      "营养液配方与管理 (Resh §4)",
      "水培与基质栽培系统 (Resh §7)"
    ],
  },
  'life/smart-agriculture': {
    title: "智慧农业",
    books: [
          "Oliver et al., \"Precision Agriculture for Sustainability\" (2018)",
          "Stafford, \"Precision Agriculture '19\" (2019)"
    ],
    chapters: [
      "智慧农业与精准农业概论 (Oliver §1)",
      "农田信息感知与传感技术 (Stafford §2)",
      "GPS与定位导航 (Stafford §3)",
      "遥感与无人机监测 (Oliver §5)",
      "变量施肥与施药 (Oliver §8)",
      "产量监测与决策支持 (Oliver §11)",
      "农业物联网与大数据 (Oliver §14)",
      "农业大数据平台与数字孪生 (Oliver §15)",
      "计算机视觉农业应用 (Oliver §17)",
      "人工智能与机器学习应用 (Oliver §16)",
      "农业机器人与农机自动导航 (Stafford §9)",
      "农产品质量追溯与区块链 (Oliver §18)"
    ],
  },
  'life/tea-science': {
    title: "茶学",
    books: [
          "Wilson & Clifford, \"Tea: Chemistry and Pharmacology\" (2018)",
          "Sharma, \"Tea: Cultivation and Processing\" (2018)"
    ],
    chapters: [
      "茶树植物学与品种 (Sharma §1)",
      "茶树育种与良种繁育 (Sharma §2)",
      "茶树栽培与修剪 (Sharma §3)",
      "茶叶加工工艺 (Sharma §7)",
      "六大茶类加工工艺（绿/红/乌龙/黑茶） (Sharma §8)",
      "茶叶深加工与综合利用（茶饮料/速溶茶/茶多酚） (Sharma §10)",
      "茶叶化学成分 (Wilson §2)",
      "茶多酚与儿茶素 (Wilson §4)",
      "茶叶香气与滋味 (Wilson §6)",
      "茶叶保健功能 (Wilson §10)",
      "茶叶品质评价与检验 (Wilson §12)",
      "茶文化与感官审评 (Wilson §14)"
    ],
  },
  'life/sericulture-apiculture': {
    title: "蚕学与蜂学",
    books: [
          "Hiware, \"Sericulture and Pest Management\" (2018)",
          "Crane, \"Bees and Beekeeping: Science, Practice and World Resources\" (1990)"
    ],
    chapters: [
      "蚕的生物学与品种 (Hiware §1)",
      "桑树栽培与养蚕 (Hiware §3)",
      "蚕病害防治 (Hiware §7)",
      "蚕茧与生丝生产 (Hiware §10)",
      "蚕桑副产品综合利用 (Hiware §12)",
      "蜜蜂生物学与社会 (Crane §3)",
      "蜂群管理与蜂具 (Crane §6)",
      "蜂种资源与蜜蜂育种 (Crane §8)",
      "蜜蜂产品：蜂蜜与蜂王浆 (Crane §11)",
      "蜂产品深加工与质量控制 (Crane §11)",
      "蜜蜂病虫害防治（蜂螨 Varroa） (Crane §14)",
      "蜜蜂授粉与保护 (Crane §13)"
    ],
  },
  'life/animal-nutrition-feed': {
    title: "动物营养与饲料科学",
    books: [
          "McDonald et al., \"Animal Nutrition\" (7th, 2011)",
          "Pond et al., \"Basic Animal Nutrition and Feeding\" (7th, 2017)"
    ],
    chapters: [
      "动物营养学基础 (McDonald §1)",
      "碳水化合物与脂类营养 (McDonald §3-4)",
      "蛋白质与氨基酸营养 (McDonald §5)",
      "维生素与矿物质 (McDonald §6-7)",
      "反刍动物瘤胃营养与微生物 (McDonald §6)",
      "能量代谢与饲料能值 (McDonald §9)",
      "饲料营养价值评定方法 (McDonald §10)",
      "饲料分类与营养价值 (Pond §11)",
      "饲料加工与添加剂 (Pond §13)",
      "饲料配方设计与加工（计算机配方） (Pond §14)",
      "饲料卫生与霉菌毒素 (Pond §15)",
      "不同动物营养需求 (McDonald §14)"
    ],
  },
  'life/animal-genetics-breeding': {
    title: "动物遗传育种与繁殖",
    books: [
          "Bourdon, \"Understanding Animal Breeding\" (3rd, 2020)",
          "Senger, \"Pathways to Pregnancy and Parturition\" (3rd, 2016)"
    ],
    chapters: [
      "动物育种遗传基础 (Bourdon §2)",
      "群体遗传与数量性状 (Bourdon §7)",
      "育种值估计与选择 (Bourdon §11)",
      "杂交与杂种优势 (Bourdon §15)",
      "基因组选择与分子育种 (Bourdon §17)",
      "动物遗传资源保护 (Bourdon §18)",
      "动物繁殖生理 (Senger §2)",
      "发情周期与人工授精 (Senger §6)",
      "妊娠与分娩 (Senger §11)",
      "繁殖障碍与不孕不育诊疗 (Senger §13)",
      "胚胎工程与繁殖生物技术细化（胚胎移植） (Senger §14)",
      "繁殖生物技术 (Senger §14)"
    ],
  },
  'life/aquaculture': {
    title: "水产养殖（深化）",
    books: [
          "Pillay & Kutty, \"Aquaculture: Principles and Practices\" (2nd, 2005)",
          "Stickney, \"Aquaculture: An Introductory Text\" (3rd, 2017)"
    ],
    chapters: [
      "水产养殖概论与发展 (Stickney §1)",
      "养殖水域环境与水质 (Pillay §3)",
      "人工繁殖与苗种培育 (Stickney §5)",
      "水产遗传育种 (Stickney §6)",
      "主要养殖鱼类各论（鲤科/罗非鱼/鲑鳟） (Pillay §6)",
      "营养与饲料 (Stickney §7)",
      "池塘与网箱养殖 (Pillay §7)",
      "藻类与贝类养殖 (Pillay §9)",
      "虾蟹与贝类养殖 (Pillay §10)",
      "鱼病防治 (Stickney §10)",
      "循环水养殖系统（RAS） (Stickney §12)",
      "观赏鱼养殖 (Pillay §11)",
      "水产养殖可持续发展 (Pillay §14)"
    ],
  },
  'life/fisheries-resources': {
    title: "捕捞学与渔业资源",
    books: [
          "Hilborn & Walters, \"Quantitative Fisheries Stock Assessment\" (1992)",
          "Jennings, Kaiser & Reynolds, \"Marine Fisheries Ecology\" (2001)"
    ],
    chapters: [
      "渔业资源与种群动态 (Hilborn §1)",
      "渔业资源评估模型 (Hilborn §4)",
      "剩余产量模型 (Hilborn §5)",
      "单位补充量产量 (Hilborn §8)",
      "渔业管理策略与决策 (Hilborn §12)",
      "渔具渔法与捕捞技术（拖网/围网/刺网/钓具） (Jennings §4)",
      "渔船与捕捞技术装备 (Jennings §4)",
      "渔具选择性与兼捕（副渔获）控制 (Jennings §14)",
      "国际渔业管理与法规（UNCLOS/RFMO） (Jennings §16)",
      "渔业经济学 (Hilborn §12)",
      "海洋生态系统与渔业 (Jennings §2)",
      "捕捞对群落的影响 (Jennings §14)",
      "渔业可持续发展 (Jennings §17)"
    ],
  },
  'life/forest-breeding-silviculture': {
    title: "林木遗传育种与森林培育",
    books: [
          "Barnes et al., \"Forest Genetics\" (2009)",
          "Smith et al., \"The Practice of Silviculture: Applied Forest Ecology\" (10th, 1997)"
    ],
    chapters: [
      "林木遗传学基础 (Barnes §1)",
      "群体遗传与地理变异 (Barnes §5)",
      "林木育种与种子园 (Barnes §10)",
      "分子标记与基因组辅助林木育种 (Barnes §11)",
      "种子园营建与苗圃育苗技术 (Barnes §12)",
      "森林立地与生态学 (Smith §3)",
      "天然林更新 (Smith §9)",
      "人工造林与苗木培育 (Smith §11)",
      "特殊立地造林 (Smith §12)",
      "混交林与异龄林经营 (Smith §14)",
      "抚育间伐 (Smith §15)",
      "森林主伐与再生 (Smith §17)"
    ],
  },
  'life/forest-protection-management': {
    title: "森林保护与经理",
    books: [
          "Edmonds, Surridge & Simpson, \"Forest Pathology\" (2018)",
          "Davis, Johnson & Bettinger, \"Forest Management\" (4th, 2001)"
    ],
    chapters: [
      "森林病理学原理 (Edmonds §1)",
      "森林病害诊断与防治 (Edmonds §5)",
      "森林害虫与综合管理 (Edmonds §8)",
      "森林检疫与外来有害生物入侵 (Edmonds §9)",
      "森林野生动物危害 (Edmonds §10)",
      "森林火灾与防火 (Davis §10)",
      "计划火烧与火生态管理 (Davis §11)",
      "森林调查与资源清查 (Davis §4)",
      "森林经营规划 (Davis §7)",
      "收获调度与可持续收获 (Davis §9)",
      "森林认证（FSC）与多功能经营 (Davis §14)",
      "林业经济分析 (Davis §12)",
      "森林可持续管理 (Davis §13)"
    ],
  },
  'life/wood-science': {
    title: "木材科学与技术",
    books: [
          "Hoadley, \"Understanding Wood: A Craftsman's Guide to Wood Technology\" (2000)",
          "Bowyer et al., \"Forest Products and Wood Science: An Introduction\" (6th, 2007)"
    ],
    chapters: [
      "木材的宏观与微观结构 (Bowyer §1)",
      "木材识别与解剖 (Bowyer §2)",
      "木材化学成分 (Bowyer §3)",
      "木材物理性质 (Hoadley §5)",
      "木材力学性质 (Bowyer §7)",
      "木材水分关系与干燥 (Hoadley §6)",
      "木材缺陷与等级评定 (Bowyer §6)",
      "木材改性（热处理/化学改性） (Bowyer §9)",
      "木材耐久性与防腐 (Bowyer §11)",
      "胶黏剂与胶合技术 (Bowyer §14)",
      "木基复合材料精细化 (Bowyer §13)",
      "木质复合材料与人造板 (Bowyer §13)",
      "木材产品加工工艺 (Bowyer §15)"
    ],
  },
  'life/food-nutrition-health': {
    title: "食品营养与健康",
    books: [
          "Belitz et al., \"Food Chemistry\" (5th, 2009)",
          "Ross et al., \"Modern Nutrition in Health and Disease\" (11th, 2014)"
    ],
    chapters: [
      "食品营养学基础 (Ross §1)",
      "蛋白质与氨基酸 (Belitz §2)",
      "脂类与脂质氧化 (Belitz §3)",
      "碳水化合物与膳食纤维 (Belitz §4)",
      "维生素与矿物质 (Ross §11-15)",
      "膳食指南与膳食结构（膳食平衡） (Ross §97)",
      "特殊人群营养（孕期/儿童/老年） (Ross §23)",
      "运动营养与植物化学物（生物活性成分） (Ross §30)",
      "食品加工对营养的影响 (Belitz §9)",
      "功能性食品与生物活性成分 (Ross §115)",
      "营养标签与法规 (Ross §98)",
      "营养与慢性病预防 (Ross §114)"
    ],
  },
  'life/food-safety-quality': {
    title: "食品安全与质量控制",
    books: [
          "Forsythe, \"The Microbiology of Safe Food\" (3rd, 2020)",
          "Mossel et al., \"Essentials of the Microbiology of Foods\" (1995)"
    ],
    chapters: [
      "食品安全微生物学 (Forsythe §1)",
      "食源性病原细菌 (Forsythe §3)",
      "食源性病毒与寄生虫 (Forsythe §6)",
      "食品腐败与货架期 (Forsythe §8)",
      "食品化学性危害（重金属/农药残留/天然毒素/加工污染物） (Forsythe §10)",
      "HACCP体系原理 (Forsythe §11)",
      "食品法规与标准（Codex/食品安全法） (Forsythe §12)",
      "食品添加剂安全性评价 (Forsythe §12)",
      "风险评估与风险管理 (Forsythe §13)",
      "食品质量管理体系 (ISO) (Forsythe §14)",
      "食品安全检测技术 (Forsythe §15)",
      "食品掺假与欺诈、过敏原管理与追溯 (Forsythe §16)"
    ],
  },
  'life/agricultural-resources-environment': {
    title: "农业资源与环境",
    books: [
          "Lal & Stewart, \"Soil Management and Agricultural Sustainability\" (2017)",
          "Matson, Parton & Power, \"Agricultural Intensification and Ecosystem Properties\" (1997)"
    ],
    chapters: [
      "农业资源环境概论 (Lal §1)",
      "土壤资源与退化 (Lal §3)",
      "盐碱化与水土流失防治 (Lal §4)",
      "水资源与农业利用 (Lal §5)",
      "农业环境质量监测与评价 (Lal §6)",
      "农业生态系统养分循环 (Matson §4)",
      "农业面源污染 (Lal §9)",
      "农田重金属污染与修复技术 (Lal §10)",
      "农用化学品环境行为与农业环保法规 (Matson §5)",
      "农业温室气体与气候变化 (Lal §11)",
      "废弃物资源化利用 (Lal §13)",
      "可持续农业管理 (Lal §16)"
    ],
  },
  'life/agroecology': {
    title: "农业生态学",
    books: [
          "Gliessman, \"Agroecology: The Ecology of Sustainable Food Systems\" (4th, 2018)",
          "Altieri, \"Agroecology: The Science of Sustainable Agriculture\" (2nd, 2018)"
    ],
    chapters: [
      "农业生态学概论 (Gliessman §1)",
      "农业生态系统结构 (Gliessman §3)",
      "农业生态系统能量流动与食物链 (Gliessman §4)",
      "种群与群落生态 (Gliessman §6)",
      "生物多样性与农业 (Altieri §4)",
      "土壤生态与养分管理 (Gliessman §9)",
      "病虫害生态管理 (Altieri §8)",
      "中国生态农业模式 (Altieri §11)",
      "农业可持续性评价 (Gliessman §17)",
      "可持续食物系统转型 (Gliessman §22)"
    ],
  },
  'life/soil-science-plant-nutrition': {
    title: "土壤学与植物营养（深化）",
    books: [
          "Brady & Weil, \"The Nature and Properties of Soils\" (15th, 2017)",
          "Marschner, \"Marschner's Mineral Nutrition of Higher Plants\" (3rd, 2012)"
    ],
    chapters: [
      "土壤组成与发生 (Brady §1-2)",
      "土壤分类系统与土壤调查（USDA/中国系统） (Brady §3)",
      "土壤物理性质 (Brady §4)",
      "土壤化学与胶体 (Brady §8)",
      "土壤酸碱度与盐碱化 (Brady §9)",
      "土壤生物学与有机质 (Brady §11)",
      "土壤养分与肥力 (Brady §14)",
      "植物必需营养元素 (Marschner §2)",
      "养分吸收与转运机制 (Marschner §3)",
      "根际过程与菌根 (Marschner §7)",
      "养分胁迫与植物适应 (Marschner §9)",
      "肥料学与施肥原理 (Marschner §14)",
      "养分诊断与推荐施肥 (Marschner §15)"
    ],
  },
  'social/agricultural-economics': {
    title: "农业经济管理",
    books: [
          "Colman & Young, \"Principles of Agricultural Economics\" (1989)",
          "Barnard & Nix, \"Farm Planning and Control\" (2nd, 1979)"
    ],
    chapters: [
      "农业经济学基础 (Colman §1)",
      "农业生产函数与要素配置 (Colman §4)",
      "供给、需求与市场价格 (Colman §6)",
      "土地经济学与地租理论 (Colman §5)",
      "农产品市场营销与价格分析 (Colman §7)",
      "农场经营决策 (Barnard §3)",
      "农场预算与财务分析（现金/部分/整体预算、盈亏平衡） (Barnard §4-5)",
      "线性规划与农场资源配置技术 (Barnard §6)",
      "风险与不确定性 (Barnard §7)",
      "农业金融、信贷与农业技术进步 (Colman §11, §13)",
      "农业政策与贸易 (Colman §12)",
      "农业合作与组织 (Colman §14)",
      "农业可持续发展经济 (Colman §15)"
    ],
  },
  'life/rural-regional-development': {
    title: "农村区域发展",
    books: [
          "Chambers, \"Rural Development: Putting the Last First\" (1983)",
          "Ellis, \"Peasant Economics\" (2nd, 1993)"
    ],
    chapters: [
      "农村发展理论 (Chambers §1)",
      "农村生计与农户经济 (Ellis §2)",
      "参与式发展方法 (Chambers §5)",
      "贫困与不平等 (Chambers §4)",
      "农村资源管理与可持续 (Chambers §7)",
      "农村产业与乡村振兴 (Chambers §8)",
      "乡村振兴战略与城乡融合发展 (Chambers §12)",
      "农村金融与保险（产业振兴） (Chambers §13)",
      "农村电商与数字乡村 (Chambers §14)",
      "农村土地制度与流转 (Ellis §9)",
      "农民合作组织 (Ellis §10)",
      "农村基础设施与服务 (Chambers §10)",
      "农村治理与制度 (Ellis §12)"
    ],
  },
  'engineering/agricultural-mechanization': {
    title: "农业机械化与自动化",
    books: [
          "Kepner, Bainer & Barger, \"Principles of Farm Machinery\" (3rd, 1978)",
          "Hunt, \"Farm Power and Machinery Management\" (11th, 2001)"
    ],
    chapters: [
      "农业机械学基础 (Kepner §1)",
      "拖拉机与动力系统 (Hunt §3)",
      "耕作与整地机械 (Kepner §4)",
      "保护性耕作与免耕播种机械 (Kepner §7)",
      "播种与施肥机械 (Kepner §6)",
      "植保机械 (Kepner §9)",
      "农业机器人（采摘/除草/温室机器人）与无人机植保 (Hunt §11)",
      "收获机械 (Kepner §11)",
      "灌溉与排灌机械（滴灌/喷灌/泵站） (Kepner §12)",
      "畜牧与饲料加工机械 (Kepner §13)",
      "谷物干燥与仓储机械 (Kepner §14)",
      "精准农业装备 (Hunt §10)",
      "农业机械化经营管理 (Hunt §12)"
    ],
  },
  'life/land-resource-management': {
    title: "土地资源管理",
    books: [
          "Dent & Young, \"Soil Survey and Land Evaluation\" (1981)",
          "FAO, \"Land Evaluation: Towards a Revised Framework\" (2007)"
    ],
    chapters: [
      "土地资源学基础 (Dent §1)",
      "土地调查与制图 (Dent §4)",
      "土壤与土地分类系统 (Dent §6)",
      "土地评价原理与方法 (FAO §2)",
      "土地利用适宜性评价 (FAO §3)",
      "土地利用变化遥感监测 (FAO §4)",
      "土地利用规划 (FAO §5)",
      "土地制度与法规、产权制度 (FAO §6)",
      "地籍管理与土地登记 (Dent §9)",
      "土地经济与土地市场（地价/估价） (Dent §10)",
      "土地整治与复垦 (Dent §11)",
      "土地资源可持续管理 (FAO §7)"
    ],
  },
  'social/econometrics': {
    title: "计量经济学",
    books: [
          "Wooldridge, \"Introductory Econometrics: A Modern Approach\" (7th, 2020)",
          "Greene, \"Econometric Analysis\" (8th, 2018)"
    ],
    chapters: [
      "简单线性回归模型 (Wooldridge §2)",
      "多元回归分析:估计与推断 (Wooldridge §3-4)",
      "多元回归的进一步问题：函数形式、预测、拟合优度与模型选择 (Wooldridge §6)",
      "虚拟变量与定性信息（交互项、分段回归） (Wooldridge §7)",
      "异方差性与OLS渐近性 (Wooldridge §5, §8)",
      "设定与数据问题 (Greene §5)",
      "时间序列回归基础：趋势、季节与平稳性及序列相关 (Wooldridge §10-12)",
      "时间序列（ARMA/ARIMA/单位根/协整） (Wooldridge §18)",
      "面板数据方法 (Wooldridge §13-14)",
      "面板数据深化（固定/随机效应） (Wooldridge §14)",
      "工具变量与两阶段最小二乘 (Wooldridge §15)",
      "联立方程模型 (Wooldridge §16)",
      "限值因变量模型（Probit/Logit/Tobit） (Wooldridge §17)",
      "GMM 广义矩估计 (Greene §13-14)"
    ],
  },
  'social/labor-economics': {
    title: "劳动经济学",
    books: [
          "Borjas, \"Labor Economics\" (8th, 2020)",
          "Cahuc, Carcillo & Zylberberg, \"Labor Economics\" (2nd, 2014)"
    ],
    chapters: [
      "劳动供给理论 (Borjas §2)",
      "劳动需求与厂商决策 (Borjas §3)",
      "劳动力市场均衡 (Borjas §4)",
      "人力资本投资与教育 (Borjas §6)",
      "补偿性工资差异 (Borjas §5)",
      "工资结构：技能溢价与工会工资差异 (Borjas §7)",
      "劳动力流动与移民 (Borjas §8)",
      "劳动市场歧视 (Borjas §9)",
      "工会与集体谈判 (Borjas §10)",
      "激励报酬与绩效工资（计件/锦标赛/委托代理） (Borjas §11)",
      "失业理论：搜索模型、效率工资与粘性工资 (Borjas §12)",
      "劳动市场制度：最低工资、就业保护与失业保险 (Cahuc 第4部分)"
    ],
  },
  'social/monetary-banking': {
    title: "货币银行学",
    books: [
          "Mishkin, \"The Economics of Money, Banking, and Financial Markets\" (12th, 2019)"
    ],
    chapters: [
      "货币与货币制度 (Mishkin §3)",
      "金融市场与利率行为 (Mishkin §4-5)",
      "利率的风险与期限结构 (Mishkin §6)",
      "股票市场与有效市场假说 (Mishkin §7)",
      "金融机构与银行业 (Mishkin §9-11)",
      "金融危机：金融脆弱性与 2008 危机 (Mishkin §12)",
      "中央银行与联邦储备体系 (Mishkin §13)",
      "货币供给过程 (Mishkin §14)",
      "货币政策工具与策略 (Mishkin §15-16)",
      "外汇市场与国际金融体系 (Mishkin §17-18)",
      "货币数量论与货币需求理论 (Mishkin §19)",
      "IS 曲线与 MP/AD 曲线 (Mishkin §20-21)",
      "AD-AS 分析框架 (Mishkin §22)",
      "货币政策理论与政策规则 (Mishkin §23)",
      "货币政策中的预期 (Mishkin §24)",
      "货币政策传导机制 (Mishkin §25)"
    ],
  },
  'social/public-finance': {
    title: "公共财政",
    books: [
          "Rosen & Gayer, \"Public Finance\" (10th, 2014)",
          "Stiglitz, \"Economics of the Public Sector\" (4th, 2015)"
    ],
    chapters: [
      "公共部门与效率 (Rosen §2-3)",
      "公共物品 (Rosen §4)",
      "外部性与环境政策 (Rosen §5)",
      "政治经济学与公共选择 (Rosen §6)",
      "政府失灵与公共选择深化 (Stiglitz §6)",
      "成本收益分析与公共支出评价 (Rosen §8)",
      "医疗公共支出 (Rosen §9)",
      "教育公共支出 (Rosen §7)",
      "收入分配、贫困与社会保障 (Rosen §11)",
      "社会保险项目：社会保障、失业保险与工伤保险 (Rosen §12-13)",
      "税收与归宿理论 (Rosen §14)",
      "税收的效率成本与超额负担 (Rosen §15)",
      "最优税收理论 (Rosen §16)",
      "公司税、消费税与财富税 (Rosen §19-21)",
      "公债与赤字财政 (Rosen §22)",
      "财政联邦主义 (Stiglitz §25)"
    ],
  },
  'social/international-economics': {
    title: "国际经济学",
    books: [
          "Krugman, Obstfeld & Melitz, \"International Economics: Theory and Policy\" (11th, 2018)"
    ],
    chapters: [
      "比较优势与李嘉图模型 (Krugman §3)",
      "特定要素模型与收入分配 (Krugman §4)",
      "要素禀赋与赫克歇尔-俄林模型 (Krugman §5)",
      "标准贸易模型：贸易条件、增长与福利 (Krugman §6)",
      "新贸易理论与外部规模经济 (Krugman §7)",
      "国际要素流动与跨国公司 (Krugman §8)",
      "贸易政策工具 (Krugman §9-10)",
      "发展中国家贸易政策与增长 (Krugman §11)",
      "国民收入核算与国际收支平衡表 (Krugman §13)",
      "汇率与开放经济的宏观经济学 (Krugman §14-16)",
      "汇率决定理论与购买力平价 (Krugman §15)",
      "固定汇率制与外汇干预 (Krugman §18)",
      "国际货币体系 (Krugman §19-20)",
      "最优货币区与欧元 (Krugman §21)",
      "发展中国家：增长与危机 (Krugman §22)"
    ],
  },
  'social/economic-history': {
    title: "经济史",
    books: [
          "Allen, \"Global Economic History: A Very Short Introduction\" (2011)",
          "North, \"Structure and Change in Economic History\" (1981)"
    ],
    chapters: [
      "经济史方法论：真实工资比较与国民核算的史学应用 (Allen 绪论)",
      "马尔萨斯陷阱、人口转型与前工业经济 (Allen §2)",
      "大分流与全球不平等 (Allen §1)",
      "工业革命与西方起飞 (Allen §3)",
      "第一次与第二次经济革命 (North §7, §10)",
      "殖民主义与世界经济 (Allen §5)",
      "北美经济史与工业欧洲（第二次工业革命） (Allen §6-7)",
      "亚洲经济史与东亚复兴 (Allen §8)",
      "大萧条、两次世界大战与 20 世纪经济史 (Allen 补充)",
      "二战后黄金时代、全球化与反全球化 (Allen 补充)",
      "经济增长的制度基础 (North §1-2)",
      "产权、国家与交易成本 (North §8)",
      "制度变迁的经济史分析 (North §13)",
      "金融与货币制度演化 (North §9)"
    ],
  },
  'social/institutional-economics': {
    title: "制度经济学",
    books: [
          "North, \"Institutions, Institutional Change and Economic Performance\" (1990)",
          "Acemoglu & Robinson, \"Why Nations Fail\" (2012)"
    ],
    chapters: [
      "制度的定义与类型 (North §1)",
      "合作、制度与经济行为 (North §2)",
      "非正式约束：文化、习俗与意识形态 (North §3)",
      "正式约束与产权制度 (North §4)",
      "制度的执行与实施机制 (North §5)",
      "制度与交易成本 (North §6)",
      "制度与经济绩效的实证 (North §7-8)",
      "路径依赖与锁入效应 (North §9)",
      "制度变迁理论 (North §10)",
      "制度变迁的驱动者：组织、企业家与政治权力 (Acemoglu 相应章)",
      "包容性与榨取性制度 (Acemoglu §1)"
    ],
  },
  'social/financial-engineering': {
    title: "金融工程",
    books: [
          "Hull, \"Options, Futures, and Other Derivatives\" (11th, 2021)"
    ],
    chapters: [
      "期货市场机制与套期保值 (Hull §2-3)",
      "利率基础与远期利率 (Hull §4)",
      "远期与期货定价 (Hull §5)",
      "利率期货 (Hull §6)",
      "互换市场 (Hull §7)",
      "期权性质与交易策略 (Hull §10-11)",
      "二叉树模型与Black-Scholes-Merton模型 (Hull §12-14)",
      "希腊字母与风险管理 (Hull §18)",
      "波动率微笑与波动率曲面 (Hull §19)",
      "风险价值 VaR (Hull §20)",
      "信用衍生品 (Hull §23)",
      "奇异期权与其他非标准产品 (Hull §22)",
      "数值方法：蒙特卡洛模拟与有限差分法 (Hull §26)",
      "利率衍生品 (Hull §28-29)",
      "利率模型：均衡模型与无套利模型 (Hull §30-31)",
      "实物期权与衍生品风险案例 (Hull §35-36)"
    ],
  },
  'social/insurance': {
    title: "保险学",
    books: [
          "Vaughan & Vaughan, \"Fundamentals of Risk and Insurance\" (10th, 2008)",
          "Rejda & McNamara, \"Principles of Risk Management and Insurance\" (13th, 2017)"
    ],
    chapters: [
      "风险与风险管理 (Rejda §1, §3)",
      "企业风险管理（ERM）与风险控制 (Rejda §4)",
      "保险公司运营：核保、理赔、营销系统与中介 (Rejda §5-6)",
      "保险公司的财务运作与精算/准备金 (Rejda §7)",
      "保险监管 (Rejda §8)",
      "保险合同原理 (Rejda §9-10)",
      "人寿保险 (Vaughan §13-14)",
      "年金与退休计划、员工福利 (Rejda §14, §17)",
      "健康保险与社会保障 (Rejda §15-16)",
      "财产与责任保险 (Rejda §19, §24-26)",
      "汽车保险与家财保险各论 (Rejda §20-23)",
      "再保险 (Vaughan §21)"
    ],
  },
  'social/comparative-politics': {
    title: "比较政治学",
    books: [
          "Lijphart, \"Patterns of Democracy\" (2nd, 2012)",
          "Hague, Harrop & McCormick, \"Comparative Government and Politics\" (10th, 2016)"
    ],
    chapters: [
      "国家建构与制度转型 (Hague §2)",
      "民主与威权政体 (Hague §3)",
      "政治文化与政治社会化 (Hague §4)",
      "政治参与：投票、抗议与公民参与 (Hague §5)",
      "选举与选民行为 (Hague §6)",
      "选举制度与政党制度 (Hague §8)",
      "利益集团与社团政治 (Hague §9)",
      "宪法、司法与违宪审查 (Hague §10)",
      "立法机关（议会）比较 (Hague §11)",
      "官僚制比较 (Hague §13)",
      "多数民主与共识民主 (Lijphart §2)",
      "总统制与议会制 (Lijphart §6)",
      "民主化理论（转型学/现代化） (Almond §7)",
      "民族冲突与革命 (Almond §9)",
      "联邦主义 (Almond §6)"
    ],
  },
  'social/international-political-economy': {
    title: "国际政治经济学",
    books: [
          "Oatley, \"International Political Economy\" (5th, 2018)",
          "Frieden & Lake, \"International Political Economy: Perspectives on Global Power and Wealth\" (5th, 2016)"
    ],
    chapters: [
      "IPE的理论传统 (Oatley §1-2)",
      "国际贸易体系的政治经济学 (Oatley §4-5)",
      "跨国公司与全球生产网络（FDI、外包、价值链） (Oatley §6)",
      "汇率制度选择的政治经济学 (Oatley §7)",
      "国际货币体系与汇率政治 (Oatley §8)",
      "国际金融机构的政治：IMF、世界银行与区域开发银行 (Oatley §9)",
      "区域经济一体化（欧盟、北美、RCEP） (Oatley §13)",
      "发展中国家债务危机与结构调整 (Frieden §10)",
      "全球化与跨国投资 (Frieden §11)",
      "发展与不平等 (Oatley §11)",
      "全球治理与金融危机 (Oatley §10)",
      "国际发展援助与援助政治 (Frieden §17)",
      "全球化反弹与保护主义 (Oatley §12)"
    ],
  },
  'social/constitutional-administrative-law': {
    title: "宪法学与行政法学",
    books: [
          "许崇德、胡锦光,《宪法学》(第7版, 2021)",
          "应松年,《行政法学教程》(第4版, 2020)"
    ],
    chapters: [
      "宪法基本理论 (许崇德 §1)",
      "宪法的基本原则与宪法史（中国宪法发展历程） (许崇德 §2-3)",
      "宪法实施与违宪审查 (许崇德 §5)",
      "国体、政体与人民代表大会制度 (许崇德 §6-7)",
      "国家结构形式：单一制、民族区域自治与特别行政区 (许崇德 §16-17)",
      "公民基本权利与义务：总论 (许崇德 §11)",
      "基本权利分论：平等权、政治权利、人身自由与社会经济权利 (许崇德 §12-14)",
      "国家机构 (许崇德 §18+)",
      "行政法基础理论 (应松年 §1)",
      "行政主体与公务员制度 (应松年 §3)",
      "行政行为 (应松年 §4)",
      "具体行政行为分论：行政许可、行政处罚与行政强制 (应松年 §5)",
      "行政程序与行政救济 (应松年 §6-7)",
      "行政复议与行政诉讼 (应松年 §8)",
      "行政赔偿与国家赔偿 (应松年 §9)"
    ],
  },
  'social/civil-commercial-law': {
    title: "民商法学",
    books: [
          "王利明,《民法学》(第9版, 2022)",
          "范健、王建文,《商法学》(第5版, 2021)"
    ],
    chapters: [
      "民法总则 (王利明 §1)",
      "物权法 (王利明 §3)",
      "债法总则与担保制度（保证、定金与担保物权） (王利明 §4)",
      "合同法 (王利明 §5)",
      "人格权法 (王利明 人格权编)",
      "婚姻家庭法与继承法 (王利明 §6)",
      "侵权责任法 (王利明 §7)",
      "商法总论：商主体、商行为、商事登记与商业账簿 (范健 §1)",
      "公司法 (范健 §3)",
      "票据法与证券法 (范健 §6-7)",
      "破产法 (范健 §9)",
      "保险法 (范健 保险编)",
      "海商法 (范健 海商编)"
    ],
  },
  'social/criminal-law': {
    title: "刑法学",
    books: [
          "高铭暄、马克昌,《刑法学》(第10版, 2022)",
          "张明楷,《刑法学》(第6版, 2021)"
    ],
    chapters: [
      "刑法概说与基本原则 (高铭暄 §1)",
      "犯罪构成要件 (张明楷 §5)",
      "犯罪主体与主观方面（故意、过失与认识错误） (高铭暄 §5)",
      "犯罪客观方面：因果关系与不作为犯 (高铭暄 §6)",
      "正当防卫与紧急避险 (高铭暄 §8)",
      "故意犯罪形态 (张明楷 §10)",
      "共同犯罪 (高铭暄 §10)",
      "罪数理论：犯罪竞合与一罪数罪 (高铭暄 §11)",
      "刑罚体系与裁量 (高铭暄 §12-13)",
      "刑罚执行与消灭：减刑、假释、时效与赦免 (高铭暄 §14-16)",
      "危害国家安全罪与危害公共安全罪 (高铭暄 §17-18)",
      "破坏社会主义市场经济秩序罪（金融与经济犯罪） (高铭暄 §19)",
      "侵犯公民人身权利罪 (高铭暄 §20)",
      "侵犯财产罪 (高铭暄 §21)",
      "妨害社会管理秩序罪 (高铭暄 §22)",
      "贪污贿赂罪与渎职罪 (高铭暄 §24-25)"
    ],
  },
  'social/international-law': {
    title: "国际法学",
    books: [
          "邵津,《国际法》(第5版, 2019)",
          "Brownlie, \"Principles of Public International Law\" (8th, 2012)"
    ],
    chapters: [
      "国际法基础理论 (邵津 §1)",
      "国际法渊源与基本原则（主权平等、不干涉、禁止使用武力） (邵津 §2)",
      "国际法主体 (邵津 §3)",
      "国际责任与国家责任法 (邵津 §4)",
      "国家领土法 (邵津 §5)",
      "海洋法 (邵津 §6)",
      "外交与领事关系法（外交特权与豁免） (邵津 §8)",
      "条约法 (邵津 §9)",
      "国际人权法 (邵津 §10)",
      "国际环境法 (邵津 §11)",
      "联合国与国际组织法：安理会与专门机构 (邵津 §12)",
      "和平解决国际争端 (邵津 §13)",
      "武装冲突法与国际人道法（战争法） (邵津 §14)",
      "引渡、庇护与国际刑法（国际刑事法院） (邵津 §15)"
    ],
  },
  'social/social-work': {
    title: "社会工作",
    books: [
          "王思斌,《社会工作导论》(第3版, 2021)",
          "Kirst-Ashman & Hull, \"Understanding Generalist Practice\" (8th, 2015)"
    ],
    chapters: [
      "社会工作发展历史与专业化过程 (王思斌 §1-3)",
      "社会工作理论框架 (王思斌 §2)",
      "社会工作价值观与伦理 (Kirst-Ashman §3)",
      "通用实务过程模式：接案—预估—计划—介入—评估 (Kirst-Ashman §5-6)",
      "个案工作方法 (王思斌 §5)",
      "小组工作 (Kirst-Ashman §8)",
      "社区工作 (王思斌 §7)",
      "社会工作行政与管理（服务机构管理与项目运作） (王思斌 §9)",
      "社会政策与社会福利 (王思斌 §10)",
      "社会工作研究方法与评估 (Kirst-Ashman 评估章)",
      "儿童与家庭社会工作 (Kirst-Ashman §11)",
      "医务与精神卫生社会工作 (Kirst-Ashman §13)",
      "老年、学校、矫正与残障社会工作 (Kirst-Ashman §14)"
    ],
  },
  'social/social-psychology': {
    title: "社会心理学",
    books: [
          "Myers & Twenge, \"Social Psychology\" (13th, 2018)",
          "Aronson, \"The Social Animal\" (12th, 2018)"
    ],
    chapters: [
      "社会心理学导论与方法 (Myers §1)",
      "自我与同一性 (Myers §2)",
      "社会信念与判断 (Myers §3)",
      "态度与说服 (Myers §4, §7)",
      "基因、文化与性别（演化与跨文化视角） (Myers §5)",
      "从众与服从 (Myers §6)",
      "群体影响 (Myers §8)",
      "偏见与刻板印象 (Myers §9)",
      "攻击行为 (Myers §10)",
      "吸引与亲密关系 (Myers §11)",
      "利他行为与亲社会行为 (Myers §12)",
      "冲突与和平解决 (Myers §13)",
      "社会心理学的应用：临床、法庭与可持续发展 (Myers §14-16)"
    ],
  },
  'social/rural-sociology': {
    title: "农村社会学",
    books: [
          "费孝通,《乡土中国》(1948)",
          "Fei, \"From the Soil: The Foundations of Chinese Society\" (trans. Hamilton & Wang, 1992)"
    ],
    chapters: [
      "乡土社会的本色 (费孝通 §1)",
      "文字下乡、再论文字下乡 (费孝通 §2-3)",
      "差序格局 (费孝通 §4)",
      "农村社会结构与家族 (费孝通 §6)",
      "男女有别 (费孝通 §7)",
      "礼治秩序 (费孝通 §8)",
      "无讼——乡土社会的纠纷解决 (费孝通 §9)",
      "无为政治与乡土权力结构 (费孝通 §10)",
      "长老统治与教化权力 (费孝通 §11)",
      "血缘与地缘 (费孝通 §12)",
      "名实的分离、从欲望到需要 (费孝通 §13-14)",
      "从传统到现代的农村变迁 (费孝通 综论)",
      "城乡关系与小城镇问题 (Fei 附录)",
      "农民工与城乡流动 (Fei 附录 补充)",
      "农村土地制度与产权改革 (当代议题)",
      "乡村治理：村民自治与基层治理 (当代议题)",
      "农村社会分层、贫困与乡村振兴 (当代议题)"
    ],
  },
  'social/criminology': {
    title: "犯罪学",
    books: [
          "Siegel, \"Criminology\" (13th, 2018)",
          "Vold, Bernard & Snipes, \"Theoretical Criminology\" (6th, 2010)"
    ],
    chapters: [
      "犯罪学理论与方法 (Siegel §1)",
      "犯罪统计与测量 (Siegel §2)",
      "古典与新古典学派：贝卡里亚、边沁与威慑理论 (Siegel §4)",
      "生物与心理犯罪理论 (Vold §6-7)",
      "社会结构与犯罪 (Siegel §6)",
      "社会过程理论（标签/社会学习/社会控制） (Siegel §7)",
      "社会冲突理论（马克思主义/左翼现实主义） (Siegel §8)",
      "发展理论：生命历程与潜在特质理论 (Siegel §9)",
      "犯罪类型：暴力、财产与白领犯罪 (Siegel §10-12)",
      "公共秩序犯罪与网络犯罪类型学 (Siegel §13)",
      "受害者学 (Vold §15)",
      "犯罪预防与环境犯罪学 (Siegel 犯罪预防章)",
      "刑罚学与矫正 (Siegel §14)",
      "刑事司法体系：警察、法院与矫正系统比较 (Siegel §15-16)"
    ],
  },
  'social/public-policy': {
    title: "公共政策",
    books: [
          "Anderson, \"Public Policymaking\" (8th, 2014)",
          "Dye, \"Understanding Public Policy\" (15th, 2017)"
    ],
    chapters: [
      "公共政策模型 (Dye §2)",
      "政策分析的方法论：理性、渐进、制度与精英模型 (Dye §1)",
      "议程设置与政策制定 (Anderson §4-5)",
      "预算与政策过程 (Anderson §6)",
      "政策执行 (Anderson §7)",
      "政策评估 (Anderson §8)",
      "政策变迁、政策学习与政策反馈 (Anderson 政策过程末章)",
      "政策工具与设计 (Dye §3)",
      "刑事司法与公共安全政策 (Dye §4)",
      "健康与教育政策 (Dye §5-6)",
      "经济政策与管制政策 (Dye §9-10)",
      "税收、贸易与移民政策 (Dye §8, §10-11)",
      "环境政策 (Dye §12)",
      "社会福利政策 (Dye §15)",
      "比较公共政策 (Dye §14)"
    ],
  },
  'social/public-administration': {
    title: "行政管理",
    books: [
          "Rosenbloom, Kravchuk & Clerkin, \"Public Administration: Understanding Management, Politics, and Law in the Public Sector\" (8th, 2015)"
    ],
    chapters: [
      "公共行政的三大途径 (Rosenbloom §1)",
      "政治与行政的关系：官僚政治、政治控制与行政责任 (Rosenbloom 政治篇)",
      "组织理论 (Rosenbloom §3)",
      "人事行政与公务员制度 (Rosenbloom §4)",
      "预算与财务行政 (Rosenbloom §5)",
      "政府间关系与央地关系管理 (Rosenbloom 央地关系章)",
      "新公共管理（NPM） (Rosenbloom §8)",
      "行政法与公共利益 (Rosenbloom §9)",
      "电子政务 (Rosenbloom §10)",
      "行政伦理与问责 (Rosenbloom §12)",
      "公共选择与制度分析在公共行政中的应用 (Rosenbloom 制度分析章)",
      "重塑政府与行政改革史 (Rosenbloom 改革篇)"
    ],
  },
  'social/human-resource-management': {
    title: "人力资源管理",
    books: [
          "Dessler, \"Human Resource Management\" (16th, 2019)",
          "Noe et al., \"Human Resource Management: Gaining a Competitive Advantage\" (11th, 2020)"
    ],
    chapters: [
      "战略人力资源管理 (Noe §1)",
      "平等就业机会与反歧视法律（EEO/肯定性行动） (Dessler §2)",
      "人力资源规划与工作分析 (Dessler §3-4)",
      "招聘与甄选 (Dessler §6-7)",
      "培训与开发 (Noe §7)",
      "绩效管理与评估 (Dessler §9)",
      "职业生涯管理与员工保留 (Dessler §10)",
      "薪酬管理 (Dessler §11)",
      "绩效薪酬与财务激励计划 (Dessler §12)",
      "福利与服务管理 (Dessler §13)",
      "员工关系与职业安全 (Noe §15)",
      "人力资源信息系统与 HR 分析（HRIS/People Analytics） (Dessler 现代章)",
      "国际人力资源管理 (Dessler §17)"
    ],
  },
  'social/tourism-management': {
    title: "旅游管理",
    books: [
          "Goeldner & Ritchie, \"Tourism: Principles, Practices, Philosophies\" (14th, 2018)",
          "Cooper et al., \"Tourism: Principles and Practice\" (5th, 2016)"
    ],
    chapters: [
      "旅游系统与基本概念 (Goeldner §3)",
      "旅游史与旅游需求预测 (Goeldner §2, §11-12)",
      "旅游动机与旅游者行为 (Cooper §5)",
      "旅游目的地开发 (Cooper §5, §11)",
      "旅游吸引物与景点管理 (Goeldner 景点章)",
      "旅游交通与基础设施 (Cooper 交通章)",
      "旅游中介：旅行社与旅游运营商 (Cooper 中介章)",
      "酒店与接待业管理 (Cooper §8)",
      "旅游营销 (Goeldner §14)",
      "旅游组织与旅游政策规划 (Goeldner 政策规划章)",
      "旅游的经济、社会文化与环境影响 (Cooper §13)",
      "可持续旅游与生态旅游 (Cooper §15)",
      "旅游业的未来趋势 (Goeldner §17)"
    ],
  },
  'social/developmental-psychology': {
    title: "发展心理学",
    books: [
          "Berk, \"Development Through the Lifespan\" (7th, 2017)",
          "Santrock, \"Life-Span Development\" (17th, 2019)"
    ],
    chapters: [
      "发展理论与研究方法 (Berk §1)",
      "遗传与环境基础 (Berk §2)",
      "产前发育、出生与新生儿 (Berk §3)",
      "婴儿期的生理与认知发展 (Berk §4)",
      "婴儿期的情绪与社会发展 (Berk §6)",
      "儿童期认知与语言发展 (Berk §5-6)",
      "学前期与童年中期的社会性发展 (Berk §8, §10)",
      "青少年期的同一性与社会化 (Berk §11)",
      "道德发展（Piaget/Kohlberg） (Berk §12)",
      "性别发展 (Berk §13)",
      "成年早期与中期的心理发展 (Berk §13-15)",
      "老年期与生命终结 (Berk §17)",
      "死亡、临终与丧亲 (Berk §19)",
      "发展障碍与特殊需要 (Santrock 相应章)"
    ],
  },
  'social/clinical-psychology': {
    title: "临床心理学",
    books: [
          "Barlow, \"Abnormal Psychology: An Integrative Approach\" (8th, 2018)",
          "Kring & Johnson, \"Abnormal Psychology\" (13th, 2018)"
    ],
    chapters: [
      "异常心理学的理论与研究方法 (Barlow §1-2)",
      "临床评估与诊断 (Barlow §3)",
      "焦虑障碍 (Kring §5)",
      "躯体症状障碍与分离障碍 (Barlow §6)",
      "心境障碍与自杀 (Barlow §7)",
      "进食障碍 (Barlow §8)",
      "物质使用障碍 (Barlow §9)",
      "人格障碍 (Barlow §10)",
      "精神分裂症与其他精神病性障碍 (Kring §11)",
      "神经发育障碍与神经认知障碍 (Barlow §12-13)",
      "性功能障碍、性别烦躁与性偏好障碍 (Barlow §14)",
      "精神卫生服务、法律与伦理 (Barlow §15)",
      "心理治疗与干预（CBT、心理动力与人本主义） (Kring §15)",
      "生物治疗与精神药理学 (Barlow 生物治疗章)"
    ],
  },
  'social/educational-psychology': {
    title: "教育心理学",
    books: [
          "Woolfolk, \"Educational Psychology\" (14th, 2019)",
          "Slavin, \"Educational Psychology: Theory and Practice\" (12th, 2018)"
    ],
    chapters: [
      "认知发展 (Woolfolk §2)",
      "社会性发展与道德发展 (Woolfolk §3)",
      "学习者差异与特殊教育 (Woolfolk §4)",
      "语言发展与语言多样性 (Woolfolk §5)",
      "文化与多元文化教育 (Woolfolk §6)",
      "学习理论:行为主义与社会认知 (Slavin §6)",
      "认知与建构主义学习观 (Woolfolk §8-9)",
      "复杂认知过程：问题解决、迁移与批判性思维 (Woolfolk §9)",
      "自我调节学习与元认知、学习科学 (Woolfolk §10)",
      "学习动机 (Slavin §10)",
      "课堂管理与教学策略 (Woolfolk §13)",
      "教学评估与标准化测试 (Slavin §14)",
      "教师心理与专业发展 (Slavin §13)"
    ],
  },
  'social/industrial-organizational-psychology': {
    title: "工业与组织心理学",
    books: [
          "Muchinsky & Culbertson, \"Psychology Applied to Work\" (11th, 2017)",
          "Spector, \"Industrial and Organizational Psychology: Research and Practice\" (7th, 2017)"
    ],
    chapters: [
      "研究方法与统计 (Muchinsky §2)",
      "工作分析与胜任力 (Muchinsky §3)",
      "人员选拔与效度验证 (Spector §4)",
      "绩效评估 (Muchinsky §5)",
      "培训与开发 (Muchinsky §7)",
      "工作动机与工作设计 (Muchinsky §8)",
      "工作满意度、组织承诺与工作态度 (Muchinsky §9)",
      "团队与群体过程、组织沟通 (Muchinsky 团队章)",
      "组织文化 (Muchinsky 文化章)",
      "领导力与管理 (Muchinsky §10)",
      "组织理论与组织行为 (Spector §10)",
      "工作压力与职业健康 (Spector §12)",
      "组织发展与变革管理 (Muchinsky §12)"
    ],
  },
  'humanities/comparative-literature': {
    title: "比较文学",
    books: [
          "Damrosch, \"What Is World Literature?\" (2003)",
          "Bassnett, \"Comparative Literature: A Critical Introduction\" (1993)"
    ],
    chapters: [
      "比较文学学科史与方法论 (Bassnett §1-2)",
      "影响研究 (Bassnett §5)",
      "平行研究与主题学 (Bassnett §6)",
      "形象学：异国形象研究 (乐黛云《比较文学原理》§形象学)",
      "文类学与比较诗学（类型学） (乐黛云《比较文学原理》§文类学)",
      "译介学 (谢天振《译介学》§1)",
      "世界文学的流通与翻译 (Damrosch §2-3)",
      "跨文化与跨学科研究 (Damrosch §10)",
      "比较文学的中国学派 (乐黛云《比较文学原理》§中国学派)",
      "翻译研究的转向 (Bassnett §8)",
      "后殖民文学与世界文学 (Damrosch §6)"
    ],
  },
  'humanities/literary-criticism': {
    title: "文学批评",
    books: [
          "Eagleton, \"Literary Theory: An Introduction\" (3rd, 2011)",
          "Abrams & Harpham, \"A Glossary of Literary Terms\" (11th, 2015)"
    ],
    chapters: [
      "俄国形式主义与新批评 (Eagleton §1)",
      "现象学、诠释学与读者反应理论 (Eagleton §2)",
      "结构主义与符号学 (Eagleton §3)",
      "后结构主义与解构 (Eagleton §4)",
      "精神分析批评 (Eagleton §5 / Abrams §Psychoanalytic)",
      "马克思主义文学批评（政治批评） (Eagleton §6)",
      "女性主义批评 (Eagleton §6)",
      "后殖民主义批评（Said/Spivak） (Eagleton §6)",
      "新历史主义与文化批评 (Eagleton 结语)",
      "后现代主义批评 (Eagleton 结语 / Jameson/Hassan)",
      "文学批评的基本概念 (Abrams §1)"
    ],
  },
  'humanities/classical-philology': {
    title: "古典文献学",
    books: [
          "杜泽逊,《文献学概要》(修订本, 2021)",
          "黄永年,《古籍整理概论》(2001)"
    ],
    chapters: [
      "目录学 (杜泽逊 §2)",
      "版本学 (杜泽逊 §3)",
      "校勘学 (杜泽逊 §4)",
      "辨伪与辑佚 (杜泽逊 §6)",
      "古籍整理方法 (黄永年 §2)",
      "文献的流传与收藏 (杜泽逊 §8)",
      "训诂与注释学 (杜泽逊 §5)",
      "古籍数字化与电子文献 (黄永年 §6)"
    ],
  },
  'humanities/historical-geography': {
    title: "历史地理学",
    books: [
          "侯仁之,《历史地理学四论》(2007)",
          "谭其骧,《中国历史地图集》(1996)"
    ],
    chapters: [
      "历史地理学理论与方法 (侯仁之 §1 / 邹逸麟 §1)",
      "历史气候变迁 (侯仁之 §2)",
      "历史自然地理：河流改道、湖泊海岸与沙漠化 (邹逸麟《中国历史地理概述》§自然地理)",
      "历史人口地理 (侯仁之 §3)",
      "历史交通地理 (侯仁之 §4)",
      "政区沿革与疆域变迁 (谭其骧 图集说明 / 邹逸麟 §政区)",
      "历史城市地理 (侯仁之 §附录)",
      "历史经济地理 (侯仁之 §5)",
      "历史军事地理 (谭其骧 图集附录)",
      "历史地图学与历史地理信息系统（HGIS） (谭其骧 图集 / 邹逸麟 §地图)"
    ],
  },
  'humanities/philology': {
    title: "文献学",
    books: [
          "张舜徽,《中国文献学》(2005)",
          "程千帆,《校雠广义》(1999)"
    ],
    chapters: [
      "文献的载体与形制 (张舜徽 §2)",
      "文献的分类与目录 (张舜徽 §3)",
      "版本与校勘 (程千帆 版本编)",
      "典藏与流传 (程千帆 藏书编)",
      "文献的辑佚与辨伪 (张舜徽 §6)",
      "古代文献的检索与利用 (张舜徽 §8)",
      "训诂与注释体例 (张舜徽 §5)",
      "类书、丛书与工具书 (张舜徽 §7)"
    ],
  },
  'humanities/cultural-relics-conservation': {
    title: "文物学与文物保护",
    books: [
          "李晓东,《文物学》(2005)",
          "王蕙贞,《文物保护学》(2009)"
    ],
    chapters: [
      "文物分类与文物学理论 (李晓东 §1-2)",
      "文物鉴定 (李晓东 §5)",
      "石质文物保护 (王蕙贞 §4)",
      "土遗址保护（土质遗址防风化加固） (王蕙贞 §5)",
      "陶瓷与玻璃器保护 (王蕙贞 §6)",
      "纸质与纺织品文物保护 (王蕙贞 §6)",
      "金属文物保护 (王蕙贞 §3)",
      "壁画与彩塑保护 (王蕙贞 §8)",
      "木质与漆器文物保护 (王蕙贞 §7)",
      "考古现场文物保护 (李晓东 §8)",
      "文物价值评估与文物管理法规（《文物保护法》与登录制度） (李晓东 §6)"
    ],
  },
  'humanities/archaeometry': {
    title: "科技考古",
    books: [
          "陈铁梅,《科技考古》(2008)",
          "Renfrew & Bahn, \"Archaeology: Theories, Methods, and Practice\" (8th, 2020)"
    ],
    chapters: [
      "考古测年：碳十四、释光、树轮与纹泥定年 (陈铁梅 §3 / Renfrew §4)",
      "地球物理勘探、遥感与GIS考古 (Renfrew §3 / 陈铁梅 §物探)",
      "文物成分分析与产地溯源 (陈铁梅 §5)",
      "稳定同位素与食谱分析 (陈铁梅 §7)",
      "动物考古 (Renfrew §10)",
      "植物考古与农业起源 (Renfrew §9)",
      "石器微痕与实验考古 (陈铁梅 §石器 / Renfrew §技术)",
      "古DNA与分子考古 (Renfrew §11)",
      "冶金考古与金属器物研究 (陈铁梅 §9)"
    ],
  },
  'humanities/paleography': {
    title: "古文字学",
    books: [
          "裘锡圭,《文字学概要》(修订本, 2013)",
          "高明,《中国古文字学通论》(1996)"
    ],
    chapters: [
      "汉字的起源与发展 (裘锡圭 §1)",
      "汉字的结构理论：六书与三书说 (裘锡圭 §汉字基本类型的划分)",
      "甲骨文 (高明 §2)",
      "金文 (高明 §3)",
      "战国文字 (裘锡圭 §4)",
      "简帛文字 (裘锡圭 §5)",
      "古文字的考释方法 (裘锡圭 §8)",
      "秦汉文字与小篆 (裘锡圭 §6)",
      "隶变与汉字演变：隶楷阶段研究 (裘锡圭 §7 / 高明 §5)"
    ],
  },
  'humanities/historical-climate': {
    title: "历史气候与环境变迁",
    books: [
          "竺可桢,《中国近五千年来气候变迁的初步研究》(1973)",
          "Brooke, \"Climate Change and the Course of Global History\" (2014)"
    ],
    chapters: [
      "气候变迁研究理论与方法 (竺可桢 §1)",
      "物候学与历史气候重建 (竺可桢 §2)",
      "中国近五千年气候分期 (竺可桢 §3)",
      "气候变化与文明兴衰 (Brooke §4)",
      "灾害史与社会响应 (Brooke §8)",
      "小冰期与近代气候 (Brooke §12)",
      "树轮气候学与历史重建 (Brooke §3)",
      "代用资料的多元重建：冰芯、石笋、沉积物与孢粉 (Brooke §3 / 竺可桢 §2)",
      "历史环境变迁：植被带迁移、沙漠化与河流改道 (竺可桢 §附录 / Brooke §环境)",
      "气候危机与社会崩溃 (Brooke §14)"
    ],
  },
  'humanities/historical-population-geography': {
    title: "历史人口地理",
    books: [
          "葛剑雄,《中国人口史》(2001)",
          "葛剑雄、吴松弟、曹树基,《中国移民史》(1997)"
    ],
    chapters: [
      "人口规模的历史变迁 (葛剑雄 §1)",
      "人口分布与人口重心 (葛剑雄 §3)",
      "人口迁移与移民史 (葛剑雄 移民史 §1)",
      "户籍制度与人口统计 (葛剑雄 §2)",
      "人口结构与城镇化 (葛剑雄 §5)",
      "战乱与人口波动 (葛剑雄 移民史 §3)",
      "人口与资源环境承载力 (葛剑雄 §7)",
      "海外移民与华人华侨 (葛剑雄 移民史 §7)"
    ],
  },
  'humanities/frontier-historical-geography': {
    title: "边疆史地",
    books: [
          "Lattimore, \"Inner Asian Frontiers of China\" (1940)",
          "马大正,《中国边疆经略史》(2000)"
    ],
    chapters: [
      "长城地带与边疆理论 (Lattimore §3)",
      "蒙古高原与游牧社会 (Lattimore §5)",
      "西域与丝绸之路 (马大正 §4)",
      "青藏高原与西藏治理：和硕特、驻藏大臣与清末西藏 (马大正 §8)",
      "东北边疆与满洲 (Lattimore §7)",
      "西南边疆与改土归流 (马大正 §6)",
      "海疆与近代边疆危机 (马大正 §9)",
      "北疆与中俄边界 (马大正 §7)",
      "边疆民族政策与朝贡体系 (Lattimore §9)"
    ],
  },
  'humanities/historical-agricultural-geography': {
    title: "历史农业地理",
    books: [
          "Chi, \"Key Economic Areas in Chinese History\" (1936)",
          "李伯重,《多视角看江南经济史》(2003)"
    ],
    chapters: [
      "基本经济区与水利事业 (Chi §2)",
      "农业起源与早期作物 (Chi §1)",
      "传统农作制度：轮作、复种与间作套种 (韩茂莉《中国历史农业地理》§农作制度)",
      "土地制度与垦殖扩张 (韩茂莉《中国历史农业地理》§土地制度)",
      "稻作农业与江南开发 (李伯重 §3)",
      "区域农业类型：华北旱作与江南水田 (韩茂莉《中国历史农业地理》§区域类型)",
      "水利与农业地理格局 (Chi §4)",
      "农业技术变迁与人口压力 (李伯重 §5)",
      "农业商品化与市镇经济 (李伯重 §7)",
      "经济作物与商品农业 (李伯重 §6)",
      "灾荒与农业波动 (李伯重 §8)"
    ],
  },
  'humanities/historiography': {
    title: "史学理论与史学史",
    books: [
          "Jenkins, \"Re-thinking History\" (3rd, 2011)",
          "白寿彝,《中国史学史》(2006)"
    ],
    chapters: [
      "历史认识论与后现代挑战 (Jenkins §2)",
      "史料学与历史考证方法（考据学） (白寿彝 §史料学 / 梁启超《中国历史研究法》)",
      "历史编纂体例：编年、纪传与纪事本末 (白寿彝 §2-3)",
      "中国史学的发展历程 (白寿彝 §1)",
      "《史记》与传统史学体例 (白寿彝 §3)",
      "近代新史学的兴起 (白寿彝 §6)",
      "西方古典史学：希罗多德、修昔底德与兰克学派 (Jenkins §5 / 白寿彝 §西方史学)",
      "西方史学流派演变 (Jenkins §5)",
      "年鉴学派（Annales） (Jenkins §4)",
      "后现代史学（怀特/安克斯密特） (Jenkins §6)",
      "马克思主义史学 (Jenkins §3)"
    ],
  },
  'humanities/chinese-intellectual-history': {
    title: "中国思想史",
    books: [
          "葛兆光,《中国思想史》(1998)",
          "侯外庐,《中国思想通史》(2011)"
    ],
    chapters: [
      "思想史的写法与方法 (葛兆光 导论)",
      "先秦诸子百家 (侯外庐 §1)",
      "汉代经学与大一统 (葛兆光 §2)",
      "魏晋玄学与佛学东传 (侯外庐 §3)",
      "隋唐佛学鼎盛与三教论衡：韩愈、柳宗元 (葛兆光 卷二 / 侯外庐 §隋唐)",
      "宋明理学 (葛兆光 §4)",
      "明清之际与近代思想转型 (侯外庐 §5)",
      "清代考据学与实学 (侯外庐 §6)",
      "近现代思想转型 (葛兆光 §6)"
    ],
  },
  'humanities/western-intellectual-history': {
    title: "西方思想史",
    books: [
          "Skinner, \"The Foundations of Modern Political Thought\" (1978)",
          "Lovejoy, \"The Great Chain of Being\" (1936)"
    ],
    chapters: [
      "古希腊思想传统 (Lovejoy §2)",
      "中世纪基督教思想 (Lovejoy §3)",
      "文艺复兴人文主义 (Skinner §1)",
      "宗教改革与政治思想 (Skinner §2)",
      "启蒙运动与理性主义 (Skinner §3)",
      "现代性的思想起源 (Skinner §4)",
      "科学革命与近代世界观 (Lovejoy §5)",
      "19 世纪思想：浪漫主义、黑格尔、马克思主义、达尔文主义与自由主义 (Lovejoy §6-7)",
      "20世纪思想与现代性批判 (Skinner §6)"
    ],
  },
  'humanities/fine-arts': {
    title: "美术学",
    books: [
          "Kleiner, \"Gardner's Art Through the Ages\" (16th, 2020)",
          "Gombrich, \"The Story of Art\" (16th, 1995)"
    ],
    chapters: [
      "史前与古代艺术 (Kleiner §1-4)",
      "中世纪艺术 (Kleiner §9)",
      "文艺复兴艺术 (Kleiner §12)",
      "巴洛克与洛可可 (Gombrich §19)",
      "19世纪艺术:从浪漫主义到印象派 (Kleiner §24)",
      "现代与当代艺术 (Kleiner §27)",
      "中国美术史 (中央美院《中国美术简史》§古代)",
      "20 世纪中国美术：近现代与新中国美术 (中央美院《中国美术简史》§近现代)",
      "美术理论与美术批评方法（造型语言、风格分析与美术史方法论） (王宏建《艺术概论》§美术批评)",
      "摄影与新媒体艺术 (Kleiner §29)"
    ],
  },
  'humanities/sculpture': {
    title: "雕塑",
    books: [
          "Boardman, \"Greek Sculpture: The Classical Period\" (1985)",
          "孙振华,《中国雕塑史》(2014)"
    ],
    chapters: [
      "雕塑的语言与基本原理 (孙振华 §1 / 西方雕塑史 §导论)",
      "古希腊雕塑：古典时期 (Boardman §2)",
      "古罗马雕塑：写实与纪念性 (西方雕塑史 §古罗马)",
      "中世纪雕塑与教堂艺术 (西方雕塑史 §中世纪)",
      "文艺复兴雕塑：多纳泰罗与米开朗琪罗 (西方雕塑史 §文艺复兴)",
      "巴洛克雕塑（贝尼尼）与新古典主义 (西方雕塑史 §巴洛克)",
      "罗丹与现代雕塑：布朗库西与亨利·摩尔 (西方雕塑史 §现代)",
      "中国古代雕塑：陵墓与宗教 (孙振华 §3)",
      "现当代中国雕塑 (孙振华 §7)",
      "当代雕塑的媒介拓展 (孙振华 §9)",
      "雕塑材料与铸造工艺 (孙振华 §10)"
    ],
  },
  'humanities/animation-digital-media': {
    title: "动画与数字媒体艺术",
    books: [
          "Wells, \"Understanding Animation\" (1998)",
          "Manovich, \"The Language of New Media\" (2001)"
    ],
    chapters: [
      "动画的历史与发展 (Wells §2)",
      "动画类型学：二维、三维、定格、实验动画与日本动画 (Wells §4-5)",
      "动画原理与叙事 (Wells §3)",
      "数字媒体的本体论 (Manovich §1)",
      "界面与交互美学 (Manovich §2)",
      "数字叙事与数据库逻辑 (Manovich §5)",
      "游戏与互动艺术 (Manovich §4)",
      "声音与影像装置 (Manovich §7)",
      "动画与新媒介融合 (Wells §8)",
      "3D 动画与计算机图形学 (Wells §6)",
      "虚拟现实与沉浸式媒介 (Manovich §8)"
    ],
  },
  'humanities/art-theory': {
    title: "艺术理论",
    books: [
          "Carroll, \"Philosophy of Art: A Contemporary Introduction\" (2nd, 2012)",
          "Danto, \"What Art Is\" (2013)"
    ],
    chapters: [
      "艺术的定义问题 (Carroll §1)",
      "艺术本体论 (Carroll §3)",
      "审美经验与艺术价值 (Danto §4)",
      "形式主义：贝尔与格林伯格、现代主义艺术理论 (Bell《艺术》/ Greenberg《艺术与文化》)",
      "图像学与艺术史理论：潘诺夫斯基 (Panofsky《图像学研究》)",
      "西方美学史脉络：康德与黑格尔的审美论 (朱光潜《西方美学史》§康德/黑格尔)",
      "艺术批评与诠释 (Carroll §6)",
      "艺术、情感与表现 (Carroll §5)",
      "艺术与社会功能 (Danto §7)",
      "艺术体制理论与历史 (Danto §5)",
      "非西方艺术理论 (Carroll §10)"
    ],
  },
  'humanities/film-theory': {
    title: "电影理论",
    books: [
          "Bordwell & Thompson, \"Film Art: An Introduction\" (11th, 2016)",
          "Stam, \"Film Theory: An Introduction\" (2000)"
    ],
    chapters: [
      "电影形式与风格 (Bordwell §2-3)",
      "电影叙事学 (Bordwell §4)",
      "蒙太奇理论与形式主义 (Stam §2)",
      "作者论与电影作者 (Stam §4)",
      "类型片理论 (Bordwell §9)",
      "精神分析、女性主义与后殖民电影理论 (Stam §7-9)",
      "现实主义电影理论（巴赞/克拉考尔） (Bordwell §2)",
      "电影符号学（Metz） (Bordwell §5)",
      "认知电影理论 (Bordwell §7)"
    ],
  },
  'humanities/cultural-heritage-museology': {
    title: "文化遗产与博物馆学",
    books: [
          "Macdonald, \"A Companion to Museum Studies\" (2006)",
          "ICOMOS, \"International Charter for the Conservation and Restoration of Monuments and Sites\" (Venice Charter, 1964)"
    ],
    chapters: [
      "博物馆的历史与发展 (Macdonald §1)",
      "新博物馆学与批判博物馆学理论 (Macdonald §3-4)",
      "藏品管理与编目 (Macdonald §5)",
      "藏品修复与预防性保护技术 (ICOMOS 宪章 / 文物保护科学)",
      "博物馆教育与公共阐释 (Macdonald §11)",
      "展览策划与诠释 (Macdonald §8)",
      "物质文化遗产保护原则 (ICOMOS Venice Charter)",
      "世界遗产体系与 UNESCO 公约框架：〈世界遗产公约〉、名录与 OUV (UNESCO《世界遗产公约》§名录)",
      "社区参与与遗产伦理 (Macdonald §17)",
      "数字博物馆与虚拟展览 (Macdonald §20)",
      "遗产地管理与旅游 (Macdonald §15)"
    ],
  },
  'humanities/intangible-cultural-heritage': {
    title: "非物质文化遗产",
    books: [
          "UNESCO, \"Convention for the Safeguarding of the Intangible Cultural Heritage\" (2003)",
          "Kurin, \"Reflections on the 2003 Convention\" (2007)"
    ],
    chapters: [
      "非遗的概念与范畴 (UNESCO §2)",
      "名录体系与申报制度：人类代表作名录与急需保护名录 (UNESCO 2003 公约 §名录)",
      "传承人制度 (UNESCO §4 / Kurin §传承人)",
      "口头传统与表现形式 (UNESCO §2.1)",
      "表演艺术 (UNESCO §2.2)",
      "社会实践、仪式与节庆 (UNESCO §2.3)",
      "传统手工艺 (UNESCO §2.5)",
      "有关自然和宇宙的知识与实践 (UNESCO §2.4)",
      "非遗的保护、传承与伦理 (Kurin §4)",
      "非遗档案化与数字化保护 (Kurin §6)",
      "非遗与旅游、商业化及知识产权 (Kurin §8 / UNESCO §知识产权)"
    ],
  },
  'humanities/philosophy-overview': {
    title: "哲学通史",
    books: [
          "Russell, \"A History of Western Philosophy\" (1945)",
          "Copleston, \"A History of Philosophy\" (9 vols, 1946-1975)"
    ],
    chapters: [
      "前苏格拉底哲学与自然哲学 (Copleston 卷1 / Russell §古代卷1)",
      "古希腊哲学:苏格拉底、柏拉图、亚里士多德 (Russell §古代)",
      "中世纪哲学:教父与经院哲学 (Copleston 卷2)",
      "文艺复兴哲学 (Copleston 卷3)",
      "近代哲学:理性主义与经验主义 (Russell §近代)",
      "启蒙运动与德国古典哲学 (Copleston 卷6)",
      "19世纪哲学:黑格尔、马克思、尼采 (Russell §近代)",
      "20世纪哲学:分析哲学与大陆哲学 (Copleston 卷8-9)",
      "中国与印度哲学概览 (冯友兰《中国哲学史》/ Radhakrishnan《Indian Philosophy》)",
      "伊斯兰哲学黄金时代 (Copleston 卷2)"
    ],
  },
  'humanities/metaphysics': {
    title: "形而上学",
    books: [
          "Loux & Crisp, \"Metaphysics: A Contemporary Introduction\" (4th, 2017)",
          "Aristotle, \"Metaphysics\" (trans. Ross)"
    ],
    chapters: [
      "存在论与本体论 (Loux §1)",
      "共相与殊相、实体与属性 (Loux §2 / Aristotle 卷Z)",
      "抽象实体：数、命题与共相之外 (Loux §3)",
      "因果性与自然律 (Loux §7)",
      "必然性与可能性 (Loux §5)",
      "时间与持存的同一性 (Loux §4)",
      "人格同一性 (Loux §6)",
      "空间与时间哲学 (Loux §8)",
      "自由意志与决定论 (Loux §9)"
    ],
  },
  'humanities/epistemology': {
    title: "认识论（知识论）",
    books: [
          "Moser, \"The Theory of Knowledge: A Thematic Introduction\" (3rd, 2017)",
          "Audi, \"Epistemology\" (4th, 2019)"
    ],
    chapters: [
      "知识的定义与盖梯尔问题 (Audi §1)",
      "怀疑论与确定性 (Moser §4)",
      "知觉与经验知识 (Audi §3)",
      "证言与社会认识论 (Audi §7)",
      "先验知识与理性主义 (Moser §6)",
      "自然化认识论 (Audi §9)",
      "证立理论（基础主义/融贯论/可靠主义） (Audi §4-5)",
      "归纳问题（Hume/Goodman） (Audi §6)",
      "信念与真理理论 (Audi §7)"
    ],
  },
  'humanities/ethics': {
    title: "伦理学",
    books: [
          "Frankena, \"Ethics\" (2nd, 1973)",
          "Cahn, \"Exploring Ethics: An Introductory Anthology\" (5th, 2019)"
    ],
    chapters: [
      "元伦理学与规范伦理学 (Frankena §1)",
      "后果主义:功利主义 (Frankena §4)",
      "义务论:康德伦理学 (Cahn §Kant)",
      "德性伦理学:亚里士多德传统 (Cahn §Aristotle)",
      "道德相对主义与道德实在论 (Frankena §6)",
      "应用伦理问题 (Cahn §Applied)",
      "社会契约论传统 (Cahn §Hobbes)",
      "道德心理学与道德情感 (Frankena §5)"
    ],
  },
  'humanities/aesthetics': {
    title: "美学",
    books: [
          "Beardsley, \"Aesthetics: Problems in the Philosophy of Criticism\" (2nd, 1981)",
          "Carroll, \"Philosophy of Art: A Contemporary Introduction\" (2nd, 2012)"
    ],
    chapters: [
      "美的概念与审美经验 (Beardsley §1)",
      "西方美学史主线：柏拉图—康德《判断力批判》—黑格尔 (朱光潜《西方美学史》§柏拉图/§康德/§黑格尔)",
      "康德《判断力批判》与审美判断力 (康德《判断力批判》§审美判断)",
      "叔本华、尼采与意志美学 (朱光潜《西方美学史》§叔本华/§尼采)",
      "艺术的定义 (Carroll §1)",
      "艺术本体论与艺术作品 (Carroll §3)",
      "艺术批评与诠释 (Beardsley §9)",
      "悲剧理论与崇高 (Beardsley §7)",
      "艺术与道德 (Carroll §7)",
      "20 世纪分析美学与现象学美学：杜夫海纳 (Beardsley §9 / Dufrenne《审美经验现象学》)",
      "环境美学 (Beardsley §环境 / 环境美学文献)",
      "后现代美学 (Carroll §后现代 / 后现代美学文献)"
    ],
  },
  'humanities/chinese-philosophy': {
    title: "中国哲学",
    books: [
          "冯友兰,《中国哲学史》(1934)",
          "Fung Yu-lan, \"A History of Chinese Philosophy\" (trans. Bodde, 1952-1953)"
    ],
    chapters: [
      "先秦儒家:孔孟荀 (冯友兰 §2)",
      "老庄道家哲学 (冯友兰 §3)",
      "墨家与名家 (冯友兰 §4)",
      "汉代哲学与经学 (冯友兰 §5)",
      "魏晋玄学：王弼、郭象与竹林玄学 (冯友兰 §6)",
      "中国佛学 (冯友兰 §7)",
      "宋明理学:程朱陆王 (冯友兰 §8)",
      "清代哲学：王夫之与戴震 (冯友兰 §清代)",
      "法家思想（韩非子） (冯友兰 §法家)",
      "道教哲学（道家宗教化） (冯友兰 §道教)",
      "现代新儒家（牟宗三/唐君毅） (冯友兰 §近现代)"
    ],
  },
  'humanities/western-philosophy': {
    title: "西方哲学史",
    books: [
          "Russell, \"A History of Western Philosophy\" (1945)",
          "Copleston, \"A History of Philosophy\" (9 vols, 1946-1975)"
    ],
    chapters: [
      "前苏格拉底与苏格拉底 (Russell §古代卷1)",
      "柏拉图与亚里士多德 (Russell §古代卷2)",
      "奥古斯丁与阿奎那 (Copleston 卷2)",
      "中世纪晚期：司各脱与奥卡姆 (Copleston 卷3)",
      "文艺复兴哲学与法国启蒙：卢梭与伏尔泰 (Copleston 卷3-4 / Russell §近代)",
      "笛卡尔与大陆理性主义 (Russell §近代卷1)",
      "洛克、休谟与英国经验主义 (Russell §近代卷2)",
      "康德与德国唯心论 (Copleston 卷6)",
      "19世纪：叔本华、尼采与克尔凯郭尔 (Russell §近代卷3)",
      "20 世纪：实用主义与分析哲学（另见相关专题） (Copleston 卷8-9)",
      "现象学与存在主义 (Copleston 卷8)"
    ],
  },
  'humanities/indian-philosophy': {
    title: "印度哲学",
    books: [
          "Radhakrishnan, \"Indian Philosophy\" (2 vols, 1923-1927)",
          "Potter, \"Encyclopedia of Indian Philosophies\" (multiple vols)"
    ],
    chapters: [
      "吠陀文献与思想起源 (Radhakrishnan 卷1 §1)",
      "奥义书的哲学 (Radhakrishnan 卷1 §3)",
      "正统六派哲学 (Radhakrishnan 卷2 §2)",
      "佛教哲学 (Radhakrishnan 卷1 §6)",
      "耆那教哲学 (Radhakrishnan 卷1 §7)",
      "印度逻辑学与认识论 (Potter 卷2)",
      "顺世论与印度唯物主义 (Radhakrishnan 卷1 §8)",
      "当代印度哲学（甘地/奥罗宾多） (Radhakrishnan 卷2 §附录)"
    ],
  },
  'humanities/islamic-philosophy': {
    title: "伊斯兰哲学",
    books: [
          "Fakhry, \"A History of Islamic Philosophy\" (3rd, 2004)",
          "Nasr, \"Three Muslim Sages\" (1964)"
    ],
    chapters: [
      "伊斯兰神学与凯拉姆 (Fakhry §2)",
      "法拉比的哲学 (Nasr §1)",
      "伊本·西那的哲学 (Nasr §2)",
      "安萨里与哲学—信仰之调和：《哲学家的矛盾》 (Fakhry §7)",
      "伊本·鲁世德与亚里士多德主义 (Fakhry §6)",
      "照明学派与苏菲主义 (Nasr §3)",
      "穆拉·萨德拉与波斯超越哲学 (Fakhry §9)",
      "伊斯兰哲学的现代发展 (Fakhry §11)",
      "伊斯玛仪派哲学 (Fakhry §5)",
      "伊斯兰伦理与政治哲学 (Nasr §4)"
    ],
  },
  'humanities/philosophy-of-mind': {
    title: "心灵哲学",
    books: [
          "Kim, \"Philosophy of Mind\" (3rd, 2011)",
          "Chalmers, \"Philosophy of Mind: Classical and Contemporary Readings\" (2002)"
    ],
    chapters: [
      "心身问题:二元论与物理主义 (Kim §2-3)",
      "行为主义与心脑同一论 (Kim §4)",
      "功能主义 (Kim §5)",
      "意识与感受质 (Chalmers §Consciousness)",
      "心理因果性与排除论证 (Kim §7)",
      "意向性与心理内容 (Kim §8)",
      "自由意志与决定论 (Chalmers §6)",
      "AI 与心灵（图灵测试/中文房间） (Chalmers §8)",
      "意识的困难问题（Chalmers §1-2）",
      "他心问题 (Chalmers §7)"
    ],
  },
  'humanities/comparative-philosophy': {
    title: "比较哲学",
    books: [
          "Larson & Deutsch, \"Interpreting Across Boundaries: New Essays in Comparative Philosophy\" (1988)",
          "Garfield, \"Empty Words: Buddhist Philosophy and Cross-Cultural Interpretation\" (2002)"
    ],
    chapters: [
      "比较哲学的方法论 (Larson §1)",
      "中西哲学比较 (Larson §3)",
      "印度思想与西方哲学 (Larson §5)",
      "佛教哲学的跨文化诠释 (Garfield §2)",
      "概念的不可通约性与翻译 (Garfield §5)",
      "普遍主义与文化相对主义 (Larson §8)",
      "跨文化诠释学 (Garfield §8)",
      "比较伦理学与跨文化价值 (Larson §6)"
    ],
  },
  'humanities/philosophy-of-mathematics': {
    title: "数学哲学",
    books: [
          "Shapiro, \"Thinking About Mathematics: The Philosophy of Mathematics\" (2000)",
          "Benacerraf & Putnam, \"Philosophy of Mathematics: Selected Readings\" (2nd, 1983)"
    ],
    chapters: [
      "数学基础与逻辑主义 (Shapiro §4)",
      "柏拉图主义与数学对象 (Shapiro §5)",
      "形式主义 (Shapiro §6)",
      "直觉主义与构造主义 (Benacerraf §Brouwer)",
      "数学真理与认识论 (Benacerraf §Benacerraf)",
      "数学应用与不可或缺性论证 (Shapiro §9)",
      "数学结构主义 (Shapiro §7)",
      "数学实践与数学哲学 (Shapiro §10)"
    ],
  },
  'humanities/philosophy-of-physics': {
    title: "物理哲学",
    books: [
          "Sklar, \"Philosophy of Physics\" (1992)",
          "Albert, \"Time and Chance\" (2000)"
    ],
    chapters: [
      "时空哲学:绝对论与关系论 (Sklar §2)",
      "相对论时空的哲学问题：狭义/广义相对论与同时性 (Sklar §1)",
      "量子力学的解释 (Sklar §5)",
      "决定论与混沌 (Sklar §4)",
      "统计力学与时间箭头 (Albert §2)",
      "概率在物理学中的角色 (Sklar §3 / Albert §2)",
      "量子非定域性与纠缠 (Albert §4)",
      "物理定律与自然类 (Sklar §7)",
      "热力学第二定律与时间 (Albert §3)",
      "量子测量与诠释比较 (Sklar §6)"
    ],
  },
  'humanities/philosophy-of-technology': {
    title: "技术哲学",
    books: [
          "Mitcham, \"Thinking Through Technology: The Path Between Engineering and Philosophy\" (1994)",
          "Ihde, \"Philosophy of Technology: An Introduction\" (1993)"
    ],
    chapters: [
      "技术的本质与定义 (Mitcham §1)",
      "工程传统与人文传统 (Mitcham §3)",
      "技术作为认识方式 (Ihde §4)",
      "技术伦理与责任 (Mitcham §7)",
      "技术与社会的共同塑造 (Ihde §6)",
      "人工智能与未来技术 (Mitcham §10)",
      "海德格尔技术哲学（\"关于技术的追问\"） (Ihde §3)",
      "法兰克福学派技术批判理论 (Ihde §5)",
      "技术决定论与社会建构论（SCOT） (Ihde §6)"
    ],
  },
  'humanities/philosophy-of-law': {
    title: "法哲学",
    books: [
          "Hart, \"The Concept of Law\" (3rd, 2012)",
          "Dworkin, \"Law's Empire\" (1986)"
    ],
    chapters: [
      "法律的概念与规则 (Hart §1-5)",
      "自然法传统 (Hart §9)",
      "法律实证主义 (Hart §7)",
      "法律作为整全性 (Dworkin §6)",
      "权利与原则 (Dworkin §4)",
      "法治与自由裁量 (Dworkin §5)",
      "法律与道德的关系 (Hart §8)",
      "法律推理与判决 (Dworkin §1)"
    ],
  },
  'humanities/philosophy-of-education': {
    title: "教育哲学",
    books: [
          "Noddings, \"Philosophy of Education\" (4th, 2018)",
          "Siegel, \"The Oxford Handbook of Philosophy of Education\" (2009)"
    ],
    chapters: [
      "教育的目的与价值 (Noddings §2)",
      "教育与民主 (Noddings §5)",
      "课程哲学 (Siegel §课程)",
      "教育公平与社会正义 (Noddings §8)",
      "批判教育学与女性主义教育 (Noddings §7)",
      "道德教育与品格培养 (Siegel §道德)",
      "杜威与实用主义教育哲学 (Noddings §4)",
      "教育认识论 (Noddings §6)"
    ],
  },
  'humanities/philosophy-of-history': {
    title: "历史哲学",
    books: [
          "Dray, \"Philosophy of History\" (2nd, 1993)",
          "Collingwood, \"The Idea of History\" (1946)"
    ],
    chapters: [
      "思辨的历史哲学 (Dray §1)",
      "历史解释与覆盖律模型 (Dray §3)",
      "历史因果与历史规律 (Dray §5)",
      "历史客观性与诠释学 (Collingwood §Epilegomena)",
      "历史叙述与叙事 (Dray §7)",
      "历史的意义与目的 (Collingwood §反思)",
      "马克思主义历史哲学 (Dray §6)",
      "后现代历史哲学（怀特） (Dray §9)"
    ],
  },
  'humanities/applied-ethics': {
    title: "应用伦理学（科技/生命/环境）",
    books: [
          "Beauchamp & Childress, \"Principles of Biomedical Ethics\" (8th, 2019)",
          "Singer, \"Practical Ethics\" (3rd, 2011)"
    ],
    chapters: [
      "生命伦理四原则（与医学伦理学专题分工） (Beauchamp §1)",
      "生命起始、生殖与基因伦理 (Beauchamp §6)",
      "基因编辑与生物技术伦理 (Singer §7 / Beauchamp §9)",
      "死亡与临终决策（详见医学伦理学专题） (Beauchamp §7)",
      "全球正义与贫困 (Singer §8)",
      "战争与武器伦理 (Singer §12)",
      "环境伦理与动物权利 (Singer §3-5)",
      "科技伦理与责任 (Singer §10)",
      "人工智能伦理 (Singer 附录)",
      "商业与企业伦理 (Singer §6)",
      "工程与计算机伦理 (Singer §11)"
    ],
  },
  'humanities/meta-ethics': {
    title: "元伦理学",
    books: [
          "Shafer-Landau, \"The Fundamentals of Ethics\" (4th, 2018)",
          "Smith, \"The Moral Problem\" (1994)"
    ],
    chapters: [
      "道德实在论与反实在论 (Shafer-Landau §14)",
      "道德相对主义 (Shafer-Landau §15)",
      "道德判断的表达主义 (Smith §3)",
      "道德动机与内在主义 (Smith §5)",
      "道德知识与直觉主义 (Shafer-Landau §17)",
      "规范性与道德理由 (Smith §6)",
      "道德语言与语义学 (Shafer-Landau §16)",
      "道德心理学与道德判断 (Smith §7)"
    ],
  },
  'humanities/phenomenology-existentialism': {
    title: "现象学与存在主义",
    books: [
          "Husserl, \"Ideas: General Introduction to Pure Phenomenology\" (1913)",
          "Heidegger, \"Being and Time\" (1927)",
          "Sartre, \"Being and Nothingness\" (1943)"
    ],
    chapters: [
      "现象学方法与悬置 (Husserl §1)",
      "意向性与本质直观 (Husserl §3)",
      "时间意识与内时间意识 (Husserl §5)",
      "此在与在世存在 (Heidegger §1-2)",
      "本真性与向死存在 (Heidegger §6)",
      "海德格尔后期思想：语言、艺术与真理 (Heidegger《林中路》/《艺术作品的起源》)",
      "自在与自为 (Sartre §1)",
      "他者与凝视 (Sartre §3)",
      "身体现象学：梅洛-庞蒂《知觉现象学》 (Merleau-Ponty《知觉现象学》§身体)",
      "伽达默尔哲学诠释学 (Gadamer《真理与方法》§诠释学循环)"
    ],
  },
  'humanities/postmodern-philosophy': {
    title: "后现代哲学",
    books: [
          "Lyotard, \"The Postmodern Condition\" (1979)",
          "Foucault, \"The Archaeology of Knowledge\" (1969)"
    ],
    chapters: [
      "后现代状况与元叙事批判 (Lyotard §1-2)",
      "解构主义与德里达：《写作与差异》《论文字学》 (Derrida《写作与差异》)",
      "差异与延异 (Derrida《论文字学》§延异)",
      "权力与知识 (Foucault §4 /《规训与惩罚》)",
      "话语分析与陈述形成 (Foucault §2)",
      "系谱学与权力谱系 (Foucault §附录)",
      "福柯《规训与惩罚》与《性史》：权力—身体分析 (Foucault《规训与惩罚》/《性史》)",
      "德勒兹差异哲学 (Deleuze《差异与重复》)",
      "鲍德里亚拟像理论 (Baudrillard《拟像与仿真》)",
      "哈贝马斯现代性批判 (Habermas《现代性的哲学话语》)",
      "罗蒂与新实用主义 (Rorty《哲学与自然之镜》)"
    ],
  },
  'humanities/analytic-continental-philosophy': {
    title: "分析哲学与大陆哲学",
    books: [
          "Soames, \"Philosophical Analysis in the Twentieth Century\" (2 vols, 2003)",
          "Critchley & Schroeder, \"A Companion to Continental Philosophy\" (1998)"
    ],
    chapters: [
      "分析哲学的起源：弗雷格、罗素与摩尔 (Soames 卷1 §1)",
      "逻辑实证主义与语言转向 (Soames 卷1 §4)",
      "维特根斯坦前后期哲学 (Soames 卷1 §5-6)",
      "日常语言哲学 (Soames 卷2 §3)",
      "美国实用主义：皮尔士、詹姆斯与杜威 (Soames 卷1 §实用主义 / James《实用主义》)",
      "当代分析哲学：奎因、克里普克与戴维森 (Soames 卷2 §奎因/§克里普克)",
      "现象学传统 (Critchley §2)",
      "存在主义与解释学 (Critchley §3)",
      "后结构主义（Derrida/Deleuze） (Critchley §后结构主义)",
      "两大传统的对话与融合 (Critchley §导论)"
    ],
  },
  'humanities/philosophy-of-action': {
    title: "行动哲学",
    books: [
          "Anscombe, \"Intention\" (2nd, 1963)",
          "Mele, \"Philosophy of Action: An Anthology\" (2019)"
    ],
    chapters: [
      "行动与行为的区分 (Anscombe §1)",
      "意向性行动 (Anscombe §5)",
      "行动的理由与动机 (Mele §Reasons)",
      "行动的因果关系理论 (Mele §Causation)",
      "实践推理与意志薄弱 (Mele §Akrasia)",
      "自由行动与代理人因果 (Mele §Free Will)",
      "集体行动与社会行动 (Mele §Collective)",
      "行动知识与道义学 (Anscombe §7)"
    ],
  },
  'humanities/normative-ethics': {
    title: "规范伦理学（功利主义/义务论/德性伦理）",
    books: [
          "James Rachels & Stuart Rachels, The Elements of Moral Philosophy (9th ed., 2018)",
          "王海明《伦理学原理》（第3版，北京大学出版社，2009）",
          "Aristotle, Nicomachean Ethics (T. Irwin trans., 2nd ed., 1999)"
    ],
    chapters: [
      "规范伦理学的定位：元伦理、规范伦理与应用伦理的三分",
      "后果主义总论：理论结构、吸引力与困难",
      "古典功利主义：边沁的量化计算与密尔的修正",
      "行为功利主义与规则功利主义",
      "义务论总论与罗斯的显见义务论",
      "康德的道德形而上学：绝对命令的三种表述",
      "契约论传统：从霍布斯、洛克到罗尔斯与斯坎伦",
      "德性伦理学：亚里士多德的中道、实践智慧与幸福",
      "当代德性伦理的复兴：安斯库姆、福特与麦金太尔",
      "关怀伦理学：吉利根与诺丁斯",
      "道德动机之争：利己主义、利他主义与心理机制",
      "规范理论的比较、冲突与综合应用"
    ],
  },
  'humanities/philosophical-logic': {
    title: "哲学逻辑（模态/道义/时态逻辑）",
    books: [
          "Ted Sider, Logic for Philosophy (Oxford University Press, 2010)",
          "Graham Priest, An Introduction to Non-Classical Logic (2nd ed., 2008)",
          "周北海《模态逻辑导论》（北京大学出版社，1997）"
    ],
    chapters: [
      "经典逻辑回顾与哲学逻辑的边界",
      "模态逻辑的语法与可能世界语义",
      "正规模态系统：K、T、S4 与 S5",
      "必然性、本质主义与克里普克语义学",
      "道义逻辑：义务、允许及其悖论",
      "时态逻辑：时间算子与时间结构模型",
      "认知逻辑：知识与信念的公理化",
      "条件句逻辑与反事实条件句",
      "直觉主义逻辑与构造性证明",
      "多值逻辑与模糊逻辑",
      "哲学逻辑在形而上学、语言学与计算机科学中的应用"
    ],
  },
  'humanities/experimental-philosophy': {
    title: "实验哲学",
    books: [
          "Joshua Knobe & Shaun Nichols (eds.), Experimental Philosophy (Oxford, 2008)",
          "Joshua Knobe & Shaun Nichols (eds.), Experimental Philosophy, Vol. 2 (Oxford, 2013)",
          "Kwame Anthony Appiah, Experiments in Ethics (Harvard, 2008)"
    ],
    chapters: [
      "实验哲学的兴起：哲学自然化与方法论转向",
      "直觉在哲学论证中的地位与来源",
      "方法论基础：问卷设计、思想实验场景与统计推断",
      "知识归因的实验研究：Gettier 案例的跨文化检验",
      "Knobe 效应：意向性判断的副作用不对称",
      "自由意志与道德责任的民间概念研究",
      "道德判断的实验研究：电车难题与双效原则",
      "因果判断与语言指称的实验研究",
      "对实验哲学的批评与回应：限制主义与专家直觉辩护",
      "实验伦理学、实验美学与实验哲学的学科前景"
    ],
  },
  'intermediate/representation-theory': {
    title: "群表示论",
    books: [
          "Jean-Pierre Serre, Linear Representations of Finite Groups (Springer, 1977)",
          "William Fulton & Joe Harris, Representation Theory: A First Course (Springer, 1991)",
          "丘维声《群表示论》（高等教育出版社，2011）"
    ],
    chapters: [
      "表示的定义、例子与基本构造",
      "完全可约性与 Maschke 定理",
      "特征标理论",
      "特征标的第一、第二正交关系",
      "群代数与正则表示的分解",
      "不可约表示的个数与类函数空间",
      "诱导表示与 Frobenius 互反律",
      "对称群的表示与杨图",
      "紧群表示与 Peter–Weyl 定理",
      "SU(2) 与 SO(3) 的表示和角动量",
      "表示论在物理、化学与数论中的应用"
    ],
  },
  'intermediate/differential-topology': {
    title: "微分拓扑",
    books: [
          "John W. Milnor, Topology from the Differentiable Viewpoint (Princeton, 1997)",
          "Victor Guillemin & Alan Pollack, Differential Topology (1974)",
          "Morris W. Hirsch, Differential Topology (Springer, 1976)"
    ],
    chapters: [
      "光滑流形与光滑映射",
      "切空间、切丛与映射的微分",
      "正则值与原像定理",
      "Sard 定理与临界值的测度",
      "浸入、浸没与子流形",
      "横截性理论",
      "Whitney 嵌入定理",
      "向量场、流与管状邻域",
      "映射度与 Brouwer 不动点定理",
      "Poincaré–Hopf 指标定理",
      "Morse 函数与配边理论初步"
    ],
  },
  'intermediate/approximation-theory': {
    title: "函数逼近论",
    books: [
          "E. Ward Cheney, Introduction to Approximation Theory (2nd ed., 1982)",
          "Lloyd N. Trefethen, Approximation Theory and Approximation Practice (SIAM, 2013)",
          "徐利治、王仁宏《函数逼近的理论与方法》（上海科学技术出版社，1983）"
    ],
    chapters: [
      "逼近论的基本问题与赋范空间框架",
      "Weierstrass 逼近定理及其证明",
      "最佳一致逼近的存在性与唯一性",
      "Chebyshev 交错定理与最小零偏差多项式",
      "插值方法：Lagrange、Newton 与 Hermite 插值",
      "正交多项式与最小二乘逼近",
      "Fourier 逼近与 Gibbs 现象",
      "样条函数与分段多项式逼近",
      "有理逼近与 Padé 逼近",
      "逼近论在数值分析与科学计算中的应用"
    ],
  },
  'intermediate/asymptotic-perturbation': {
    title: "渐近分析与摄动方法",
    books: [
          "Carl M. Bender & Steven A. Orszag, Advanced Mathematical Methods for Scientists and Engineers (1978)",
          "E. John Hinch, Perturbation Methods (Cambridge, 1991)",
          "Ali Hasan Nayfeh, Perturbation Methods (Wiley, 1973)"
    ],
    chapters: [
      "渐近级数：定义、运算与最优截断",
      "量级符号、规范函数与渐近序列",
      "积分渐近：分部积分法与 Watson 引理",
      "Laplace 方法与鞍点（最速下降）法",
      "驻相法与振荡积分",
      "正则摄动：代数方程与微分方程",
      "奇异摄动与边界层理论",
      "匹配渐近展开与合成展开",
      "多重尺度方法与平均法",
      "WKB 近似及其在量子力学中的应用"
    ],
  },
  'intermediate/monte-carlo-methods': {
    title: "蒙特卡罗方法与随机模拟",
    books: [
          "Christian P. Robert & George Casella, Monte Carlo Statistical Methods (2nd ed., Springer, 2004)",
          "Dirk P. Kroese et al., Handbook of Monte Carlo Methods (Wiley, 2011)",
          "George S. Fishman, Monte Carlo: Concepts, Algorithms, and Applications (Springer, 1996)"
    ],
    chapters: [
      "蒙特卡罗方法的思想与大数定律、中心极限定理基础",
      "伪随机数生成器与统计检验",
      "逆变换法、舍选抽样法与常见分布抽样",
      "方差缩减：对偶变量、控制变量与分层抽样",
      "重要性抽样与自归一化估计",
      "马尔可夫链蒙特卡罗（MCMC）的基本原理",
      "Metropolis–Hastings 算法",
      "Gibbs 抽样与切片采样",
      "MCMC 的收敛诊断与蒙特卡罗误差估计",
      "序贯蒙特卡罗与粒子滤波",
      "拟蒙特卡罗方法与物理、金融中的应用"
    ],
  },
  'intermediate/multivariate-statistics': {
    title: "多元统计分析",
    books: [
          "Theodore W. Anderson, An Introduction to Multivariate Statistical Analysis (3rd ed., Wiley, 2003)",
          "Richard A. Johnson & Dean W. Wichern, Applied Multivariate Statistical Analysis (6th ed., 2007)",
          "张尧庭、方开泰《多元统计分析引论》（科学出版社，1982）"
    ],
    chapters: [
      "多元数据的矩阵表示与样本几何",
      "多元正态分布及其性质",
      "Wishart 分布与 Hotelling T² 检验",
      "多元线性回归与多元方差分析（MANOVA）",
      "主成分分析（PCA）",
      "因子分析",
      "判别分析：Fisher 判别与贝叶斯判别",
      "聚类分析：系统聚类与 K 均值",
      "典型相关分析",
      "对应分析与多维标度",
      "高维与稀疏情形下的多元统计方法"
    ],
  },
  'intermediate/symbolic-computation': {
    title: "符号计算与计算机代数",
    books: [
          "Keith O. Geddes, Stephen R. Czapor & George Labahn, Algorithms for Computer Algebra (1992)",
          "Joachim von zur Gathen & Jürgen Gerhard, Modern Computer Algebra (3rd ed., 2013)",
          "David A. Cox, John Little & Donal O'Shea, Ideals, Varieties, and Algorithms (4th ed., 2015)"
    ],
    chapters: [
      "符号计算的对象、表示与代数系统概览",
      "多项式算术：稠密与稀疏表示、快速乘法",
      "多项式的欧几里得算法与最大公因子",
      "子结式与结式",
      "多项式因式分解：有限域与整数环上的算法",
      "Gröbner 基与 Buchberger 算法",
      "符号积分与 Risch 算法",
      "代数方程组求解与三角化方法",
      "符号微分方程求解初步",
      "计算机代数系统与自动定理证明应用"
    ],
  },
  'intermediate/wavelet-analysis': {
    title: "小波分析",
    books: [
          "Ingrid Daubechies, Ten Lectures on Wavelets (SIAM, 1992)",
          "Stéphane Mallat, A Wavelet Tour of Signal Processing (3rd ed., 2009)",
          "程正兴《小波分析算法与应用》（西安交通大学出版社，1998）"
    ],
    chapters: [
      "从 Fourier 分析到时频分析：窗口 Fourier 变换",
      "连续小波变换与容许条件",
      "框架理论与离散小波",
      "多分辨分析（MRA）",
      "正交小波基的构造：Haar 小波与 Daubechies 小波",
      "Mallat 快速分解与重构算法",
      "双正交小波与提升格式",
      "小波包与最优基选择",
      "小波阈值去噪与统计估计",
      "应用：图像压缩（JPEG 2000）、特征提取与机器学习"
    ],
  },
  'intermediate/optimal-transport': {
    title: "最优传输理论",
    books: [
          "Cédric Villani, Optimal Transport: Old and New (Springer, 2009)",
          "Filippo Santambrogio, Optimal Transport for Applied Mathematicians (Birkhäuser, 2015)",
          "Gabriel Peyré & Marco Cuturi, Computational Optimal Transport (2019)"
    ],
    chapters: [
      "Monge 问题：起源、表述与困难",
      "Kantorovich 松弛与传输耦合",
      "对偶理论与最优性条件",
      "Wasserstein 距离及其诱导的拓扑",
      "Brenier 定理与凸函数梯度映射",
      "Wasserstein 梯度流与 JKO 格式",
      "熵正则化与 Sinkhorn 算法",
      "计算最优传输的其他方法：切片 Wasserstein 与网络算法",
      "应用：图像配准、色彩迁移与 Wasserstein GAN",
      "Gromov–Wasserstein 距离与非配对比较问题"
    ],
  },
  'intermediate/group-theory-in-physics': {
    title: "物理学中的群论",
    books: [
          "Wu-Ki Tung, Group Theory in Physics (World Scientific, 1985)",
          "Howard Georgi, Lie Algebras in Particle Physics (2nd ed., 1999)",
          "韩其智、孙洪洲《群论》（北京大学出版社，1987）"
    ],
    chapters: [
      "对称性与群论基础",
      "有限群表示论要点",
      "点群与晶体的宏观对称性",
      "空间群与固体中的对称操作",
      "转动群 SO(3) 与角动量理论",
      "SU(2)：自旋与同位旋",
      "SU(3) 与强子分类（八重法）",
      "Lorentz 群与 Poincaré 群",
      "对称性、简并与守恒律",
      "群论在分子光谱、固体与粒子物理中的应用"
    ],
  },
  'advanced/soft-matter-physics': {
    title: "软物质物理",
    books: [
          "Masao Doi, Soft Matter Physics (Oxford, 2013)",
          "Richard A. L. Jones, Soft Condensed Matter (Oxford, 2002)",
          "Paul M. Chaikin & Tom C. Lubensky, Principles of Condensed Matter Physics (Cambridge, 1995)"
    ],
    chapters: [
      "软物质的概念、特征与特征尺度",
      "聚合物物理：链构象与熵弹性",
      "聚合物溶液与熔体的统计理论",
      "胶体与分散体系的稳定性（DLVO 理论）",
      "液晶物理：向列相、弹性与取向序",
      "表面活性剂、双层膜与自组装",
      "玻璃化转变与阻塞（jamming）转变",
      "软物质流变学",
      "活性物质与生物软物质",
      "软物质在食品、日化与医药中的应用"
    ],
  },
  'advanced/phase-transitions-critical-phenomena': {
    title: "相变与临界现象",
    books: [
          "Nigel Goldenfeld, Lectures on Phase Transitions and the Renormalization Group (1992)",
          "John Cardy, Scaling and Renormalization in Statistical Physics (Cambridge, 1996)",
          "于渌、郝柏林、陈晓松《边缘奇迹：相变和临界现象》（科学出版社，2005）"
    ],
    chapters: [
      "相变的分类与序参量",
      "临界现象与临界指数的实验事实",
      "Landau 平均场理论",
      "涨落与 Ginzburg 判据",
      "Ising 模型：一维精确解与二维 Onsager 解",
      "标度假设与 Widom 标度律",
      "重整化群的物理图像：Kadanoff 与 Wilson",
      "动量空间重整化群与 ε 展开",
      "普适性与普适类",
      "动力学临界现象与有限尺寸标度"
    ],
  },
  'advanced/quantum-metrology-sensing': {
    title: "量子精密测量与量子传感",
    books: [
          "C. L. Degen, F. Reinhard & P. Cappellaro, Quantum sensing, Reviews of Modern Physics 89, 035002 (2017)",
          "V. Giovannetti, S. Lloyd & L. Maccone, Advances in quantum metrology, Nature Photonics 5, 222 (2011)",
          "Howard M. Wiseman & Gerard J. Milburn, Quantum Measurement and Control (Cambridge, 2010)"
    ],
    chapters: [
      "量子测量的基本概念与投影测量",
      "标准量子极限与散粒噪声",
      "海森堡极限与量子 Cramér–Rao 界",
      "压缩态及其在干涉仪中的应用",
      "原子钟与光钟：频率计量的量子飞跃",
      "NV 色心与固态量子传感",
      "量子磁力计、重力仪与惯性传感",
      "量子成像与量子照明",
      "引力波探测中的量子噪声抑制",
      "量子传感网络与工程化展望"
    ],
  },
  'intermediate/electrochemistry-principles': {
    title: "电化学原理（电极过程动力学）",
    books: [
          "Allen J. Bard & Larry R. Faulkner, Electrochemical Methods: Fundamentals and Applications (2nd ed., 2001)",
          "John O'M. Bockris & Amulya K. N. Reddy, Modern Electrochemistry (2nd ed., 1998)",
          "李荻《电化学原理》（第3版，北京航空航天大学出版社，2008）"
    ],
    chapters: [
      "电解质溶液：活度、离子迁移与电导",
      "电化学体系与可逆电池",
      "电极电势与 Nernst 方程",
      "双电层结构模型",
      "电极过程动力学基础：Butler–Volmer 方程",
      "Tafel 关系与电荷转移控制步骤",
      "传质过程：扩散、对流与电迁移",
      "电化学测量方法：循环伏安与交流阻抗",
      "腐蚀电化学与电化学防护",
      "应用：电解、电镀与化学电源"
    ],
  },
  'intermediate/animal-behavior': {
    title: "动物行为学",
    books: [
          "John Alcock, Animal Behavior: An Evolutionary Approach (10th ed., 2013)",
          "尚玉昌《动物行为学》（第2版，北京大学出版社，2014）",
          "N. B. Davies, J. R. Krebs & S. A. West, An Introduction to Behavioural Ecology (4th ed., 2012)"
    ],
    chapters: [
      "行为学的奠基：Lorenz、Tinbergen 与 von Frisch",
      "Tinbergen 四问：机制、发育、功能与演化",
      "本能、固定动作型与关键刺激",
      "学习行为：习惯化、条件反射与认知能力",
      "印记与敏感期",
      "动物通讯：信号、诚实性与信息",
      "觅食行为与最优觅食理论",
      "生殖行为、交配系统与性选择",
      "亲代抚育、亲属选择与汉密尔顿法则",
      "社群行为、利他与合作的演化",
      "行为的遗传基础与神经生理机制"
    ],
  },
  'intermediate/molecular-evolution-phylogenetics': {
    title: "分子进化与系统发育学",
    books: [
          "Roderic D. M. Page & Edward C. Holmes, Molecular Evolution: A Phylogenetic Approach (1998)",
          "Joseph Felsenstein, Inferring Phylogenies (Sinauer, 2004)",
          "Ziheng Yang, Molecular Evolution: A Statistical Approach (Oxford, 2014)"
    ],
    chapters: [
      "分子进化的中性理论与近中性理论",
      "核苷酸替换模型与进化距离校正",
      "分子钟假说及其松弛模型",
      "多序列比对方法与可靠性",
      "系统发育树构建：距离法与最大简约法",
      "最大似然建树与进化模型选择",
      "贝叶斯系统发育推断与后验概率",
      "基因树与物种树不一致、不完全谱系分选与网状进化",
      "系统发育比较方法与性状演化分析",
      "应用：病原体溯源、分子系统地理学与保护遗传学"
    ],
  },
  'advanced/biomechanics': {
    title: "生物力学",
    books: [
          "Y. C. Fung（冯元桢）, Biomechanics: Mechanical Properties of Living Tissues (2nd ed., 1993)",
          "Steven Vogel, Comparative Biomechanics: Life's Physical World (2nd ed., Princeton, 2013)",
          "Y. C. Fung, Biomechanics: Circulation (2nd ed., 1997)"
    ],
    chapters: [
      "生物力学的研究对象与本构关系思想",
      "骨与软骨的力学性质",
      "软组织力学：皮肤、肌腱与韧带",
      "心血管生物力学与血管壁力学",
      "血液流变学与微循环力学",
      "呼吸系统力学",
      "骨骼肌力学与 Hill 方程",
      "运动生物力学：步态与动作技术分析",
      "细胞与分子生物力学",
      "植介入器械力学与组织工程"
    ],
  },
  'advanced/aging-biology': {
    title: "衰老生物学",
    books: [
          "Carlos López-Otín et al., The Hallmarks of Aging, Cell 153, 1194 (2013)",
          "Robert Arking, Biology of Aging: Observations and Principles (3rd ed., Oxford, 2006)",
          "Edward J. Masoro & Steven N. Austad (eds.), Handbook of the Biology of Aging (8th ed., 2016)"
    ],
    chapters: [
      "衰老研究简史与衰老标志（Hallmarks）框架",
      "衰老的进化理论：突变积累、拮抗多效与一次性体细胞",
      "端粒缩短与复制性衰老",
      "细胞衰老与衰老相关分泌表型（SASP）",
      "基因组不稳定与 DNA 损伤修复",
      "表观遗传改变与表观遗传时钟",
      "蛋白质稳态丧失与线粒体功能障碍",
      "营养感应通路：胰岛素/IGF-1、mTOR、AMPK 与 sirtuins",
      "干细胞耗竭与组织再生能力衰退",
      "衰老干预：热量限制、雷帕霉素与 senolytics"
    ],
  },
  'advanced/microbiome-science': {
    title: "微生物组学",
    books: [
          "Julian R. Marchesi, The Human Microbiota and Microbiome (CABI, 2014)",
          "Human Microbiome Project Consortium, Structure, function and diversity of the healthy human microbiome, Nature 486, 207 (2012)",
          "Christopher Quince et al., Shotgun metagenomics, from sampling to analysis, Nature Biotechnology 35, 833 (2017)"
    ],
    chapters: [
      "微生物组的概念与人类微生物组计划（HMP）",
      "16S rRNA 扩增子测序与 OTU/ASV 分析",
      "宏基因组学：组装、分箱与功能注释",
      "群落生态分析：α/β 多样性、差异丰度与共现网络",
      "肠道微生物组与宿主代谢",
      "微生物组与免疫系统的发育和调节",
      "肠–脑轴与神经行为",
      "微生物组与疾病：关联、因果与粪菌移植（FMT）",
      "口腔、皮肤与生殖道微生物组",
      "环境与工程微生物组、合成菌群设计"
    ],
  },
  'intermediate/isotope-geochronology': {
    title: "同位素地质年代学",
    books: [
          "Alan P. Dickin, Radiogenic Isotope Geology (3rd ed., Cambridge, 2018)",
          "Gunter Faure & Teresa M. Mensing, Isotopes: Principles and Applications (3rd ed., 2005)",
          "William M. White, Isotope Geochemistry (Wiley, 2015)"
    ],
    chapters: [
      "放射性衰变定律与地质定年原理",
      "Rb–Sr 与 Sm–Nd 等时线定年",
      "U–Th–Pb 体系与锆石定年",
      "K–Ar 与 ⁴⁰Ar/³⁹Ar 定年",
      "¹⁴C 放射性碳定年",
      "宇宙成因核素暴露定年（¹⁰Be 等）",
      "裂变径迹与 (U–Th)/He 低温热年代学",
      "Sr–Nd–Pb–Hf 同位素示踪与源区判别",
      "稳定同位素（C/O/S）在地质过程中的应用",
      "地质年代表与深时定年的综合约束"
    ],
  },
  'foundations/observational-astronomy': {
    title: "观测天文学与天文技术",
    books: [
          "C. R. Kitchin, Astrophysical Techniques (6th ed., CRC Press, 2013)",
          "Pierre Léna et al., Observational Astrophysics (2nd ed., Springer, 1998)",
          "D. Scott Birney, Guillermo Gonzalez & David Oesper, Observational Astronomy (2nd ed., 2006)"
    ],
    chapters: [
      "天球坐标系、岁差章动与时间系统",
      "大气窗口、视宁度与天文台址",
      "光学望远镜：结构、像差与现代大型望远镜",
      "探测器：CCD/CMOS 与光子计数技术",
      "测光：测光系统、定标与大气消光",
      "天文光谱学与大规模光谱巡天",
      "射电望远镜与甚长基线干涉（VLBI）",
      "红外、毫米波与亚毫米波观测",
      "X 射线、γ 射线与宇宙线天文",
      "空间天文台：从哈勃到詹姆斯·韦布",
      "多信使天文学：引力波、中微子与天文数据处理"
    ],
  },
  'cs/mlops-llmops': {
    title: "MLOps 与 LLMOps",
    books: [
          "Mark Treveil et al., Introducing MLOps (O'Reilly, 2020)",
          "Chip Huyen, Designing Machine Learning Systems (O'Reilly, 2022)",
          "Noah Gift & Alfredo Deza, Practical MLOps (O'Reilly, 2021)"
    ],
    chapters: [
      "MLOps 的概念、生命周期与角色分工",
      "数据版本管理（DVC）与特征平台",
      "实验跟踪与模型注册（MLflow/W&B）",
      "机器学习流水线与编排（Kubeflow/Airflow）",
      "CI/CD for ML：自动化测试与持续部署",
      "模型部署模式：批推理、在线服务与影子/金丝雀发布",
      "模型监控：数据漂移、概念漂移与告警",
      "LLMOps：提示管理、评估体系与安全护栏",
      "LLM 微调与 RAG 流水线的工程化",
      "GPU 资源调度、成本优化与 MLOps 平台选型"
    ],
  },
  'cs/privacy-preserving-computing': {
    title: "隐私计算（差分隐私/同态加密/安全多方计算）",
    books: [
          "Cynthia Dwork & Aaron Roth, The Algorithmic Foundations of Differential Privacy (2014)",
          "David Evans, Vladimir Kolesnikov & Mike Rosulek, A Pragmatic Introduction to Secure Multi-Party Computation (2018)",
          "Craig Gentry, A Fully Homomorphic Encryption Scheme (Stanford PhD thesis, 2009)"
    ],
    chapters: [
      "隐私威胁模型：再识别、属性推断与成员推断攻击",
      "去标识化与 k-匿名、ℓ-多样性及其局限",
      "差分隐私：形式化定义与敏感度",
      "拉普拉斯/高斯机制与组合定理",
      "本地差分隐私与中心化差分隐私",
      "同态加密：部分同态、层次型与全同态",
      "安全多方计算：混淆电路与秘密分享",
      "隐私求交（PSI）与不经意传输",
      "联邦学习中的隐私保护与可信执行环境（TEE）",
      "隐私计算的法规、标准与工程实践"
    ],
  },
  'advanced/code-intelligence': {
    title: "代码智能与 AI 辅助编程",
    books: [
          "Mark Chen et al., Evaluating Large Language Models Trained on Code (Codex, 2021)",
          "Zhang et al., Unifying the Perspectives of NLP and Software Engineering: A Survey on Language Models for Code (TSE, 2024)",
          "Carlos E. Jimenez et al., SWE-bench: Can Language Models Resolve Real-World GitHub Issues? (2024)"
    ],
    chapters: [
      "代码智能概览：任务谱系、数据集与评测格局",
      "代码的表示学习：CodeBERT 与 GraphCodeBERT",
      "代码生成大模型：从 Codex 到 Code LLaMA/DeepSeek-Coder",
      "代码补全与 IDE 集成的工程实践",
      "评测基准：HumanEval、MBPP 与 pass@k 估计",
      "仓库级代码理解与长上下文建模",
      "检索增强的代码生成",
      "单元测试生成与自动化程序修复",
      "代码审查、缺陷检测与安全审计",
      "Agent 化编程：SWE-agent 与工具链编排",
      "SWE-bench 与真实仓库 issue 修复评测",
      "工程落地：Copilot/Cursor 模式、许可证与安全风险"
    ],
  },
  'engineering/mems-microfabrication': {
    title: "微机电系统与微纳制造（MEMS/NEMS）",
    books: [
          "Stephen D. Senturia, Microsystem Design (Springer, 2001)",
          "Marc J. Madou, Fundamentals of Microfabrication and Nanotechnology (3rd ed., 2011)",
          "Chang Liu, Foundations of MEMS (2nd ed., 2011)"
    ],
    chapters: [
      "MEMS 的概念、缩放律与典型器件",
      "体硅微加工与表面微加工工艺",
      "光刻、薄膜沉积与刻蚀技术",
      "LIGA 工艺与非硅微加工",
      "微传感器：加速度计、压力传感器与陀螺仪",
      "微执行器：静电、电热与压电驱动",
      "RF MEMS 与光 MEMS",
      "微流控芯片与 BioMEMS",
      "纳机电系统（NEMS）与微纳制造前沿",
      "MEMS 封装、可靠性与产业化"
    ],
  },
  'engineering/reliability-engineering': {
    title: "可靠性工程",
    books: [
          "Patrick O'Connor & Andre Kleyner, Practical Reliability Engineering (5th ed., Wiley, 2012)",
          "Elsayed A. Elsayed, Reliability Engineering (2nd ed., Wiley, 2012)",
          "曾声奎等《系统可靠性设计分析教程》（北京航空航天大学出版社，2001）"
    ],
    chapters: [
      "可靠性的基本概念与指标体系",
      "失效规律与寿命分布：指数、Weibull 与对数正态分布",
      "系统可靠性模型：串联、并联、表决与旁联系统",
      "可靠性框图与马尔可夫模型",
      "FMEA/FMECA 与故障树分析（FTA）",
      "可靠性预计、分配与冗余设计",
      "可靠性试验与加速寿命试验",
      "可靠性增长：Duane 模型与 AMSAA 模型",
      "维修性、可用性与综合保障工程",
      "软件可靠性与 MBSE 中的可靠性工作"
    ],
  },
  'life/vaccinology': {
    title: "疫苗与免疫预防",
    books: [
          "Stanley A. Plotkin, Walter A. Orenstein & Paul A. Offit, Vaccines (7th ed., 2018)",
          "CDC, Epidemiology and Prevention of Vaccine-Preventable Diseases (The Pink Book, 14th ed., 2021)"
    ],
    chapters: [
      "免疫预防简史：从 Jenner 牛痘到 mRNA 疫苗",
      "疫苗免疫学基础：先天与适应性免疫应答",
      "疫苗技术路线：灭活疫苗与减毒活疫苗",
      "重组蛋白、多糖结合与病毒载体疫苗",
      "核酸疫苗（DNA/mRNA）与新型技术平台",
      "佐剂与抗原递送系统",
      "国家免疫规划与免疫程序",
      "疫苗冷链、接种实施与接种率监测",
      "疑似预防接种异常反应（AEFI）的监测与处置",
      "群体免疫、疾病消除与根除策略",
      "疫苗犹豫、风险沟通与新疫苗研发展望"
    ],
  },
  'life/evidence-based-medicine': {
    title: "循证医学",
    books: [
          "Sharon E. Straus et al., Evidence-Based Medicine: How to Practice and Teach EBM (5th ed., 2019)",
          "李幼平主编《循证医学》（人民卫生出版社）",
          "Julian P. T. Higgins & James Thomas (eds.), Cochrane Handbook for Systematic Reviews of Interventions (2nd ed., 2019)"
    ],
    chapters: [
      "循证医学的缘起、理念与五步实践模式",
      "临床问题的构建：PICO 框架",
      "证据的分级与质量评价体系",
      "临床研究设计谱系：从病例报告到随机对照试验",
      "文献检索策略与常用数据库",
      "治疗性研究评价：偏倚风险与结果解读",
      "系统评价与 Meta 分析方法",
      "诊断试验与预后研究的评价",
      "临床实践指南与 GRADE 证据分级",
      "共同决策、患者价值观与循证医学的批评和发展"
    ],
  },
  'life/general-surgery': {
    title: "普通外科学",
    books: [
          "陈孝平、汪建平、赵继宗主编《外科学》（第9版，人民卫生出版社，2018）",
          "F. Charles Brunicardi et al., Schwartz's Principles of Surgery (11th ed., 2019)",
          "Courtney M. Townsend et al., Sabiston Textbook of Surgery (21st ed., 2022)"
    ],
    chapters: [
      "外科基础：无菌术与围手术期处理",
      "水电解质与酸碱平衡紊乱、外科营养支持",
      "外科休克与多器官功能障碍综合征",
      "甲状腺与甲状旁腺疾病",
      "乳腺疾病",
      "腹外疝与腹壁疾病",
      "急腹症的诊断与处理、腹部创伤",
      "胃十二指肠疾病与胃肠道肿瘤",
      "肝脏、胆道与胰腺疾病",
      "结直肠与肛管疾病",
      "腹腔镜与微创外科技术"
    ],
  },
  'life/plastic-burn-surgery': {
    title: "整形外科与烧伤外科",
    books: [
          "王炜主编《整形外科学》（浙江科学技术出版社，1999）",
          "Peter C. Neligan, Plastic Surgery (4th ed., 2018)",
          "David N. Herndon, Total Burn Care (5th ed., 2018)"
    ],
    chapters: [
      "整形外科的基本原则与组织移植",
      "皮瓣学：随意皮瓣、轴型皮瓣与游离皮瓣",
      "显微外科与断指（肢）再植",
      "创面愈合、瘢痕形成与防治",
      "烧伤的面积深度评估与现场急救",
      "烧伤休克与液体复苏",
      "烧伤创面处理与感染防治",
      "体表肿瘤与先天性畸形的修复",
      "美容外科：基本原则、常见术式与伦理",
      "组织工程与再生医学在修复重建中的应用"
    ],
  },
  'social/procedural-law': {
    title: "诉讼法学（民事诉讼/刑事诉讼/行政诉讼）",
    books: [
          "张卫平《民事诉讼法》（法律出版社）",
          "陈光中主编《刑事诉讼法》（北京大学出版社、高等教育出版社）",
          "姜明安主编《行政法与行政诉讼法》（北京大学出版社、高等教育出版社）"
    ],
    chapters: [
      "诉讼与诉讼法：程序正义的基本理论",
      "民事诉讼法的基本原则与基本制度",
      "民事诉讼的管辖与当事人制度",
      "民事证据与证明责任分配",
      "民事审判程序：一审、二审与再审",
      "刑事诉讼的理念与基本原则",
      "刑事强制措施与辩护制度",
      "刑事证据规则与非法证据排除",
      "侦查、起诉与刑事审判程序",
      "行政诉讼：受案范围、举证责任与审判程序",
      "执行程序与多元化纠纷解决（调解与仲裁）"
    ],
  },
  'social/economic-law': {
    title: "经济法学",
    books: [
          "杨紫烜主编《经济法》（北京大学出版社、高等教育出版社）",
          "李昌麒主编《经济法学》（法律出版社）",
          "马工程教材《经济法学》编写组《经济法学》（高等教育出版社）"
    ],
    chapters: [
      "经济法的产生、概念与调整对象",
      "经济法的基本原则与体系结构",
      "市场主体规制：公司与企业法律制度",
      "反垄断法：垄断协议、滥用市场支配地位与经营者集中",
      "反不正当竞争法",
      "消费者权益保护法",
      "产品质量与食品安全法律制度",
      "宏观调控法：财政、税收与预算法制",
      "金融法制与金融监管",
      "市场监管体制与平台经济的法律规制"
    ],
  },
  'social/comparative-education': {
    title: "比较教育学",
    books: [
          "顾明远、薛理银《比较教育导论——教育与国家发展》（人民教育出版社，1996）",
          "王承绪、顾明远主编《比较教育》（人民教育出版社）",
          "Robert F. Arnove & Carlos A. Torres (eds.), Comparative Education: The Dialectic of the Global and the Local (4th ed., 2013)"
    ],
    chapters: [
      "比较教育的学科发展与方法论传统",
      "国别研究：美国教育制度与改革",
      "国别研究：英国、法国与德国教育",
      "国别研究：日本、俄罗斯与新加坡教育",
      "北欧教育模式与福利国家",
      "发展中国家教育与教育公平",
      "国际组织与全球教育治理（UNESCO/OECD/世界银行）",
      "国际大型教育测评：PISA、TIMSS 及其影响",
      "教育国际化、跨境教育与留学研究",
      "比较视野下的中国教育改革"
    ],
  },
  'social/higher-education': {
    title: "高等教育学",
    books: [
          "潘懋元、王伟廉主编《高等教育学》（福建教育出版社，1995）",
          "Clark Kerr, The Uses of the University (5th ed., Harvard, 2001)",
          "Burton R. Clark, The Higher Education System: Academic Organization in Cross-National Perspective (1983)"
    ],
    chapters: [
      "高等教育的历史演进：从古典大学到现代大学",
      "大学理念：洪堡、纽曼与蔡元培",
      "高等教育发展理论与马丁·特罗大众化阶段论",
      "高等教育的结构、类型与层次",
      "大学治理：学术权力与行政权力",
      "高等学校的课程与教学",
      "研究生教育与大学科研组织",
      "学术职业与高校教师发展",
      "高等教育质量保障与评估",
      "世界一流大学建设与高等教育国际化"
    ],
  },
  'social/advertising': {
    title: "广告学",
    books: [
          "威廉·阿伦斯《当代广告学》（中译本，人民邮电出版社）",
          "陈培爱主编《广告学概论》（高等教育出版社）",
          "David Ogilvy, Ogilvy on Advertising (1983)"
    ],
    chapters: [
      "广告的本质、功能与发展简史",
      "经典广告理论：USP、品牌形象与定位",
      "消费者行为与广告心理",
      "广告调查与市场研究",
      "广告策划与广告预算",
      "广告创意：策略与表现",
      "广告文案与视觉设计",
      "广告媒介策略与程序化购买",
      "广告效果测评",
      "广告伦理、法规与社会责任"
    ],
  },
  'social/science-technology-society': {
    title: "科学技术与社会（STS）",
    books: [
          "Sergio Sismondo, An Introduction to Science and Technology Studies (2nd ed., 2010)",
          "Sheila Jasanoff et al. (eds.), Handbook of Science and Technology Studies (Sage, 1995)",
          "Bruno Latour & Steve Woolgar, Laboratory Life: The Construction of Scientific Facts (1979)"
    ],
    chapters: [
      "STS 的学科形成与建制化",
      "科学知识的实验室研究（Latour & Woolgar）",
      "爱丁堡学派与科学知识社会学的强纲领",
      "社会建构论及其争论（科学大战）",
      "行动者网络理论（ANT）",
      "技术的社会塑造（SCOT）与技术系统论",
      "风险社会、专家治理与公众参与科学",
      "性别、后殖民视角与科学的权力分析",
      "科技政策、国家创新体系与负责任创新",
      "STS 前沿：平台社会与人工智能治理"
    ],
  },
  'humanities/chinese-language-philology': {
    title: "汉语言文字学（古代/现代汉语与汉语史）",
    books: [
          "王力主编《古代汉语》（校订重排本，中华书局）",
          "黄伯荣、廖序东主编《现代汉语》（增订六版，高等教育出版社）",
          "王力《汉语史稿》（中华书局）"
    ],
    chapters: [
      "汉字的起源与性质",
      "六书与汉字的构造",
      "汉字形体的演变：从甲骨文到楷书",
      "现代汉语语音：声母、韵母、声调与音变",
      "现代汉语词汇：构词法与词义系统",
      "现代汉语语法：词类与句法结构",
      "修辞与语用",
      "古代汉语通论：文字、词汇与语法",
      "音韵学基础：中古音系与上古音系",
      "训诂学基础：释义方法与训诂体例",
      "汉语史概要：语音、词汇与语法的历时演变",
      "汉语方言、共同语与语言文字规范"
    ],
  },
  'humanities/editing-publishing': {
    title: "编辑出版学",
    books: [
          "全国出版专业职业资格考试办公室《出版专业基础（中级）》（崇文书局）",
          "John B. Thompson, Merchants of Culture: The Publishing Business in the Twenty-First Century (2nd ed., 2021)"
    ],
    chapters: [
      "出版的概念、属性与中外出版史概要",
      "编辑工作与编辑流程总览",
      "选题策划与组稿",
      "审稿、编辑加工与内容质量管理",
      "校对与装帧设计",
      "著作权与版权贸易",
      "出版物成本核算与定价",
      "发行渠道与图书营销",
      "期刊出版与学术出版",
      "数字出版、融合出版与出版法规职业道德"
    ],
  },
  'frontier/financial-technology': {
    title: "金融科技（FinTech）",
    books: [
          "Douglas W. Arner, Jànos Barberis & Ross P. Buckley, The Evolution of FinTech: A New Post-Crisis Paradigm? (2016)",
          "Susanne Chishti & Janos Barberis (eds.), The FINTECH Book (Wiley, 2016)",
          "BIS, Central Bank Digital Currencies: Foundational Principles and Core Features (2020)"
    ],
    chapters: [
      "金融科技的定义与演进阶段",
      "支付科技：移动支付、二维码与实时清算体系",
      "网络借贷、众筹与数字信贷",
      "智能投顾与量化财富管理",
      "数字货币：加密资产、稳定币与央行数字货币（CBDC）",
      "区块链在金融中的应用",
      "大数据征信与智能风控",
      "监管科技（RegTech）与合规科技",
      "开放银行、API 经济与嵌入式金融",
      "金融科技的风险、伦理与全球监管比较"
    ],
  },
  'life/field-experiment-biostatistics': {
    title: "田间试验与生物统计（农业试验设计与统计分析）",
    books: [
          "盖钧镒主编《试验统计方法》，中国农业出版社，2000（面向21世纪课程教材）",
          "莫惠栋《农业试验统计》，上海科学技术出版社，1992",
          "杜荣骞《生物统计学》，高等教育出版社"
    ],
    chapters: [
      "试验设计基本原理（重复、随机、局部控制）",
      "完全随机设计与随机区组设计",
      "拉丁方设计与裂区设计",
      "正交试验设计",
      "田间试验的实施、管理与误差控制",
      "统计资料整理与特征数",
      "概率分布与抽样分布",
      "参数估计与假设检验",
      "方差分析（单因素与多因素）",
      "卡方检验与适合性检验",
      "相关与回归分析",
      "多元回归与通径分析",
      "抽样调查与样本量确定",
      "统计软件实现（R/SAS）"
    ],
  },
  'life/plant-animal-quarantine': {
    title: "动植物检疫（植物检疫/动物检验检疫与国门生物安全）",
    books: [
          "许志刚主编《植物检疫学》，中国农业出版社，2003",
          "柳增善、任洪林、张守印主编《动物检疫检验学》，科学出版社，2012",
          "鞠兴荣主编《动植物检验检疫学》，中国轻工业出版社，2010"
    ],
    chapters: [
      "检疫性有害生物与风险分析（PRA）",
      "检疫法规体系与 WTO/SPS 协定",
      "植物检疫程序、产地检疫与除害处理（熏蒸/热处理/辐照）",
      "检疫性植物病害",
      "检疫性害虫",
      "检疫性杂草",
      "进境种苗与繁殖材料检疫",
      "WOAH 动物疫病名录与通报",
      "动物及动物产品检疫检验",
      "人兽共患病检疫",
      "口岸检疫与国门生物安全",
      "检疫性有害生物分子鉴定技术",
      "入侵生物预警与应急处置",
      "检疫处理与无害化"
    ],
  },
  'life/traditional-chinese-veterinary-medicine': {
    title: "中兽医学",
    books: [
          "刘钟杰、许剑琴主编《中兽医学》（第四版），中国农业出版社，2020（面向21世纪课程教材）",
          "杨英主编《兽医针灸学》，高等教育出版社，2006",
          "钟秀会主编《中兽医学实验指导》，中国农业出版社，2016"
    ],
    chapters: [
      "阴阳五行学说",
      "脏腑与经络学说",
      "病因与病机",
      "四诊（望闻问切）",
      "辨证论治（八纲辨证/脏腑辨证/卫气营血辨证）",
      "中兽药性能、炮制与配伍禁忌",
      "常用中兽药（解表/清热/补益等）",
      "兽医方剂学",
      "动物针灸基础与常用穴位",
      "家畜常见病辨证施治（脾胃病/咳喘/泄泻）",
      "中药饲料添加剂与减抗替抗",
      "中西兽医结合"
    ],
  },
  'life/veterinary-public-health': {
    title: "兽医公共卫生学",
    books: [
          "张彦明主编《兽医公共卫生学》（第三版），中国农业出版社，2019（十三五规划教材）",
          "柳增善主编《兽医公共卫生学》，中国轻工业出版社，2010",
          "张彦明、佘锐平主编《动物性食品卫生学》（第四版），中国农业出版社，2012"
    ],
    chapters: [
      "One Health 理念与兽医公共卫生定位",
      "人兽共患传染病（狂犬病/布鲁氏菌病/高致病性禽流感/结核）",
      "人兽共患寄生虫病（弓形虫/旋毛虫/囊尾蚴）",
      "动物性食品污染与卫生检验",
      "肉品卫生与屠宰检疫",
      "乳与乳制品卫生",
      "蛋品与水产品卫生",
      "兽药残留与食品安全",
      "动物福利与动物伦理",
      "养殖场生物安全与废弃物处理",
      "环境污染与生态平衡",
      "兽医在突发公共卫生事件中的职能",
      "比较医学"
    ],
  },
  'life/aquatic-animal-medicine': {
    title: "水生动物医学（水产动物病害学）",
    books: [
          "战文斌主编《水产动物病害学》（第二版），中国农业出版社，2011（十一五国家级规划教材）",
          "黄琪琰主编《水产动物疾病学》，上海科学技术出版社，2004",
          "麦康森主编《水产动物营养与饲料学》（第二版），中国农业出版社，2011（营养与病害关联参考）"
    ],
    chapters: [
      "水产动物病原学（病毒/细菌/真菌/寄生虫）",
      "鱼类免疫学基础",
      "水产动物病理学基础",
      "疾病诊断技术（临床检查/病原分离/分子诊断）",
      "鱼类病毒性疾病（出血病/淋巴囊肿病）",
      "细菌性疾病（烂鳃病/肠炎病/败血症）",
      "寄生虫病（小瓜虫/指环虫/车轮虫）",
      "虾蟹病害（白斑综合征/急性肝胰腺坏死病）",
      "贝类与两栖爬行类病害",
      "水质恶化与应激性疾病",
      "渔药药理学与规范用药",
      "微生态制剂与免疫增强剂",
      "疫苗与免疫防控",
      "养殖场生物安全与健康管理"
    ],
  },
  'humanities/history-of-agriculture': {
    title: "农学史（中国农业科技史）",
    books: [
          "梁家勉主编《中国农业科学技术史稿》，农业出版社，1989",
          "游修龄《中国稻作史》，中国农业出版社，1995",
          "陈文华《中国古代农业科技史图谱》，农业出版社，1991"
    ],
    chapters: [
      "农业起源与作物驯化（稻/粟/黍）",
      "新石器时代农具与原始耕作",
      "先秦农学与《吕氏春秋·上农》四篇",
      "汉代代田法与区田法",
      "《齐民要术》与魏晋南北朝农学",
      "唐宋曲辕犁与江南水田农业体系",
      "占城稻推广与宋代农业变革",
      "元代《农桑辑要》与王祯《农书》",
      "徐光启《农政全书》与《授时通考》",
      "美洲作物传入（玉米/甘薯/马铃薯/烟草）",
      "传统多熟制与耕作制度演变",
      "农田水利史（都江堰/陂塘/坎儿井）",
      "蚕桑、茶与畜牧兽医史",
      "近代农学教育、农事试验场与农业改良",
      "绿色革命与现代农业科技转型"
    ],
  },
  'life/farming-systems': {
    title: "耕作学与农作制度",
    books: [
          "刘巽浩《耕作学》，农业出版社，1994",
          "曹敏建主编《耕作学》，中国农业出版社（面向21世纪课程教材）",
          "刘巽浩、高旺盛《农作学》，中国农业大学出版社"
    ],
    chapters: [
      "耕作制度的概念、功能与类型",
      "作物布局与种植结构",
      "复种与多熟制",
      "间作、混作与套作",
      "轮作与连作障碍",
      "土壤耕作（少耕/免耕/深松）",
      "种养结合与农牧复合系统",
      "农作制度的区域分异",
      "耕地保护与撂荒治理",
      "保护性农业（Conservation Agriculture）",
      "可持续集约化",
      "气候智慧型农作制度"
    ],
  },
  'life/veterinary-pharmacology': {
    title: "兽医药理学与动物药学",
    books: [
          "陈杖榴主编《兽医药理学》（第四版），中国农业出版社，2017",
          "沈建忠、肖希龙主编《兽医药理学》（研究生用书），中国农业大学出版社",
          "胡功政主编《兽药制剂学》，中国农业出版社"
    ],
    chapters: [
      "兽药代谢动力学",
      "兽药效应动力学",
      "抗菌药物与细菌耐药性",
      "抗寄生虫药",
      "解热镇痛抗炎药",
      "作用于神经系统的药物与麻醉药",
      "消化系统与呼吸系统药物",
      "激素类药物与繁殖调控",
      "解毒药",
      "兽药制剂与剂型",
      "兽药残留监控与食品安全",
      "新兽药研发、注册与 GLP/GCP"
    ],
  },
  'life/laboratory-animal-science': {
    title: "实验动物学",
    books: [
          "秦川、谭毅、张连峰编《医学实验动物学》（第2版），人民卫生出版社，2015（十二五规划教材）",
          "刘恩岐主编《医学实验动物学》，科学出版社，2008",
          "刘恩岐主编《人类疾病动物模型》，人民卫生出版社"
    ],
    chapters: [
      "实验动物分类与微生物学等级（普通级/清洁级/SPF/无菌）",
      "常用实验动物生物学特性（小鼠/大鼠/豚鼠/兔/犬/小型猪）",
      "实验动物遗传质量控制（近交系/封闭群/杂交群）",
      "微生物与寄生虫质量监测",
      "实验动物环境与设施（屏障系统/独立通气笼 IVC）",
      "实验动物营养与饲料",
      "动物模型制备原理",
      "基因工程动物（转基因/基因敲除/条件性敲入）",
      "人类疾病动物模型（肿瘤/代谢/神经退行）",
      "动物实验伦理与 3R 原则",
      "动物实验基本操作技术",
      "GLP 规范与实验动物法规"
    ],
  },
  'life/tobacco-science': {
    title: "烟草科学与工程",
    books: [
          "刘国顺主编《烟草栽培学》（第二版），中国农业出版社，2017",
          "宫长荣编著《烟草调制学》（第二版），中国农业出版社，2017",
          "于建军主编《卷烟工艺学》，中国农业出版社，2009"
    ],
    chapters: [
      "烟草类型与品种（烤烟/白肋烟/香料烟/晒晾烟）",
      "烟草生物学特性与栽培生理",
      "育苗与移栽",
      "烟草营养与施肥",
      "烟草育种学",
      "烟草病虫害防治",
      "烟草调制学（烤房与烘烤工艺）",
      "烟叶分级与质量评价",
      "烟草化学与烟气分析",
      "卷烟工艺学（制丝/卷接/包装）",
      "烟叶醇化与发酵",
      "烟草经济与控烟政策",
      "新型烟草制品（电子烟/加热不燃烧）"
    ],
  },
  'life/invasion-biology': {
    title: "入侵生物学",
    books: [
          "万方浩主编《入侵生物学》，科学出版社，2011",
          "徐海根、强胜主编《生物入侵》，科学出版社",
          "万方浩、郑小波、郭建英《重要农林外来入侵物种的生物学与控制》，科学出版社"
    ],
    chapters: [
      "生物入侵概念与入侵过程（传入/定殖/扩散/暴发）",
      "入侵机制（入侵种内在特性/群落可入侵性/天敌逃逸假说）",
      "重要入侵植物（紫茎泽兰/薇甘菊/加拿大一枝黄花）",
      "重要入侵动物（红火蚁/福寿螺/草地贪夜蛾）",
      "入侵病原（松材线虫/稻水象甲携带病原）",
      "生物入侵与全球变化互作",
      "入侵风险评估与预警",
      "监测检测与分子溯源技术",
      "防控技术（物理/化学/生物防治/生态替代）",
      "入侵生物管理的法规与国际公约（CBD/IPPC）",
      "入侵对生物多样性与生态系统服务的影响",
      "典型入侵事件案例与治理"
    ],
  },
  'social/agricultural-policy': {
    title: "农业政策学",
    books: [
          "钟甫宁主编《农业政策学》（第二版），中国农业大学出版社，2013",
          "孔祥智主编《农业政策学》，高等教育出版社",
          "张广胜主编《农业政策学》，高等教育出版社"
    ],
    chapters: [
      "农业政策过程与政策评估方法",
      "粮食安全与粮食政策",
      "农业补贴与支持保护政策",
      "农产品价格支持与市场干预",
      "农村土地制度与承包经营制度",
      "新型农业经营主体与合作社政策",
      "农业保险与农业风险管理",
      "农业科技与教育政策",
      "农产品贸易政策与 WTO 农业协定",
      "乡村振兴战略与政策体系",
      "农业环境与生态补偿政策",
      "国际农业政策比较（美国/欧盟/日本）"
    ],
  },
  'life/animal-products-processing': {
    title: "畜产品加工学（肉/乳/蛋加工）",
    books: [
          "周光宏主编《畜产品加工学》（第二版），中国农业出版社，2011",
          "李晓东主编《乳品工艺学》，中国轻工业出版社，2024（十四五规划教材）",
          "孔保华主编《肉品科学与技术》，中国轻工业出版社"
    ],
    chapters: [
      "肉的组织结构、化学组成与宰后生化（尸僵/成熟）",
      "肉品加工工艺（腌腊/酱卤/熏烤/发酵肉制品）",
      "肉品贮藏保鲜与冷链",
      "乳的化学组成与微生物",
      "原料乳验收与预处理",
      "液态乳（巴氏杀菌/超高温灭菌）",
      "发酵乳与益生菌乳制品",
      "干酪加工",
      "乳粉、炼乳与奶油",
      "冰淇淋与冷冻乳制品",
      "蛋的构造、品质与保鲜",
      "蛋制品加工（皮蛋/咸蛋/液蛋/蛋粉）",
      "畜产品质量安全与控制体系"
    ],
  },
  'life/grain-oil-processing-storage': {
    title: "粮食油脂与植物蛋白工程（含粮油储藏）",
    books: [
          "《粮油储藏学》（第三版），中国轻工业出版社（河南工业大学统编教材）",
          "陈复生等《蛋白质化学与工艺学》，中国轻工业出版社",
          "黄亚伟主编《粮油仓储工艺与设备》（第二版），中国轻工业出版社"
    ],
    chapters: [
      "稻谷加工与碾米工艺",
      "小麦制粉与专用粉",
      "淀粉生产与变性淀粉",
      "油脂制取（压榨法/浸出法）",
      "油脂精炼与改性",
      "植物蛋白提取与功能性质",
      "大豆蛋白制品（分离蛋白/组织蛋白）",
      "粮油储藏生理与品质变化",
      "储粮害虫与防治",
      "气调储藏与低温储藏",
      "粮油仓储工艺与设备",
      "粮油品质检验与标准",
      "全谷物食品与主食工业化"
    ],
  },
  'life/postharvest-physiology-storage': {
    title: "园艺产品采后生理与贮藏加工（农产品加工及贮藏工程）",
    books: [
          "罗云波、蔡同一主编《园艺产品贮藏加工学》（贮藏篇/加工篇），中国农业大学出版社",
          "赵丽芹主编《果蔬加工工艺学》，中国轻工业出版社，2007",
          "陈昆松等《园艺产品采后生物学基础》，科学出版社"
    ],
    chapters: [
      "采后呼吸生理与蒸腾",
      "乙烯生理与成熟衰老调控",
      "采后侵染性病害与生理性病害（冷害/褐变）",
      "采收、分级、包装与预冷",
      "冷藏与气调贮藏（CA/MA）",
      "保鲜剂、涂膜与 1-MCP 处理",
      "冷链物流与货架期",
      "果蔬干制与脱水加工",
      "果蔬汁、果酱与罐藏",
      "果蔬糖制与腌制",
      "速冻果蔬加工",
      "鲜切果蔬（最小加工）",
      "加工副产物综合利用"
    ],
  },
  'life/viticulture-enology': {
    title: "葡萄与葡萄酒工程",
    books: [
          "李华、王华、袁春龙、王树生《葡萄酒工艺学》（第二版），科学出版社，2023",
          "李华主编《葡萄栽培学》，中国农业出版社，2008（十一五国家级规划教材）",
          "李华《葡萄酒品尝学》，科学出版社，2006"
    ],
    chapters: [
      "葡萄种与品种（酿酒葡萄/鲜食葡萄）",
      "葡萄栽培学（架式/整形修剪/土肥水管理）",
      "果实发育、成熟调控与采收期确定",
      "葡萄酒微生物（酿酒酵母/乳酸菌/腐败菌）",
      "酒精发酵与苹果酸-乳酸发酵",
      "红葡萄酒酿造工艺",
      "白葡萄酒与桃红葡萄酒工艺",
      "起泡葡萄酒与特种酒（冰酒/贵腐）",
      "陈酿、橡木桶与葡萄酒成熟",
      "葡萄酒稳定、澄清与灌装",
      "葡萄酒分析检验与质量控制",
      "葡萄酒感官品评",
      "产区、风土（terroir）与地理标志",
      "葡萄皮渣与副产物综合利用"
    ],
  },
  'advanced/applied-chemistry': {
    title: "应用化学与精细化工（精细化学品化学）",
    books: [
          "宋启煌等主编《精细化工工艺学》（第五版），化学工业出版社，2024",
          "李和平、葛虹主编《精细化工工艺学》，科学出版社，1997",
          "冯亚青、王世荣主编《精细有机合成》（第三版），化学工业出版社"
    ],
    chapters: [
      "精细化工的范畴、特点与产品分类",
      "精细化工工艺学基础与新产品技术开发",
      "精细化工绿色化与原子经济性",
      "表面活性剂",
      "合成材料助剂（增塑剂/抗氧剂/阻燃剂）",
      "食品添加剂",
      "胶黏剂",
      "涂料",
      "香料与香精",
      "染料与颜料",
      "电子化学品",
      "化妆品与日用化学品",
      "精细有机合成单元反应与技术"
    ],
  },
  'advanced/energy-chemistry': {
    title: "能源化学",
    books: [
          "陈军、陶占良编著《能源化学》（第二版），化学工业出版社，2014",
          "李文翠等编《能源化学工程概论》，化学工业出版社，2015",
          "陈军、陶占良、苟兴龙编著《化学电源：原理、技术与应用》，化学工业出版社，2005"
    ],
    chapters: [
      "能源结构、能源转换与化学热力学基础",
      "化石燃料化学（煤/石油/天然气）",
      "氢能：制氢、储氢与氢化学",
      "燃料电池原理与电催化",
      "化学电源（锂离子电池与新型电池）",
      "太阳能电池与光电化学转换",
      "光催化与人工光合作用（太阳能燃料）",
      "生物质能与生物燃料",
      "核能化学基础",
      "CO2 捕获、转化与利用化学",
      "能源材料与储能器件"
    ],
  },
  'intermediate/group-theory-in-chemistry': {
    title: "群论与分子对称性（群论在化学中的应用）",
    books: [
          "F. Albert Cotton, Chemical Applications of Group Theory, 3rd ed., Wiley, 1990（中译本《群论在化学中的应用》）",
          "David M. Bishop, Group Theory and Chemistry, Dover, 1993",
          "Alan Vincent, Molecular Symmetry and Group Theory, 2nd ed., Wiley, 2001"
    ],
    chapters: [
      "对称操作与对称元素",
      "点群分类与分子归属",
      "群的表示与特征标表",
      "可约表示的约化",
      "投影算符与对称性匹配线性组合（SALC）",
      "群论与分子轨道理论",
      "振动光谱（IR/Raman）的群论分析",
      "电子光谱与选择定则",
      "配位场中的轨道分裂",
      "对称性与化学反应性（轨道对称守恒）",
      "空间群与晶体对称性简介"
    ],
  },
  'intermediate/heterocyclic-chemistry': {
    title: "杂环化学",
    books: [
          "John A. Joule, Keith Mills, Heterocyclic Chemistry, 5th ed., Wiley, 2010",
          "Thomas L. Gilchrist, Heterocyclic Chemistry, 3rd ed., Pearson, 1997",
          "Joule & Mills 著《杂环化学》（中译本），科学出版社"
    ],
    chapters: [
      "杂环化合物的命名、分类与芳香性",
      "五元单杂环：呋喃、噻吩、吡咯",
      "稠合五元杂环：吲哚、苯并呋喃、苯并噻吩",
      "六元杂环：吡啶、喹啉、异喹啉",
      "含两个及以上杂原子的环系（咪唑/噻唑/嘧啶/嘌呤）",
      "杂环的经典合成方法",
      "杂环的亲电/亲核取代与区域选择性",
      "杂环的金属化与偶联反应",
      "杂环在药物与天然产物中的应用",
      "七元环与饱和杂环（氮杂环/氧杂环）"
    ],
  },
  'intermediate/corrosion-electrochemistry': {
    title: "腐蚀电化学与金属防护",
    books: [
          "曹楚南《腐蚀电化学原理》（第三版），化学工业出版社，2008",
          "Mars G. Fontana, Corrosion Engineering, 3rd ed., McGraw-Hill, 1986",
          "Denny A. Jones, Principles and Prevention of Corrosion, 2nd ed., Prentice Hall, 1996"
    ],
    chapters: [
      "腐蚀热力学与电位-pH 图（Pourbaix 图）",
      "腐蚀电极过程动力学与极化曲线",
      "钝化与钝化膜",
      "电偶腐蚀",
      "点蚀与缝隙腐蚀",
      "晶间腐蚀与选择性腐蚀",
      "应力腐蚀开裂与氢脆",
      "大气、海水与土壤腐蚀",
      "阴极保护与阳极保护",
      "缓蚀剂",
      "金属涂层与表面处理",
      "腐蚀试验与电化学监测方法"
    ],
  },
  'humanities/philosophy-of-chemistry': {
    title: "化学哲学",
    books: [
          "Eric Scerri, The Periodic Table: Its Story and Its Significance, Oxford University Press, 2007（2nd ed. 2019）",
          "Davis Baird, Eric Scerri, Lee McIntyre (eds.), Philosophy of Chemistry: Synthesis of a New Discipline, Springer (Boston Studies), 2006",
          "Jaap van Brakel, Philosophy of Chemistry, Leuven University Press, 2000"
    ],
    chapters: [
      "化学能否还原为物理学：还原论之争",
      "元素概念与元素周期律的哲学",
      "化学键与分子结构的实在论问题",
      "物质、纯度与混合物的形而上学",
      "化学分类与自然类",
      "化学合成（making）的认识论地位",
      "化学定律与化学解释的独特性",
      "化学中的模型与表征（结构式/轨道图）",
      "化学史案例的哲学分析（燃素说/氧化学说）",
      "化学与纳米技术的哲学"
    ],
  },
  'social/chemistry-education': {
    title: "化学教育（化学教学论）",
    books: [
          "刘知新主编《化学教学论》（第五版），高等教育出版社，2018",
          "王后雄主编《新理念化学教学论》，华中师范大学出版社，2011",
          "ACS, Journal of Chemical Education（期刊与教学资源体系）"
    ],
    chapters: [
      "化学课程论：课程理念与课程标准",
      "化学教材分析与使用",
      "化学学习论与学习心理",
      "化学教学设计与教学模式",
      "化学基本概念与理论教学",
      "化学实验教学",
      "探究式教学与科学探究",
      "化学教育测量与评价",
      "信息技术与化学教学",
      "化学教师专业发展",
      "国际化学教育比较（NGSS/IB/AP 化学）"
    ],
  },
  'intermediate/chemical-laboratory-safety': {
    title: "化学实验室安全与健康",
    books: [
          "Robert H. Hill Jr., David C. Finster, Laboratory Safety for Chemistry Students, 2nd ed., Wiley, 2016",
          "National Research Council, Prudent Practices in the Laboratory: Handling and Management of Chemical Hazards, National Academies Press, 2011",
          "ACS Guidelines for Chemical Laboratory Safety in Academic Institutions, American Chemical Society, 2016"
    ],
    chapters: [
      "GHS 全球化学品统一分类和标签制度",
      "化学品储存与相容性",
      "通风柜与实验室通风",
      "个人防护装备（PPE）",
      "防火防爆与易燃化学品",
      "压缩气体与低温液体安全",
      "化学废弃物分类与处理",
      "辐射、激光与高压设备安全",
      "实验室生物危害",
      "事故应急响应与急救",
      "风险评估（RAMP 原则）与安全管理体系"
    ],
  },
  'advanced/carbohydrate-chemistry': {
    title: "碳水化合物化学（糖化学）",
    books: [
          "John F. Robyt, Essentials of Carbohydrate Chemistry, Springer, 1998",
          "Robert V. Stick, Spencer J. Williams, Carbohydrates: The Essential Molecules of Life, 2nd ed., Elsevier, 2009",
          "Benjamin G. Davis, Antony J. Fairbanks, Carbohydrate Chemistry, Oxford Chemistry Primers, Oxford University Press, 2002"
    ],
    chapters: [
      "单糖的结构、构型与构象",
      "糖苷键与糖基化反应",
      "寡糖与多糖的结构（淀粉/纤维素/几丁质/透明质酸）",
      "糖化学中的保护基策略",
      "寡糖的化学合成与自动固相合成",
      "酶法糖基化与化学酶法合成",
      "糖复合物：糖蛋白与糖脂",
      "糖的谱学分析与结构鉴定",
      "糖与分子识别（凝集素-糖相互作用）",
      "糖药物与糖疫苗"
    ],
  },
  'advanced/combustion-chemistry': {
    title: "燃烧化学",
    books: [
          "Irvin Glassman, Richard A. Yetter, Nick G. Glumac, Combustion, 5th ed., Academic Press, 2015",
          "Kenneth K. Kuo, Principles of Combustion, 2nd ed., Wiley, 2005",
          "Stephen R. Turns, An Introduction to Combustion: Concepts and Applications, 3rd ed., McGraw-Hill, 2012"
    ],
    chapters: [
      "燃烧热化学与化学平衡",
      "燃烧化学动力学与链式反应",
      "着火、熄火与可燃极限",
      "层流预混火焰",
      "扩散火焰",
      "湍流燃烧基础",
      "爆燃与爆轰",
      "污染物生成化学（NOx/碳烟/CO）",
      "固体推进剂与含能材料燃烧",
      "燃烧激光诊断技术"
    ],
  },
  'advanced/plasma-chemistry': {
    title: "等离子体化学",
    books: [
          "Alexander Fridman, Plasma Chemistry, Cambridge University Press, 2008",
          "Alexander Fridman, Lawrence A. Kennedy, Plasma Physics and Engineering, 2nd ed., CRC Press, 2011"
    ],
    chapters: [
      "等离子体中的基元过程与化学活性物种",
      "气体放电类型与等离子体产生",
      "热等离子体与非平衡等离子体化学",
      "等离子体刻蚀与微电子加工",
      "等离子体增强化学气相沉积（PECVD）",
      "等离子体催化",
      "等离子体 CO2 转化与固氮",
      "等离子体医学与表面改性",
      "大气压等离子体与等离子体炬",
      "等离子体化学的工业应用与反应器"
    ],
  },
  'cs/computer-science-overview': {
    title: "计算机科学导论（计算思维）",
    books: [
          "J. Glenn Brookshear, Dennis Brylow《Computer Science: An Overview》（Pearson, 13th ed., 2019）",
          "Robert Sedgewick, Kevin Wayne《Computer Science: An Interdisciplinary Approach》（Princeton University Press, 2016）",
          "Behrouz Forouzan《Foundations of Computer Science》（Cengage Learning, 4th ed., 2018）"
    ],
    chapters: [
      "计算思维与问题求解方法",
      "数据的表示与存储（二进制/字符编码/图像与声音数字化）",
      "计算机硬件组成（冯·诺依曼体系/指令周期）",
      "操作系统基础（进程/存储管理）",
      "算法与程序设计入门",
      "数据结构与抽象数据类型",
      "计算机网络与互联网基础",
      "数据库与信息管理基础",
      "软件工程与软件开发流程",
      "人工智能概览",
      "计算理论初步（图灵机与可计算性）",
      "信息安全与隐私保护",
      "计算的社会、伦理与职业问题"
    ],
  },
  'advanced/artificial-intelligence': {
    title: "经典人工智能（符号主义：搜索/规划/知识表示与推理）",
    books: [
          "Stuart Russell, Peter Norvig《Artificial Intelligence: A Modern Approach》（Pearson, 4th ed., 2021）",
          "David Poole, Alan Mackworth《Artificial Intelligence: Foundations of Computational Agents》（Cambridge University Press, 2nd ed., 2017）",
          "George F. Luger《Artificial Intelligence: Structures and Strategies for Complex Problem Solving》（Addison-Wesley, 6th ed., 2008）"
    ],
    chapters: [
      "智能体概念与问题求解范式",
      "无信息搜索（BFS/DFS/一致代价搜索）",
      "启发式搜索（A*/贪心最佳优先）",
      "局部搜索与元启发式（爬山法/模拟退火/遗传算法）",
      "对抗搜索与博弈（极小极大/α-β剪枝）",
      "约束满足问题（回溯/弧一致性）",
      "命题逻辑与归结推理",
      "一阶逻辑与自动定理证明",
      "知识表示（语义网络/框架/描述逻辑）",
      "自动规划（STRIPS/状态空间规划/分层任务网络）",
      "不确定性推理与贝叶斯网络",
      "专家系统与产生式规则系统"
    ],
  },
  'humanities/history-of-computing': {
    title: "计算机科学史（计算史）",
    books: [
          "Paul E. Ceruzzi《A History of Modern Computing》（MIT Press, 2nd ed., 2003）",
          "Martin Campbell-Kelly, William Aspray, Nathan Ensmenger, Jeffrey Yost《Computer: A History of the Information Machine》（Westview Press, 3rd ed., 2013）",
          "Subrata Dasgupta《It Began with Babbage: The Genesis of Computer Science》（Oxford University Press, 2014）"
    ],
    chapters: [
      "前机械时代的计算工具（算筹/算盘/对数表）",
      "机械计算器（帕斯卡/莱布尼茨）",
      "巴贝奇差分机与分析机、艾达·洛夫莱斯",
      "穿孔卡片制表机与霍列瑞斯",
      "机电计算机（Zuse Z 系列/哈佛 Mark I）",
      "电子管时代：ENIAC 与 EDVAC",
      "冯·诺依曼报告与存储程序计算机",
      "晶体管、集成电路与 IBM System/360",
      "小型机、分时系统与 UNIX 的诞生",
      "微处理器与个人计算机革命",
      "编程语言与操作系统史",
      "ARPANET、互联网与万维网简史",
      "人工智能史（达特茅斯会议/专家系统/AI 寒冬/深度学习复兴）"
    ],
  },
  'cs/network-security': {
    title: "网络安全（协议安全与攻防）",
    books: [
          "William Stallings《Cryptography and Network Security: Principles and Practice》（Pearson, 8th ed., 2020）",
          "Charlie Kaufman, Radia Perlman, Mike Speciner《Network Security: Private Communication in a Public World》（Prentice Hall, 2nd ed., 2002）",
          "Ross Anderson《Security Engineering: A Guide to Building Dependable Distributed Systems》（Wiley, 3rd ed., 2020）"
    ],
    chapters: [
      "网络安全威胁模型与 CIA 三性",
      "认证协议与密钥分发（Kerberos）",
      "公钥基础设施（PKI）与数字证书",
      "TLS/SSL 协议族",
      "IPsec 与 VPN",
      "防火墙与入侵检测/防御系统（IDS/IPS）",
      "无线网络安全（WEP/WPA2/WPA3）",
      "Web 安全（XSS/CSRF/SQL 注入/会话安全）",
      "恶意软件、僵尸网络与蠕虫",
      "DDoS 攻击与防护",
      "渗透测试与攻防演练方法论",
      "数字取证基础",
      "安全协议的形式化分析（BAN 逻辑/模型检验）"
    ],
  },
  'cs/digital-image-processing': {
    title: "数字图像处理",
    books: [
          "Rafael C. Gonzalez, Richard E. Woods《Digital Image Processing》（Pearson, 4th ed., 2018）",
          "Milan Sonka, Vaclav Hlavac, Roger Boyle《Image Processing, Analysis, and Machine Vision》（Cengage Learning, 4th ed., 2014）"
    ],
    chapters: [
      "图像获取与数字化（采样/量化/成像几何）",
      "灰度变换与直方图处理",
      "空间域滤波（平滑/锐化）",
      "频率域处理（傅里叶变换/频域滤波器）",
      "图像复原与重建（退化模型/逆滤波/CT 重建）",
      "彩色图像处理",
      "小波与多分辨率分析",
      "图像压缩（JPEG/JPEG2000）",
      "形态学图像处理",
      "图像分割（阈值/区域生长/边缘检测/活动轮廓）",
      "局部特征提取与描述（SIFT/HOG）",
      "图像配准与拼接",
      "表示与描述（边界/区域/纹理特征）"
    ],
  },
  'cs/mobile-computing': {
    title: "移动计算与无线网络",
    books: [
          "Jochen Schiller《Mobile Communications》（Addison-Wesley, 2nd ed., 2003）",
          "Theodore S. Rappaport《Wireless Communications: Principles and Practice》（Prentice Hall, 2nd ed., 2001）",
          "Asoke K. Talukder, Hasan Ahmed, Roopa R. Yavagal《Mobile Computing: Technology, Applications and Service Creation》（McGraw-Hill, 2nd ed., 2010）"
    ],
    chapters: [
      "无线传输基础（信道衰落/调制/扩频）",
      "蜂窝系统原理与频率复用",
      "移动通信网络演进（2G GSM → 5G NR）",
      "无线局域网（IEEE 802.11 族）",
      "蓝牙与无线个域网",
      "移动 IP 与移动性管理",
      "无线环境下的 TCP 与传输层优化",
      "移动计算模型与断接操作",
      "上下文感知与普适计算",
      "定位技术与位置服务",
      "移动边缘计算（MEC）",
      "无线传感器网络基础"
    ],
  },
  'cs/real-time-systems': {
    title: "实时系统",
    books: [
          "Jane W. S. Liu《Real-Time Systems》（Prentice Hall, 2000）",
          "Giorgio C. Buttazzo《Hard Real-Time Computing Systems: Predictable Scheduling Algorithms and Applications》（Springer, 3rd ed., 2011）",
          "Hermann Kopetz《Real-Time Systems: Design Principles for Distributed Embedded Applications》（Springer, 2nd ed., 2011）"
    ],
    chapters: [
      "实时系统概念与分类（硬/软/固实时）",
      "任务模型与可调度性分析",
      "速率单调调度（RM）与截止期单调调度（DM）",
      "最早截止期优先（EDF）调度",
      "资源共享与优先级反转问题",
      "优先级继承协议与优先级天花板协议",
      "非周期与偶发任务调度（轮询/偶发服务器）",
      "多处理器与多核实时调度",
      "实时操作系统（RTOS）内核机制",
      "时间触发架构（TTA）与实时通信",
      "分布式实时系统与容错时钟同步",
      "最坏执行时间（WCET）分析"
    ],
  },
  'advanced/multiagent-systems': {
    title: "多智能体系统（经典 MAS）",
    books: [
          "Michael Wooldridge《An Introduction to MultiAgent Systems》（Wiley, 2nd ed., 2009）",
          "Yoav Shoham, Kevin Leyton-Brown《Multiagent Systems: Algorithmic, Game-Theoretic, and Logical Foundations》（Cambridge University Press, 2009）",
          "Gerhard Weiss (ed.)《Multiagent Systems: A Modern Approach to Distributed Artificial Intelligence》（MIT Press, 2nd ed., 2013）"
    ],
    chapters: [
      "智能体概念与体系结构（反应式/慎思式/混合式）",
      "BDI 模型与实用推理",
      "多智能体交互与协作问题",
      "智能体通信语言（KQML/FIPA-ACL）",
      "协商与谈判协议",
      "拍卖机制与VCG 机制",
      "联盟形成与任务分配",
      "分布式问题求解与分布式约束",
      "博弈论在多智能体系统中的应用",
      "论证、信任与声誉模型",
      "群体行为与涌现现象",
      "多智能体学习及与深度 MARL 的衔接"
    ],
  },
  'cs/performance-evaluation': {
    title: "计算机系统性能评估",
    books: [
          "Raj Jain《The Art of Computer Systems Performance Analysis: Techniques for Experimental Design, Measurement, Simulation, and Modeling》（Wiley, 1991）",
          "David J. Lilja《Measuring Computer Performance: A Practitioner's Guide》（Cambridge University Press, 2000）",
          "Edward D. Lazowska, John Zahorjan, G. Scott Graham, Kenneth C. Sevcik《Quantitative System Performance: Computer System Analysis Using Queueing Network Models》（Prentice Hall, 1984）"
    ],
    chapters: [
      "性能指标（响应时间/吞吐量/利用率/加速比）",
      "测量技术与工具（计数器/剖析器/追踪）",
      "工作负载特征化",
      "基准测试体系（SPEC/TPC）",
      "实验设计与方差分析",
      "性能数据的统计分析与置信区间",
      "模拟与仿真方法",
      "排队论模型基础（M/M/1、Little 定律）",
      "开排队网络与闭排队网络",
      "均值分析（MVA）算法",
      "容量规划与性能预测",
      "常见性能评估误区与陷阱"
    ],
  },
  'cs/game-development': {
    title: "游戏引擎与游戏开发",
    books: [
          "Jason Gregory《Game Engine Architecture》（CRC Press, 3rd ed., 2018）",
          "Ian Millington《AI for Games》（CRC Press, 3rd ed., 2019）",
          "David H. Eberly《3D Game Engine Design: A Practical Approach to Real-Time Computer Graphics》（CRC Press, 2nd ed., 2006）"
    ],
    chapters: [
      "游戏引擎总体架构与主循环",
      "实时渲染管线",
      "游戏数学（向量/矩阵/四元数）",
      "物理引擎与碰撞检测",
      "动画系统（骨骼动画/动画混合）",
      "音频系统",
      "游戏 AI（A* 寻路/行为树/效用系统）",
      "场景管理与空间数据结构（BVH/八叉树）",
      "资源系统与资产管线",
      "脚本系统与游戏逻辑",
      "多人游戏网络同步架构",
      "工具链与关卡编辑器"
    ],
  },
  'cs/discrete-event-simulation': {
    title: "离散事件系统仿真",
    books: [
          "Jerry Banks, John S. Carson, Barry L. Nelson, David M. Nicol《Discrete-Event System Simulation》（Pearson, 5th ed., 2010）",
          "Averill M. Law《Simulation Modeling and Analysis》（McGraw-Hill, 5th ed., 2014）",
          "Bernard P. Zeigler, Herbert Praehofer, Tag Gon Kim《Theory of Modeling and Simulation》（Academic Press, 2nd ed., 2000）"
    ],
    chapters: [
      "建模与仿真的基本概念与分类",
      "离散事件仿真原理（事件调度/活动扫描/过程交互）",
      "随机数生成与统计检验",
      "随机变量生成（逆变换法/拒绝采样法）",
      "输入数据分析与概率分布拟合",
      "排队系统仿真",
      "库存与制造系统仿真案例",
      "输出数据分析（终止型/稳态型仿真）",
      "方差缩减技术",
      "仿真的验证、确认与认证（VV&A）",
      "DEVS 形式化体系",
      "基于智能体的仿真（Agent-based Simulation）"
    ],
  },
  'intermediate/geodynamics': {
    title: "地球动力学（Geodynamics）",
    books: [
          "Turcotte & Schubert, Geodynamics, Cambridge University Press, 3rd ed., 2014",
          "Schubert, Turcotte & Olson, Mantle Convection in the Earth and Planets, Cambridge University Press, 2001",
          "Fowler, The Solid Earth: An Introduction to Global Geophysics, Cambridge University Press, 2nd ed., 2005"
    ],
    chapters: [
      "板块构造的定量框架",
      "岩石圈弯曲与挠曲均衡",
      "热传导与地球热史",
      "地幔对流的流体动力学方程",
      "瑞利数与对流形态",
      "俯冲带的力学与热结构",
      "地幔柱与热点动力学",
      "洋中脊与海底扩张动力学",
      "大陆岩石圈流变学",
      "造山带动力学与大陆碰撞",
      "后冰期回弹与黏弹性地球",
      "重力、应力场与地球形状",
      "核幔边界与地核动力学"
    ],
  },
  'intermediate/hydrogeology': {
    title: "水文地质学（地下水科学）",
    books: [
          "Freeze & Cherry, Groundwater, Prentice-Hall, 1979",
          "Fetter, Applied Hydrogeology, Pearson, 4th ed., 2000",
          "Domenico & Schwartz, Physical and Chemical Hydrogeology, Wiley, 2nd ed., 1997"
    ],
    chapters: [
      "孔隙度、渗透性与含水层",
      "达西定律与地下水流方程",
      "水头、势函数与流网",
      "非饱和带与入渗",
      "井流力学（Theis/Jacob 方法）",
      "抽水试验与参数反演",
      "地下水化学与水文地球化学过程",
      "溶质运移与弥散",
      "污染物运移与场地修复",
      "地下水数值模拟（MODFLOW 思想）",
      "区域地下水流系统（Tóth 理论）",
      "地下水补给、资源评价与管理",
      "岩溶水文地质与裂隙介质"
    ],
  },
  'intermediate/applied-geophysics': {
    title: "应用地球物理学（勘探地球物理）",
    books: [
          "Telford, Geldart & Sheriff, Applied Geophysics, Cambridge University Press, 2nd ed., 1990",
          "Kearey, Brooks & Hill, An Introduction to Geophysical Exploration, Blackwell Science, 3rd ed., 2002",
          "Reynolds, An Introduction to Applied and Environmental Geophysics, Wiley, 2nd ed., 2011"
    ],
    chapters: [
      "重力勘探与密度结构反演",
      "磁法勘探与磁异常解释",
      "直流电法与激发极化法",
      "大地电磁测深（MT）",
      "地震反射法原理与采集",
      "地震资料处理（叠加/偏移）",
      "地震折射法与广角反射",
      "测井方法（电/声/放射性测井）",
      "浅层与工程地球物理（GPR）",
      "航空与地面电磁法",
      "微震监测与非常规勘探",
      "海洋地球物理勘探",
      "综合物探解释与油藏地球物理"
    ],
  },
  'advanced/geophysical-inverse-theory': {
    title: "地球物理反演理论",
    books: [
          "Menke, Geophysical Data Analysis: Discrete Inverse Theory, Academic Press, 3rd ed., 2012",
          "Tarantola, Inverse Problem Theory and Methods for Model Parameter Estimation, SIAM, 2005",
          "Aster, Borchers & Thurber, Parameter Estimation and Inverse Problems, Academic Press, 3rd ed., 2019"
    ],
    chapters: [
      "正问题与不适定反问题",
      "最小二乘与线性反演",
      "模型分辨矩阵与协方差",
      "奇异值分解与广义逆",
      "正则化（Tikhonov）与折衷曲线",
      "非线性反问题与局部线性化",
      "贝叶斯反演与后验分布",
      "蒙特卡洛与马尔可夫链采样",
      "层析成像（Radon 变换/旅行时反演）",
      "全波形反演（FWI）概念",
      "电磁与重力联合反演",
      "不确定度量化",
      "机器学习辅助反演"
    ],
  },
  'intermediate/geostatistics': {
    title: "地统计学（空间统计学）",
    books: [
          "Journel & Huijbregts, Mining Geostatistics, Academic Press, 1978",
          "Cressie, Statistics for Spatial Data, Wiley, 1993",
          "Chilès & Delfiner, Geostatistics: Modeling Spatial Uncertainty, Wiley, 2nd ed., 2012"
    ],
    chapters: [
      "区域化变量与二阶平稳假设",
      "变异函数（半方差）估计与拟合",
      "克里金（普通/简单/泛克里金）",
      "协克里金与多变量估计",
      "指示克里金与非参数方法",
      "储量估算与品位插值",
      "随机模拟（序贯高斯模拟）",
      "点过程与空间点模式",
      "空间自回归与格点数据模型",
      "地统计在土壤制图与污染评估中的应用",
      "地统计与 GIS 集成",
      "不确定性传播与风险评估"
    ],
  },
  'intermediate/basin-analysis': {
    title: "沉积盆地分析",
    books: [
          "Allen & Allen, Basin Analysis: Principles and Application to Petroleum Play Assessment, Wiley-Blackwell, 3rd ed., 2013",
          "Miall, The Geology of Fluvial Deposits, Springer, 1996",
          "Einsele, Sedimentary Basins: Evolution, Facies, and Sediment Budget, Springer, 2nd ed., 2000"
    ],
    chapters: [
      "盆地分类（McKenzie 拉张/Wilson 旋回）",
      "岩石圈伸展与裂谷盆地",
      "前陆盆地与挠曲沉降",
      "走滑与克拉通盆地",
      "沉积充填与层序格架",
      "沉降史反演与回剥分析",
      "盆地热流与热史",
      "成熟度与烃源岩评价",
      "沉积物源—汇系统",
      "盆地模拟方法",
      "被动边缘与盐构造",
      "中国含油气盆地实例"
    ],
  },
  'intermediate/petroleum-geology': {
    title: "石油地质学",
    books: [
          "Gluyas & Swennen, Petroleum Geoscience, Blackwell Publishing, 2004",
          "柳广弟等，《石油地质学》，石油工业出版社，第4版，2009",
          "Selley & Sonnenberg, Elements of Petroleum Geology, Academic Press, 3rd ed., 2014"
    ],
    chapters: [
      "油气成因与烃源岩",
      "生烃动力学与成熟度",
      "储集层物理与储层类型",
      "盖层与封闭机制",
      "圈闭分类与油气藏类型",
      "油气运移（初次/二次）",
      "成藏组合与含油气系统",
      "非常规油气（页岩油气/致密油/煤层气）",
      "油气勘探方法与经济评价",
      "油藏描述与地质建模",
      "测井与地震在油气评价中的应用",
      "天然气水合物"
    ],
  },
  'intermediate/isotope-geochemistry': {
    title: "同位素地球化学",
    books: [
          "White, Isotope Geochemistry, Wiley, 2015",
          "Faure & Mensing, Isotopes: Principles and Applications, Wiley, 3rd ed., 2005",
          "Rollinson, Using Geochemical Data, Routledge, 2nd ed., 2014"
    ],
    chapters: [
      "同位素丰度与δ值表示",
      "质量相关/质量无关分馏",
      "平衡与动力学分馏",
      "放射性衰变体系（Rb-Sr/Sm-Nd/U-Pb/Lu-Hf）",
      "放射成因同位素示踪地幔源区",
      "稳定同位素（H/O/C/S/N）",
      "氧同位素与古温度",
      "非传统稳定同位素（Fe/Mo/Li）",
      "宇宙成因核素与暴露测年",
      "同位素质谱（TIMS/MC-ICP-MS）",
      "壳幔循环的同位素证据",
      "环境与气候研究中的同位素"
    ],
  },
  'intermediate/atmospheric-radiation': {
    title: "大气辐射学",
    books: [
          "Liou, An Introduction to Atmospheric Radiation, Academic Press, 2nd ed., 2002",
          "Petty, A First Course in Atmospheric Radiation, Sundog Publishing, 2nd ed., 2006",
          "Thomas & Stamnes, Radiative Transfer in the Atmosphere and Ocean, Cambridge University Press, 2nd ed., 2017"
    ],
    chapters: [
      "黑体辐射与普朗克定律",
      "比尔—布格—朗伯定律",
      "分子吸收光谱（H2O/CO2/O3）",
      "加宽机制与谱线形状",
      "辐射传输方程",
      "二流近似与离散纵标法",
      "瑞利散射与米氏散射",
      "气溶胶辐射效应",
      "云的光学性质",
      "温室效应的定量计算",
      "卫星遥感反演原理",
      "地表辐射收支与能量平衡"
    ],
  },
  'intermediate/boundary-layer-meteorology': {
    title: "大气边界层气象学",
    books: [
          "Stull, An Introduction to Boundary Layer Meteorology, Kluwer Academic, 1988",
          "Arya, Introduction to Micrometeorology, Academic Press, 2nd ed., 2001",
          "Garratt, The Atmospheric Boundary Layer, Cambridge University Press, 1992"
    ],
    chapters: [
      "湍流的统计描述与谱",
      "湍动能收支方程",
      "通量—梯度关系与莫宁—奥布霍夫相似理论",
      "近地面层结构",
      "对流边界层与混合层增长",
      "稳定边界层",
      "边界层云与夹卷",
      "陆面—大气相互作用与能量收支",
      "城市边界层",
      "海洋大气边界层",
      "大涡模拟（LES）",
      "湍流观测技术（涡度相关）"
    ],
  },
  'intermediate/mesoscale-meteorology': {
    title: "中尺度气象学",
    books: [
          "Markowski & Richardson, Mesoscale Meteorology in Midlatitudes, Wiley-Blackwell, 2010",
          "Lackmann, Midlatitude Synoptic Meteorology, American Meteorological Society, 2011",
          "Trapp, Mesoscale-Convective Processes in the Atmosphere, Cambridge University Press, 2013"
    ],
    chapters: [
      "中尺度系统的尺度分类与观测",
      "浮力、CAPE 与热力学图解",
      "深对流的启动机制",
      "超级单体风暴动力学",
      "飑线与中尺度对流系统（MCS）",
      "龙卷动力学",
      "中尺度对流涡旋（MCV）",
      "地形降水与山谷风",
      "海陆风环流",
      "锋面动力学与雨带",
      "重力波与对流耦合",
      "中尺度数值模拟（WRF）"
    ],
  },
  'advanced/numerical-weather-prediction': {
    title: "数值天气预报与资料同化",
    books: [
          "Kalnay, Atmospheric Modeling, Data Assimilation and Predictability, Cambridge University Press, 2003",
          "Coiffier, Fundamentals of Numerical Weather Prediction, Cambridge University Press, 2011",
          "Warner, Numerical Weather and Climate Prediction, Cambridge University Press, 2011"
    ],
    chapters: [
      "控制方程组与滤波近似",
      "水平离散化（谱方法/格点）",
      "时间积分格式与稳定性（CFL）",
      "半拉格朗日方法",
      "物理过程参数化（积云/微物理/边界层）",
      "初始条件与客观分析",
      "最优插值与三维变分（3D-Var）",
      "四维变分（4D-Var）",
      "集合卡尔曼滤波（EnKF）",
      "观测系统（卫星辐射率同化）",
      "集合预报与可预报性",
      "次季节—季节预测",
      "机器学习天气预报模型"
    ],
  },
  'intermediate/cloud-precipitation-physics': {
    title: "云降水物理学",
    books: [
          "Rogers & Yau, A Short Course in Cloud Physics, Butterworth-Heinemann, 3rd ed., 1989",
          "Pruppacher & Klett, Microphysics of Clouds and Precipitation, Kluwer Academic, 2nd ed., 1997",
          "Wallace & Hobbs, Atmospheric Science: An Introductory Survey（云物理章）, Academic Press, 2nd ed., 2006"
    ],
    chapters: [
      "湿空气热力学与假绝热过程",
      "核化理论与云凝结核",
      "科勒曲线与液滴活化",
      "云滴凝结增长",
      "碰并增长与随机碰并",
      "冰晶形态与贝吉隆过程",
      "降水粒子（雪/霰/雹）",
      "云的微物理观测",
      "云滴谱与雷达定量估测",
      "暖云/冷云降水机制",
      "云在气候系统中的作用",
      "人工增雨与播云原理"
    ],
  },
  'intermediate/upper-atmosphere-physics': {
    title: "中高层大气物理学（含电离层）",
    books: [
          "Kelley, The Earth's Ionosphere: Plasma Physics and Electrodynamics, Academic Press, 2nd ed., 2009",
          "Rees, Physics and Chemistry of the Upper Atmosphere, Cambridge University Press, 1989",
          "Brasseur & Solomon, Aeronomy of the Middle Atmosphere, Springer, 3rd ed., 2005"
    ],
    chapters: [
      "中性大气的垂直结构",
      "平流层臭氧化学",
      "行星波与平流层爆发性增温",
      "中间层与极地中间层云",
      "热层结构与逃逸层",
      "大气潮汐与重力波上传",
      "电离层的形成与分层（D/E/F区）",
      "电离层电动力学与发电机区",
      "电离层暴与闪烁",
      "无线电波传播与通信影响",
      "中层大气光化学（气辉）",
      "中层—低层耦合"
    ],
  },
  'foundations/limnology': {
    title: "湖泊学（内陆水体科学）",
    books: [
          "Wetzel, Limnology: Lake and River Ecosystems, Academic Press, 3rd ed., 2001",
          "Kalff, Limnology: Inland Water Ecosystems, Prentice Hall, 2002",
          "Dodds & Whiles, Freshwater Ecology: Concepts and Environmental Applications of Limnology, Academic Press, 3rd ed., 2019"
    ],
    chapters: [
      "湖泊的形态测量与成因类型",
      "湖水的光热结构与分层",
      "湖泊水动力（湖震/环流）",
      "河流的水文与地貌",
      "淡水化学与离子组成",
      "营养盐循环（N/P/Si）",
      "初级生产与浮游生物",
      "富营养化机理与控制",
      "湖泊食物网与下行/上行效应",
      "沉积记录与湖泊演化",
      "湖泊与气候变化",
      "湿地与河口过渡带",
      "流域管理与湖泊修复"
    ],
  },
  'intermediate/cartography': {
    title: "地图学",
    books: [
          "Slocum et al., Thematic Cartography and Geovisualization, Pearson Prentice Hall, 3rd ed., 2009",
          "Kraak & Ormeling, Cartography: Visualization of Geospatial Data, CRC Press, 4th ed., 2020",
          "王家耀等，《地图学原理与方法》，科学出版社，2006"
    ],
    chapters: [
      "地球椭球体与地图数学基础",
      "地图投影的分类与变形",
      "常用投影（墨卡托/兰伯特/等积）",
      "地图符号学与视觉变量",
      "地图概括（综合）",
      "专题地图编制方法",
      "色彩理论与制图配色",
      "地形图与等高线表示",
      "数字制图与自动综合",
      "地理可视化与虚拟地球",
      "地图认知与可用性评价",
      "地图史与制图文化"
    ],
  },
  'advanced/meteoritics-cosmochemistry': {
    title: "陨石学与宇宙化学",
    books: [
          "McSween & Huss, Cosmochemistry, Cambridge University Press, 2010",
          "Lodders & Fegley, The Planetary Scientist's Companion, Oxford University Press, 1998",
          "Palme & Jones, Treatise on Geochemistry, Volume 1: Meteorites, Comets, and Planets, Elsevier, 2nd ed., 2014"
    ],
    chapters: [
      "元素与同位素的宇宙丰度",
      "核合成（恒星/超新星/宇宙线）",
      "太阳星云凝聚序列",
      "球粒陨石与CAI包体",
      "无球粒陨石与行星分异",
      "铁陨石与行星核",
      "前太阳颗粒（presolar grains）",
      "短寿命核素与太阳系年表",
      "小行星与彗星采样返回",
      "月球岩石与月壤",
      "火星陨石",
      "地外有机物与生命前化学",
      "撞击过程与冲击变质"
    ],
  },
  'advanced/climate-earth-system-modeling': {
    title: "气候与地球系统数值模拟",
    books: [
          "Goosse, Climate System Dynamics and Modelling, Cambridge University Press, 2015",
          "McGuffie & Henderson-Sellers, A Climate Modelling Primer, Wiley, 4th ed., 2014",
          "Washington & Parkinson, An Introduction to Three-Dimensional Climate Modeling, University Science Books, 2nd ed., 2005"
    ],
    chapters: [
      "气候模式的谱系（EBM→GCM→ESM）",
      "大气环流模式的动力框架",
      "海洋环流模式（坐标与混合参数化）",
      "海冰模式",
      "陆面过程模式",
      "碳循环与生物地球化学模块",
      "气溶胶与大气化学模块",
      "耦合技术与通量校正",
      "初始漂移与模式气候态",
      "敏感性试验与反馈分析",
      "CMIP 框架与情景试验",
      "降尺度（动力/统计）",
      "模式评估与不确定度",
      "地球系统模式的机器学习组件"
    ],
  },
  'humanities/history-of-earth-science': {
    title: "地球科学史",
    books: [
          "Gohau, A History of Geology, Rutgers University Press, 1990",
          "O'Hara, A Brief History of Geology, Cambridge University Press, 2018",
          "Oldroyd, Thinking About the Earth: A History of Ideas in Geology, Harvard University Press, 1996"
    ],
    chapters: [
      "古代的地学思想",
      "文艺复兴时期的化石之争",
      "水成论与火成论（维尔纳/赫顿）",
      "居维叶与灾变论",
      "莱伊尔与均变论",
      "地质年代表的建立",
      "达尔文与地质学的互动",
      "魏格纳与大陆漂移",
      "海底扩张与板块构造革命",
      "冰期理论的形成",
      "放射性测年与地球年龄之争",
      "深空时代：阿波罗与行星科学的诞生",
      "气象学史（从温度计到数值预报）"
    ],
  },
  'engineering/electric-circuit-analysis': {
    title: "电路分析基础（电路原理）",
    books: [
          "邱关源, 《电路》第5版, 高等教育出版社, 2006",
          "Charles K. Alexander & Matthew N.O. Sadiku, \"Fundamentals of Electric Circuits\" (6th ed., McGraw-Hill, 2017)",
          "William H. Hayt, Jack E. Kemmerly, Steven M. Durbin, \"Engineering Circuit Analysis\" (8th ed., McGraw-Hill, 2012)"
    ],
    chapters: [
      "电路模型与基本变量（电压/电流/功率）",
      "基尔霍夫定律与电阻等效变换",
      "线性电阻电路分析：节点法与网孔法",
      "电路定理：叠加/戴维宁/诺顿/最大功率传输",
      "运算放大器与含受控源电路",
      "一阶电路（RC/RL）时域分析",
      "二阶电路与暂态响应",
      "正弦稳态分析与相量法",
      "正弦稳态功率与三相电路",
      "频率响应、谐振与滤波器",
      "互感与变压器电路",
      "拉普拉斯变换与网络函数",
      "二端口网络参数"
    ],
  },
  'engineering/analog-digital-electronics': {
    title: "模拟与数字电子技术（电子线路基础）",
    books: [
          "童诗白、华成英, 《模拟电子技术基础》第5版, 高等教育出版社, 2015",
          "阎石, 《数字电子技术基础》第6版, 高等教育出版社, 2016",
          "Adel S. Sedra & Kenneth C. Smith, \"Microelectronic Circuits\" (7th ed., Oxford University Press, 2015)"
    ],
    chapters: [
      "半导体器件：二极管/BJT/MOSFET 特性与模型",
      "基本放大电路与共射/共集/共基组态",
      "多级放大与频率响应",
      "差分放大与电流源",
      "负反馈放大电路",
      "集成运放与信号运算/处理电路",
      "功率放大与直流稳压电源",
      "数制与逻辑代数基础",
      "门电路与组合逻辑设计",
      "触发器与时序逻辑电路",
      "计数器/寄存器/存储器与可编程逻辑器件",
      "脉冲产生整形与 ADC/DAC"
    ],
  },
  'engineering/electromagnetic-fields-waves': {
    title: "电磁场与电磁波（工程电磁场）",
    books: [
          "David K. Cheng, \"Field and Wave Electromagnetics\" (2nd ed., Addison-Wesley, 1989)",
          "谢处方、饶克谨, 《电磁场与电磁波》第4版, 高等教育出版社, 2006",
          "Constantine A. Balanis, \"Advanced Engineering Electromagnetics\" (2nd ed., Wiley, 2012)"
    ],
    chapters: [
      "矢量分析与场论基础",
      "静电场与恒定电场",
      "恒定磁场与电感",
      "时变电磁场与 Maxwell 方程组",
      "平面电磁波在理想介质/导电媒质中的传播",
      "波的反射、折射与极化",
      "导行电磁波：TEM/TE/TM 模与传输线理论",
      "矩形/圆波导与谐振腔",
      "电磁辐射与天线基础",
      "准静态场与工程电磁场数值方法简介"
    ],
  },
  'engineering/engineering-thermodynamics': {
    title: "工程热力学",
    books: [
          "沈维道、童钧耕, 《工程热力学》第5版, 高等教育出版社, 2016",
          "Yunus A. Çengel & Michael A. Boles, \"Thermodynamics: An Engineering Approach\" (9th ed., McGraw-Hill, 2019)",
          "Michael J. Moran & Howard N. Shapiro, \"Fundamentals of Engineering Thermodynamics\" (9th ed., Wiley, 2018)"
    ],
    chapters: [
      "基本概念与热力学第零定律/温标",
      "热力学第一定律与能量方程",
      "理想气体与实际气体性质",
      "热力学第二定律与卡诺循环",
      "熵与㶲（可用能）分析",
      "水蒸气与湿空气性质",
      "气体动力循环：Otto/Diesel/Brayton",
      "蒸汽动力循环：朗肯循环与再热回热",
      "制冷与热泵循环",
      "混合气体与化学热力学初步",
      "喷管与压气机中的流动"
    ],
  },
  'engineering/heat-and-mass-transfer': {
    title: "传热学（含传质基础）",
    books: [
          "杨世铭、陶文铨, 《传热学》第4版, 高等教育出版社, 2006",
          "Frank P. Incropera & David P. DeWitt, \"Fundamentals of Heat and Mass Transfer\" (7th ed., Wiley, 2011)",
          "J.P. Holman, \"Heat Transfer\" (10th ed., McGraw-Hill, 2010)"
    ],
    chapters: [
      "导热基本定律与稳态导热",
      "非稳态导热与集总参数法",
      "导热数值解法初步",
      "对流换热原理与边界层理论",
      "管内强制对流与外掠换热",
      "自然对流换热",
      "凝结与沸腾换热",
      "热辐射基本定律与黑体辐射",
      "灰体间辐射换热与角系数",
      "换热器设计与分析（LMTD/ε-NTU）",
      "传质基础与类比律"
    ],
  },
  'engineering/systems-engineering': {
    title: "系统工程",
    books: [
          "Benjamin S. Blanchard & Wolter J. Fabrycky, \"Systems Engineering and Analysis\" (5th ed., Pearson, 2011)",
          "INCOSE, \"Systems Engineering Handbook\" (4th ed., Wiley, 2015)",
          "汪应洛, 《系统工程》第4版, 机械工业出版社, 2008"
    ],
    chapters: [
      "系统与系统工程概念、生命周期模型",
      "系统方法论：霍尔三维结构与硬/软系统方法",
      "需求分析与利益相关者分析",
      "功能分析与系统体系结构设计",
      "V 模型与基于模型的系统工程（MBSE/SysML）",
      "接口管理与技术状态管理",
      "系统权衡分析与多准则决策",
      "可靠性/维修性/保障性（RMS）工程",
      "系统建模与仿真",
      "验证、确认与系统测试",
      "系统评价与层次分析法（AHP）",
      "大型工程项目中的系统工程案例（航天/国防）"
    ],
  },
  'engineering/aerodynamics': {
    title: "空气动力学",
    books: [
          "John D. Anderson, \"Fundamentals of Aerodynamics\" (6th ed., McGraw-Hill, 2017)",
          "钱翼稷, 《空气动力学》, 北京航空航天大学出版社, 2004",
          "John D. Anderson, \"Introduction to Flight\" (8th ed., McGraw-Hill, 2016)"
    ],
    chapters: [
      "空气动力学基本原理与标准大气",
      "流体力学基础：连续/动量/能量方程",
      "无黏不可压流与位流理论",
      "翼型绕流与库塔-茹科夫斯基定理",
      "有限翼展机翼与升力线理论",
      "黏性流动与边界层",
      "可压缩流基础与激波/膨胀波",
      "高速翼型与临界马赫数",
      "亚/跨/超声速机翼气动特性",
      "阻力构成与减阻技术",
      "风洞试验与气动相似准则",
      "计算流体力学（CFD）入门"
    ],
  },
  'engineering/computational-mechanics': {
    title: "计算力学与有限元方法",
    books: [
          "O.C. Zienkiewicz, R.L. Taylor & J.Z. Zhu, \"The Finite Element Method: Its Basis and Fundamentals\" (7th ed., Butterworth-Heinemann, 2013)",
          "王勖成, 《有限单元法》, 清华大学出版社, 2003",
          "Klaus-Jürgen Bathe, \"Finite Element Procedures\" (2nd ed., 2014)"
    ],
    chapters: [
      "弹性力学基本方程（应力/应变/本构）",
      "能量原理与变分法（最小势能/虚功原理）",
      "加权残值法与 Galerkin 方法",
      "一维杆/梁单元直接刚度法",
      "平面问题的三角形/四边形单元",
      "等参数单元与数值积分",
      "轴对称与三维实体单元",
      "板壳单元（Kirchhoff/Mindlin）",
      "结构动力学有限元（模态/瞬态）",
      "材料非线性与几何非线性",
      "接触问题与有限元软件实践（ANSYS/Abaqus）",
      "误差估计与网格自适应"
    ],
  },
  'engineering/mechanical-vibration': {
    title: "机械振动（振动工程基础）",
    books: [
          "Singiresu S. Rao, \"Mechanical Vibrations\" (6th ed., Pearson, 2017)",
          "William T. Thomson & Marie Dillon Dahleh, \"Theory of Vibration with Applications\" (5th ed., 1998)",
          "倪振华, 《振动力学》, 西安交通大学出版社, 1989"
    ],
    chapters: [
      "单自由度系统自由振动",
      "单自由度强迫振动与频响函数",
      "隔振与振动测量",
      "多自由度系统：固有频率与振型",
      "模态分析与振型叠加法",
      "连续系统振动：弦/杆/梁",
      "近似方法：Rayleigh 法与 Ritz 法",
      "转子动力学基础与临界转速",
      "随机振动基础",
      "非线性振动初步",
      "振动试验与模态测试技术",
      "振动主动/被动控制简介"
    ],
  },
  'engineering/communication-principles': {
    title: "通信原理",
    books: [
          "樊昌信、曹丽娜, 《通信原理》第7版, 国防工业出版社, 2012",
          "John G. Proakis & Masoud Salehi, \"Digital Communications\" (5th ed., McGraw-Hill, 2008)",
          "Simon Haykin & Michael Moher, \"Communication Systems\" (5th ed., Wiley, 2009)"
    ],
    chapters: [
      "通信系统模型与信道容量（香农公式）",
      "确知信号与随机过程基础",
      "模拟调制：AM/DSB/SSB/FM/PM",
      "模拟信号数字化：抽样/量化/PCM",
      "数字基带传输与码间串扰（Nyquist 准则）",
      "数字带通调制：ASK/FSK/PSK/QAM",
      "最佳接收与匹配滤波器",
      "信道编码：分组码/卷积码/Turbo/LDPC",
      "同步原理（载波/位/帧同步）",
      "衰落信道与分集/均衡",
      "多路复用与多址技术（FDM/TDM/CDMA）",
      "现代通信系统案例（蜂窝/卫星）"
    ],
  },
  'engineering/microwave-and-antennas': {
    title: "微波技术与天线",
    books: [
          "David M. Pozar, \"Microwave Engineering\" (4th ed., Wiley, 2012)",
          "Constantine A. Balanis, \"Antenna Theory: Analysis and Design\" (4th ed., Wiley, 2016)",
          "梁昌洪, 《简明微波》, 高等教育出版社, 2006"
    ],
    chapters: [
      "传输线理论与 Smith 圆图",
      "微波网络与 S 参数",
      "阻抗匹配与调谐",
      "波导与微波传输线（微带/带状线）",
      "微波无源器件：耦合器/功分器/滤波器",
      "微波有源电路与低噪声放大器基础",
      "天线基本参数（增益/方向图/极化）",
      "线天线与口径天线",
      "天线阵与相控阵原理",
      "微带天线与反射面天线",
      "微波测量技术"
    ],
  },
  'engineering/combustion': {
    title: "燃烧学",
    books: [
          "Stephen R. Turns, \"An Introduction to Combustion: Concepts and Applications\" (3rd ed., McGraw-Hill, 2012)",
          "Irvin Glassman, Richard A. Yetter & Nick G. Glumac, \"Combustion\" (5th ed., Academic Press, 2015)",
          "Chung K. Law, \"Combustion Physics\" (Cambridge University Press, 2006)"
    ],
    chapters: [
      "燃烧热化学与绝热火焰温度",
      "化学动力学与链式反应机理",
      "传质与守恒方程（Shvab-Zeldovich）",
      "预混气体层流火焰与火焰传播速度",
      "预混火焰稳定与着火/熄火",
      "扩散火焰与液滴蒸发燃烧",
      "湍流燃烧基础",
      "气体爆炸与爆轰",
      "固体燃料（煤/生物质）燃烧",
      "污染物生成与控制（NOx/碳烟/CO）",
      "内燃机与燃气轮机中的燃烧",
      "燃烧诊断技术"
    ],
  },
  'engineering/nuclear-reactor-engineering': {
    title: "核反应堆工程（反应堆物理与热工水力）",
    books: [
          "John R. Lamarsh & Anthony J. Baratta, \"Introduction to Nuclear Engineering\" (4th ed., Pearson, 2017)",
          "谢仲生, 《核反应堆物理分析》（修订本/第5版）, 西安交通大学出版社, 2004",
          "于平安 等, 《核反应堆热工分析》第3版, 上海交通大学出版社, 2002"
    ],
    chapters: [
      "核物理基础：中子与原子核相互作用",
      "链式裂变反应与反应堆类型",
      "中子慢化与扩散理论",
      "反应堆临界理论与几何/材料曲率",
      "非均匀堆与栅格计算",
      "反应性系数与燃耗",
      "反应性控制（控制棒/硼浓度）",
      "反应堆动力学（点堆方程/缓发中子）",
      "堆芯热工水力：冷却剂流动与传热",
      "核燃料循环与燃料管理",
      "反应堆安全分析（LOCA/纵深防御）",
      "压水堆/沸水堆/快堆/高温气冷堆系统"
    ],
  },
  'engineering/aircraft-design': {
    title: "飞行器设计",
    books: [
          "Daniel P. Raymer, \"Aircraft Design: A Conceptual Approach\" (6th ed., AIAA, 2018)",
          "Jan Roskam, \"Airplane Design\" (Parts I-VIII, DARcorporation)",
          "Egbert Torenbeek, \"Synthesis of Subsonic Airplane Design\" (Delft University Press, 1982)"
    ],
    chapters: [
      "设计流程：概念/初步/详细设计",
      "需求分析与设计指标（任务剖面）",
      "重量估算与重心控制",
      "推重比与翼载选择、总体参数权衡",
      "气动布局设计（机翼/尾翼/机身）",
      "推进系统选型与一体化",
      "结构与材料选型",
      "稳定性与操纵性设计",
      "性能分析：起飞/爬升/巡航/着陆",
      "载荷、起落架与系统布置",
      "设计权衡、优化与造价分析",
      "无人机/高超声速飞行器设计特点"
    ],
  },
  'engineering/inorganic-nonmetallic-materials': {
    title: "无机非金属材料工程（陶瓷/玻璃/水泥）",
    books: [
          "W.D. Kingery, H.K. Bowen & D.R. Uhlmann, \"Introduction to Ceramics\" (2nd ed., Wiley, 1976)",
          "Michel W. Barsoum, \"Fundamentals of Ceramics\" (2nd ed., CRC Press, 2019)",
          "陆佩文, 《无机材料科学基础》, 武汉工业大学出版社, 1996"
    ],
    chapters: [
      "陶瓷晶体结构与硅酸盐化学",
      "玻璃结构与非晶态形成",
      "陶瓷中的缺陷与扩散",
      "相图与陶瓷相平衡",
      "烧结机理与致密化动力学",
      "陶瓷粉体制备与成型工艺",
      "结构陶瓷（氧化物/氮化物/碳化物）",
      "功能陶瓷（介电/压电/铁电/磁性）",
      "玻璃制备与加工工艺",
      "水泥与混凝土化学（硅酸盐水泥水化）",
      "耐火材料",
      "陶瓷的力学性能与脆性断裂"
    ],
  },
  'engineering/disaster-prevention-mitigation': {
    title: "防灾减灾工程与防护工程",
    books: [
          "Anil K. Chopra, \"Dynamics of Structures: Theory and Applications to Earthquake Engineering\" (5th ed., Pearson, 2020)",
          "Ray W. Clough & Joseph Penzien, \"Dynamics of Structures\" (3rd ed., Computers & Structures, 1995)",
          "李爱群、丁幼亮, 《工程结构抗震设计》, 中国建筑工业出版社, 2018"
    ],
    chapters: [
      "地震工程基础：震源/震级/地震动参数",
      "单自由度体系地震反应与反应谱",
      "多自由度体系振型分解反应谱法",
      "结构抗震概念设计与延性设计",
      "隔震与消能减震技术",
      "工程结构抗风设计（风荷载/风振）",
      "建筑防火与结构抗火",
      "爆炸冲击与结构抗爆防护",
      "地基基础抗震与场地效应",
      "桥梁抗震与生命线工程",
      "灾害风险评估与韧性城市",
      "工程结构检测、鉴定与加固"
    ],
  },
  'engineering/human-factors-ergonomics': {
    title: "人因工程（工效学）",
    books: [
          "Gavriel Salvendy & Waldemar Karwowski (eds.), \"Handbook of Human Factors and Ergonomics\" (5th ed., Wiley, 2021)",
          "Christopher D. Wickens et al., \"Engineering Psychology and Human Performance\" (5th ed., Routledge, 2021)",
          "郭伏、钱省三, 《人因工程学》, 机械工业出版社, 2006"
    ],
    chapters: [
      "人因工程概念与人-机-环境系统",
      "人体测量学与工作空间设计",
      "感觉与知觉（视觉/听觉显示设计）",
      "人的信息加工与决策",
      "人体生物力学与体力作业",
      "工作负荷与疲劳",
      "控制器与显示器的人机界面设计",
      "人为差错与可靠性",
      "作业环境：照明/噪声/微气候",
      "安全工效学与事故预防",
      "认知工效学与自动化中的人因",
      "可用性评估与用户体验测试"
    ],
  },
  'engineering/engineering-management-economics': {
    title: "工程管理与工程经济",
    books: [
          "William G. Sullivan, Elin M. Wicks & C. Patrick Koelling, \"Engineering Economy\" (17th ed., Pearson, 2019)",
          "Project Management Institute, \"A Guide to the Project Management Body of Knowledge (PMBOK Guide)\" (7th ed., PMI, 2021)",
          "Jack R. Meredith & Samuel J. Mantel, \"Project Management: A Managerial Approach\" (9th ed., Wiley, 2015)"
    ],
    chapters: [
      "工程经济基本原理与资金时间价值",
      "投资方案评价（NPV/IRR/回收期）",
      "设备更新与折旧、不确定性分析",
      "项目可行性研究",
      "工程项目组织与治理结构",
      "项目范围与进度管理（WBS/CPM/PERT）",
      "项目成本管理与挣值分析",
      "工程质量管理（PDCA/全面质量管理）",
      "工程招投标与合同管理（FIDIC 条款）",
      "项目风险管理",
      "工程安全与职业健康管理",
      "BIM 与数字化项目管理"
    ],
  },
  'engineering/tribology': {
    title: "摩擦学",
    books: [
          "Gwidon W. Stachowiak & Andrew W. Batchelor, \"Engineering Tribology\" (4th ed., Butterworth-Heinemann, 2013)",
          "Bharat Bhushan, \"Introduction to Tribology\" (2nd ed., Wiley, 2013)",
          "温诗铸、黄平, 《摩擦学原理》第5版, 清华大学出版社, 2018"
    ],
    chapters: [
      "表面形貌与真实接触面积",
      "摩擦机理与摩擦定律",
      "磨损机制：黏着/磨粒/疲劳/腐蚀磨损",
      "流体动压润滑与 Reynolds 方程",
      "弹性流体动力润滑（EHL）",
      "边界润滑与固体润滑",
      "润滑剂与添加剂",
      "轴承设计（滑动/滚动轴承）",
      "密封与摩擦副材料",
      "摩擦磨损试验方法与失效分析",
      "微纳摩擦学与生物摩擦学"
    ],
  },
  'engineering/sensors-and-detection': {
    title: "传感器与检测技术",
    books: [
          "Jacob Fraden, \"Handbook of Modern Sensors: Physics, Designs, and Applications\" (5th ed., Springer, 2016)",
          "Alan S. Morris & Reza Langari, \"Measurement and Instrumentation: Theory and Application\" (3rd ed., Academic Press, 2020)",
          "John G. Webster & Halit Eren (eds.), \"Measurement, Instrumentation, and Sensors Handbook\" (2nd ed., CRC Press, 2014)"
    ],
    chapters: [
      "测量系统与误差理论、不确定度评定",
      "传感器静态/动态特性与标定",
      "电阻式传感器（应变片/热电阻）",
      "电容式与电感式传感器",
      "压电/磁电/霍尔传感器",
      "热电式与光电式传感器",
      "半导体与 MEMS 传感器",
      "信号调理：电桥/放大/滤波/隔离",
      "数据采集与 ADC/接口总线",
      "虚拟仪器与自动测试系统",
      "智能传感器与物联网传感网络"
    ],
  },
  'engineering/fire-protection-engineering': {
    title: "消防工程",
    books: [
          "SFPE, \"Handbook of Fire Protection Engineering\" (5th ed., Springer, 2016)",
          "Dougal Drysdale, \"An Introduction to Fire Dynamics\" (3rd ed., Wiley, 2011)",
          "Vytenis Babrauskas, \"Ignition Handbook\" (Fire Science Publishers, 2003)"
    ],
    chapters: [
      "燃烧与火灾科学基础（点火/火焰蔓延/轰燃）",
      "烟气运动与烟气控制",
      "火灾探测与自动报警系统",
      "建筑防火设计（防火分区/疏散）",
      "自动灭火系统：喷淋/气体/泡沫",
      "结构抗火与耐火设计",
      "性能化防火设计方法",
      "工业火灾与爆炸防护（泄爆/抑爆）",
      "危化品火灾与储罐区消防",
      "人员疏散模拟与应急照明",
      "森林与草原火灾",
      "火灾调查与风险评估"
    ],
  },
  'engineering/underwater-acoustic-engineering': {
    title: "水声工程",
    books: [
          "Robert J. Urick, \"Principles of Underwater Sound\" (3rd ed., McGraw-Hill, 1983)",
          "刘伯胜、雷家煜, 《水声学原理》第2版, 哈尔滨工程大学出版社, 2010",
          "Paul C. Etter, \"Underwater Acoustic Modeling and Simulation\" (5th ed., CRC Press, 2018)"
    ],
    chapters: [
      "海洋声学环境（声速剖面/声道）",
      "声波在海洋中的传播：射线与简正波理论",
      "声呐方程与声呐参数",
      "海洋混响与环境噪声",
      "声传播起伏与目标强度",
      "水声换能器与基阵（指向性/波束形成）",
      "主动声呐与目标检测",
      "被动声呐与目标参数估计",
      "水声通信与组网",
      "水声定位与导航（USBL/LBL）",
      "匹配场处理与水下目标识别"
    ],
  },
  'engineering/engineering-ethics': {
    title: "工程伦理",
    books: [
          "Mike W. Martin & Roland Schinzinger, \"Ethics in Engineering\" (4th ed., McGraw-Hill, 2005)",
          "李正风、丛杭青、王前 等, 《工程伦理》第2版, 清华大学出版社, 2019",
          "Charles E. Harris et al., \"Engineering Ethics: Concepts and Cases\" (6th ed., Cengage, 2019)"
    ],
    chapters: [
      "工程职业与工程师的角色责任",
      "伦理理论框架（功利/义务/德性）在工程中的应用",
      "工程中的安全、风险与可接受性",
      "工程诚实与学术/职业诚信",
      "利益冲突与保密义务",
      "告发（whistleblowing）与组织 disobedience 案例（挑战者号）",
      "环境与可持续发展责任",
      "信息与人工智能工程的伦理问题",
      "工程师的跨国责任与全球工程规范",
      "工程社团伦理守则（NSPE/IEEE）"
    ],
  },
  'humanities/history-of-technology': {
    title: "工程技术史（技术史）",
    books: [
          "Charles Singer et al. (eds.), \"A History of Technology\" (7 vols., Oxford University Press, 1954-1978)",
          "李约瑟, 《中国科学技术史》, 科学出版社",
          "Donald Cardwell, \"The Norton History of Technology\" (W.W. Norton, 1995)"
    ],
    chapters: [
      "古代技术：农业/冶金/建筑的起源",
      "中国四大发明与传统工程技术体系",
      "希腊罗马技术与中世纪技术复兴",
      "工业革命：蒸汽机与机器制造",
      "电力时代与电气工程的兴起",
      "化学工业与材料革命的历程",
      "交通运输工程史（铁路/汽车/航空）",
      "通信与信息工程史（电报/电话/无线电）",
      "土木与水利工程史（桥梁/大坝/摩天楼）",
      "军事技术与两次世界大战的技术动员",
      "航天与核能工程的诞生",
      "技术转移、创新体系与技术史方法论"
    ],
  },
  'frontier/system-dynamics': {
    title: "系统动力学",
    books: [
          "Sterman《Business Dynamics: Systems Thinking and Modeling for a Complex World》（McGraw-Hill, 2000）",
          "Forrester《Industrial Dynamics》（MIT Press, 1961）",
          "王其藩《系统动力学》（上海财经大学出版社，修订版 2009）"
    ],
    chapters: [
      "系统思考与反馈回路",
      "因果回路图与基模",
      "存量流量图与 Vensim/Stella 建模",
      "一阶与二阶系统动态",
      "S 形增长与寻的结构",
      "振荡、延迟与不稳定结构",
      "模型检验与真实性验证",
      "市场与供应链动力学（牛鞭效应）",
      "人口—资源—环境模型（World3 与增长的极限）",
      "企业动力学与战略微世界",
      "公共政策设计与杠杆点",
      "系统动力学与 ABM/GIS 的集成"
    ],
  },
  'frontier/agent-based-modeling': {
    title: "基于主体的建模与复杂适应系统",
    books: [
          "Wilensky & Rand《An Introduction to Agent-Based Modeling: Modeling Natural, Social, and Engineered Complex Systems with NetLogo》（MIT Press, 2015）",
          "Railsback & Grimm《Agent-Based and Individual-Based Modeling: A Practical Introduction》（Princeton University Press, 2nd ed. 2019）",
          "Epstein & Axtell《Growing Artificial Societies: Social Science from the Bottom Up》（MIT Press, 1996）"
    ],
    chapters: [
      "复杂适应系统与涌现概念",
      "ABM 范式与 NetLogo 基础",
      "主体—环境交互与调度",
      "ODD 模型描述协议",
      "从模式出发的建模（pattern-oriented modeling）",
      "模型校准、验证与复现",
      "生态系统个体模型（IBM）",
      "社会模拟：隔离、扩散与合作演化",
      "基于主体的计算经济学（ACE）",
      "传染病 ABM 与政策评估",
      "行为空间探索与灵敏度分析",
      "大规模 ABM 与并行计算（Repast/MASON）",
      "ABM 与机器学习混合建模"
    ],
  },
  'intermediate/modeling-and-simulation': {
    title: "建模与仿真",
    books: [
          "Banks, Carson, Nelson & Nicol《Discrete-Event System Simulation》（Pearson, 5th ed. 2010）",
          "Law《Simulation Modeling and Analysis》（McGraw-Hill, 5th ed. 2014）",
          "Zeigler, Muzy & Kofman《Theory of Modeling and Simulation: Discrete Event and Iterative System Computational Foundations》（Academic Press/Elsevier, 3rd ed. 2019）"
    ],
    chapters: [
      "建模与仿真的分类与范式",
      "伪随机数生成与随机变量采样",
      "离散事件仿真（事件调度/进程交互）",
      "输入数据建模与分布拟合",
      "输出分析：终态与稳态仿真",
      "置信区间与重复运行设计",
      "方差缩减技术",
      "DEVS 形式体系",
      "连续系统仿真与混合仿真",
      "并行与分布式仿真（HLA）",
      "仿真优化（排序选择与元启发式）",
      "验证、确认与可信度评估（VV&A）",
      "数字孪生中的在线仿真"
    ],
  },
  'intermediate/uncertainty-quantification': {
    title: "不确定性量化",
    books: [
          "Smith《Uncertainty Quantification: Theory, Implementation, and Applications》（SIAM, 2014）",
          "Sullivan《Introduction to Uncertainty Quantification》（Springer, 2015）",
          "Ghanem, Higdon & Owhadi (eds)《Handbook of Uncertainty Quantification》（Springer, 2017）"
    ],
    chapters: [
      "不确定性来源与偶然/认知分类",
      "概率框架与贝叶斯框架",
      "参数辨识与统计反问题",
      "多项式混沌展开（PCE）",
      "随机配点与稀疏网格方法",
      "高斯过程代理模型",
      "全局灵敏度分析（Sobol 指数）",
      "贝叶斯反演与 MCMC",
      "模型校准、确认与外推",
      "降阶模型与多保真建模",
      "深度学习中的不确定性（BNN/集成）",
      "UQ 在工程与气候模型中的应用"
    ],
  },
  'frontier/urban-computing': {
    title: "城市计算与城市科学",
    books: [
          "郑宇《Urban Computing》（MIT Press, 2019）",
          "Bettencourt《Introduction to Urban Science: Evidence and Theory of Cities as Complex Systems》（MIT Press, 2021）",
          "Batty《The New Science of Cities》（MIT Press, 2013）"
    ],
    chapters: [
      "城市感知与多源城市数据",
      "时空数据管理与索引",
      "轨迹数据挖掘与地图匹配",
      "人流预测与 OD 估计",
      "城市空气质量细粒度推断",
      "共享出行与智能调度",
      "城市标度律（urban scaling）",
      "城市网络与空间交互模型",
      "智慧城市平台与数据中台",
      "城市数字孪生",
      "城市韧性与应急管理应用",
      "城市计算的隐私与治理"
    ],
  },
  'frontier/industrial-ecology': {
    title: "工业生态学",
    books: [
          "Graedel & Allenby《Industrial Ecology and Sustainable Engineering》（Prentice Hall, 2010）",
          "Ayres & Ayres (eds)《A Handbook of Industrial Ecology》（Edward Elgar, 2002）",
          "邓南圣、吴峰主编《工业生态学：理论与应用》（化学工业出版社，2002）"
    ],
    chapters: [
      "产业代谢与产业生态系统",
      "物质流分析（MFA）",
      "生命周期评价（LCA）框架",
      "过程清单与影响评价方法",
      "生态工业园与产业共生（卡伦堡模式）",
      "环境扩展投入产出分析（EEIO-LCA）",
      "面向环境的设计（DfE/生态设计）",
      "循环性指标与物质循环率",
      "碳足迹、水足迹与生态足迹核算",
      "城市代谢",
      "生产者责任延伸与政策工具",
      "工业生态学与碳中和路径"
    ],
  },
  'frontier/planetary-health': {
    title: "行星健康",
    books: [
          "Myers & Frumkin (eds)《Planetary Health: Protecting Nature to Protect Ourselves》（Island Press, 2020）",
          "Haines & Frumkin《Planetary Health: Safeguarding Human Health and the Environment in the Anthropocene》（Cambridge University Press, 2021）",
          "Whitmee et al.《Safeguarding human health in the Anthropocene epoch: report of The Rockefeller Foundation–Lancet Commission on planetary health》（The Lancet, 2015）"
    ],
    chapters: [
      "行星健康概念与人类世背景",
      "地球系统边界（planetary boundaries）",
      "气候变化与健康结局",
      "生物多样性丧失与人兽共患病溢出",
      "土地系统变化与食物系统",
      "淡水系统与涉水健康风险",
      "海洋健康与海产品安全",
      "空气污染与化学品暴露",
      "营养转型与可持续膳食",
      "城市化、建成环境与健康",
      "气候变化迁移与健康公平",
      "行星健康指标、治理与教育"
    ],
  },
  'frontier/risk-analysis': {
    title: "风险分析与风险科学",
    books: [
          "Aven《Foundations of Risk Analysis: A Knowledge and Decision-Oriented Perspective》（Wiley, 2nd ed. 2012）",
          "Bedford & Cooke《Probabilistic Risk Analysis: Foundations and Methods》（Cambridge University Press, 2001）",
          "Modarres《Risk Analysis in Engineering: Techniques, Tools, and Trends》（CRC Press, 2006）"
    ],
    chapters: [
      "风险概念体系与定量定义（Kaplan & Garrick）",
      "概率风险评价（PRA）流程",
      "故障树与事件树分析",
      "贝叶斯方法与风险更新",
      "专家判断结构化引出",
      "风险感知的心理测量范式",
      "风险的社会放大与风险沟通",
      "可接受风险与 ALARP 原则",
      "极值与尾部风险建模",
      "系统性风险与级联失效",
      "新兴风险与深度不确定性",
      "风险治理框架（IRGC）与韧性"
    ],
  },
  'advanced/affective-computing': {
    title: "情感计算",
    books: [
          "Picard《Affective Computing》（MIT Press, 1997）",
          "Calvo, D'Mello, Gratch & Kappas (eds)《The Oxford Handbook of Affective Computing》（Oxford University Press, 2015）",
          "Schuller & Batliner《Computational Paralinguistics: Emotion, Affect and Personality in Speech and Language Processing》（Wiley, 2013）"
    ],
    chapters: [
      "情绪理论：离散模型与维度模型",
      "情感的心理生理学基础",
      "面部表情识别（FACS 与深度学习）",
      "语音情感与副语言计算",
      "生理信号情感识别（ECG/EDA/EEG）",
      "文本情感分析与情绪检测",
      "多模态情感融合",
      "情感生成与虚拟人表达",
      "情感机器人与具身交互",
      "心理健康与人机交互应用",
      "数据集、标注与评测基准",
      "情感计算的伦理、隐私与操纵风险"
    ],
  },
  'humanities/big-history': {
    title: "大历史",
    books: [
          "Christian, Brown & Benjamin《Big History: Between Nothing and Everything》（McGraw-Hill, 2014）",
          "Christian《Maps of Time: An Introduction to Big History》（University of California Press, 2004）",
          "Spier《Big History and the Future of Humanity》（Wiley-Blackwell, 2nd ed. 2015）"
    ],
    chapters: [
      "大历史的学科定位与八个复杂度阈值",
      "宇宙大爆炸与时空起源",
      "恒星演化与化学元素生成",
      "太阳系与地球的形成",
      "生命起源与达尔文进化",
      "人类起源与集体学习",
      "采集社会与早期人类扩散",
      "农业革命与定居文明",
      "城市、国家与文明网络",
      "现代革命与化石能源文明",
      "人类世与未来学视角",
      "大历史的证据方法与教学设计"
    ],
  },
  'humanities/medical-humanities': {
    title: "医学人文与叙事医学",
    books: [
          "Cole, Carlin & Carson《Medical Humanities: An Introduction》（Cambridge University Press, 2015）",
          "Charon《Narrative Medicine: Honoring the Stories of Illness》（Oxford University Press, 2006）",
          "Whitehead & Woods (eds)《The Edinburgh Companion to the Critical Medical Humanities》（Edinburgh University Press, 2016）"
    ],
    chapters: [
      "医学人文的学科史与定义",
      "叙事医学：细读与倾听",
      "病史采集与叙事能力",
      "文学中的疾病与疯狂书写",
      "艺术、音乐与视觉文化中的医学",
      "医学人类学与跨文化照护",
      "疼痛、残疾与具身体验",
      "衰老、临终与死亡叙事",
      "医患沟通与反思性实践教学",
      "医学人文课程体系设计",
      "批判医学人文与社会决定因素",
      "中国医学人文与新医科实践"
    ],
  },
  'social/science-communication': {
    title: "科学传播与公众理解科学",
    books: [
          "Bucchi & Trench (eds)《Routledge Handbook of Public Communication of Science and Technology》（Routledge, 3rd ed. 2021）",
          "National Academies of Sciences《Communicating Science Effectively: A Research Agenda》（National Academies Press, 2017）",
          "Burns, O'Connor & Stocklmayer《Science Communication: A Contemporary Definition》（Public Understanding of Science, 2003，权威综述）"
    ],
    chapters: [
      "科学传播的历史与学科化",
      "缺失模型、对话模型与参与模型",
      "公众理解科学与科学素质测评",
      "科学新闻的生产与变迁",
      "博物馆、科学中心与非正式学习",
      "数字媒体、短视频与科学网红",
      "风险沟通与健康传播",
      "科学辟谣与错误信息纠偏",
      "公民科学（citizen science）",
      "争议性科技议题传播（疫苗/转基因/AI）",
      "科学传播效果评估方法",
      "中国科普体系：科普法与全民科学素质行动"
    ],
  },
  'humanities/applied-linguistics': {
    title: "应用语言学（第二语言习得与外语教学）",
    books: [
          "Rod Ellis, \"The Study of Second Language Acquisition\" (2nd ed., Oxford University Press, 2008)",
          "Norbert Schmitt (ed.), \"An Introduction to Applied Linguistics\" (2nd ed., Routledge, 2010)",
          "Jack C. Richards & Theodore S. Rodgers, \"Approaches and Methods in Language Teaching\" (3rd ed., Cambridge University Press, 2014)"
    ],
    chapters: [
      "应用语言学的学科范围与历史 (Schmitt Ch 1-2)",
      "第一语言习得与第二语言习得的对比 (Ellis Ch 1-3)",
      "中介语理论与错误分析 (Ellis Ch 3-4)",
      "SLA 的认知视角：输入、互动与输出假说 (Ellis Ch 6-8)",
      "学习者个体差异：动机、学能、策略 (Ellis Ch 9-11)",
      "语法教学法史：语法翻译法、听说法、交际法 (Richards & Rodgers Part I-II)",
      "任务型语言教学与内容型教学 (Richards & Rodgers Ch 9-10)",
      "语言测试：信度、效度与构念 (Schmitt Ch 语言测试)",
      "语言规划与语言政策 (Schmitt Ch 语言政策)",
      "语料库在语言教学与词典编纂中的应用 (Schmitt Ch 语料库)",
      "多语制与全球英语（English as a Lingua Franca）",
      "应用语言学研究方法：定量与定性 (Schmitt Ch 方法)"
    ],
  },
  'humanities/historical-linguistics-typology': {
    title: "历史语言学与语言类型学",
    books: [
          "Lyle Campbell, \"Historical Linguistics: An Introduction\" (4th ed., Edinburgh University Press, 2021)",
          "Bernard Comrie, \"Language Universals and Linguistic Typology\" (2nd ed., Blackwell, 1989)",
          "William Croft, \"Typology and Universals\" (2nd ed., Cambridge University Press, 2003)"
    ],
    chapters: [
      "语言演变与语言谱系：印欧语系的发现 (Campbell Ch 1-2)",
      "语音演变与音变定律（新语法学派假说） (Campbell Ch 2-3)",
      "类推变化与词汇借用 (Campbell Ch 4-5)",
      "比较法与原始语重建 (Campbell Ch 6)",
      "语言年代学与同源词统计法 (Campbell Ch 8)",
      "语言接触与克里奥尔化 (Campbell Ch 7)",
      "语言演变的社会与认知动因 (Campbell Ch 9-10)",
      "类型学的基本概念：跨语言比较与语言共性 (Comrie Ch 1-2)",
      "语序类型学：SVO/SOV 与 Greenberg 共性 (Comrie Ch 3-4；Croft Ch 3)",
      "形态类型：孤立/黏着/屈折/多式综合 (Comrie Ch 1)",
      "主宾格与作格配列类型 (Comrie Ch 5；Croft Ch 5)",
      "蕴涵共性与标记性理论 (Croft Ch 4)",
      "语法化理论与单向性假说 (Croft Ch 8)"
    ],
  },
  'humanities/sociolinguistics': {
    title: "社会语言学",
    books: [
          "Ronald Wardhaugh & Janet M. Fuller, \"An Introduction to Sociolinguistics\" (7th ed., Wiley-Blackwell, 2015)",
          "William Labov, \"Sociolinguistic Patterns\" (University of Pennsylvania Press, 1972)",
          "Florian Coulmas, \"Sociolinguistics: The Study of Speakers' Choices\" (Cambridge University Press, 2005)"
    ],
    chapters: [
      "社会语言学的对象与方法：语言变项概念 (Wardhaugh Ch 1-2；Labov Ch 1-2)",
      "语言、方言与变体：言语共同体 (Wardhaugh Ch 2-3)",
      "语言变异的社会分层：阶级、年龄与性别 (Labov Ch 4-6)",
      "Labov 纽约市百货公司调查与变异研究方法 (Labov Ch 3)",
      "语言变异的显像时间与进行中的变化 (Labov Ch 7-8)",
      "语码转换与双语现象 (Wardhaugh Ch 4)",
      "语言与性别： Lakoff 假说及其检验 (Wardhaugh Ch 13)",
      "语言态度与语言认同 (Wardhaugh Ch 11)",
      "语言保持、转用与消亡 (Wardhaugh Ch 9)",
      "皮钦语与克里奥尔语 (Wardhaugh Ch 3)",
      "语言规划：地位规划与本体规划 (Coulmas Part II)",
      "礼貌原则与会话分析的社会维度 (Wardhaugh Ch 12)"
    ],
  },
  'humanities/corpus-linguistics': {
    title: "语料库语言学",
    books: [
          "Douglas Biber, Susan Conrad & Randi Reppen, \"Corpus Linguistics: Investigating Language Structure and Use\" (Cambridge University Press, 1998)",
          "Tony McEnery & Andrew Hardie, \"Corpus Linguistics: Method, Theory and Practice\" (Cambridge University Press, 2012)",
          "Graeme Kennedy, \"An Introduction to Corpus Linguistics\" (Longman, 1998)"
    ],
    chapters: [
      "语料库语言学史：从 Brown 与 LOB 到 BNC/COCA (McEnery & Hardie Ch 1)",
      "语料库类型：通用/专门、单语/平行、历时语料库 (McEnery & Hardie Ch 2)",
      "语料库设计：代表性、平衡性与抽样 (Biber et al. Ch 1)",
      "标注体系：词性标注与句法标注 (McEnery & Hardie Ch 3)",
      "词频表、关键词表与统计显著性 (Kennedy Ch 2)",
      "搭配与类联接：MI 值、t 值与对数似然 (Biber et al. Ch 2)",
      "索引行（KWIC）分析与词汇语法 (McEnery & Hardie Ch 4)",
      "语域变异的多维分析（Biber MD 分析） (Biber et al. Ch 5)",
      "学习者语料库与中介语对比分析 (McEnery & Hardie Ch 8)",
      "平行语料库与翻译研究 (McEnery & Hardie Ch 7)",
      "语料库驱动的词典编纂 (Kennedy Ch 6)",
      "语料库方法的批评与三角互证 (McEnery & Hardie Ch 10)"
    ],
  },
  'humanities/graphemics-writing-systems': {
    title: "普通文字学与世界文字系统",
    books: [
          "Florian Coulmas, \"The Writing Systems of the World\" (Blackwell, 1989)",
          "Henry Rogers, \"Writing Systems: A Linguistic Approach\" (Blackwell, 2005)",
          "周有光《比较文字学初探》(语文出版社, 1998)"
    ],
    chapters: [
      "文字的定义、功能与文字系统分类 (Rogers Ch 1-2)",
      "文字的起源：从图画记事到楔形文字 (Coulmas Ch 1-2)",
      "语素文字：汉字与玛雅文字 (Rogers Ch 汉字部分)",
      "音节文字：日文假名与线性文字 B (Rogers Ch 音节文字)",
      "辅音文字：腓尼基、阿拉米与阿拉伯字母 (Coulmas Ch 4)",
      "全音素文字：希腊字母的起源与传播 (Coulmas Ch 5)",
      "拉丁、西里尔字母的演变与扩散 (Rogers Ch 拉丁字母)",
      "印度系文字与婆罗米字母体系 (Coulmas Ch 6)",
      "文字的形体类型学（周有光「三相」分类） (周有光 Ch 2-4)",
      "文字改革与现代化：汉字简化、越南国语字 (周有光 Ch 改革部分)",
      "书写与语言的关系：文字能否独立于口语 (Rogers Ch 2)",
      "失读文字与文字破译：商博良与罗塞塔石碑 (Coulmas Ch 3)"
    ],
  },
  'humanities/rhetoric': {
    title: "修辞学",
    books: [
          "Edward P.J. Corbett & Robert J. Connors, \"Classical Rhetoric for the Modern Student\" (4th ed., Oxford University Press, 1999)",
          "George A. Kennedy, \"A New History of Classical Rhetoric\" (Princeton University Press, 1994)",
          "陈望道《修辞学发凡》(上海教育出版社, 1932/多次再版)"
    ],
    chapters: [
      "修辞学的起源：西西里与智者学派 (Kennedy Ch 1)",
      "亚里士多德《修辞学》：ethos/pathos/logos 三诉诸 (Kennedy Ch 2；陈望道 题旨情境论)",
      "修辞五艺：取材/布局/风格/记忆/发表 (Corbett Part I)",
      "论证模式：恩梯墨玛（省略三段论）与例证 (Corbett Ch 取材)",
      "西塞罗与昆体良的演说术体系 (Kennedy Ch 3-4)",
      "文体风格论：三大风格层次与辞格 (Corbett Ch 风格)",
      "比喻与隐喻理论：从亚里士多德到 Richards (陈望道 譬喻格；Corbett Ch 辞格)",
      "中国修辞学传统：《文心雕龙》与历代修辞论 (陈望道 引论)",
      "陈望道两大分野：消极修辞与积极修辞 (陈望道 篇四-五)",
      "新修辞学：Perelman 论辩修辞与 Burke 认同理论 (Corbett 结语)",
      "修辞与公共话语：政治演说与媒介修辞 (Kennedy Ch 近现代)",
      "修辞与写作教学：当代 composition studies (Corbett Part II)"
    ],
  },
  'humanities/semiotics': {
    title: "符号学",
    books: [
          "Umberto Eco, \"A Theory of Semiotics\" (Indiana University Press, 1976)",
          "Daniel Chandler, \"Semiotics: The Basics\" (2nd ed., Routledge, 2007)",
          "赵毅衡《符号学：原理与推演》(南京大学出版社, 2011)"
    ],
    chapters: [
      "符号学的两大学统：索绪尔语言学与皮尔斯逻辑学 (Chandler Ch 1)",
      "能指/所指与符号的任意性 (Chandler Ch 2)",
      "皮尔斯三分：像似符/指示符/规约符 (Chandler Ch 2；赵毅衡 Ch 1)",
      "符号过程（semiosis）与无限衍义 (Eco §0-1)",
      "代码理论：s-代码与规则代码 (Eco §2)",
      "符号生产理论：识别、记号、复制 (Eco §3)",
      "组合轴与聚合轴（横组合/纵聚合） (Chandler Ch 4)",
      "外延/内涵与神话（巴特第二级符号系统） (Chandler Ch 5；赵毅衡 Ch 5)",
      "文本间性与元语言 (Chandler Ch 6)",
      "符号双轴与文本的伴随文本 (赵毅衡 Ch 6-7)",
      "叙述的符号学分析 (赵毅衡 Ch 8)",
      "符号学在艺术、广告与媒介分析中的应用 (Chandler Ch 7)"
    ],
  },
  'humanities/narratology': {
    title: "叙事学",
    books: [
          "Gérard Genette, \"Narrative Discourse: An Essay in Method\" (Cornell University Press, 1980)",
          "Mieke Bal, \"Narratology: Introduction to the Theory of Narrative\" (3rd ed., University of Toronto Press, 2009)",
          "申丹《叙述学与小说文体学研究》(北京大学出版社, 1998, 多次修订再版)"
    ],
    chapters: [
      "叙事学的兴起：俄国形式主义与法国结构主义 (Bal Introduction)",
      "故事/话语的区分：fabula 与 sjuzhet (Bal Ch 1)",
      "热奈特时序理论：顺序、预叙、倒叙 (Genette Part I)",
      "时距：概要、场景、省略与停顿 (Genette Part II)",
      "频率：单一/重复/反复叙事 (Genette Part III)",
      "语式：距离与投影（聚焦理论） (Genette Part IV；Bal Ch 2)",
      "语态：叙述者层级与叙述类型（同故事/异故事） (Genette Part V)",
      "隐含作者、隐含读者与不可靠叙述 (申丹 Ch 叙述者)",
      "人物视点与聚焦模式的文本分析 (Bal Ch 3)",
      "叙事时间与小说文体的交叉分析 (申丹 Ch 时间/视角)",
      "后经典叙事学：认知叙事学与跨媒介叙事 (Bal 后经典部分)",
      "叙事学在非虚构与史学叙述中的应用 (申丹 结语)"
    ],
  },
  'humanities/stylistics': {
    title: "文体学（语言学与文学交叉）",
    books: [
          "Geoffrey Leech & Mick Short, \"Style in Fiction: A Linguistic Introduction to English Fictional Prose\" (2nd ed., Pearson/Longman, 2007)",
          "Paul Simpson, \"Stylistics: A Resource Book for Students\" (Routledge, 2004)",
          "胡壮麟《理论文体学》(外语教学与研究出版社, 2000)"
    ],
    chapters: [
      "文体的概念：变异、选择与发展前景化 (Leech & Short Ch 1)",
      "文体分析的语言学清单：词汇/语法/修辞/衔接 (Leech & Short Ch 2)",
      "前景化与平行结构、偏离 (Simpson Ch 1-2)",
      "小说中的话语呈现：直接/间接/自由间接引语 (Leech & Short Ch 10)",
      "思维模式（mind style）与认知文体学 (Simpson Ch 7)",
      "视角与叙事层面的文体标记 (Leech & Short Ch 6)",
      "会话含义与戏剧对话分析 (Simpson Ch 5)",
      "语域理论与功能文体学 (胡壮麟 Ch 功能文体)",
      "诗歌文体：音系模式与格律分析 (Simpson Ch 3)",
      "语料库文体学与作者风格统计 (Leech & Short 新版增补)",
      "批评文体学与意识形态分析 (Simpson Ch 8)",
      "文体的社会历史维度 (胡壮麟 Ch 文体史)"
    ],
  },
  'humanities/oral-tradition-epic': {
    title: "口头传统与史诗研究（口头诗学）",
    books: [
          "Milman Parry, \"The Making of Homeric Verse: The Collected Papers of Milman Parry\" (ed. Adam Parry, Oxford University Press, 1971)",
          "Albert B. Lord, \"The Singer of Tales\" (Harvard University Press, 1960)",
          "John Miles Foley, \"The Theory of Oral Composition: History and Methodology\" (Indiana University Press, 1988)"
    ],
    chapters: [
      "荷马问题：分析派与统一派之争 (Parry 绪论；Foley Ch 1)",
      "帕里的南斯拉夫田野与程式（formula）概念 (Parry 1928-1935 论文)",
      "程式系统、节俭原则与格律约束 (Parry 论文集核心)",
      "洛德的表演理论：演唱中的创编 (Lord Ch 2-4)",
      "主题（theme）与故事范型 (Lord Ch 4-7)",
      "口头与书写的对比：文本概念的解构 (Lord Ch 8；Foley Ch 3)",
      "口头传统的类型学：史诗、歌谣、叙事诗 (Foley Ch 2)",
      "大词（traditional referentiality）与传统指涉性 (Foley Ch 4)",
      "荷马史诗分析案例：《伊利亚特》的程式结构 (Parry/Lord 案例)",
      "中国口头史诗：《格萨尔》《江格尔》《玛纳斯》研究 (Foley 世界传统章)",
      "民族志诗学与口头文类界定 (Foley Ch 5)",
      "口头性与非遗保护 (Lord 后记与当代延伸)"
    ],
  },
  'humanities/teaching-chinese-international': {
    title: "汉语国际教育（对外汉语教学）",
    books: [
          "刘珣《对外汉语教育学引论》(北京语言大学出版社, 2000)",
          "赵金铭主编《对外汉语教学概论》(商务印书馆, 2004)",
          "周小兵《对外汉语教学入门》(中山大学出版社, 2004)"
    ],
    chapters: [
      "学科性质与名称演变：对外汉语教学→汉语国际教育→国际中文教育 (刘珣 Ch 1)",
      "学科理论基础：语言学、教育学、心理学、文化学 (刘珣 Ch 2)",
      "第二语言教学主要流派在汉语教学中的应用 (刘珣 Ch 6)",
      "汉语作为第二语言习得研究：偏误分析 (刘珣 Ch 5)",
      "汉字教学的特殊性与教学方法 (周小兵 Ch 汉字)",
      "语音教学：声调、轻声与语流音变 (周小兵 Ch 语音)",
      "词汇教学与语素教学法 (赵金铭 Ch 词汇教学)",
      "语法教学：把字句、了着过等难点处理 (周小兵 Ch 语法)",
      "听力/口语/阅读/写作技能训练 (赵金铭 Ch 技能训练)",
      "教材编写原则与评估 (赵金铭 Ch 教材)",
      "汉语水平考试（HSK）与语言测试 (刘珣 Ch 测试)",
      "跨文化交际与文化教学 (刘珣 Ch 文化)",
      "国际中文教师专业发展 (赵金铭 Ch 教师)"
    ],
  },
  'humanities/cultural-history': {
    title: "文化史（新文化史）",
    books: [
          "Peter Burke, \"What is Cultural History?\" (3rd ed., Polity, 2019)",
          "Peter Burke, \"Varieties of Cultural History\" (Polity, 1997)",
          "冯天瑜、何晓明、周积明《中华文化史》(上海人民出版社, 1990)"
    ],
    chapters: [
      "经典文化史：布克哈特与赫伊津哈 (Burke, What is Cultural History? Ch 1)",
      "文化史的社会学转向：艺术社会史传统 (Burke Ch 2)",
      "年鉴学派与心态史（histoire des mentalités） (Burke Ch 3)",
      "新文化史的兴起：历史人类学转向 (Burke Ch 4)",
      "微观史学：金茨堡《奶酪与蛆虫》 (Burke, Varieties Ch 微观史)",
      "日常生活史与物质文化 (Burke, Varieties)",
      "阅读史与书籍史 (Burke, Varieties)",
      "身体史与情感史 (Burke, What is Cultural History? 新版章)",
      "记忆、纪念与历史表征 (Burke 新版章)",
      "文化史的方法论问题：表征与实践 (Burke Ch 5)",
      "中华文化史的分期与结构 (冯天瑜《中华文化史》)",
      "中国近世文化转型与近代文化史 (冯天瑜 下册)"
    ],
  },
  'humanities/environmental-history': {
    title: "环境史",
    books: [
          "J. Donald Hughes, \"What is Environmental History?\" (2nd ed., Polity, 2016)",
          "J.R. McNeill, \"Something New Under the Sun: An Environmental History of the Twentieth-Century World\" (W.W. Norton, 2000)",
          "Donald Worster, \"Nature's Economy: A History of Ecological Ideas\" (2nd ed., Cambridge University Press, 1994)"
    ],
    chapters: [
      "环境史的定义与三大研究主题 (Hughes Ch 1)",
      "环境史的先驱：地理决定论与年鉴学派 (Hughes Ch 2)",
      "哥伦布大交换：生态帝国主义 (Hughes Ch 4)",
      "美国环境史：边疆、荒野与保护运动 (Hughes Ch 5)",
      "生态思想史：从林奈到生态学 (Worster 全书主线)",
      "尘暴（Dust Bowl）与大平原生态史 (Worster 研究范式)",
      "20 世纪能源体制的环境后果 (McNeill Part I)",
      "大气、水圈与土壤的世纪变迁 (McNeill Part II)",
      "生物圈变化：物种入侵与生物多样性丧失 (McNeill Part III)",
      "中国环境史：大象的退却与黄河史 (Hughes 中国章)",
      "环境史与全球史、大历史的交叉 (McNeill 结语)",
      "环境正义与后殖民环境史 (Hughes 新版章)"
    ],
  },
  'humanities/oral-history': {
    title: "口述历史",
    books: [
          "Paul Thompson (with Joanna Bornat), \"The Voice of the Past: Oral History\" (4th ed., Oxford University Press, 2017)",
          "Donald A. Ritchie, \"Doing Oral History\" (3rd ed., Oxford University Press, 2015)",
          "Alessandro Portelli, \"The Death of Luigi Trastulli and Other Stories: Form and Meaning in Oral History\" (SUNY Press, 1991)"
    ],
    chapters: [
      "口述史的历史：从修昔底德到现代口述史运动 (Thompson Ch 1)",
      "口述史的成就与局限：证据问题 (Thompson Ch 2-3)",
      "记忆的科学：自传记忆与遗忘 (Thompson 新版 Ch 记忆)",
      "访谈的设计与实施：提问技术与录音规范 (Ritchie Ch 3-4)",
      "口述史的法律与伦理：知情同意与版权 (Ritchie Ch 5)",
      "口述史档案的整理、转录与著录 (Ritchie Ch 6)",
      "记忆的不可靠性与口述史的独特价值 (Portelli Ch 1-2)",
      "「错误」的史料价值：Trastulli 案例分析 (Portelli 标题章)",
      "口述史与社会史：底层、劳工与妇女史 (Thompson Ch 4-5)",
      "社区口述史项目与公共史学实践 (Ritchie Ch 7)",
      "数字时代的口述史：音视频与在线档案 (Ritchie Ch 8)",
      "口述史在中国：现当代史与非遗口述 (Thompson 国际章)"
    ],
  },
  'humanities/ethnomusicology': {
    title: "民族音乐学（Ethnomusicology）",
    books: [
          "Bruno Nettl, \"The Study of Ethnomusicology: Thirty-Three Discussions\" (3rd ed., University of Illinois Press, 2015)",
          "Helen Myers (ed.), \"Ethnomusicology: An Introduction\" (W.W. Norton, 1992)",
          "Jeff Todd Titon (ed.), \"Worlds of Music: An Introduction to the Music of the World's Peoples\" (6th ed., Cengage, 2016)"
    ],
    chapters: [
      "学科史：比较音乐学到民族音乐学 (Nettl Part I)",
      "田野工作方法与民族志记录 (Nettl Part II；Myers Ch 田野)",
      "音乐记谱与分析：跨文化记谱问题 (Myers Ch 记谱)",
      "音乐作为文化：Merriam 三重模式 (Nettl Ch 概念)",
      "乐器学与乐器分类（Hornbostel-Sachs） (Myers Ch 乐器)",
      "口头传承与音乐的记忆机制 (Nettl Ch 传承)",
      "世界音乐文化区：东亚/南亚/中东/非洲 (Titon 各文化章)",
      "中国传统音乐：民歌、戏曲、器乐 (Titon 中国章)",
      "仪式音乐与音乐的社会功能 (Myers Ch 功能)",
      "城市民族音乐学与流行音乐研究 (Nettl 新版章)",
      "全球化、流散与音乐认同 (Nettl 新版章)",
      "应用民族音乐学与非遗保护 (Nettl 结语)"
    ],
  },
  'humanities/music-theory-composition': {
    title: "音乐理论与作曲技术理论",
    books: [
          "Walter Piston, \"Harmony\" (5th ed., revised by Mark DeVoto, W.W. Norton, 1987)",
          "Samuel Adler, \"The Study of Orchestration\" (4th ed., W.W. Norton, 2016)",
          "Arnold Schoenberg, \"Fundamentals of Musical Composition\" (ed. Gerald Strang & Leonard Stein, Faber & Faber, 1967)"
    ],
    chapters: [
      "音程、音阶与调式体系 (Piston Ch 1-4)",
      "三和弦与和声进行：正格/变格/半成 (Piston Ch 5-8)",
      "七和弦与转位、终止式 (Piston Ch 9-12)",
      "转调与变化和弦 (Piston Ch 转调部分)",
      "对位法基础：分类对位（福克斯传统） (Piston 对位章)",
      "曲式学：乐段、单二部、单三部 (Schoenberg Part I)",
      "奏鸣曲式与回旋曲式 (Schoenberg Part II-III)",
      "主题发展与动机展开技术 (Schoenberg Ch 动机)",
      "管弦乐器法：弦/木管/铜管/打击乐性能 (Adler Part I)",
      "配器法：乐器组合与管弦乐织体 (Adler Part II)",
      "总谱读法与移调乐器 (Adler Part III)",
      "20 世纪作曲技术概览：序列主义与音色音乐 (Adler 新版章)"
    ],
  },
  'humanities/sociology-of-art': {
    title: "艺术社会学",
    books: [
          "Victoria D. Alexander, \"Sociology of the Arts: Exploring Fine and Popular Forms\" (Blackwell, 2003)",
          "Howard S. Becker, \"Art Worlds\" (University of California Press, 1982)",
          "Pierre Bourdieu, \"The Rules of Art: Genesis and Structure of the Literary Field\" (Stanford University Press, 1996)"
    ],
    chapters: [
      "艺术社会学的路径：反映论与塑造论 (Alexander Ch 1)",
      "文化菱形：艺术-创作者-分配者-接受者 (Alexander Ch 2)",
      "艺术界（art world）作为协作网络 (Becker Ch 1-2)",
      "惯例、资源与艺术的集体行动 (Becker Ch 3-5)",
      "国家、赞助人与艺术市场 (Alexander Ch 3-4)",
      "文化生产视角：文化产业与把关人 (Alexander Ch 5)",
      "文化资本、惯习与趣味区隔 (Bourdieu 相关章节)",
      "文学场的生成与自主性 (Bourdieu Part I)",
      "艺术的接受与消费：受众研究 (Alexander Ch 6-7)",
      "艺术的社会边界：高雅/通俗/民间 (Becker Ch 7-8)",
      "艺术体制论与艺术的定义之争 (Becker/Alexander 综合)",
      "全球化与数字时代的艺术生产 (Alexander 新版章)"
    ],
  },
  'humanities/postcolonial-studies': {
    title: "后殖民理论与后殖民研究",
    books: [
          "Edward W. Said, \"Orientalism\" (Pantheon, 1978)",
          "Bill Ashcroft, Gareth Griffiths & Helen Tiffin, \"The Empire Writes Back: Theory and Practice in Post-Colonial Literatures\" (2nd ed., Routledge, 2002)",
          "Robert J.C. Young, \"Postcolonialism: An Historical Introduction\" (Blackwell, 2001)"
    ],
    chapters: [
      "东方学：作为话语的东方 (Said 绪论-Ch 1)",
      "殖民话语分析：从法农到萨义德 (Young Part I)",
      "斯皮瓦克与底层研究：「底层人能说话吗」 (Young Ch 底层)",
      "巴巴的杂糅性、模拟与第三空间 (Young Ch 杂糅)",
      "逆写帝国：后殖民文学的挪用与重置 (Ashcroft Ch 2-4)",
      "语言问题：弃用英语与本土语言写作之争 (Ashcroft Ch 2)",
      "后殖民民族文学与经典重构 (Ashcroft Ch 5-6)",
      "后殖民女性主义与第三世界女性书写 (Young Part III)",
      "流散、移民与文化认同 (Young Part IV)",
      "新殖民主义与全球化批判 (Young Part V)",
      "东方主义批评及其争议 (Said 后记)",
      "后殖民理论在中国学界的接受 (综合)"
    ],
  },
  'humanities/cultural-memory-studies': {
    title: "文化记忆研究",
    books: [
          "Astrid Erll & Ansgar Nünning (eds.), \"Cultural Memory Studies: An International and Interdisciplinary Handbook\" (De Gruyter, 2008)",
          "Jan Assmann, \"Cultural Memory and Early Civilization: Writing, Remembrance, and Political Imagination\" (Cambridge University Press, 2011)",
          "Astrid Erll, \"Memory in Culture\" (Palgrave Macmillan, 2011)"
    ],
    chapters: [
      "哈布瓦赫：集体记忆的社会框架 (Erll Handbook 历史章)",
      "阿斯曼：交往记忆与文化记忆的区分 (Jan Assmann Ch 1-2)",
      "记忆的媒介：文字、图像、身体与空间 (Erll, Memory in Culture Ch 2-4)",
      "文化记忆与认同建构 (Jan Assmann Ch 3)",
      "记忆的存储方式：功能记忆与存储记忆 (Aleida Assmann 相关章)",
      "文学作为文化记忆的媒介 (Erll Handbook 文学章)",
      "创伤记忆与大屠杀记忆研究 (Erll Handbook 创伤章)",
      "记忆之场（lieux de mémoire）与纪念政治 (Nora 传统；Handbook 空间章)",
      "跨国记忆与记忆的旅行 (Erll Ch 旅行记忆)",
      "数字记忆与媒介生态变迁 (Erll Handbook 媒介章)",
      "记忆与历史书写的关系 (Jan Assmann Ch 4)",
      "中国的文化记忆研究：国族叙事与地方记忆 (综合)"
    ],
  },
  'humanities/gender-studies': {
    title: "性别研究",
    books: [
          "Susan M. Shaw & Janet Lee, \"Women's Voices, Feminist Visions: Classic and Contemporary Readings\" (6th ed., McGraw-Hill, 2014)",
          "Judith Butler, \"Gender Trouble: Feminism and the Subversion of Identity\" (Routledge, 1990)",
          "Raewyn Connell, \"Gender and Power: Society, the Person and Sexual Politics\" (Polity, 1987)"
    ],
    chapters: [
      "性与性别：生理/社会性别的区分 (Shaw & Lee Ch 1-2)",
      "女性主义理论谱系：自由派/激进/社会主义/后结构 (Shaw & Lee Ch 理论)",
      "性别化的制度：家庭、教育与劳动 (Connell Part I)",
      "性征理论：性别的历史建构 (Connell Part II)",
      "巴特勒的性别操演理论 (Butler Ch 1-3)",
      "交叉性（intersectionality）：种族、阶级与性别 (Shaw & Lee Ch 交叉性)",
      "男性气质研究与支配性男性气质 (Connell Ch 男性气质)",
      "性别与文学：女性书写传统与经典修正 (Shaw & Lee 文学章)",
      "性别与媒介表征 (Shaw & Lee 媒介章)",
      "酷儿理论导论 (Butler 后续发展)",
      "跨国女性主义与后殖民性别研究 (Shaw & Lee 全球章)",
      "中国语境下的性别研究 (综合)"
    ],
  },
  'intermediate/biostatistics': {
    title: "生物统计学与实验设计",
    books: [
          "Zar, \"Biostatistical Analysis\" (5th ed., Pearson, 2010)",
          "杜荣骞《生物统计学》（第4版，高等教育出版社，2014）",
          "Glantz, \"Primer of Biostatistics\" (7th ed., McGraw-Hill, 2012)"
    ],
    chapters: [
      "数据类型与描述统计",
      "概率分布（二项/Poisson/正态）",
      "抽样分布与参数估计",
      "假设检验原理",
      "t 检验与非参数检验",
      "单因素与多因素方差分析",
      "区组设计与拉丁方",
      "析因实验设计",
      "相关与回归分析",
      "卡方检验与列联表",
      "计数数据与广义线性模型",
      "多元统计初步（PCA/判别）",
      "生存分析基础",
      "功效分析与样本量估计",
      "R/统计软件实践"
    ],
  },
  'intermediate/cancer-biology': {
    title: "癌症生物学（基础）",
    books: [
          "Weinberg, \"The Biology of Cancer\" (2nd ed., Garland Science, 2014)",
          "Pecorino, \"Molecular Biology of Cancer\" (4th ed., Oxford UP, 2021)",
          "DeVita, Lawrence & Rosenberg, \"DeVita, Hellman, and Rosenberg's Cancer: Principles & Practice of Oncology\" (12th ed., Wolters Kluwer, 2023)"
    ],
    chapters: [
      "肿瘤的多步发生与克隆演化",
      "癌基因（ras/myc/src）",
      "抑癌基因（Rb/p53）",
      "细胞周期失控与凋亡逃逸",
      "端粒酶与永生化",
      "血管生成",
      "侵袭与转移",
      "肿瘤微环境",
      "肿瘤免疫与免疫逃逸",
      "基因组不稳定与突变特征",
      "肿瘤代谢（Warburg 效应）",
      "肿瘤干细胞",
      "表观遗传改变",
      "靶向治疗与耐药机制",
      "免疫检查点与 CAR-T 原理"
    ],
  },
  'humanities/history-of-biology': {
    title: "生物学史",
    books: [
          "洛伊斯·N. 玛格纳《生命科学史》（刘学礼等译，上海人民出版社，2012）",
          "Mayr, \"The Growth of Biological Thought\" (Harvard UP, 1982)",
          "加兰·E. 艾伦《20世纪的生命科学史》（复旦大学出版社，2001）"
    ],
    chapters: [
      "古希腊自然哲学与亚里士多德生物学",
      "文艺复兴解剖学革命（维萨里）",
      "哈维与血液循环",
      "显微镜时代与微生物发现",
      "林奈分类体系",
      "胚胎学之争（预成论/渐成论）",
      "细胞学说",
      "达尔文与进化论",
      "孟德尔遗传学的重新发现",
      "生理学与实验生物学兴起",
      "分子生物学革命（DNA 双螺旋）",
      "重组 DNA 与基因组计划",
      "现代综合进化论",
      "中国近现代生物学发展"
    ],
  },
  'intermediate/ornithology': {
    title: "鸟类学",
    books: [
          "Gill & Prum, \"Ornithology\" (4th ed., W. H. Freeman, 2019)",
          "郑光美《鸟类学》（第2版，北京师范大学出版社，2012）",
          "Lovette & Fitzpatrick (eds.), \"Handbook of Bird Biology\" (3rd ed., Wiley/Cornell Lab, 2016)"
    ],
    chapters: [
      "鸟类的起源与演化（恐龙-鸟类过渡）",
      "形态与飞行适应",
      "羽毛与换羽",
      "呼吸系统与代谢",
      "鸣声与通讯",
      "繁殖行为与繁殖系统",
      "巢址选择与育雏",
      "迁徙与导航",
      "食性与群落生态",
      "鸟类系统分类",
      "种群动态与保护",
      "观鸟与环志方法"
    ],
  },
  'intermediate/mammalogy': {
    title: "哺乳动物学",
    books: [
          "Vaughan, Ryan & Czaplewski, \"Mammalogy\" (6th ed., Jones & Bartlett, 2015)",
          "Feldhamer et al., \"Mammalogy: Adaptation, Diversity, Ecology\" (4th ed., Johns Hopkins UP, 2015)",
          "Wilson & Mittermeier (eds.), \"Handbook of the Mammals of the World\" (Lynx Edicions, 2009–2019)"
    ],
    chapters: [
      "哺乳动物起源与中生代演化",
      "单孔类/有袋类/真兽类三大支系",
      "皮肤、毛与腺体",
      "体温调节与能量代谢",
      "感觉与回声定位",
      "生殖策略与胎盘多样性",
      "社会行为与通讯",
      "食性适应与消化系统",
      "主要目级分类（啮齿/食肉/灵长/鲸偶蹄等）",
      "生物地理与区系",
      "保护现状与人兽冲突",
      "野外调查方法（红外相机/无线电追踪）"
    ],
  },
  'intermediate/herpetology': {
    title: "两栖爬行动物学",
    books: [
          "Pough et al., \"Herpetology\" (4th ed., Sinauer/Oxford UP, 2016)",
          "Vitt & Caldwell, \"Herpetology: An Introductory Biology of Amphibians and Reptiles\" (4th ed., Academic Press, 2014)",
          "费梁、叶昌媛、江建平《中国两栖动物及其分布彩色图鉴》（四川科学技术出版社，2012）"
    ],
    chapters: [
      "四足动物起源与登陆",
      "两栖纲分类（无尾/有尾/蚓螈）",
      "爬行纲分类（龟鳖/鳞龙/鳄）",
      "皮肤呼吸与渗透调节",
      "变温生理与热生态",
      "繁殖模式与亲代抚育",
      "鸣声与求偶行为",
      "蛇类感觉与毒液系统",
      "生活史与变态",
      "两栖类全球衰退与壶菌病",
      "区系与生物地理",
      "保护与人工繁育"
    ],
  },
  'intermediate/protistology': {
    title: "原生生物学",
    books: [
          "Hausmann, Hülsmann & Radek, \"Protistology\" (3rd ed., E. Schweizerbart'sche, 2003)",
          "Lee, Leedale & Bradbury (eds.), \"An Illustrated Guide to the Protozoa\" (2nd ed., Society of Protozoologists, 2000)",
          "Patterson, \"Free-Living Freshwater Protozoa\" (Manson, 1996)"
    ],
    chapters: [
      "原生生物在真核生物树中的位置",
      "鞭毛虫类",
      "变形虫与有孔虫",
      "纤毛虫",
      "孢子虫（顶复门）",
      "硅藻与甲藻",
      "细胞骨架与摄食结构",
      "无性/有性生殖与生活史",
      "共生与寄生",
      "原生生物在食物网中的角色",
      "环境指示与污水生物处理",
      "显微镜与培养方法"
    ],
  },
  'intermediate/behavioral-genetics': {
    title: "行为遗传学",
    books: [
          "Knopik, Neiderhiser, DeFries & Plomin, \"Behavioral Genetics\" (7th ed., Worth, 2017)",
          "Anholt & Mackay, \"Principles of Behavioral Genetics\" (Academic Press, 2010)",
          "Falconer & Mackay, \"Introduction to Quantitative Genetics\" (4th ed., Pearson, 1996)"
    ],
    chapters: [
      "孟德尔定律与超越孟德尔",
      "数量性状与多基因遗传",
      "遗传度概念与误用",
      "双生子研究设计",
      "收养研究与家庭研究",
      "动物模型（果蝇/小鼠）",
      "连锁与关联分析",
      "GWAS 与多基因评分",
      "基因-环境交互（G×E）",
      "基因-环境相关（rGE）",
      "智力与认知的遗传研究",
      "精神疾病的遗传学",
      "人格与行为的遗传基础",
      "分子遗传学方法（敲除/光遗传筛选）"
    ],
  },
  'intermediate/ecotoxicology': {
    title: "生态毒理学",
    books: [
          "Newman, \"Fundamentals of Ecotoxicology\" (4th ed., CRC Press, 2015)",
          "Walker et al., \"Principles of Ecotoxicology\" (4th ed., CRC Press, 2012)",
          "Newman & Unger, \"Fundamentals of Ecotoxicology: The Science of Pollution\" 配套案例卷 (CRC Press)"
    ],
    chapters: [
      "污染物环境归趋与迁移",
      "生物富集与生物放大",
      "剂量-反应关系与毒性终点",
      "急性与慢性毒性试验",
      "重金属毒性",
      "农药与有机污染物",
      "内分泌干扰物",
      "微塑料与新兴污染物",
      "种群与群落水平效应",
      "生物标志物",
      "生态风险评估框架",
      "沉积物与土壤生态毒理",
      "法规毒理与化学品管理"
    ],
  },
  'intermediate/reproductive-biology': {
    title: "生殖生物学",
    books: [
          "Plant & Zeleznik (eds.), \"Knobil and Neill's Physiology of Reproduction\" (4th ed., Academic Press, 2015)",
          "Johnson & Everitt, \"Essential Reproduction\" (8th ed., Wiley-Blackwell, 2018)",
          "Jones & Lopez, \"Human Reproductive Biology\" (4th ed., Academic Press, 2014)"
    ],
    chapters: [
      "下丘脑-垂体-性腺轴",
      "精子发生与卵子发生",
      "性腺激素合成与作用",
      "性决定与性分化",
      "受精的分子机制",
      "着床与胎盘形成",
      "妊娠维持与分娩",
      "泌乳",
      "生殖行为的神经内分泌基础",
      "季节性繁殖",
      "比较生殖生物学（脊椎动物）",
      "生殖衰老",
      "辅助生殖原理"
    ],
  },
  'intermediate/chronobiology': {
    title: "时间生物学（生物钟与生物节律）",
    books: [
          "Refinetti, \"Circadian Physiology\" (3rd ed., CRC Press, 2016)",
          "Foster & Kreitzman, \"Circadian Rhythms: A Very Short Introduction\" (Oxford UP, 2017)",
          "Dunlap, Loros & DeCoursey (eds.), \"Chronobiology: Biological Timekeeping\" (Sinauer, 2004)"
    ],
    chapters: [
      "昼夜节律现象与概念（自由运行/授时）",
      "核心钟基因与转录-翻译反馈环（CLOCK/BMAL1/PER/CRY）",
      "视交叉上核主钟",
      "光授时与褪黑素",
      "外周振荡器与器官时钟",
      "睡眠-觉醒调控双过程模型",
      "进食与代谢节律",
      "季节节律与光周期",
      "潮汐/月节律",
      "细胞自主振荡器",
      "轮班、时差与健康",
      "时间治疗学（chronotherapy）"
    ],
  },
  'intermediate/landscape-ecology': {
    title: "景观生态学",
    books: [
          "Forman, \"Land Mosaics: The Ecology of Landscapes and Regions\" (Cambridge UP, 1995)",
          "Turner, Gardner & O'Neill, \"Landscape Ecology in Theory and Practice\" (Springer, 2001)",
          "傅伯杰等《景观生态学原理及应用》（第2版，科学出版社，2011）"
    ],
    chapters: [
      "景观概念与尺度",
      "斑块-廊道-基质模型",
      "空间异质性格局",
      "景观格局指数与度量",
      "边缘效应",
      "景观连接度",
      "干扰与景观动态",
      "岛屿生物地理与集合种群",
      "源-汇动态",
      "土地利用变化驱动",
      "景观与生态系统服务",
      "GIS 与遥感在景观生态中的应用",
      "景观规划与保护网络设计"
    ],
  },
  'intermediate/special-functions': {
    title: "特殊函数",
    books: [
          "王竹溪、郭敦仁《特殊函数概论》，北京大学出版社",
          "G. E. Andrews, R. Askey & R. Roy, Special Functions, Cambridge University Press, 1999",
          "E. T. Whittaker & G. N. Watson, A Course of Modern Analysis, Cambridge University Press (4th ed., 1927)"
    ],
    chapters: [
      "Gamma 函数与 Beta 函数",
      "超几何函数",
      "合流超几何函数",
      "Legendre 函数与球谐函数",
      "Bessel 函数（三类柱函数）",
      "正交多项式（Hermite / Laguerre / Jacobi）",
      "椭圆积分与椭圆函数",
      "Mathieu 函数与 Lamé 函数",
      "特殊函数的积分表示与围道积分",
      "渐近展开与最速下降法",
      "生成函数方法",
      "q-级数与基本超几何函数"
    ],
  },
  'intermediate/integral-equations': {
    title: "积分方程",
    books: [
          "R. Kress, Linear Integral Equations, Springer (3rd ed., 2014)",
          "F. G. Tricomi, Integral Equations, Interscience, 1957 (Dover 重印)"
    ],
    chapters: [
      "积分方程的分类（Fredholm 型 / Volterra 型）",
      "逐次逼近法与 Neumann 级数",
      "退化核方程",
      "Fredholm 择一定理",
      "Hilbert–Schmidt 理论",
      "对称核的展开定理",
      "Volterra 方程与预解核",
      "奇异积分方程",
      "Wiener–Hopf 方法",
      "积分方程与微分方程边值问题的等价",
      "积分方程的数值方法（Nyström 法 / 配置法）"
    ],
  },
  'intermediate/computability-theory': {
    title: "可计算性理论（递归论）",
    books: [
          "S. B. Cooper, Computability Theory, Chapman & Hall/CRC, 2004",
          "R. I. Soare, Recursively Enumerable Sets and Degrees, Springer, 1987",
          "N. Cutland, Computability, Cambridge University Press, 1980"
    ],
    chapters: [
      "Turing 机与 Church–Turing 论题",
      "原始递归函数与部分递归函数",
      "可计算枚举集",
      "停机问题与不可判定问题",
      "多一归约与 Turing 归约",
      "跳跃算子与 Turing 度",
      "优先方法（有穷损害 / 无穷损害）",
      "算术分层",
      "相对可计算性与 oracle 计算",
      "Post 问题与度的结构",
      "算法随机性（Martin-Löf 随机性）引论",
      "可计算模型论与可计算分析简介"
    ],
  },
  'intermediate/proof-theory': {
    title: "证明论",
    books: [
          "A. S. Troelstra & H. Schwichtenberg, Basic Proof Theory, Cambridge University Press (2nd ed., 2000)",
          "G. Takeuti, Proof Theory, North-Holland (2nd ed., 1987)",
          "S. R. Buss (ed.), Handbook of Proof Theory, Elsevier, 1998"
    ],
    chapters: [
      "自然演绎系统",
      "Gentzen 序贯演算（LK / LJ）",
      "截消定理（cut elimination）",
      "规范化与子公式性质",
      "直觉主义逻辑",
      "Peano 算术与归纳",
      "Gentzen 一致性证明与 ε₀",
      "序数分析引论",
      "Curry–Howard 对应",
      "Gödel 不完全性定理的证明论视角",
      "证明挖掘（proof mining）简介"
    ],
  },
  'intermediate/algebraic-k-theory': {
    title: "代数 K 理论",
    books: [
          "J. Rosenberg, Algebraic K-Theory and Its Applications, Springer GTM 147, 1994",
          "C. A. Weibel, The K-book: An Introduction to Algebraic K-theory, AMS GSM 145, 2013",
          "H. Bass, Algebraic K-Theory, Benjamin, 1968"
    ],
    chapters: [
      "投射模与 Grothendieck 群 K₀",
      "K₁ 与 Whitehead 引理",
      "Steinberg 群与 K₂",
      "Milnor K 理论",
      "Swan 定理（拓扑与代数的联系）",
      "Quillen 的 +-构造",
      "Quillen 的 Q-构造",
      "高阶 K 群的谱序列",
      "Bass–Heller–Swan 定理",
      "群环的 K 理论与 Whitehead 挠率",
      "与代数数论的联系（类群与调节子）",
      "拓扑 K 理论概览"
    ],
  },
  'intermediate/riemann-surfaces': {
    title: "黎曼曲面",
    books: [
          "O. Forster, Lectures on Riemann Surfaces, Springer GTM 81, 1981",
          "R. Miranda, Algebraic Curves and Riemann Surfaces, AMS GSM 5, 1995",
          "H. M. Farkas & I. Kra, Riemann Surfaces, Springer GTM 71 (2nd ed., 1992)"
    ],
    chapters: [
      "黎曼曲面的定义与例子",
      "全纯映射与分歧覆盖",
      "Riemann–Hurwitz 公式",
      "微分形式与留数定理",
      "层与层上同调初步",
      "Riemann–Roch 定理",
      "Serre 对偶",
      "Abel 定理与 Jacobi 反演",
      "单值化定理",
      "紧黎曼曲面与代数曲线",
      "椭圆曲线作为黎曼曲面",
      "模空间与 Teichmüller 理论引论"
    ],
  },
  'intermediate/discrete-and-convex-geometry': {
    title: "离散与凸几何",
    books: [
          "J. Matoušek, Lectures on Discrete Geometry, Springer GTM 212, 2002",
          "R. Schneider, Convex Bodies: The Brunn–Minkowski Theory, Cambridge University Press (2nd ed., 2014)",
          "B. Grünbaum, Convex Polytopes, Springer GTM 221 (2nd ed., 2003)"
    ],
    chapters: [
      "凸集与分离定理",
      "Carathéodory / Helly / Radon 定理",
      "凸多面体与 f-向量",
      "格点几何与 Minkowski 基本定理",
      "Ehrhart 多项式",
      "Brunn–Minkowski 不等式",
      "等周不等式",
      "Borsuk–Ulam 定理及其组合应用",
      "填充与覆盖问题",
      "组合几何中的 Erdős 型问题",
      "VC 维与 ε-网",
      "与线性规划及计算几何的联系"
    ],
  },
  'intermediate/inverse-problems': {
    title: "反问题与正则化方法",
    books: [
          "A. Kirsch, An Introduction to the Mathematical Theory of Inverse Problems, Springer (3rd ed., 2021)",
          "H. W. Engl, M. Hanke & A. Neubauer, Regularization of Inverse Problems, Kluwer, 1996",
          "P. C. Hansen, Discrete Inverse Problems: Insight and Algorithms, SIAM, 2010"
    ],
    chapters: [
      "反问题与 Hadamard 适定性",
      "紧算子的奇异系统",
      "Tikhonov 正则化",
      "截断奇异值分解（TSVD）",
      "正则化参数选择（偏差原理 / L 曲线）",
      "迭代正则化（Landweber / 共轭梯度）",
      "全变差与稀疏正则化",
      "Radon 变换与计算机层析成像",
      "逆散射问题引论",
      "贝叶斯反问题",
      "典型应用：医学成像与地球物理反演"
    ],
  },
  'intermediate/enumerative-combinatorics': {
    title: "计数组合学与概率方法",
    books: [
          "R. P. Stanley, Enumerative Combinatorics, Vol. 1 & 2, Cambridge University Press, 1997/1999 (Vol. 1, 2nd ed. 2011)",
          "N. Alon & J. H. Spencer, The Probabilistic Method, Wiley (4th ed., 2016)",
          "J. H. van Lint & R. M. Wilson, A Course in Combinatorics, Cambridge University Press (2nd ed., 2001)"
    ],
    chapters: [
      "普通生成函数与指数生成函数",
      "递推关系求解",
      "容斥原理",
      "整数分拆",
      "Catalan 数与格路计数",
      "Pólya 计数定理",
      "对称函数引论",
      "概率方法：期望与删除法",
      "第二矩方法",
      "Lovász 局部引理",
      "相关不等式（FKG）",
      "Turán 定理与 Ramsey 数的概率下界"
    ],
  },
  'intermediate/nonparametric-statistics': {
    title: "非参数统计",
    books: [
          "L. Wasserman, All of Nonparametric Statistics, Springer, 2006",
          "M. Hollander, D. A. Wolfe & E. Chicken, Nonparametric Statistical Methods, Wiley (3rd ed., 2014)"
    ],
    chapters: [
      "次序统计量及其分布",
      "符号检验与 Wilcoxon 符号秩检验",
      "Wilcoxon 秩和检验与 Mann–Whitney U",
      "Kruskal–Wallis 与 Friedman 检验",
      "Kolmogorov–Smirnov 拟合优度检验",
      "置换检验与自助法（bootstrap）",
      "U 统计量",
      "核密度估计与带宽选择",
      "非参数回归：核与局部多项式",
      "光滑样条",
      "渐近相对效率（Pitman ARE）",
      "半参数模型引论"
    ],
  },
  'intermediate/design-of-experiments': {
    title: "试验设计与方差分析",
    books: [
          "D. C. Montgomery, Design and Analysis of Experiments, Wiley (10th ed., 2019)",
          "G. E. P. Box, J. S. Hunter & W. G. Hunter, Statistics for Experimenters, Wiley (2nd ed., 2005)",
          "C. F. J. Wu & M. S. Hamada, Experiments: Planning, Analysis, and Optimization, Wiley (2nd ed., 2009)"
    ],
    chapters: [
      "试验设计基本原则（随机化/重复/区组）",
      "单因素方差分析",
      "多重比较",
      "随机化区组与拉丁方设计",
      "多因素方差分析与交互作用",
      "2^k 与 3^k 因子设计",
      "部分因子设计与混杂",
      "正交表与正交试验设计",
      "裂区设计",
      "响应面方法",
      "稳健参数设计（田口方法）",
      "最优设计（D-最优）引论"
    ],
  },
  'intermediate/survival-analysis': {
    title: "生存分析",
    books: [
          "J. D. Kalbfleisch & R. L. Prentice, The Statistical Analysis of Failure Time Data, Wiley (2nd ed., 2002)",
          "J. P. Klein & M. L. Moeschberger, Survival Analysis: Techniques for Censored and Truncated Data, Springer (2nd ed., 2003)",
          "T. R. Fleming & D. P. Harrington, Counting Processes and Survival Analysis, Wiley, 1991"
    ],
    chapters: [
      "生存函数与危险函数",
      "删失与截尾机制",
      "Kaplan–Meier 估计",
      "log-rank 检验",
      "Cox 比例风险模型",
      "部分似然推断",
      "参数生存模型（指数 / Weibull）",
      "加速失效时间模型",
      "时依协变量",
      "竞争风险",
      "脆弱模型（frailty）",
      "计数过程与鞅方法引论"
    ],
  },
  'advanced/random-matrix-theory': {
    title: "随机矩阵理论",
    books: [
          "M. L. Mehta, Random Matrices, Elsevier (3rd ed., 2004)",
          "G. W. Anderson, A. Guionnet & O. Zeitouni, An Introduction to Random Matrices, Cambridge University Press, 2010",
          "T. Tao, Topics in Random Matrix Theory, AMS GSM 132, 2012"
    ],
    chapters: [
      "Wigner 半圆律",
      "高斯系综（GOE / GUE / GSE）",
      "特征值联合密度与 Vandermonde 行列式",
      "行列式点过程",
      "Tracy–Widom 分布",
      "Marchenko–Pastur 律",
      "自由概率引论",
      "Dyson 布朗运动",
      "普适性问题",
      "稀疏随机矩阵",
      "与数论（L 函数零点）的联系",
      "在无线通信与高维统计中的应用"
    ],
  },
  'intermediate/geometric-measure-theory': {
    title: "几何测度论",
    books: [
          "P. Mattila, Geometry of Sets and Measures in Euclidean Spaces, Cambridge University Press, 1995",
          "H. Federer, Geometric Measure Theory, Springer, 1969",
          "L. C. Evans & R. F. Gariepy, Measure Theory and Fine Properties of Functions, CRC Press (revised ed., 2015)"
    ],
    chapters: [
      "Hausdorff 测度与 Hausdorff 维数",
      "覆盖定理（Vitali / Besicovitch）",
      "密度定理",
      "Rademacher 定理与 Lipschitz 函数",
      "整流（rectifiable）集与切空间",
      "面积公式与余面积公式",
      "Frostman 引理与维数估计",
      "变分几何（varifolds）引论",
      "电流（currents）引论",
      "极小曲面正则性引论",
      "Marstrand 射影定理",
      "与分形几何的联系"
    ],
  },
  'intermediate/markov-chains-and-mixing-times': {
    title: "马尔可夫链与混合时间",
    books: [
          "D. A. Levin, Y. Peres & E. L. Wilmer, Markov Chains and Mixing Times, AMS (2nd ed., 2017)",
          "J. R. Norris, Markov Chains, Cambridge University Press, 1997"
    ],
    chapters: [
      "有限马尔可夫链回顾",
      "不可约性与周期性",
      "平稳分布与细致平衡",
      "全变差距离",
      "耦合方法与混合时间上界",
      "谱隙与特征值方法",
      "对数 Sobolev 不等式引论",
      "洗牌模型",
      "Glauber 动力学与 Ising 模型",
      "Metropolis–Hastings 算法",
      "MCMC 收敛诊断",
      "cutoff 现象"
    ],
  },
  'intermediate/finite-group-theory': {
    title: "有限群论",
    books: [
          "I. M. Isaacs, Finite Group Theory, AMS GSM 92, 2008",
          "D. J. S. Robinson, A Course in the Theory of Groups, Springer GTM 80 (2nd ed., 1996)",
          "D. Gorenstein, Finite Groups, AMS Chelsea (2nd ed., 1980)"
    ],
    chapters: [
      "Sylow 定理",
      "p-群的结构",
      "幂零群",
      "可解群",
      "群作用与置换群",
      "传递群与本原群",
      "合成列与 Jordan–Hölder 定理",
      "有限单群分类定理概览",
      "交错群与散在单群（Mathieu 群）",
      "Frobenius 定理引论",
      "转移（transfer）与同调初步",
      "局部分析与融合引论"
    ],
  },
  'intermediate/universal-algebra': {
    title: "泛代数（万有代数）",
    books: [
          "S. Burris & H. P. Sankappanavar, A Course in Universal Algebra, Springer GTM 78, 1981",
          "C. Bergman, Universal Algebra: Fundamentals and Selected Topics, CRC Press, 2012",
          "R. McKenzie, G. McNulty & W. Taylor, Algebras, Lattices, Varieties, Vol. I, AMS Chelsea, 1987"
    ],
    chapters: [
      "代数与同类（type）",
      "子代数与同态",
      "同构定理",
      "同余格",
      "自由代数",
      "项与等式",
      "簇（variety）与 Birkhoff HSP 定理",
      "等式逻辑的完备性",
      "Mal'cev 条件",
      "判别性（primality）引论",
      "克隆（clone）理论初步",
      "与格论及模型论的联系"
    ],
  },
  'intermediate/potential-theory': {
    title: "位势论",
    books: [
          "L. L. Helms, Potential Theory, Springer Universitext (2nd ed., 2014)",
          "J. L. Doob, Classical Potential Theory and Its Probabilistic Counterpart, Springer, 1984",
          "T. Ransford, Potential Theory in the Complex Plane, Cambridge University Press, 1995"
    ],
    chapters: [
      "调和函数回顾",
      "次调和函数",
      "Green 函数",
      "容量与能量",
      "平衡测度",
      "Dirichlet 问题的 Perron 方法",
      "正则边界点",
      "Riesz 分解定理",
      "细拓扑引论",
      "Martin 边界",
      "对数位势与复分析的联系",
      "概率位势论：与布朗运动的联系"
    ],
  },
  'intermediate/difference-equations': {
    title: "差分方程",
    books: [
          "S. Elaydi, An Introduction to Difference Equations, Springer (3rd ed., 2005)",
          "W. G. Kelley & A. C. Peterson, Difference Equations: An Introduction with Applications, Academic Press (2nd ed., 2001)"
    ],
    chapters: [
      "一阶线性差分方程",
      "高阶线性差分方程",
      "Z 变换",
      "平衡解与稳定性判据",
      "线性差分方程组",
      "相图与周期解",
      "Logistic 方程与混沌",
      "非线性差分方程",
      "振荡理论初步",
      "函数方程（Cauchy / Jensen 方程）",
      "微分方程离散化与差分格式",
      "应用：人口模型与经济动态"
    ],
  },
  'intermediate/nonstandard-analysis': {
    title: "非标准分析",
    books: [
          "R. Goldblatt, Lectures on the Hyperreals, Springer GTM 188, 1998",
          "A. Robinson, Non-standard Analysis, Princeton University Press (rev. ed., 1996)",
          "A. E. Hurd & P. A. Loeb, An Introduction to Nonstandard Real Analysis, Academic Press, 1985"
    ],
    chapters: [
      "超实数的构造（超滤子）",
      "转移原理",
      "无穷小与无穷大量",
      "标准部分映射",
      "连续性与极限的非标准刻画",
      "微积分基本定理的非标准证明",
      "紧致性与饱和性",
      "超有限集",
      "Loeb 测度",
      "非标准拓扑初步",
      "与标准分析的保守性",
      "应用：组合数论与随机过程简介"
    ],
  },
  'foundations/how-to-prove': {
    title: "数学证明导论（证明方法）",
    books: [
          "D. J. Velleman, How to Prove It: A Structured Approach, Cambridge University Press (3rd ed., 2019)",
          "R. Hammack, Book of Proof (3rd ed., 2018)"
    ],
    chapters: [
      "命题与逻辑联结词",
      "量词与否定",
      "直接证明",
      "逆否命题法",
      "反证法",
      "数学归纳法与强归纳法",
      "存在性与唯一性证明",
      "集合、关系与函数",
      "等价关系与划分",
      "基数初步与可数性",
      "反例的构造",
      "数学写作规范"
    ],
  },
  'intermediate/physiology': {
    title: "生理学",
    books: [
          "王庭槐主编《生理学》（第9版，人民卫生出版社，2018）",
          "Guyton & Hall, Textbook of Medical Physiology（Elsevier，第14版，2020）",
          "Boron & Boulpaep, Medical Physiology（Elsevier，第3版，2016）"
    ],
    chapters: [
      "细胞的基本功能（膜电位/信号转导）",
      "血液生理",
      "血液循环生理",
      "呼吸生理",
      "消化与吸收",
      "能量代谢与体温",
      "尿的生成与排出",
      "感觉器官生理",
      "神经系统生理",
      "内分泌生理",
      "生殖生理",
      "稳态与生理调节"
    ],
  },
  'intermediate/pharmacology': {
    title: "药理学",
    books: [
          "杨宝峰主编《药理学》（第9版，人民卫生出版社，2018）",
          "Brunton et al., Goodman & Gilman's The Pharmacological Basis of Therapeutics（McGraw-Hill，第13版，2018）",
          "Katzung et al., Basic & Clinical Pharmacology（McGraw-Hill，第16版，2024）"
    ],
    chapters: [
      "药物效应动力学",
      "药物代谢动力学",
      "传出神经系统药物",
      "中枢神经系统药物",
      "心血管系统药物",
      "血液与造血系统药物",
      "内分泌系统药物",
      "化学治疗药物（抗菌/抗病毒/抗肿瘤）",
      "抗炎与免疫调节药物",
      "影响自体活性物质药物",
      "新药研发与临床试验概论"
    ],
  },
  'life/medical-psychology': {
    title: "医学心理学",
    books: [
          "姚树桥、杨彦春主编《医学心理学》（第7版，人民卫生出版社，2018）",
          "姜乾金主编《医学心理学》（人民卫生出版社，八年制规划教材）"
    ],
    chapters: [
      "心理学基础（认知/情绪/人格）",
      "心理应激与应对",
      "心身疾病",
      "心理健康与心理障碍总论",
      "心理评估（量表与访谈）",
      "心理咨询与心理治疗",
      "医患关系与医患沟通",
      "病人心理与角色适应",
      "疼痛与心理",
      "睡眠与心理",
      "临终心理与哀伤辅导"
    ],
  },
  'life/child-adolescent-and-maternal-health': {
    title: "儿少卫生与妇幼保健学",
    books: [
          "季成叶主编《儿童少年卫生学》（人民卫生出版社，预防医学规划教材）",
          "Kotch, Maternal and Child Health: Programs, Problems, and Policy in Public Health（Jones & Bartlett）"
    ],
    chapters: [
      "儿童生长发育规律与评价",
      "青春期发育与卫生",
      "儿童心理行为发育",
      "学校卫生与学习环境",
      "儿童常见病防治（近视/龋齿/肥胖）",
      "儿童意外伤害预防",
      "婚前与孕前保健",
      "孕产期保健与高危妊娠管理",
      "新生儿与婴幼儿保健",
      "母乳喂养与婴幼儿喂养",
      "妇女常见病防治与两癌筛查",
      "妇幼保健指标与体系建设"
    ],
  },
  'life/health-education-and-promotion': {
    title: "健康教育与健康促进",
    books: [
          "傅华主编《健康教育学》（第3版，人民卫生出版社，2017）",
          "Glanz, Rimer & Viswanath, Health Behavior: Theory, Research, and Practice（Jossey-Bass，第5版）"
    ],
    chapters: [
      "健康行为理论（知信行/健康信念模式/行为阶段改变）",
      "健康传播学基础",
      "健康教育计划设计（PRECEDE-PROCEED 模式）",
      "社区健康教育",
      "学校健康教育",
      "职业场所健康促进",
      "医院健康教育与患者教育",
      "健康促进与《渥太华宪章》",
      "健康城市与健康场所建设",
      "健康素养评估与提升",
      "控烟与生活方式干预",
      "健康教育项目效果评价"
    ],
  },
  'life/clinical-pharmacology': {
    title: "临床药理学",
    books: [
          "李俊主编《临床药理学》（第6版，人民卫生出版社，2018）",
          "Rowland & Tozer, Clinical Pharmacokinetics and Pharmacodynamics（Wolters Kluwer，第4版）"
    ],
    chapters: [
      "临床药物代谢动力学",
      "治疗药物监测（TDM）",
      "药物相互作用",
      "药品不良反应与药物警戒",
      "特殊人群用药（妊娠/儿童/老年）",
      "肝肾功能不全患者用药调整",
      "药物基因组学与个体化用药",
      "新药临床试验（I–IV期）设计",
      "生物等效性评价",
      "抗菌药物临床应用管理",
      "循证用药与合理用药评价"
    ],
  },
  'life/radiation-oncology': {
    title: "放射肿瘤学（肿瘤放射治疗）",
    books: [
          "殷蔚伯、余子豪主编《肿瘤放射治疗学》（第4版，中国协和医科大学出版社，2008）",
          "Halperin et al., Perez and Brady's Principles and Practice of Radiation Oncology（Wolters Kluwer，第7版）"
    ],
    chapters: [
      "放射物理学基础",
      "放射生物学基础（4R/线性二次模型）",
      "临床剂量学与治疗计划",
      "外照射技术（IMRT/IGRT/SBRT）",
      "近距离放射治疗",
      "头颈部肿瘤放疗",
      "胸部肿瘤放疗（肺癌/食管癌）",
      "腹部与盆腔肿瘤放疗",
      "妇科肿瘤放疗",
      "放射反应与正常组织防护",
      "放疗与化疗/免疫综合治疗"
    ],
  },
  'life/interventional-radiology': {
    title: "介入放射学",
    books: [
          "郭启勇主编《介入放射学》（人民卫生出版社，医学影像学专业规划教材）",
          "Kandarpa & Machan, Handbook of Interventional Radiologic Procedures（Wolters Kluwer，第5版）"
    ],
    chapters: [
      "血管介入基本技术（Seldinger 技术/导管导丝）",
      "血管造影与血管栓塞",
      "肿瘤介入治疗（TACE/消融）",
      "血管成形术与支架置入",
      "经皮穿刺引流与活检",
      "非血管管腔成形（胆道/食管/气道）",
      "神经介入（取栓/动脉瘤栓塞）",
      "出血性疾病的急诊栓塞",
      "静脉介入（滤器/TIPS/输液港）",
      "介入并发症与围手术期管理"
    ],
  },
  'life/pain-medicine': {
    title: "疼痛医学",
    books: [
          "谭冠先主编《疼痛诊疗学》（第3版，人民卫生出版社，2011；第4版郭政、王国年主编）",
          "Fishman et al., Bonica's Management of Pain（Wolters Kluwer，第5版）"
    ],
    chapters: [
      "疼痛的基础理论（闸门学说/神经可塑性）",
      "疼痛评估与诊断学基础",
      "疼痛药物治疗（NSAIDs/阿片类/辅助用药）",
      "神经阻滞与局部注射治疗",
      "微创介入镇痛技术",
      "头面部疼痛",
      "颈肩腰背痛",
      "神经病理性疼痛",
      "癌性疼痛三阶梯治疗",
      "术后镇痛",
      "分娩镇痛",
      "疼痛的多学科综合管理"
    ],
  },
  'life/sleep-medicine': {
    title: "睡眠医学",
    books: [
          "赵忠新、叶京英主编《睡眠医学》（第2版，人民卫生出版社，2022）",
          "Kryger, Roth & Dement, Principles and Practice of Sleep Medicine（Elsevier，第7版）"
    ],
    chapters: [
      "睡眠生理与生物节律",
      "多导睡眠监测（PSG）与睡眠分期",
      "失眠障碍与认知行为治疗（CBT-I）",
      "睡眠呼吸障碍（OSA/中枢性）",
      "中枢性嗜睡（发作性睡病）",
      "异态睡眠（梦游/REM 睡眠行为障碍）",
      "睡眠相关运动障碍（不宁腿）",
      "昼夜节律睡眠-觉醒障碍",
      "儿童睡眠障碍",
      "睡眠障碍与精神疾病共病",
      "睡眠与慢性病的双向关系"
    ],
  },
  'life/transfusion-medicine': {
    title: "输血医学",
    books: [
          "Cohn et al., Technical Manual（AABB，第21版，2023）",
          "Shaz et al., Transfusion Medicine and Hemostasis: Clinical and Laboratory Aspects（Elsevier，第3版）"
    ],
    chapters: [
      "血型系统与免疫血液学（ABO/Rh）",
      "血液成分制备与保存",
      "交叉配血与抗体筛查",
      "全血与成分输血指征",
      "自体输血与血液保护",
      "输血不良反应（溶血/TA-GVHD/TRALI）",
      "输血传播感染与血液安全",
      "新生儿溶血病",
      "大量输血与创伤救治输血",
      "治疗性单采与血浆置换",
      "造血干细胞移植相关输血",
      "合理用血与输血质量管理"
    ],
  },
  'life/tropical-medicine': {
    title: "热带医学与旅行医学",
    books: [
          "Farrar et al., Manson's Tropical Diseases（Elsevier，第24版，2023）",
          "Ryan et al., Hunter's Tropical Medicine and Emerging Infectious Diseases（Elsevier，第10版）"
    ],
    chapters: [
      "疟疾",
      "血吸虫病",
      "利什曼病与锥虫病",
      "淋巴丝虫病与盘尾丝虫病",
      "登革热及其他虫媒病毒病",
      "肠道原虫与蠕虫感染",
      "霍乱与旅行者腹泻",
      "麻风病",
      "被忽视热带病（NTDs）控制策略",
      "旅行前咨询与预防用药",
      "气候变化与热带病分布变迁"
    ],
  },
  'life/disaster-medicine': {
    title: "灾难医学",
    books: [
          "Ciottone et al., Ciottone's Disaster Medicine（Elsevier，第2版，2016）",
          "Koenig & Schultz, Disaster Medicine: Comprehensive Principles and Practices（Cambridge University Press，第2版）"
    ],
    chapters: [
      "灾难分类与医学救援体系",
      "灾害现场检伤分类（START）",
      "批量伤员的组织与转运",
      "创伤急救与损伤控制",
      "挤压综合征与地震伤",
      "核与辐射事故医学处置",
      "化学与生物恐怖应对",
      "灾后防疫与公共卫生应急",
      "心理危机干预",
      "国际人道主义医疗救援",
      "应急预案与模拟演练"
    ],
  },
  'life/optometry': {
    title: "眼视光学",
    books: [
          "瞿佳主编《眼视光学理论和方法》（第3版，人民卫生出版社，2018）",
          "Benjamin & Borish, Borish's Clinical Refraction（Elsevier，第2版）"
    ],
    chapters: [
      "眼球光学与屈光基础",
      "视力与视觉功能检查",
      "客观与主观验光",
      "屈光不正与矫正（框架眼镜）",
      "角膜接触镜学",
      "双眼视觉与视功能异常",
      "斜视与弱视的视光学处理",
      "老视与渐进镜",
      "低视力康复",
      "屈光手术概论",
      "儿童青少年近视防控",
      "视光门诊与眼病筛查转诊"
    ],
  },
  'life/audiology-speech-language-pathology': {
    title: "听力学与言语语言病理学",
    books: [
          "李胜利、陈卓铭主编《语言治疗学》（第3版，人民卫生出版社，2018）",
          "Martin & Clark, Introduction to Audiology（Pearson，第13版）"
    ],
    chapters: [
      "听觉系统解剖生理",
      "听力学检测（纯音测听/声导抗/耳声发射/ABR）",
      "听力障碍分类与干预策略",
      "助听器验配",
      "人工耳蜗与听觉康复",
      "言语产生机制与语音学",
      "失语症评估与治疗",
      "构音障碍与运动性言语障碍",
      "儿童语言发育迟缓",
      "流畅度障碍（口吃）",
      "吞咽障碍评估与治疗",
      "嗓音障碍与嗓音训练"
    ],
  },
  'life/addiction-medicine': {
    title: "成瘾医学",
    books: [
          "Miller et al., The ASAM Principles of Addiction Medicine（Wolters Kluwer，第6版）",
          "Johnson, Addiction Medicine: Science and Practice（Springer，2011）"
    ],
    chapters: [
      "成瘾的神经生物学（奖赏环路/多巴胺）",
      "酒精使用障碍",
      "阿片类使用障碍与美沙酮维持",
      "兴奋剂与新型精神活性物质",
      "烟草依赖与戒烟治疗",
      "镇静催眠药依赖",
      "行为成瘾（赌博/游戏障碍）",
      "成瘾筛查与 DSM-5 诊断",
      "脱毒治疗与急性戒断管理",
      "动机访谈与认知行为治疗",
      "复吸预防与长期康复",
      "成瘾与精神障碍共病"
    ],
  },
  'humanities/history-of-medicine': {
    title: "医学史",
    books: [
          "张大庆《医学史十五讲》（北京大学出版社，2007）",
          "Roy Porter, The Greatest Benefit to Mankind: A Medical History of Humanity（W.W. Norton，1997）",
          "张大庆主编《医学史》（北京大学医学出版社，医学人文规划教材）"
    ],
    chapters: [
      "原始医学与古代文明医学（埃及/美索不达米亚）",
      "古希腊罗马医学（希波克拉底/盖伦）",
      "中医学的形成与经典体系",
      "中世纪医学与伊斯兰医学",
      "文艺复兴与解剖学革命（维萨里）",
      "血液循环的发现与生理学奠基",
      "临床医学的诞生（医院医学/物理诊断）",
      "细菌学说与传染病学的革命",
      "麻醉、无菌技术与现代外科",
      "疫苗与免疫学的兴起",
      "公共卫生运动史",
      "20 世纪生物医学（抗生素/影像/分子医学）",
      "医学职业与医疗制度史"
    ],
  },
  'life/medical-education': {
    title: "医学教育学",
    books: [
          "Swanwick et al. (eds.), Understanding Medical Education: Evidence, Theory and Practice（Wiley-Blackwell，第3版）",
          "Dent, Harden & Hunt (eds.), A Practical Guide for Medical Teachers（Elsevier，第6版）"
    ],
    chapters: [
      "医学教育史与 Flexner 报告",
      "课程体系设计（整合课程/PBL/器官系统课程）",
      "胜任力导向医学教育（CBME）",
      "临床技能教学与模拟医学教育",
      "床旁教学与临床带教",
      "形成性评价与 OSCE 考核",
      "住院医师规范化培训制度",
      "继续医学教育与终身学习",
      "医学教育研究方法",
      "教育技术与在线医学教育",
      "医学教师发展",
      "医学教育认证与质量保障"
    ],
  },
  'humanities/critical-thinking': {
    title: "批判性思维与非形式逻辑",
    books: [
          "Brooke Noel Moore and Richard Parker, \"Critical Thinking\" (13th ed., McGraw-Hill, 2020)",
          "Douglas N. Walton, \"Informal Logic: A Pragmatic Approach\" (2nd ed., Cambridge University Press, 2008)",
          "Irving M. Copi, Carl Cohen and Kenneth McMahon, \"Introduction to Logic\" (14th ed., Routledge, 2010)"
    ],
    chapters: [
      "论证的识别、结构与重构",
      "演绎论证与有效性",
      "归纳论证与归纳强度",
      "非形式谬误（相干/预设/歧义三大类）",
      "定义与语言澄清",
      "类比论证与法律/道德论证",
      "因果推理与密尔方法",
      "统计论证与概率推理",
      "诉诸权威的评估与信源可信度",
      "科学推理与假说检验",
      "决策、认知偏差与理性讨论规则",
      "批判性阅读与论证写作"
    ],
  },
  'humanities/axiology': {
    title: "价值论（价值哲学）",
    books: [
          "李德顺, 《价值论——一种主体性的研究》（第3版, 中国人民大学出版社, 2013）",
          "袁贵仁, 《价值学引论》（北京师范大学出版社, 1991）",
          "Nicholas Rescher, \"Introduction to Value Theory\" (Prentice-Hall, 1969)"
    ],
    chapters: [
      "价值与事实的区分：休谟问题",
      "价值的本质：主观主义、客观主义与关系论",
      "价值的分类：功利/道德/审美/宗教价值",
      "内在价值与工具价值",
      "价值评价与评价标准",
      "价值认知与价值真理",
      "价值冲突与价值排序",
      "善、正当与权利的优先性之争",
      "价值的主体性与人的需要",
      "价值观的形成与社会价值体系",
      "多元主义与价值相对主义批判",
      "马克思主义价值论"
    ],
  },
  'humanities/business-ethics': {
    title: "商业伦理（企业伦理）",
    books: [
          "Manuel G. Velasquez, \"Business Ethics: Concepts and Cases\" (8th ed., Pearson, 2017)",
          "Tom L. Beauchamp, Norman E. Bowie and Denis G. Arnold, \"Ethical Theory and Business\" (9th ed., Pearson, 2012)",
          "周祖城, 《企业伦理学》（第3版, 清华大学出版社, 2015）"
    ],
    chapters: [
      "伦理理论与商业决策框架",
      "市场中的道德：自由市场批判与辩护",
      "企业社会责任（CSR）之争：Friedman 与利益相关者理论",
      "雇佣关系中的伦理：歧视、隐私与举报",
      "消费者保护、广告与营销伦理",
      "环境责任与可持续经营",
      "公司治理与内部人伦理",
      "会计诚信与财务舞弊",
      "全球供应链与跨国经营伦理",
      "技术、数据与算法带来的商业伦理新问题",
      "道德文化与合规体系建设"
    ],
  },
  'humanities/decision-theory': {
    title: "决策论与理性选择理论",
    books: [
          "Martin Peterson, \"An Introduction to Decision Theory\" (2nd ed., Cambridge University Press, 2017)",
          "Richard C. Jeffrey, \"The Logic of Decision\" (2nd ed., University of Chicago Press, 1983)",
          "R. Duncan Luce and Howard Raiffa, \"Games and Decisions\" (Wiley, 1957; Dover reprint, 1989)"
    ],
    chapters: [
      "决策问题的形式化：状态、行动与结果",
      "确定性、风险与不确定性下的决策",
      "期望效用理论与 vNM 公理",
      "主观概率与拉姆齐-德菲内蒂路径",
      "Allais 悖论与 Ellsberg 悖论",
      "贝叶斯决策理论与证据更新",
      "博弈论基础与纳什均衡的哲学地位",
      "囚徒困境、集体行动与社会选择",
      "Newcomb 问题与因果/证据决策论之争",
      "有限理性与描述性决策理论",
      "决策论在认识论与伦理学中的应用"
    ],
  },
  'humanities/philosophy-of-probability': {
    title: "概率哲学与归纳逻辑",
    books: [
          "Ian Hacking, \"An Introduction to Probability and Inductive Logic\" (Cambridge University Press, 2001)",
          "Donald Gillies, \"Philosophical Theories of Probability\" (Routledge, 2000)",
          "Brian Skyrms, \"Choice and Chance: An Introduction to Inductive Logic\" (4th ed., Wadsworth, 2000)"
    ],
    chapters: [
      "休谟归纳问题及其现代回应",
      "概率的古典解释与逻辑解释",
      "频率解释及其困难",
      "倾向解释",
      "主观（贝叶斯）解释与荷兰赌论证",
      "贝叶斯确证理论",
      "归纳逻辑：Carnap 体系及其遗产",
      "统计推断哲学：频率派 vs 贝叶斯派",
      "证据、相关与因果",
      "最佳解释推理（IBE）",
      "概率在法律、医学与日常推理中的应用"
    ],
  },
  'humanities/paradoxes': {
    title: "悖论",
    books: [
          "R.M. Sainsbury, \"Paradoxes\" (3rd ed., Cambridge University Press, 2009)",
          "Michael Clark, \"Paradoxes from A to Z\" (3rd ed., Routledge, 2012)",
          "Nicholas Rescher, \"Paradoxes: Their Roots, Range, and Resolution\" (Open Court, 2001)"
    ],
    chapters: [
      "悖论的概念与分类",
      "芝诺悖论与无穷",
      "说谎者悖论与语义封闭",
      "集合论悖论：罗素悖论",
      "语义悖论的解决：Tarski 与 Kripke",
      "连锁（堆垛）悖论与模糊性",
      "意外考试悖论与知识",
      "彩票悖论与序言悖论",
      "纽康姆悖论与决策",
      "谷堆/忒修斯之船与同一性",
      "全能悖论与宗教哲学中的悖论",
      "自指、对角化方法与不定点"
    ],
  },
  'humanities/ancient-greek-philosophy': {
    title: "古希腊罗马哲学",
    books: [
          "G.S. Kirk, J.E. Raven and M. Schofield, \"The Presocratic Philosophers\" (2nd ed., Cambridge University Press, 1983)",
          "W.K.C. Guthrie, \"A History of Greek Philosophy\" (6 vols., Cambridge University Press, 1962-1981)",
          "苗力田 主编, 《古希腊哲学》（中国人民大学出版社, 1989）"
    ],
    chapters: [
      "米利都学派与早期自然哲学",
      "赫拉克利特与毕达哥拉斯学派",
      "爱利亚学派：巴门尼德与芝诺",
      "多元论者：恩培多克勒与阿那克萨戈拉",
      "原子论：德谟克利特",
      "智者运动与普罗泰戈拉",
      "苏格拉底与德行即知识",
      "柏拉图：理念论、知识与政制",
      "亚里士多德：形而上学、伦理学与逻辑学",
      "希腊化哲学：伊壁鸠鲁学派与斯多亚学派",
      "怀疑论与犬儒学派",
      "新柏拉图主义：普罗提诺"
    ],
  },
  'humanities/medieval-philosophy': {
    title: "中世纪哲学",
    books: [
          "赵敦华, 《基督教哲学1500年》（人民出版社, 1994）",
          "Norman Kretzmann, Anthony Kenny and Jan Pinborg (eds.), \"The Cambridge History of Later Medieval Philosophy\" (Cambridge University Press, 1982)",
          "John Marenbon, \"Medieval Philosophy: An Historical and Philosophical Introduction\" (Routledge, 2007)"
    ],
    chapters: [
      "教父哲学与奥古斯丁",
      "波爱修与中世纪逻辑的开端",
      "中世纪早期：爱留根纳与加洛林复兴",
      "安瑟尔谟与本体论证明",
      "阿拉伯-犹太哲学的影响：阿维森纳、阿维洛伊、迈蒙尼德",
      "经院哲学的兴起与大学",
      "托马斯·阿奎那：存在、本质与自然法",
      "司各脱与唯意志论",
      "奥卡姆与唯名论革命",
      "中世纪共相之争",
      "中世纪晚期与向文艺复兴的过渡"
    ],
  },
  'humanities/german-classical-philosophy': {
    title: "德国古典哲学",
    books: [
          "俞吾金 等, 《德国古典哲学》（人民出版社, 2009）",
          "杨祖陶, 《德国古典哲学的逻辑进程》（修订版, 武汉大学出版社, 2003）",
          "邓晓芒, 《德国古典哲学讲演录》（湖南教育出版社, 2010）"
    ],
    chapters: [
      "德国古典哲学的产生背景与启蒙",
      "康德哲学的形成与总问题",
      "康德先验感性论与时空观",
      "康德先验分析论：范畴与图型",
      "康德先验辩证论与理性批判",
      "康德道德哲学：绝对命令",
      "康德判断力批判与美学/目的论",
      "费希特的知识学",
      "谢林的自然哲学与同一哲学",
      "黑格尔精神现象学",
      "黑格尔逻辑学与辩证法",
      "黑格尔法哲学与历史哲学",
      "费尔巴哈与德国古典哲学的终结"
    ],
  },
  'humanities/neo-confucianism': {
    title: "宋明理学",
    books: [
          "陈来, 《宋明理学》（第2版, 华东师范大学出版社, 2004）",
          "钱穆, 《宋明理学概述》（台湾学生书局, 1977；九州出版社简体版, 2010）",
          "张立文, 《宋明理学研究》（中国人民大学出版社, 1985）"
    ],
    chapters: [
      "宋明理学的产生：韩愈、李翱与儒学复兴",
      "北宋五子：周敦颐与《太极图说》",
      "邵雍的象数学",
      "张载的气论与《西铭》",
      "程颢、程颐与天理本体",
      "朱熹：理气论、心性论与格物致知",
      "陆九渊与心即理",
      "朱陆之争与鹅湖之会",
      "王阳明：心即理、知行合一与致良知",
      "阳明后学的分化",
      "罗钦顺、王廷相的气学回应",
      "刘宗周与明清之际的理学总结"
    ],
  },
  'humanities/african-philosophy': {
    title: "非洲哲学",
    books: [
          "Kwasi Wiredu (ed.), \"A Companion to African Philosophy\" (Blackwell, 2004)",
          "Samuel Oluoch Imbo, \"An Introduction to African Philosophy\" (Rowman & Littlefield, 1998)"
    ],
    chapters: [
      "非洲哲学是否存在：学科合法性问题",
      "民族哲学（ethnophilosophy）之争",
      "智者哲学：Ogotemmêli 与 Oruka 的方案",
      "专业哲学与批判传统：Hountondji",
      "Ubuntu 伦理学",
      "非洲形而上学：存在、人格与时间观",
      "非洲政治哲学：公社主义与 Nyerere",
      "埃及与埃塞俄比亚古典文本传统",
      "非洲逻辑与认识论问题",
      "后殖民哲学与去殖民化",
      "非洲哲学与西方哲学的对话"
    ],
  },
  'humanities/latin-american-philosophy': {
    title: "拉丁美洲哲学",
    books: [
          "Susana Nuccetelli, \"Latin American Thought: Philosophical Problems and Arguments\" (Rowman & Littlefield, 2002)",
          "Jorge J.E. Gracia and Elizabeth Millán-Zaibert (eds.), \"Latin American Philosophy for the 21st Century\" (Prometheus Books, 2004)"
    ],
    chapters: [
      "殖民时期的经院传统",
      "独立运动与实证主义",
      "二十世纪「正常性」之争：拉美是否有自己的哲学",
      "存在主义与现象学在拉美",
      "解放哲学：Enrique Dussel",
      "解放神学及其哲学基础",
      "拉美马克思主义与依赖理论",
      "Mariátegui 与印第安主义",
      "分析哲学在拉美的兴起",
      "身份、 mestizaje 与文化哲学",
      "去殖民性（decoloniality）思潮"
    ],
  },
  'humanities/social-epistemology': {
    title: "社会认识论",
    books: [
          "Alvin I. Goldman, \"Knowledge in a Social World\" (Oxford University Press, 1999)",
          "Alvin I. Goldman and Dennis Whitcomb (eds.), \"Social Epistemology: Essential Readings\" (Oxford University Press, 2011)",
          "Miranda Fricker, \"Epistemic Injustice: Power and the Ethics of Knowing\" (Oxford University Press, 2007)"
    ],
    chapters: [
      "社会认识论的兴起与定位",
      "证言认识论",
      "专家意见的分歧与应对",
      "群体信念与集体认识主体",
      "社会制度与真理追求（veritism）",
      "互联网与信息生态",
      "回音室、假信息与认识污染",
      "认识不正义：证言与诠释的不正义",
      "科学共同体的认识结构",
      "民主的认识论基础",
      "社会认识论与女性主义认识论的交叉"
    ],
  },
  'humanities/philosophy-of-ai': {
    title: "人工智能哲学",
    books: [
          "Margaret A. Boden (ed.), \"The Philosophy of Artificial Intelligence\" (Oxford University Press, 1990)",
          "Margaret A. Boden, \"AI: Its Nature and Future\" (Oxford University Press, 2016)"
    ],
    chapters: [
      "图灵测试及其批评",
      "强 AI 与中文屋论证（Searle）",
      "计算主义与心灵的计算理论",
      "符号主义 vs 联结主义的哲学意涵",
      "意向性、意义与接地问题",
      "机器意识的可能性",
      "具身性与情境认知对 AI 的挑战",
      "创造力与美感能否被机器实现",
      "大语言模型带来的哲学新问题",
      "AI 与自由意志、道德主体地位",
      "奇点论证与超级智能的哲学评估"
    ],
  },
  'humanities/history-of-western-ethics': {
    title: "西方伦理思想史",
    books: [
          "宋希仁 主编, 《西方伦理思想史》（中国人民大学出版社, 2004）",
          "万俊人, 《现代西方伦理学史》（上、下卷, 北京大学出版社, 1990-1992）",
          "Alasdair MacIntyre, \"A Short History of Ethics\" (2nd ed., Routledge, 1998)"
    ],
    chapters: [
      "古希腊伦理学：从荷马到苏格拉底",
      "柏拉图与亚里士多德的德性伦理",
      "希腊化时期：伊壁鸠鲁与斯多亚伦理学",
      "基督教伦理：奥古斯丁与阿奎那",
      "近代自然法与社会契约伦理",
      "英国道德感学派：沙夫茨伯里到休谟",
      "康德的义务论体系",
      "功利主义的形成：边沁与密尔",
      "黑格尔与马克思的伦理思想",
      "尼采的道德批判",
      "二十世纪元伦理学转向",
      "当代规范理论的复兴与德性伦理复兴"
    ],
  },
  'intermediate/experimental-physics-methods': {
    title: "物理实验方法与误差分析",
    books: [
          "J. R. Taylor, An Introduction to Error Analysis: The Study of Uncertainties in Physical Measurements, 2nd ed., University Science Books, 1997",
          "P. R. Bevington & D. K. Robinson, Data Reduction and Error Analysis for the Physical Sciences, 3rd ed., McGraw-Hill, 2003",
          "A. C. Melissinos & J. Napolitano, Experiments in Modern Physics, 2nd ed., Academic Press, 2003"
    ],
    chapters: [
      "测量、单位与不确定度表示",
      "随机误差与统计分布（高斯/泊松）",
      "误差传播定律",
      "最小二乘法与数据拟合",
      "系统误差识别与实验设计",
      "微弱信号检测与锁相放大",
      "真空技术基础",
      "低温获得与恒温控制",
      "实验电子学与数据采集",
      "经典近代物理实验（密立根油滴/弗兰克-赫兹/光电效应）",
      "精密干涉测量技术",
      "蒙特卡洛方法评估不确定度"
    ],
  },
  'advanced/laser-physics': {
    title: "激光物理",
    books: [
          "O. Svelto, Principles of Lasers, 5th ed., Springer, 2010",
          "A. E. Siegman, Lasers, University Science Books, 1986",
          "周炳琨、高以智等《激光原理》，国防工业出版社（第7版，2014）"
    ],
    chapters: [
      "受激辐射与爱因斯坦系数",
      "速率方程与粒子数反转",
      "光学谐振腔与模式",
      "高斯光束及其变换",
      "增益饱和与输出功率",
      "调 Q 技术",
      "锁模与超短脉冲",
      "激光线宽与频率稳定",
      "半导体激光器",
      "光纤激光器与放大器",
      "固体与气体激光器",
      "激光与物质相互作用"
    ],
  },
  'advanced/low-temperature-physics': {
    title: "低温物理",
    books: [
          "F. Pobell, Matter and Methods at Low Temperatures, 3rd ed., Springer, 2007",
          "C. Enss & S. Hunklinger, Low-Temperature Physics, Springer, 2005",
          "G. K. White & P. J. Meeson, Experimental Techniques in Low-Temperature Physics, 4th ed., Oxford University Press, 2002"
    ],
    chapters: [
      "气体液化与制冷循环",
      "3He/4He 稀释制冷机",
      "绝热去磁制冷",
      "低温温度计量",
      "固体低温比热与热导",
      "超流 4He 与二流体模型",
      "超流 3He 与 p 波配对",
      "量子液体概述",
      "低温下的量子输运",
      "超导电子学与 SQUID",
      "毫开尔文技术与核退磁"
    ],
  },
  'advanced/accelerator-physics': {
    title: "加速器物理",
    books: [
          "D. A. Edwards & M. J. Syphers, An Introduction to the Physics of High Energy Accelerators, Wiley, 1993",
          "H. Wiedemann, Particle Accelerator Physics, 4th ed., Springer, 2015",
          "M. Conte & W. W. MacKay, An Introduction to the Physics of Particle Accelerators, 2nd ed., World Scientific, 2008"
    ],
    chapters: [
      "加速器发展简史与分类",
      "横向束流光学与传输矩阵",
      "同步加速器原理与纵向动力学",
      "射频腔与相位稳定",
      "束流发射度与亮度",
      "同步辐射及其阻尼效应",
      "储存环与对撞机",
      "直线加速器",
      "束流不稳定性与集体效应",
      "加速器磁体（含超导磁体）",
      "束流诊断与测量",
      "自由电子激光原理"
    ],
  },
  'advanced/stellar-structure-evolution': {
    title: "恒星结构与演化",
    books: [
          "R. Kippenhahn, A. Weigert & A. Weiss, Stellar Structure and Evolution, 2nd ed., Springer, 2012",
          "黄润乾《恒星物理》（第二版），中国科学技术大学出版社，2012",
          "D. Prialnik, An Introduction to the Theory of Stellar Structure and Evolution, 2nd ed., Cambridge University Press, 2009"
    ],
    chapters: [
      "恒星观测特征与 H-R 图",
      "流体静力学平衡与位力定理",
      "恒星物态方程与简并物质",
      "不透明度与能量传输（辐射/对流）",
      "核反应速率与 pp 链、CNO 循环",
      "恒星结构方程组与数值模型",
      "主序星结构与质光关系",
      "后主序演化：红巨星与 AGB",
      "恒星脉动",
      "白矮星",
      "超新星爆发",
      "中子星与恒星质量黑洞",
      "双星相互作用演化"
    ],
  },
  'advanced/galactic-physics-and-dynamics': {
    title: "星系物理与星系动力学",
    books: [
          "L. S. Sparke & J. S. Gallagher, Galaxies in the Universe: An Introduction, 2nd ed., Cambridge University Press, 2007",
          "J. Binney & S. Tremaine, Galactic Dynamics, 2nd ed., Princeton University Press, 2008",
          "H. Mo, F. van den Bosch & S. White, Galaxy Formation and Evolution, Cambridge University Press, 2010"
    ],
    chapters: [
      "银河系结构与银盘银晕",
      "星系形态分类与哈勃序列",
      "星系测光与光谱观测",
      "恒星轨道理论",
      "无碰撞玻尔兹曼方程与 Jeans 方程",
      "旋涡结构与密度波理论",
      "旋转曲线与暗物质",
      "星系核球与超大质量黑洞",
      "活动星系核",
      "星系群与星系团",
      "星系形成与等级成团",
      "星系演化与大尺度结构"
    ],
  },
  'advanced/high-energy-astrophysics': {
    title: "高能天体物理与宇宙线",
    books: [
          "M. S. Longair, High Energy Astrophysics, 3rd ed., Cambridge University Press, 2011",
          "T. K. Gaisser, R. Engel & E. Resconi, Cosmic Rays and Particle Physics, 2nd ed., Cambridge University Press, 2016",
          "G. B. Rybicki & A. P. Lightman, Radiative Processes in Astrophysics, Wiley-VCH, 2004（1979 首版）"
    ],
    chapters: [
      "同步辐射与曲率辐射",
      "逆康普顿散射",
      "轫致辐射与热辐射",
      "吸积盘物理",
      "相对论性喷流",
      "费米加速与激波加速",
      "宇宙线成分、能谱与各向异性",
      "广延大气簇射",
      "伽马射线天文（地面切伦科夫/卫星）",
      "高能中微子天体物理",
      "超高能宇宙线与 GZK 截断",
      "暗物质间接探测"
    ],
  },
  'advanced/topological-matter': {
    title: "拓扑物态与拓扑绝缘体",
    books: [
          "B. A. Bernevig & T. L. Hughes, Topological Insulators and Topological Superconductors, Princeton University Press, 2013",
          "J. K. Asbóth, L. Oroszlány & A. Pályi, A Short Course on Topological Insulators, Springer, 2016",
          "S.-Q. Shen, Topological Insulators: Dirac Equation in Condensed Matter, 2nd ed., Springer, 2017"
    ],
    chapters: [
      "整数量子霍尔效应与 TKNN 不变量",
      "Berry 相位与 Berry 曲率",
      "Chern 数与拓扑不变量",
      "SSH 模型与一维拓扑相",
      "量子自旋霍尔效应与 Z2 不变量",
      "三维拓扑绝缘体",
      "拓扑超导体与 Majorana 零模",
      "外尔与狄拉克半金属",
      "对称性保护与十重分类",
      "拓扑相变与边界态",
      "分数陈绝缘体与拓扑序"
    ],
  },
  'advanced/medical-physics': {
    title: "医学物理与放射物理",
    books: [
          "F. M. Khan & J. P. Gibbons, Khan's The Physics of Radiation Therapy, 5th ed., Wolters Kluwer/LWW, 2014",
          "E. B. Podgorsak, Radiation Physics for Medical Physicists, 3rd ed., Springer, 2016",
          "J. T. Bushberg et al., The Essential Physics of Medical Imaging, 3rd ed., LWW, 2011"
    ],
    chapters: [
      "电离辐射与物质相互作用",
      "辐射剂量学与腔理论",
      "X 射线产生与能谱",
      "医用直线加速器",
      "外照射治疗计划与剂量计算",
      "调强放射治疗（IMRT）与容积旋转调强",
      "近距离放射治疗",
      "放射防护与剂量限值",
      "X 射线与 CT 成像物理",
      "核医学成像（PET/SPECT）",
      "磁共振成像物理",
      "医学超声物理"
    ],
  },
  'advanced/spintronics': {
    title: "自旋电子学",
    books: [
          "I. Žutić, J. Fabian & S. Das Sarma, Spintronics: Fundamentals and applications, Reviews of Modern Physics 76, 323 (2004)",
          "C. Felser & G. H. Fecher (eds.), Spintronics: From Materials to Devices, Springer, 2013",
          "J. Stöhr & H. C. Siegmann, Magnetism: From Fundamentals to Nanoscale Dynamics, Springer, 2006"
    ],
    chapters: [
      "自旋极化与自旋输运基础",
      "巨磁电阻（GMR）",
      "隧穿磁电阻（TMR）",
      "自旋注入与探测",
      "自旋轨道耦合（Rashba/Dresselhaus）",
      "自旋霍尔效应与逆自旋霍尔效应",
      "自旋转移力矩与自旋轨道力矩",
      "磁畴壁运动与赛道存储",
      "稀磁半导体与自旋场效应管",
      "磁随机存储器（MRAM）",
      "反铁磁自旋电子学"
    ],
  },
  'advanced/x-ray-physics': {
    title: "X 射线物理与同步辐射",
    books: [
          "J. Als-Nielsen & D. McMorrow, Elements of Modern X-ray Physics, 2nd ed., Wiley, 2011",
          "D. Attwood, Soft X-Rays and Extreme Ultraviolet Radiation: Principles and Applications, Cambridge University Press, 1999",
          "P. Willmott, An Introduction to Synchrotron Radiation: Techniques and Applications, Wiley, 2011"
    ],
    chapters: [
      "X 射线与物质相互作用（光电/康普顿/瑞利）",
      "X 射线源：轫致辐射与特征谱",
      "同步辐射原理与插入件",
      "X 射线光学元件（反射镜/单色器/波带片）",
      "X 射线衍射运动学理论",
      "小角 X 射线散射（SAXS）",
      "X 射线吸收精细结构（XAFS/XANES）",
      "相干衍射成像",
      "X 射线成像与 CT 原理",
      "X 射线自由电子激光",
      "共振非弹性 X 射线散射"
    ],
  },
  'advanced/quantum-foundations': {
    title: "量子力学基础与诠释",
    books: [
          "A. Peres, Quantum Theory: Concepts and Methods, Kluwer Academic, 1993",
          "J. S. Bell, Speakable and Unspeakable in Quantum Mechanics, 2nd ed., Cambridge University Press, 2004",
          "G. Auletta, Foundations and Interpretation of Quantum Mechanics, World Scientific, 2000"
    ],
    chapters: [
      "测量问题与波包坍缩",
      "EPR 佯谬",
      "Bell 不等式及其实验检验",
      "退相干理论",
      "哥本哈根诠释",
      "多世界诠释",
      "隐变量理论与 Bohm 力学",
      "自发坍缩模型（GRW）",
      "量子贝叶斯主义（QBism）",
      "宏观量子叠加与薛定谔猫实验",
      "弱测量与量子态层析",
      "语境性与 Kochen-Specker 定理"
    ],
  },
  'advanced/electronic-structure-theory': {
    title: "第一性原理计算与电子结构理论",
    books: [
          "R. M. Martin, Electronic Structure: Basic Theory and Practical Methods, Cambridge University Press, 2004（2nd ed. 2020）",
          "D. S. Sholl & J. A. Steckel, Density Functional Theory: A Practical Introduction, Wiley, 2009",
          "E. Kaxiras, Atomic and Electronic Structure of Solids, Cambridge University Press, 2003"
    ],
    chapters: [
      "Hartree-Fock 近似",
      "Hohenberg-Kohn 定理",
      "Kohn-Sham 方程",
      "交换关联泛函（LDA/GGA/杂化泛函）",
      "赝势与平面波基组",
      "能带结构与态密度计算",
      "密度泛函微扰理论与声子",
      "从头算分子动力学",
      "GW 近似与 Bethe-Salpeter 方程",
      "强关联体系与 DFT+U/DMFT",
      "高通量计算与材料数据库"
    ],
  },
  'advanced/metrology': {
    title: "计量学与 SI 单位制",
    books: [
          "M. Gläser & M. Kochsiek (eds.), Handbook of Metrology, Wiley-VCH, 2010",
          "BIPM, Le Système international d'unités / The International System of Units (SI Brochure), 9th ed., BIPM, 2019",
          "T. Quinn, From Artefacts to Atoms: The BIPM and the Search for Ultimate Measurement Standards, Oxford University Press, 2011"
    ],
    chapters: [
      "计量学基本概念与量值传递",
      "SI 单位制的历史演变",
      "2019 年 SI 新定义与基本常数固定",
      "时间频率计量与原子钟",
      "长度计量与稳频激光干涉",
      "质量计量与基布尔（瓦特）天平",
      "电学量子基准（约瑟夫森效应/量子霍尔效应）",
      "温度计量与玻尔兹曼常数测定",
      "测量不确定度评定（GUM 框架）",
      "国际比对与计量溯源链"
    ],
  },
  'social/physics-education-research': {
    title: "物理教育研究",
    books: [
          "L. C. McDermott & E. F. Redish, Resource Letter: PER-1: Physics Education Research, American Journal of Physics 67, 755 (1999)",
          "L. C. McDermott & the Physics Education Group, Physics by Inquiry, Wiley, 1996",
          "R. D. Knight, Five Easy Lessons: Strategies for Successful Physics Teaching, Addison-Wesley, 2002"
    ],
    chapters: [
      "学生前概念与错误概念研究",
      "力学概念诊断（FCI/FMCE）",
      "概念转变理论",
      "同伴教学法（Peer Instruction）",
      "探究式实验课程设计",
      "多表征学习",
      "认知负荷理论在物理教学中的应用",
      "学生认识论发展",
      "物理课程评价体系",
      "PER 的量化与质化研究方法",
      "物理教师教育"
    ],
  },
  'social/communication-and-negotiation': {
    title: "沟通表达与谈判",
    books: [
          "Adler & Rodman《沟通的艺术：看入人里，看出人外》(Understanding Human Communication, 世界图书出版公司中译本)",
          "Fisher & Ury《谈判力》(Getting to Yes, 中信出版社)",
          "Cialdini《影响力》(Influence: The Psychology of Persuasion, 中国人民大学出版社)"
    ],
    chapters: [
      "沟通的过程与常见障碍",
      "倾听与同理心反馈",
      "自我袒露与关系建立",
      "非语言沟通（表情/姿态/空间）",
      "语言沟通与人际冲突管理",
      "立场与利益：原则式谈判",
      "BATNA（最佳替代方案）与谈判准备",
      "分配式谈判与整合式谈判",
      "说服六原则（互惠/承诺一致/社会认同/喜好/权威/稀缺）",
      "跨文化沟通",
      "职场沟通与线上沟通礼仪"
    ],
  },
  'social/practical-and-academic-writing': {
    title: "实用写作与学术写作",
    books: [
          "Strunk & White《风格的要素》(The Elements of Style)",
          "Booth, Colomb & Williams《研究是一门艺术》(The Craft of Research, 新华出版社中译本)",
          "Minto《金字塔原理》(The Minto Pyramid Principle, 南海出版公司)"
    ],
    chapters: [
      "写作目的与读者分析",
      "金字塔结构与 SCQA 框架",
      "观点—理由—证据的论证结构",
      "段落组织与过渡衔接",
      "简明风格原则（删冗词/主动语态）",
      "摘要、报告与方案写作",
      "文献综述与研究问题确立",
      "引用规范与学术诚信",
      "公文、邮件与职场文书",
      "修改、润色与同行反馈"
    ],
  },
  'social/reading-and-knowledge-management': {
    title: "阅读方法与知识管理",
    books: [
          "Adler & Van Doren《如何阅读一本书》(How to Read a Book, 商务印书馆)",
          "Ahrens《卡片笔记写作法》(How to Take Smart Notes, 人民邮电出版社)",
          "奥野宣之《如何有效阅读一本书》(江西人民出版社)"
    ],
    chapters: [
      "阅读的四个层次（基础/检视/分析/主题）",
      "主动阅读与四个基本提问",
      "书籍分类与差异化阅读策略",
      "结构笔记与概念笔记",
      "康奈尔笔记法",
      "卡片盒笔记法（Zettelkasten）与双向链接",
      "个人知识库工具与工作流",
      "速读与精读的适用场景",
      "主题阅读与知识综合",
      "输出（写作/讲授）与间隔复习"
    ],
  },
  'social/career-planning-and-job-hunting': {
    title: "职业规划与求职技能",
    books: [
          "Bolles《你的降落伞是什么颜色？》(What Color Is Your Parachute?, 中信出版社)",
          "《大学生职业发展与就业指导》（高等教育出版社统编教材）",
          "Super《生涯发展理论》相关研究综述（学术依据）"
    ],
    chapters: [
      "自我认知：兴趣、能力与价值观",
      "霍兰德职业兴趣类型（RIASEC）",
      "Super 生涯发展阶段理论",
      "职业信息探索与行业分析",
      "职业目标设定与行动计划",
      "简历撰写与作品集准备",
      "面试应对（STAR 法则与结构化面试）",
      "求职渠道、内推与人脉经营",
      "Offer 比较与薪酬谈判",
      "职业转换与终身生涯管理"
    ],
  },
  'life/photography-and-video-production': {
    title: "摄影与短视频拍摄制作",
    books: [
          "颜志刚《摄影技艺教程》（复旦大学出版社）",
          "任金州《电视摄像》（中国传媒大学出版社）",
          "傅正义《影视剪辑编辑艺术》（北京广播学院出版社）"
    ],
    chapters: [
      "相机成像原理与曝光三要素",
      "镜头焦距与景深控制",
      "构图法则与视觉引导",
      "自然光与人造光运用、基础布光",
      "景别与镜头语言",
      "运动镜头与场面调度",
      "同期声与录音基础",
      "剪辑节奏与蒙太奇",
      "剪辑软件实操（Premiere/剪映）",
      "基础调色",
      "短视频叙事结构与平台适配"
    ],
  },
  'life/gymnastics': {
    title: "体操（竞技体操与艺术体操）",
    books: [
          "全国体育院校教材委员会审定《体操》（体育院校通用教材，人民体育出版社，2014）",
          "《竞技体操高级教程》（体育院校专修通用教材，人民体育出版社，2002）",
          "FIG《竞技体操评分规则》(Code of Points)"
    ],
    chapters: [
      "体操发展史与项目分类",
      "徒手体操与队列队形",
      "轻器械体操",
      "垫上运动与前滚翻类技术",
      "支撑跳跃",
      "单杠练习",
      "双杠练习",
      "平衡木与自由体操",
      "体操保护与帮助方法",
      "体操动作教学与创编",
      "竞赛规则与裁判法",
      "艺术体操器械（绳/圈/球/棒/带）与编排"
    ],
  },
  'life/weightlifting': {
    title: "举重运动",
    books: [
          "杨世勇《举重运动教程》（体育院校通用教材，人民体育出版社，2014）",
          "IWF《举重技术与竞赛规则》"
    ],
    chapters: [
      "举重运动史与体重级别划分",
      "抓举技术分解",
      "挺举技术分解（提铃/上挺）",
      "辅助力量练习（深蹲/硬拉/推举）",
      "举重技术教学法与常见错误纠正",
      "训练计划与周期安排",
      "举重动作的生物力学分析",
      "青少年选材与基础训练",
      "举重损伤预防",
      "竞赛规则与裁判法"
    ],
  },
  'life/shooting-sport': {
    title: "射击运动",
    books: [
          "国家体育总局编《射击》（人民体育出版社）",
          "ISSF《国际射击竞赛规则》"
    ],
    chapters: [
      "射击项目分类（步枪/手枪/飞碟）",
      "射击原理与内弹道常识",
      "射击姿势与据枪稳定性",
      "瞄准与击发技术",
      "呼吸控制与心理调控",
      "10 米气步枪与气手枪训练",
      "飞碟射击（双向/多向）",
      "竞赛规则与环值判定",
      "射击运动心理训练",
      "枪支管理法规与射击安全"
    ],
  },
  'life/mind-sports-chess-bridge': {
    title: "棋类与智力运动（围棋/中国象棋/国际象棋/桥牌）",
    books: [
          "聂卫平围棋道场系列教程（辽宁科学技术出版社）",
          "人民体育出版社棋牌入门系列（围棋/象棋/国际象棋/桥牌）",
          "ACBL 桥牌教学体系教材（美国定约桥牌联盟）"
    ],
    chapters: [
      "围棋规则与气、提子、打劫",
      "围棋基本死活与手筋",
      "围棋布局常识与中盘攻防",
      "官子与形势判断",
      "中国象棋基本杀法与实用残局",
      "象棋开局体系（中炮/屏风马）",
      "国际象棋基本战术（牵制/击双/闪击）",
      "国际象棋开局原则与残局基础",
      "桥牌自然叫牌法体系",
      "桥牌做庄与防守基本技术",
      "智力运动竞赛、段位与等级分制度"
    ],
  },
  'life/karate': {
    title: "空手道",
    books: [
          "WKF《空手道竞赛规则》（型与组手）",
          "中国空手道协会段位制培训教程"
    ],
    chapters: [
      "空手道源流与四大流派（松涛馆/刚柔/糸东/和道）",
      "道场礼仪与基本功（站架/移动）",
      "基本技：冲拳与击打",
      "基本技：踢技",
      "基本技：受（格挡）",
      "型（套路）的学习与演练",
      "约束组手与自由组手",
      "段位审查制度",
      "WKF 竞技规则（型比赛/组手比赛）",
      "体能、柔韧训练与损伤预防"
    ],
  },
  'life/dancesport-and-aerobics': {
    title: "体育舞蹈与健美操",
    books: [
          "张清澍《体育舞蹈》（北京体育大学出版社）",
          "《健美操》（高等教育出版社高校体育教材）"
    ],
    chapters: [
      "体育舞蹈分类（标准舞/拉丁舞）",
      "标准舞基本技术（华尔兹/探戈/维也纳华尔兹）",
      "拉丁舞基本技术（伦巴/恰恰恰/桑巴）",
      "音乐节奏识别与舞蹈风格表现",
      "竞技健美操规则与难度动作",
      "大众健美操成套动作与创编",
      "形体训练基础",
      "竞赛编排与评分标准",
      "教学组织与口令",
      "常见舞蹈损伤与预防"
    ],
  },
  'engineering/turning-and-milling-operations': {
    title: "车工与铣工（普通金属切削机床操作）",
    books: [
          "《车工工艺与技能训练》（中国劳动社会保障出版社，职业教育规划教材）",
          "《铣工工艺与技能训练》（中国劳动社会保障出版社，同系列）",
          "《金属切削原理与刀具》（机械工业出版社，高校机制专业教材）"
    ],
    chapters: [
      "车床结构与安全操作规程",
      "车刀几何角度与刃磨",
      "外圆与端面车削",
      "孔加工（钻孔/镗孔/铰孔）",
      "螺纹车削",
      "圆锥面与成形面车削",
      "铣床类型与铣刀选用",
      "平面与台阶面铣削",
      "沟槽、键槽与切断加工",
      "分度头使用与简单齿轮铣削",
      "切削用量（速度/进给/背吃刀量）选择",
      "常用量具使用与精度测量",
      "机械加工工艺规程基础"
    ],
  },
  'life/sewing-and-garment-making': {
    title: "缝纫与服装制作",
    books: [
          "张文斌《服装工艺学：成衣工艺分册》（中国纺织出版社）",
          "刘瑞璞《服装纸样设计原理与应用》（中国纺织出版社）",
          "《服装制作工》职业技能培训教材（中国劳动社会保障出版社）"
    ],
    chapters: [
      "面料识别与辅料选用",
      "人体测量与服装号型",
      "服装纸样基础（原型法/比例法）",
      "裁剪与排料",
      "手缝基础工艺",
      "工业/家用缝纫机使用与维护",
      "部件工艺：领、袖、口袋、开衩",
      "裙装与裤装制作",
      "衬衫与上衣制作",
      "熨烫与定型工艺",
      "服装改款、修补与旧衣改造"
    ],
  },
  'life/opticianry-and-eyeglasses-dispensing': {
    title: "眼镜验光与配镜（眼镜验光员/眼镜定配工）",
    books: [
          "齐备《眼镜验光员》（国家职业资格培训教程，中国劳动社会保障出版社，2008）",
          "《眼镜定配工》（国家职业资格培训教程，中国劳动社会保障出版社）",
          "人社部《眼镜验光员国家职业技能标准（2018年版）》"
    ],
    chapters: [
      "眼球解剖与屈光原理（近视/远视/散光/老视）",
      "视力检查与眼部初步检查",
      "客观验光：电脑验光与检影验光",
      "主观验光与红绿平衡",
      "散光轴位精调（交叉圆柱镜）",
      "双眼视功能检查",
      "处方原则与常见眼病转诊指征",
      "镜片材料、折射率与镀膜",
      "镜架选择、瞳距与瞳高测量",
      "磨边、装配与整形校配",
      "配装眼镜质量检测",
      "隐形眼镜验配常识"
    ],
  },
  'engineering/building-painting-and-coating': {
    title: "油漆工与建筑涂装",
    books: [
          "《油漆工》（建筑工人职业技能培训教材，中国建筑工业出版社）",
          "GB 50210《建筑装饰装修工程质量验收标准》（国家标准）"
    ],
    chapters: [
      "涂料组成与分类（乳胶漆/木器漆/防锈漆/防火涂料）",
      "基层处理与批刮腻子",
      "打磨与砂光工艺",
      "刷涂、滚涂与喷涂工艺",
      "木器清漆与混油涂装",
      "金属表面防腐涂装",
      "墙面涂饰施工与分色",
      "裱糊与软包基础",
      "色彩基础与配色",
      "质量通病防治（流坠/起泡/开裂/泛碱）",
      "涂装作业安全与职业防护（VOC/防火）"
    ],
  },
  'life/intimate-relationships-and-marriage': {
    title: "亲密关系与婚姻家庭",
    books: [
          "Rowland Miller《亲密关系》(Intimate Relationships, 第 6 版，人民邮电出版社中译本)",
          "Gottman & Silver《幸福的婚姻》(The Seven Principles for Making Marriage Work, 浙江人民出版社中译本)",
          "Bowlby 依恋理论经典研究（学术依据）"
    ],
    chapters: [
      "亲密关系的社会科学研究方法",
      "吸引力与择偶机制",
      "成人依恋类型（安全/焦虑/回避）",
      "斯滕伯格爱情三角理论",
      "亲密沟通与自我表露",
      "冲突模式与修复尝试（Gottman 四骑士）",
      "关系中的权力、公平与家务分工",
      "承诺与关系维持",
      "婚姻法律常识（结婚/财产/离婚）",
      "家庭暴力识别与求助",
      "分手、离异与重组家庭"
    ],
  },
  'life/home-renovation-and-interior-design': {
    title: "家庭装修与室内设计实务",
    books: [
          "陆震纬、来增祥《室内设计原理》（中国建筑工业出版社）",
          "JGJ 367《住宅室内装饰装修设计规范》与 GB 50327《住宅装饰装修工程施工规范》",
          "《住宅室内装饰装修工程质量验收规范》JGJ/T 304"
    ],
    chapters: [
      "户型分析与功能空间规划",
      "装修风格与色彩搭配",
      "装修流程、工期与预算编制",
      "水电改造与隐蔽工程验收",
      "泥瓦工程（防水/贴砖）验收要点",
      "木作与油漆工序",
      "主材选购（瓷砖/地板/门窗/卫浴）",
      "定制家具与收纳系统设计",
      "照明设计与开关插座布局",
      "环保材料与室内空气检测治理",
      "装修合同、增项与维权",
      "竣工验收与保修"
    ],
  },
  'social/history-of-economic-thought': {
    title: "经济思想史",
    books: [
          "马克·布劳格《经济理论的回顾》（Economic Theory in Retrospect, Cambridge University Press，第5版1997）",
          "斯坦利·布鲁、兰迪·格兰特《经济思想史》（The Evolution of Economic Thought，北京大学出版社中译本，第8版）",
          "约瑟夫·熊彼特《经济分析史》（History of Economic Analysis, Oxford University Press, 1954；商务印书馆中译本）"
    ],
    chapters: [
      "前古典经济学：重商主义与重农学派",
      "斯密与古典经济学的创立",
      "李嘉图体系与古典分配理论",
      "马尔萨斯、萨伊与古典宏观争论",
      "约翰·斯图亚特·穆勒的综合",
      "边际革命：杰文斯、门格尔、瓦尔拉斯",
      "马歇尔与新古典综合",
      "奥地利学派传统",
      "凯恩斯革命",
      "货币主义与新古典宏观经济学",
      "制度主义传统：凡勃伦到加尔布雷思",
      "福利经济学与公共选择思想源流",
      "一般均衡理论的成熟：阿罗-德布鲁",
      "博弈论与信息经济学的思想脉络"
    ],
  },
  'social/public-economics': {
    title: "公共经济学",
    books: [
          "加雷斯·迈尔斯《公共经济学》（Public Economics, Cambridge University Press, 1995）",
          "约瑟夫·斯蒂格利茨《公共部门经济学》（Economics of the Public Sector, W. W. Norton，第4版2014）",
          "让-雅克·拉丰《激励理论：委托-代理模型》（The Theory of Incentives, Princeton University Press, 2002）"
    ],
    chapters: [
      "市场失灵与政府干预的边界",
      "公共品理论与最优供给",
      "外部性与科斯定理",
      "公共选择理论：投票、官僚与寻租",
      "最优税收理论：商品税与所得税",
      "税收归宿与税收效率分析",
      "收入分配与再分配政策",
      "社会保障的经济学分析",
      "财政分权与地方公共品（蒂布特模型）",
      "成本-收益分析",
      "信息不对称与机制设计",
      "政府规制经济学"
    ],
  },
  'social/mathematical-economics': {
    title: "数量经济学（数理经济学）",
    books: [
          "蒋中一（Alpha C. Chiang）《数理经济学的基本方法》（Fundamental Methods of Mathematical Economics, McGraw-Hill，第4版；北京大学出版社中译本）",
          "蒋中一《动态最优化基础》（Elements of Dynamic Optimization；商务印书馆中译本）",
          "高山晟（Akira Takayama）《数理经济学》（Mathematical Economics, Cambridge University Press，第2版1985）"
    ],
    chapters: [
      "矩阵代数与线性经济模型",
      "比较静态分析与隐函数定理",
      "无约束与等式约束最优化",
      "库恩-塔克条件与非线性规划",
      "凹规划与二阶条件",
      "包络定理及其应用",
      "差分方程与经济动态学",
      "微分方程与相位图分析",
      "最优控制理论与汉密尔顿函数",
      "动态规划与贝尔曼方程",
      "一般均衡的数学表述",
      "不确定性下的决策：期望效用理论"
    ],
  },
  'social/environmental-resource-economics': {
    title: "环境与资源经济学",
    books: [
          "张帆、李东《环境与自然资源经济学》（格致出版社/上海人民出版社，第3版2016）",
          "罗杰·珀曼等《自然资源与环境经济学》（Natural Resource and Environmental Economics, Pearson，第4版2011）",
          "汤姆·蒂坦伯格、琳恩·刘易斯《环境与自然资源经济学》（Environmental and Natural Resource Economics，中国人民大学出版社中译本，第10版）"
    ],
    chapters: [
      "环境问题的经济学本质：外部性再审视",
      "庇古税与污染最优控制",
      "可交易排污许可制度",
      "环境规制工具比较：命令控制与经济激励",
      "环境价值评估：支付意愿与条件价值法",
      "旅行费用法与特征价格法",
      "可耗竭资源的最优开采路径（霍特林法则）",
      "可再生资源经济学：渔业与森林",
      "环境库兹涅茨曲线争论",
      "气候变化经济学：碳定价与贴现率之争",
      "成本-收益分析在环境政策中的应用",
      "中国的环境经济政策实践"
    ],
  },
  'social/health-economics': {
    title: "健康经济学",
    books: [
          "舍曼·富兰德、艾伦·古德曼、迈伦·斯坦诺《卫生经济学》（The Economics of Health and Health Care, Routledge/Pearson，第8版；中国人民大学出版社中译本）",
          "杰伊·巴塔查里亚等《健康经济学》（Health Economics, Palgrave Macmillan, 2014）",
          "迈克尔·格罗斯曼《健康需求的人力资本模型》（The Demand for Health, NBER/Columbia University Press 经典论文与著作传统）"
    ],
    chapters: [
      "健康的人力资本模型（格罗斯曼模型）",
      "医疗服务需求与道德风险",
      "医疗保险：逆向选择与风险分担",
      "医疗供给方诱导需求",
      "医院与医生的激励机制",
      "药品市场与专利制度",
      "卫生技术评估与成本效果分析",
      "医疗支付制度改革：DRG 与按人头付费",
      "健康不平等及其测量",
      "传染病防控的经济学分析",
      "公共卫生干预的经济评价",
      "中国医改的经济学分析"
    ],
  },
  'social/comparative-economic-systems': {
    title: "比较经济体制与转轨经济学",
    books: [
          "雅诺什·科尔奈《社会主义体制：共产主义政治经济学》（The Socialist System, Princeton University Press, 1992；中央编译出版社中译本）",
          "热拉尔·罗兰《转型与经济学》（Transition and Economics: Politics, Markets, and Firms, MIT Press, 2000；北京大学出版社中译本）",
          "青木昌彦《比较制度分析》（Comparative Institutional Analysis, MIT Press, 2001；上海远东出版社中译本）"
    ],
    chapters: [
      "经济体制的分类与比较维度",
      "计划经济体制：短缺经济学与软预算约束",
      "市场经济的多样性：英美模式与莱茵模式",
      "东亚发展型政府模式",
      "转轨的路径之争：休克疗法与渐进主义",
      "中国渐进式改革的双轨制逻辑",
      "产权改革与国有企业重组",
      "价格自由化与宏观经济稳定",
      "法与金融：法律制度对金融发展的影响",
      "比较制度分析：博弈论视角的制度多样性",
      "新比较经济学：政府的'掠夺之手'",
      "体制绩效的测度与历史教训"
    ],
  },
  'social/jurisprudence': {
    title: "法理学（法学理论）",
    books: [
          "张文显主编《法理学》（高等教育出版社/北京大学出版社，第5版2018）",
          "H. L. A. 哈特《法律的概念》（The Concept of Law, Oxford University Press，第3版2012；法律出版社中译本）",
          "埃德加·博登海默《法理学：法律哲学与法律方法》（Jurisprudence: The Philosophy and Method of the Law；中国政法大学出版社中译本，邓正来译）"
    ],
    chapters: [
      "法的概念与本质：分析法学传统",
      "自然法理论：从阿奎那到菲尼斯",
      "法律实证主义：奥斯丁、凯尔森、哈特",
      "德沃金的解释主义与权利理论",
      "法律现实主义与批判法学",
      "法律的要素：规则、原则与概念",
      "法律体系与法律渊源",
      "法律关系、权利与义务",
      "法律责任与制裁",
      "法律解释与法律推理",
      "立法、司法与执法理论",
      "法治的概念与原则",
      "法律与道德的关系之争",
      "中国社会主义法治理论"
    ],
  },
  'social/legal-history': {
    title: "法律史（中国法制史与外国法制史）",
    books: [
          "张晋藩主编《中国法制史》（中国政法大学出版社，第3版）",
          "何勤华主编《外国法制史》（法律出版社，第7版）",
          "曾宪义、赵晓耕主编《中国法制史》（中国人民大学出版社，法学 21 世纪系列教材）"
    ],
    chapters: [
      "中国法律的起源与夏商周法制",
      "春秋战国的成文法运动",
      "秦汉法律：睡虎地秦简与汉律",
      "魏晋南北朝法律的儒家化",
      "《唐律疏议》与中华法系",
      "宋元明清法制的演变",
      "清末修律与法律近代化",
      "中华民国六法体系",
      "革命根据地法制与新中国法制建设",
      "古代两河、希伯来与希腊法律",
      "罗马法：市民法、万民法与《国法大全》",
      "中世纪教会法与商法",
      "英国普通法与衡平法传统",
      "大陆法系的形成：《法国民法典》与《德国民法典》"
    ],
  },
  'social/law-and-economics': {
    title: "法律经济学",
    books: [
          "理查德·波斯纳《法律的经济分析》（Economic Analysis of Law, Wolters Kluwer，第9版2014；中国大百科全书出版社中译本）",
          "罗伯特·考特、托马斯·尤伦《法和经济学》（Law & Economics, Pearson，第6版；格致出版社中译本）",
          "斯蒂文·沙维尔《法律经济分析的基础理论》（Foundations of Economic Analysis of Law, Harvard University Press, 2004）"
    ],
    chapters: [
      "科斯定理与交易成本",
      "产权的经济分析",
      "财产法的经济学：占有、使用与征收",
      "合同法的经济学：效率违约与救济",
      "侵权法的经济学：过失责任与严格责任",
      "犯罪与刑罚的经济分析（贝克尔模型）",
      "公司法与证券监管的经济分析",
      "反垄断法的经济学基础",
      "诉讼与和解的经济分析",
      "法律程序的成本与激励",
      "法律制度与经济发展",
      "行为法律经济学"
    ],
  },
  'social/history-of-political-thought': {
    title: "政治思想史",
    books: [
          "乔治·萨拜因《政治学说史》（A History of Political Theory；商务印书馆中译本）",
          "列奥·施特劳斯、约瑟夫·克罗波西主编《政治哲学史》（History of Political Philosophy, University of Chicago Press，第3版1987；法律出版社/河北人民出版社中译本）",
          "萧公权《中国政治思想史》（商务印书馆/新星出版社）"
    ],
    chapters: [
      "古希腊政治思想：柏拉图与亚里士多德",
      "罗马与中世纪政治思想：西塞罗到奥古斯丁、阿奎那",
      "马基雅维利与现代政治的开端",
      "近代自然法与社会契约：霍布斯、洛克、卢梭",
      "启蒙时代的政治思想：孟德斯鸠与联邦党人",
      "保守主义传统：柏克及其后继者",
      "自由主义传统：从边沁到密尔、托克维尔",
      "社会主义思潮：从空想到马克思主义",
      "民族主义、帝国主义与法西斯主义的思想根源",
      "20 世纪政治思想：罗尔斯、诺齐克、哈耶克、阿伦特",
      "先秦政治思想：儒家、法家、道家",
      "秦汉至明清政治思想的演变",
      "中国近代政治思想：从康梁到三民主义",
      "政治概念史与剑桥学派方法"
    ],
  },
  'social/social-research-methods': {
    title: "社会研究方法",
    books: [
          "艾尔·巴比《社会研究方法》（The Practice of Social Research, Cengage，第15版；华夏出版社中译本，邱泽奇译）",
          "风笑天《社会研究方法》（中国人民大学出版社，第5版2018）",
          "袁方主编《社会研究方法教程》（北京大学出版社，重排本）"
    ],
    chapters: [
      "社会科学研究的逻辑：演绎与归纳",
      "研究设计与选题操作化",
      "概念化、操作化与测量",
      "抽样原理与抽样设计",
      "问卷设计与调查研究",
      "实验法与准实验设计",
      "实地研究与参与观察",
      "深度访谈与焦点小组",
      "内容分析与文献研究",
      "社会统计学基础：描述统计与推断统计",
      "相关与回归分析",
      "测量的信度与效度",
      "研究伦理与学术规范",
      "混合方法与计算社会科学方法"
    ],
  },
  'social/sociological-theory': {
    title: "社会学理论（古典与当代）",
    books: [
          "乔治·瑞泽尔《社会学理论》（Sociological Theory, McGraw-Hill，第9版；北京大学出版社/上海古籍出版社中译本）",
          "侯钧生主编《西方社会学理论教程》（南开大学出版社，第4版）",
          "乔纳森·特纳《社会学理论的结构》（The Structure of Sociological Theory, Wadsworth，第10版；华夏出版社中译本）"
    ],
    chapters: [
      "孔德与社会学的创立",
      "涂尔干：社会事实、分工与自杀论",
      "马克思：阶级、异化与意识形态",
      "韦伯：理解社会学、理性化与支配类型",
      "齐美尔：形式社会学",
      "帕森斯与结构功能主义",
      "冲突理论：达伦多夫与科塞",
      "符号互动论：米德、布鲁默与戈夫曼",
      "交换理论与理性选择：霍曼斯、布劳、科尔曼",
      "法兰克福学派批判理论",
      "布迪厄：场域、惯习与资本",
      "吉登斯：结构化理论",
      "哈贝马斯：交往行动理论",
      "福柯与后现代转向",
      "社会学理论的中国本土化"
    ],
  },
  'social/organizational-behavior': {
    title: "组织行为学",
    books: [
          "斯蒂芬·罗宾斯、蒂莫西·贾奇《组织行为学》（Organizational Behavior, Pearson，第18版；中国人民大学出版社中译本）",
          "弗雷德·卢森斯《组织行为学》（Organizational Behavior: An Evidence-Based Approach, McGraw-Hill，第12版；人民邮电出版社中译本）",
          "埃德加·沙因《组织文化与领导力》（Organizational Culture and Leadership, Jossey-Bass，第5版2016；中国人民大学出版社中译本）"
    ],
    chapters: [
      "组织行为学的学科基础与研究方法",
      "个体行为：人格、能力与价值观",
      "知觉与归因",
      "态度与工作满意度",
      "情绪与工作压力",
      "激励的内容理论：马斯洛、赫茨伯格、麦克利兰",
      "激励的过程理论：期望理论、公平理论、目标设置",
      "群体行为与团队动力",
      "沟通与人际过程",
      "领导理论：特质、行为与权变模型",
      "变革型领导与领导-成员交换",
      "权力与政治行为",
      "冲突与谈判",
      "组织结构与设计",
      "组织文化、组织变革与发展"
    ],
  },
  'social/strategic-management': {
    title: "战略管理",
    books: [
          "迈克尔·希特、R. 杜安·爱尔兰、罗伯特·霍斯基森《战略管理：概念与案例》（Strategic Management: Concepts and Cases, Cengage，第12版；中国人民大学出版社中译本）",
          "弗雷德·戴维《战略管理》（Strategic Management: Concepts and Cases, Pearson，第16版；清华大学出版社中译本）",
          "杰伊·巴尼、威廉·赫斯特里《战略管理》（Strategic Management and Competitive Advantage, Pearson，第6版）"
    ],
    chapters: [
      "战略与战略管理过程",
      "愿景、使命与战略目标",
      "外部环境分析：PEST 与五力模型",
      "内部资源与能力分析：VRIO 框架",
      "资源基础观与核心竞争力",
      "业务层战略：成本领先、差异化与聚焦",
      "蓝海战略与价值创新",
      "公司层战略：一体化、多元化与并购",
      "国际化战略",
      "战略联盟与合作竞争",
      "商业模式创新",
      "战略实施：结构、文化与控制系统",
      "战略领导与公司治理",
      "平台战略与生态系统竞争"
    ],
  },
  'social/management-information-systems': {
    title: "管理信息系统",
    books: [
          "肯尼斯·劳东、简·劳东《管理信息系统》（Management Information Systems: Managing the Digital Firm, Pearson，第16版；机械工业出版社中译本）",
          "薛华成主编《管理信息系统》（清华大学出版社，第7版）",
          "R. Kelly Rainer 等《信息系统导论》（Introduction to Information Systems, Wiley；中国人民大学出版社中译本）"
    ],
    chapters: [
      "信息系统与组织：技术、组织、管理三维度",
      "信息系统战略与竞争优势",
      "企业级系统：ERP 与业务流程",
      "供应链管理系统与客户关系管理",
      "电子商务与数字市场",
      "决策支持系统与商务智能",
      "大数据在管理中的应用",
      "知识管理系统",
      "信息系统规划与开发方法论",
      "IT 治理与 IT 外包",
      "信息安全管理与隐私",
      "企业数字化转型"
    ],
  },
  'social/history-of-education': {
    title: "教育史（中国教育史与外国教育史）",
    books: [
          "孙培青主编《中国教育史》（华东师范大学出版社，第4版2019）",
          "吴式颖、李明德主编《外国教育史教程》（人民教育出版社，第3版2015）",
          "张斌贤主编《外国教育史》（教育科学出版社，第2版）"
    ],
    chapters: [
      "中国古代教育的起源与官学私学制度",
      "孔子与儒家教育思想",
      "稷下学宫与诸子教育思想",
      "科举制度的创立与演变",
      "书院制度与理学教育",
      "西学东渐与近代学制：癸卯学制、壬戌学制",
      "蔡元培、陶行知与现代教育思潮",
      "古希腊罗马教育：苏格拉底、柏拉图、昆体良",
      "中世纪大学与经院教育",
      "文艺复兴与宗教改革时期的教育",
      "夸美纽斯、洛克、卢梭的教育思想",
      "赫尔巴特与科学教育学的奠基",
      "杜威与进步主义教育运动",
      "20 世纪欧美教育改革：要素主义、永恒主义与新传统派"
    ],
  },
  'social/curriculum-and-instruction': {
    title: "课程与教学论",
    books: [
          "王本陆主编《课程与教学论》（高等教育出版社，第4版2023）",
          "施良方《课程理论：课程的基础、原理与问题》（教育科学出版社，1996）",
          "拉尔夫·泰勒《课程与教学的基本原理》（Basic Principles of Curriculum and Instruction, University of Chicago Press, 1949/2013；中国轻工业出版社中译本）"
    ],
    chapters: [
      "课程与教学的概念及其关系",
      "课程论的历史发展：学科中心、学生中心、社会中心",
      "泰勒原理与目标模式",
      "课程开发的模式：过程模式与实践模式",
      "课程类型与课程结构：显性课程与隐性课程",
      "课程标准与教科书制度",
      "校本课程开发",
      "教学过程的本质与规律",
      "教学原则与教学方法体系",
      "教学模式：发现学习、掌握学习、有意义学习",
      "教学设计理论：加涅与迪克-凯里模型",
      "课堂管理与教学组织形式",
      "教学评价：诊断性、形成性与总结性评价",
      "核心素养导向的课程教学改革"
    ],
  },
  'social/experimental-psychology': {
    title: "实验心理学",
    books: [
          "郭秀艳《实验心理学》（人民教育出版社，第2版2019）",
          "朱滢主编《实验心理学》（北京大学出版社，第4版2022）",
          "坎特威茨等《实验心理学》（Experimental Psychology, Wadsworth，第10版；华东师范大学出版社中译本）"
    ],
    chapters: [
      "实验心理学的科学性质与历史",
      "心理学实验的变量与设计：被试内与被试间",
      "实验的信度与效度、额外变量控制",
      "心理物理学：费希纳与传统心理物理法",
      "信号检测论",
      "反应时法：减法法、加法因素法与开窗技术",
      "注意实验：过滤器理论与双作业范式",
      "感知觉实验：知觉组织与错觉研究",
      "记忆实验：感觉记忆、短时记忆与内隐记忆",
      "思维与问题解决实验",
      "情绪实验与面部表情研究",
      "眼动、脑电与 fMRI 技术在心理学实验中的应用",
      "实验报告的撰写与研究伦理"
    ],
  },
  'social/psychometrics': {
    title: "心理统计与测量",
    books: [
          "戴海崎、张锋主编《心理与教育测量》（暨南大学出版社，第4版2018）",
          "张厚粲、徐建平《现代心理与教育统计学》（北京师范大学出版社，第5版2020）",
          "郑日昌、蔡永红、周益群《心理测量学》（人民教育出版社，1999）"
    ],
    chapters: [
      "心理测量的性质与历史：高尔顿、比内传统",
      "经典测量理论：真分数模型",
      "测量的信度：重测、复本、内部一致性",
      "测量的效度：内容、构想与效标关联效度",
      "测验的项目分析：难度与区分度",
      "常模与分数解释",
      "智力测验：斯坦福-比内、韦克斯勒量表",
      "人格测验：自陈量表与投射测验",
      "项目反应理论（IRT）",
      "心理统计基础：描述统计与概率分布",
      "假设检验、t 检验与方差分析",
      "相关与回归分析",
      "卡方检验与非参数检验",
      "测量等值与测验公平性"
    ],
  },
  'social/cognitive-psychology': {
    title: "认知心理学",
    books: [
          "王甦、汪安圣《认知心理学》（北京大学出版社，重排本2006）",
          "罗伯特·斯滕伯格《认知心理学》（Cognitive Psychology, Cengage，第7版；中国轻工业出版社中译本）",
          "罗伯特·索尔所、金伯利·麦克林《认知心理学》（Cognitive Psychology, Pearson，第8版；上海人民出版社中译本）"
    ],
    chapters: [
      "认知心理学的兴起：信息加工范式",
      "认知神经科学方法：ERP、fMRI 与脑损伤研究",
      "知觉：模式识别与知觉加工",
      "注意：选择性注意与注意资源理论",
      "记忆结构：感觉记忆、短时记忆与工作记忆",
      "长时记忆：编码、存储与提取",
      "知识的表征：概念、图式与语义网络",
      "表象与心理旋转",
      "语言：语言理解与产生的认知机制",
      "问题解决与创造性",
      "推理与决策",
      "认知发展与个体差异",
      "认知心理学的应用：教育、人机交互与临床"
    ],
  },
  'social/personality-psychology': {
    title: "人格心理学",
    books: [
          "杰里·伯格（Jerry M. Burger）《人格心理学》（Personality, Cengage，第10版；中国轻工业出版社中译本）",
          "黄希庭《人格心理学》（浙江教育出版社，2002）",
          "兰迪·拉森、戴维·巴斯《人格心理学》（Personality Psychology: Domains of Knowledge About Human Nature, McGraw-Hill，第6版；人民邮电出版社中译本）"
    ],
    chapters: [
      "人格心理学的研究对象与方法",
      "精神分析理论：弗洛伊德的人格结构与发展",
      "新精神分析：荣格、阿德勒、霍妮与埃里克森",
      "特质理论：奥尔波特、卡特尔与艾森克",
      "大五人格模型与人格的结构",
      "生物学取向：气质、进化与人格的遗传基础",
      "人本主义：马斯洛与罗杰斯",
      "行为主义与社会学习理论：斯金纳、班杜拉",
      "认知取向：凯利的个人构念理论",
      "人格测量：问卷、投射与行为评定",
      "人格的稳定性与毕生发展",
      "人格与文化：跨文化人格研究",
      "人格障碍的基础知识"
    ],
  },
  'social/abnormal-psychology': {
    title: "变态心理学",
    books: [
          "钱铭怡主编《变态心理学》（北京大学出版社，2006）",
          "戴维·巴洛、马克·杜兰德《异常心理学》（Abnormal Psychology: An Integrative Approach, Cengage，第8版；中国轻工业出版社中译本）",
          "苏珊·诺伦-霍克西玛《变态心理学》（Abnormal Psychology, McGraw-Hill，第8版；人民邮电出版社中译本）"
    ],
    chapters: [
      "变态心理学的研究对象与历史",
      "心理障碍的分类与诊断：DSM-5 与 ICD-11",
      "变态心理学的理论模型：生物、心理、社会与整合模型",
      "临床评估与心理测验",
      "焦虑障碍：广泛性焦虑、惊恐障碍与恐惧症",
      "强迫症与创伤后应激障碍",
      "心境障碍：抑郁障碍与双相障碍",
      "精神分裂症谱系障碍",
      "分离障碍与躯体症状障碍",
      "进食障碍与睡眠障碍",
      "人格障碍",
      "神经发育障碍：自闭症谱系与多动症",
      "物质使用与成瘾障碍",
      "自杀与危机干预"
    ],
  },
  'engineering/semiconductor-manufacturing': {
    title: "半导体制造工艺与装备",
    books: [
          "Quirk, Serda, \"Semiconductor Manufacturing Technology\" (2001)（中译《半导体制造技术》）",
          "Van Zant, \"Microchip Fabrication\" (6th ed., 2014)（中译《芯片制造：半导体工艺制程实用教程》）",
          "Sze (ed.), \"VLSI Technology\" (2nd ed., 1988)"
    ],
    chapters: [
      "半导体材料与晶体生长（直拉/区熔单晶、晶圆制备）",
      "洁净室技术与污染控制（颗粒/金属离子/静电）",
      "氧化工艺（热氧化、栅氧与场氧）",
      "光刻工艺（涂胶/曝光/显影，工艺链中的定位）",
      "刻蚀技术（湿法刻蚀、等离子体与反应离子刻蚀 RIE）",
      "薄膜沉积（PVD 溅射、CVD/LPCVD/PECVD、ALD）",
      "掺杂工艺（热扩散与离子注入、退火）",
      "金属化与多层互连（铝互连、铜大马士革工艺）",
      "化学机械抛光（CMP）与平坦化",
      "CMOS 工艺集成（前道 FEOL/后道 BEOL 全流程）",
      "量测与缺陷检测（膜厚/关键尺寸 CD/套刻量测）",
      "良率工程与统计过程控制（SPC）",
      "先进封装（引线键合/倒装焊/2.5D/3D 集成、Chiplet）",
      "制造装备总览（光刻机/刻蚀机/薄膜沉积/离子注入/量测设备）"
    ],
  },
  'engineering/lithography-technology': {
    title: "光刻技术与光刻机",
    books: [
          "Levinson, \"Principles of Lithography\" (4th ed., SPIE, 2019)",
          "Mack, \"Fundamental Principles of Optical Lithography\" (Wiley, 2007)",
          "Bakshi (ed.), \"EUV Lithography\" (2nd ed., SPIE, 2018)"
    ],
    chapters: [
      "光刻原理与投影成像光学基础",
      "分辨率理论与工艺窗口（Rayleigh 准则、k1 因子、焦深 DOF）",
      "曝光光源演进（g/i 线、KrF 248nm、ArF 193nm、EUV 13.5nm）",
      "EUV 光源（激光等离子体 LPP、锡滴发生、驱动激光与收集镜）",
      "投影物镜（超高 NA 光学、像差校正、浸没式物镜）",
      "照明系统与分辨率增强（离轴照明、自由形式照明）",
      "掩模（掩模基板、吸收层、缺陷检测与修补、EUV 反射掩模）",
      "光刻胶化学（化学放大胶、EUV 胶、显影与线宽粗糙度 LER）",
      "涂胶显影（Track 工艺与烘烤控制）",
      "超精密运动系统（双工件台、掩模台、气浮/磁悬浮、nm 级同步扫描）",
      "对准与套刻控制（Alignment/Overlay 测量与前馈反馈）",
      "调平调焦与伺服控制、整机热管理与减振",
      "浸没式光刻（浸液流场、缺陷控制）",
      "多重图形化（LELE/SADP/SAQP 与自对准工艺）",
      "计算光刻（OPC 光学邻近效应校正、光源掩模协同优化 SMO、反向光刻 ILT）",
      "EUV 整机系统集成（真空系统、污染控制、高 NA EUV 展望）",
      "非光学光刻（电子束直写、纳米压印、X 射线光刻）"
    ],
  },
  'engineering/aircraft-maintenance-engineering': {
    title: "航空器维修工程（机务）",
    books: [
          "FAA, \"Aviation Maintenance Technician Handbook—General\" (FAA-H-8083-30B, 2023)",
          "FAA, \"Aviation Maintenance Technician Handbook—Airframe\" (FAA-H-8083-31A, 2023)",
          "FAA, \"Aviation Maintenance Technician Handbook—Powerplant\" (FAA-H-8083-32A, 2023)"
    ],
    chapters: [
      "维修体系与适航法规（CCAR-145 维修单位、CCAR-66 执照、FAA/EASA 体系）",
      "维修理论与大纲（MSG-3、以可靠性为中心的维修 RCM）",
      "飞机结构与站位识别、维修分区",
      "钣金结构修理（铆接、损伤评估与 SRM 结构修理手册）",
      "复合材料结构维修（损伤检测、铺贴与固化修理）",
      "机体系统维修（液压、起落架、飞行操纵、燃油、环境控制）",
      "航空发动机维修（涡扇发动机单元体、孔探、试车）",
      "航空电气与航电系统维修（线路标准施工、排故）",
      "无损检测（渗透/磁粉/涡流/超声/射线 NDT）",
      "技术文件体系（AMM/CMM/IPC/SRM、工卡与服务通告 SB/AD）",
      "航线维护与定检（过站/A 检/C 检/D 检组织与实施）",
      "维修人为因素与维修差错（墨菲定律、脏十二）",
      "机队维修计划、航材管理与可靠性工程"
    ],
  },
  'engineering/design-methodology': {
    title: "机械设计方法学与系统设计",
    books: [
          "Pahl, Beitz, \"Engineering Design: A Systematic Approach\" (3rd ed., 2007)（中译《工程设计方法学》）",
          "Dieter, Schmidt, \"Engineering Design\" (6th ed., 2020)",
          "Ulrich, Eppinger, \"Product Design and Development\" (7th ed., 2020)"
    ],
    chapters: [
      "设计过程模型与系统设计方法（需求—功能—原理—结构）",
      "需求分析、质量功能展开（QFD）与技术规格书",
      "功能结构分解与物理原理解求解",
      "概念设计与方案评价（形态学矩阵、决策矩阵）",
      "创新设计方法（TRIZ 发明原理、矛盾矩阵与进化法则）",
      "参数化、模块化与产品平台设计",
      "优化设计方法（数学规划在机械设计中的应用）",
      "可靠性设计与失效模式分析（FMEA/FTA）",
      "稳健设计（田口方法与容差设计）",
      "设计公理（Axiomatic Design：独立公理与信息公理）",
      "面向制造与装配的设计（DFMA、公差叠加分析）",
      "全生命周期设计（绿色设计、可维修性、可回收性）",
      "智能设计（AI 辅助概念生成、生成式设计与拓扑优化）"
    ],
  },
  'life/flight-pilot-training': {
    title: "飞行驾驶与飞行员训练",
    books: [
          "FAA, \"Pilot's Handbook of Aeronautical Knowledge\" (FAA-H-8083-25C, 2023)（中译《飞行员航空知识手册》）",
          "FAA, \"Airplane Flying Handbook\" (FAA-H-8083-3C, 2021)",
          "FAA, \"Instrument Flying Handbook\" (FAA-H-8083-15B, 2013)"
    ],
    chapters: [
      "飞行原理（升力/阻力/失速、稳定性与操纵性）",
      "飞机系统与动力装置（活塞/涡桨/涡扇、电气与燃油系统）",
      "飞行性能与载重平衡计算",
      "航空气象（天气系统、危险天气、气象报文识读 METAR/TAF）",
      "空中领航（推测领航、无线电领航、RNAV/GNSS）",
      "航空法规与空域运行（CCAR-61/91、ICAO 体系）",
      "陆空通话与飞行程序（标准通话、离场进场进近程序）",
      "目视飞行（VFR）操作训练（起落航线、机动飞行、转场）",
      "仪表飞行（IFR）训练（仪表扫视、等待、ILS/RNP 进近）",
      "人的因素与机组资源管理（CRM、情景意识、疲劳管理）",
      "特情与应急程序（发动机失效、失压、火警、迫降）",
      "执照与训练体系（私照 PPL/仪表等级/商照 CPL/航线执照 ATPL、教员等级）",
      "高性能与多发机型改装训练",
      "航线运行基础（签派放行、运行规范、与签派/管制的协同）"
    ],
  },
  'engineering/semiconductor-device-physics': {
    title: "半导体器件物理",
    books: [
          "Sze, Ng, \"Physics of Semiconductor Devices\" (3rd ed., 2007)（中译《半导体器件物理》）",
          "Streetman, Banerjee, \"Solid State Electronic Devices\" (7th ed., 2016)",
          "Neamen, \"Semiconductor Physics and Devices\" (4th ed., 2012)"
    ],
    chapters: [
      "能带与载流子统计回顾（与《半导体物理》衔接）",
      "PN 结（内建电势、耗尽层、I-V/C-V 特性、击穿）",
      "金属-半导体接触（肖特基势垒与欧姆接触）",
      "双极晶体管 BJT（放大原理、厄利效应、频率特性）",
      "MOS 电容（平带/阈值电压、C-V 分析）",
      "MOSFET 长沟道模型（平方律、亚阈值、体效应）",
      "短沟道效应与器件 scaling（DIBL、速度饱和、热载流子）",
      "FinFET 与多栅器件（静电控制、三维器件物理）",
      "GAA 纳米片/纳米线器件与 CFET 展望",
      "存储器件物理（浮栅/电荷俘获、DRAM 单元、铁电器件）",
      "化合物半导体器件（HEMT、HBT、光电器件概览）",
      "器件可靠性（NBTI、TDDB、电迁移）与表征方法"
    ],
  },
  'engineering/digital-ic-design': {
    title: "数字集成电路设计",
    books: [
          "Rabaey, Chandrakasan, Nikolic, \"Digital Integrated Circuits: A Design Perspective\" (2nd ed., 2003)",
          "Weste, Harris, \"CMOS VLSI Design: A Circuits and Systems Perspective\" (4th ed., 2010)"
    ],
    chapters: [
      "CMOS 反相器（静态特性、动态特性、功耗三成分）",
      "制造工艺与版图基础（设计规则、与《半导体制造工艺》衔接）",
      "互连（RC 延迟模型、Elmore 延迟、串扰）",
      "组合逻辑（静态 CMOS、传输门、动态/多米诺逻辑）",
      "时序逻辑（寄存器、锁存器、建立/保持时间）",
      "时序分析与时钟（时钟偏差、抖动、时序收敛概念）",
      "数据通路运算单元（加法器族、乘法器、移位器）",
      "存储器阵列（6T SRAM 单元、感放、译码与外围电路）",
      "低功耗设计（DVFS、时钟门控、电源门控、多阈值）",
      "设计方法学（RTL→综合→版图流程、标准单元库）",
      "可测试性设计（扫描链、BIST、ATPG 概念）",
      "I/O 与 ESD 保护、片上传输线效应"
    ],
  },
  'engineering/analog-ic-design': {
    title: "模拟集成电路设计",
    books: [
          "Razavi, \"Design of Analog CMOS Integrated Circuits\" (2nd ed., 2017)（中译《模拟 CMOS 集成电路设计》）",
          "Gray, Hurst, Lewis, Meyer, \"Analysis and Design of Analog Integrated Circuits\" (6th ed., 2024)"
    ],
    chapters: [
      "MOS 器件小信号模型与大信号模型回顾",
      "单级放大器（共源/共栅/源跟随、电流源负载）",
      "共源共栅（Cascode）与增益提升",
      "差分放大器（差分对、共模抑制、失调）",
      "电流镜与偏置技术（匹配、温度特性）",
      "频率响应（极点/零点、Miller 效应）",
      "噪声（热噪声/闪烁噪声、输入参考噪声）",
      "反馈放大器（四种反馈拓扑、稳定性判据）",
      "运算放大器（两级运放、折叠 Cascode、频率补偿）",
      "带隙基准与偏置生成",
      "比较器与开关电容电路",
      "数据转换器基础（ADC/DAC 架构与指标）",
      "模拟版图艺术（匹配、对称、共质心、衬底噪声）"
    ],
  },
  'engineering/memory-technology': {
    title: "存储器技术（DRAM/Flash/HBM/新型存储）",
    books: [
          "Sharma, \"Semiconductor Memories: Technology, Testing, and Reliability\" (IEEE Press, 1997)",
          "Keeth, Baker, Johnson, Lin, \"DRAM Circuit Design: Fundamental and High-Speed Topics\" (2nd ed., 2008)",
          "Cappelletti, Gola (eds.), \"Flash Memories\" (Springer, 1999)"
    ],
    chapters: [
      "存储层次与存储器指标（容量/带宽/延迟/耐久/成本）",
      "SRAM（6T 单元读写分析、稳定性 SNM、外围电路）",
      "DRAM 单元与阵列（1T1C、刷新、读出放大器）",
      "DRAM 接口演进（SDR→DDR5、LPDDR、GDDR）",
      "HBM（TSV 堆叠、宽接口、与 GPU/AI 芯片的协同）",
      "NAND Flash 单元（浮栅/电荷俘获、编程/擦除机理）",
      "多值存储（MLC/TLC/QLC）与读扰动/保持力",
      "3D NAND（沟道孔刻蚀、层数演进、串堆叠）",
      "NOR Flash 与嵌入式存储（eFlash 的 scaling 困境）",
      "新型非易失存储（PCM、RRAM、MRAM、FeRAM 原理与现状）",
      "存储可靠性（磨损均衡、ECC、LDPC 纠错）与主控",
      "存内计算与近存计算（CIM/PIM、CXL 内存池化）"
    ],
  },
  'engineering/advanced-packaging-chiplet': {
    title: "先进封装与 Chiplet",
    books: [
          "Lau, \"3D IC Integration and Packaging\" (McGraw-Hill, 2016)",
          "Tummala (ed.), \"Fundamentals of Device and Systems Packaging: Technologies and Applications\" (2nd ed., 2019)",
          "UCIe Consortium, \"UCIe Specification\"（Die-to-Die 互连标准）"
    ],
    chapters: [
      "封装的层级与功能（芯片→封装→板→系统）",
      "传统封装（引线键合、BGA、倒装焊 Flip-Chip）",
      "晶圆级封装（WLP/Fan-in/Fan-out）",
      "2.5D 集成（硅中介层、CoWoS、EMIB、重布线层 RDL）",
      "3D 集成与硅通孔 TSV（制造、应力、热问题）",
      "混合键合（Hybrid Bonding，Cu-Cu 直接互连）",
      "HBM 集成（堆叠、与逻辑芯片的 2.5D 共封装）",
      "Chiplet 理念（良率经济学、异构组合、KGD 已知良品芯粒）",
      "Die-to-Die 接口标准（UCIe、BoW、AIB）",
      "功率完整性与散热（供电网络、热界面材料、液冷封装）",
      "共封装光学（CPO）与光电异构集成",
      "系统级封装 SiP 与封装设计流程"
    ],
  },
  'engineering/power-semiconductor': {
    title: "功率半导体器件（IGBT/SiC/GaN）",
    books: [
          "Baliga, \"Fundamentals of Power Semiconductor Devices\" (2nd ed., 2019)",
          "Lutz et al., \"Semiconductor Power Devices: Physics, Characteristics, Reliability\" (2nd ed., 2018)",
          "Baliga, \"The IGBT Device\" (2nd ed., 2022)"
    ],
    chapters: [
      "功率器件的指标与根本折中（耐压 vs 导通电阻、Baliga 极限）",
      "功率二极管（PiN、肖特基、快恢复）",
      "功率 MOSFET（平面栅/沟槽栅、体二极管）",
      "超结 MOSFET（电荷平衡原理）",
      "IGBT（MOS 栅控双极导通、电导调制、拖尾电流、闩锁）",
      "晶闸管与 GTO（高压大电流场景）",
      "SiC 材料特性与 SiC MOSFET/SBD（临界电场、高温高频优势）",
      "GaN HEMT（二维电子气、常关型实现、动态导通电阻）",
      "栅极驱动与保护（隔离驱动、退饱和保护、短路耐受）",
      "功率模块封装与热设计（热阻网络、双面散热）",
      "典型应用拓扑（整流/逆变/DCDC、新能源车电控与充电桩）",
      "可靠性与失效分析（功率循环、栅氧退化、宇宙射线）"
    ],
  },
  'engineering/advanced-process-integration': {
    title: "先进制程与器件集成（FinFET→GAA→CFET）",
    books: [
          "Colinge (ed.), \"FinFETs and Other Multi-Gate Transistors\" (Springer, 2008)",
          "Xiao, \"3D IC Devices, Technologies, and Manufacturing\" (SPIE, 2016)",
          "IRDS（International Roadmap for Devices and Systems）年度路线图 + imec 技术报告"
    ],
    chapters: [
      "摩尔定律 scaling 简史与 Dennard 缩放终结",
      "晶体管架构演进：平面 → FinFET → GAA 纳米片 → CFET",
      "high-k/金属栅（栅氧 scaling 困境与解决方案）",
      "沟道工程（应变硅、SiGe/Ge 沟道、III-V 沟道探索）",
      "源漏工程与接触电阻（S/D 外延、硅化物、接触 scaling）",
      "互连演进（铝→铜大马士革→钴/钌、low-k 介质、互连 RC 瓶颈）",
      "EUV 工艺整合（与《光刻技术与光刻机》衔接：图形化策略）",
      "背面供电网络（BSPDN）与埋入式电源轨（BPR）",
      "DTCO/STCO（设计-工艺-系统协同优化方法论）",
      "3D 单片集成与晶圆键合（顺序式 3D、与封装 3D 的分工）",
      "制程命名与营销（「3nm/2nm」的真实含义、PPA 指标）",
      "路线图展望（IRDS/imec：A14/A10 节点、CFET、2D 材料沟道）"
    ],
  },
  'cs/cpu-microarchitecture': {
    title: "CPU 微架构（乱序/分支预测/存储一致性）",
    books: [
          "Hennessy, Patterson, \"Computer Architecture: A Quantitative Approach\" (6th ed., 2017)",
          "Shen, Lipasti, \"Modern Processor Design: Fundamentals of Superscalar Processors\" (2005)",
          "Patterson, Hennessy, \"Computer Organization and Design\" (RISC-V ed., 2020)"
    ],
    chapters: [
      "ISA 与微架构的分工、流水线基础（五级流水线）",
      "冒险与 forwarding、流水线控制",
      "分支预测（两位饱和计数器、TAGE、感知机预测器）",
      "超标量发射与动态调度（记分板、Tomasulo 算法）",
      "寄存器重命名与重排序缓冲（ROB）、精确异常",
      "推测执行与安全侧信道（Spectre/Meltdown 及缓解）",
      "存储层次（缓存组织、映射、替换策略、写策略）",
      "预取（硬件预取器、软件预取）",
      "存储一致性模型（SC/TSO/弱模型）与缓存一致性协议（MESI/MOESI/目录）",
      "多核与片上互连（环形/Mesh/NoC）",
      "SIMD 与向量扩展（SSE/AVX/RVV/SVE）",
      "性能建模与评估（IPC 分析、Roofline、微基准测试）"
    ],
  },
  'advanced/gpu-architecture-cuda': {
    title: "GPU 架构与 CUDA 并行编程",
    books: [
          "Kirk, Hwu, \"Programming Massively Parallel Processors\" (4th ed., 2022)",
          "NVIDIA, \"CUDA C++ Programming Guide\"（随 CUDA 版本更新的官方指南）",
          "Hennessy, Patterson, \"Computer Architecture: A Quantitative Approach\" (6th ed., §4 GPU 章)"
    ],
    chapters: [
      "GPU 简史：从图形管线到通用计算（GPGPU）",
      "SIMT 执行模型与硬件层次（SM/warp/线程束调度）",
      "CUDA 编程模型（grid/block/thread、kernel 启动）",
      "内存层次（全局/共享/常量/纹理内存、合并访问）",
      "占用率与延迟隐藏（并行度量化分析）",
      "共享内存与 bank conflict、同步原语",
      "Tensor Core 与矩阵运算（WMMA/MMA、与 H100/B200 博文联动）",
      "流、事件与并发执行（计算/传输重叠）",
      "统一内存与新特性（页迁移、动态并行、协作组）",
      "性能分析与调优（Nsight、Roofline、内存/计算受限判定）",
      "多 GPU 编程（NVLink/NVSwitch、NCCL、与集群博文衔接）",
      "图形管线架构概述（光栅化、光线追踪核心 RT Core）"
    ],
  },
  'cs/eda-algorithms': {
    title: "EDA 算法与芯片设计流程",
    books: [
          "Wang, Chang, Cheng, \"Electronic Design Automation: Synthesis, Verification, and Test\" (Morgan Kaufmann, 2009)",
          "Kahng, Lienig, Markov, Hu, \"VLSI Physical Design: From Graph Partitioning to Timing Closure\" (2nd ed., 2022)",
          "Lavagno, Martin, Scheffer (eds.), \"Electronic Design Automation for IC System Design, Verification, and Testing\" (2006)"
    ],
    chapters: [
      "设计流程总览（规格→RTL→综合→布局布线→签核→流片）",
      "逻辑综合（两级/多级逻辑优化、工艺映射）",
      "高层次综合 HLS（调度、分配、绑定）",
      "布图规划与布局（划分、模拟退火、解析式布局器）",
      "时钟树综合（CTS、偏斜控制）",
      "布线（Steiner 树、全局/详细布线、轨道分配）",
      "静态时序分析 STA（时序图、RC 提取、OCV/AOCV）",
      "物理验证（DRC/LVS/ERC、天线效应）",
      "仿真与验证（事件驱动仿真、覆盖率、UVM 方法学）",
      "形式验证（等价性检查、模型检验）",
      "可制造性设计 DFM（OPC 交互、良率感知设计）",
      "ML for EDA（布局/布线/良率预测的机器学习方法）"
    ],
  },
  'social/semiconductor-industry': {
    title: "半导体产业与供应链（芯片战争）",
    books: [
          "Miller, \"Chip War: The Fight for the World's Most Critical Technology\" (2022)（中译《芯片战争》）",
          "O'Mara, \"The Code: Silicon Valley and the Remaking of America\" (2019)",
          "SIA（美国半导体行业协会）/ McKinsey 半导体产业报告与 IRDS 路线图"
    ],
    chapters: [
      "产业模式演进（IDM → fabless/foundry 垂直分工的诞生）",
      "台积电模式（纯代工的商业创新、产能与良率的规模壁垒）",
      "设备供应链（ASML、应用材料、泛林、东京电子、KLA 的垄断格局）",
      "材料与耗材（信越/SUMCO 硅片、光刻胶、特种气体）",
      "设计生态（ARM 与 IP 授权模式、EDA 三巨头、RISC-V 的变量）",
      "市场结构（存储的周期性与寡头化、代工/逻辑/模拟分层）",
      "摩尔定律经济学（建厂成本曲线、先进节点的客户收窄）",
      "产业政策（美国 CHIPS 法案、中国大基金、欧盟/日韩补贴竞赛）",
      "出口管制与技术主权（实体清单、EUV 禁运、算力管制）",
      "地缘格局（台湾海峡集中度风险、供应链「去风险化」）",
      "人才与研发模式（产学联盟 imec/SEMATECH 的兴衰）",
      "中国大陆的追赶路径（成熟制程、设备国产化、先进封装换道）"
    ],
  },
  'engineering/rocket-propulsion': {
    title: "火箭推进工程",
    books: [
          "Sutton, Biblarz, \"Rocket Propulsion Elements\" (9th ed., 2017)（中译《火箭推进原理》）",
          "Huzel, Huang, \"Modern Engineering for Design of Liquid-Propellant Rocket Engines\" (AIAA, 1992)",
          "NASA SP 系列液体/固体发动机设计手册（NASA SP-8000 系列公开技术文件）"
    ],
    chapters: [
      "火箭推进基础（推力方程、比冲、齐奥尔科夫斯基公式）",
      "喷管流动与膨胀比（特征速度、推力系数、高度补偿喷管）",
      "液体推进剂（液氧煤油/液氧甲烷/液氢/肼类，性能与贮存）",
      "液体发动机循环（燃气发生器/分级燃烧/全流量分级燃烧/膨胀/抽气循环）",
      "涡轮泵与阀门（气蚀、轴封、诱导轮设计）",
      "推力室与燃烧不稳定（喷注器、再生冷却、烧蚀）",
      "固体火箭发动机（装药型面、内弹道、推力矢量）",
      "固液混合与电推进（霍尔/离子/MPD，高比冲深空推进）",
      "推进剂供应与增压系统（贮箱、气瓶、落压/泵压）",
      "推力矢量控制与姿态动力系统（游动发动机/RCS）",
      "发动机试车与可靠性（试车台、测量、故障案例）",
      "典型发动机谱系（Merlin/Raptor/RS-25/RD-180/YF-100/LE-9 对比）"
    ],
  },
  'engineering/astrodynamics-attitude-control': {
    title: "航天轨道力学与姿态控制",
    books: [
          "Vallado, \"Fundamentals of Astrodynamics and Applications\" (4th ed., 2013)",
          "Bate, Mueller, White, Saylor, \"Fundamentals of Astrodynamics\" (2nd ed., 2020)",
          "Sidi, \"Spacecraft Dynamics and Control: A Practical Engineering Approach\" (1997)"
    ],
    chapters: [
      "二体问题与开普勒轨道（轨道六根数、星下点）",
      "轨道机动（霍曼转移、双椭圆转移、平面改变）",
      "轨道确定与摄动（J2、大气阻力、太阳光压、三体引力）",
      "发射窗口与交会对接轨道设计",
      "行星际转移（兰伯特问题、引力弹弓、霍曼链）",
      "轨道类型谱系（LEO/MEO/GEO/SSO/大椭圆/晕轨道/拉格朗日点）",
      "再入与返回弹道（气动减速、过载与热流）",
      "姿态运动学（欧拉角/四元数）与姿态动力学",
      "姿态确定（太阳/地球/星敏感器、陀螺、卡尔曼滤波）",
      "姿态控制执行机构（反作用轮/磁力矩器/推力器）与控制律",
      "轨道维持与星座构型（相位保持、Walker 星座）",
      "空间碎片环境与规避机动"
    ],
  },
  'engineering/spacecraft-design': {
    title: "航天器总体设计",
    books: [
          "Wertz, Larson (eds.), \"Space Mission Analysis and Design\" (SMAD, 3rd ed., 1999)",
          "Fortescue, Swinerd, Stark, \"Spacecraft Systems Engineering\" (4th ed., 2011)",
          "Griffin, French, \"Space Vehicle Design\" (2nd ed., AIAA, 2004)"
    ],
    chapters: [
      "空间任务分析与总体设计流程（任务需求→轨道→载荷→平台）",
      "空间环境（真空、热、辐射、微流星、原子氧）",
      "结构与机构（构型、动静载荷、展开机构、分离装置）",
      "热控分系统（被动/主动热控、热管、百叶窗、低温制冷）",
      "电源分系统（太阳电池阵、蓄电池、RTG、配电）",
      "测控与数传（TT&C、链路预算、地面站）",
      "星载数据管理（OBDH、总线、容错计算）",
      "推进分系统（单组元/双组元/电推进选型）",
      "有效载荷集成（光学校准、微振动、电磁兼容）",
      "可靠性、安全性与质量保证（冗余、FMEA、余度管理）",
      "空间环境试验与鉴定（振动/噪声/热真空/EMC 试验体系）",
      "总体方案权衡与成本模型（质量/功率/链路三大预算）"
    ],
  },
  'engineering/satellite-systems': {
    title: "卫星工程与应用（通信/遥感/导航）",
    books: [
          "Maral, Bousquet, Sun, \"Satellite Communications Systems\" (7th ed., 2020)",
          "Elachi, van Zyl, \"Introduction to the Physics and Techniques of Remote Sensing\" (3rd ed., 2021)",
          "Misra, Enge, \"Global Positioning System: Signals, Measurements, and Performance\" (2nd ed., 2006)"
    ],
    chapters: [
      "卫星通信系统（透明/再生转发器、频段分配 C/Ku/Ka）",
      "星地链路预算与调制编码（DVB-S2、雨衰、多波束天线）",
      "高通量卫星与低轨星座（Starlink/OneWeb/星网的架构）",
      "光学遥感（推扫/摆扫、分辨率与幅宽折中、定标）",
      "微波遥感（SAR 成像原理、干涉 InSAR、散射计/高度计）",
      "遥感数据应用链（辐射校正→几何校正→反演→产品）",
      "卫星导航原理（伪距测量、卫星钟差、精密定位 PPP/RTK）",
      "GNSS 系统对比（GPS/GLONASS/伽利略/北斗的信号与体制）",
      "气象与海洋卫星（静止/极轨双体系、载荷谱系）",
      "科学卫星与空间天文（巡天、行星际中继）",
      "小卫星与立方星（CubeSat 标准、搭载发射、编队飞行）",
      "在轨服务与碎片清除（交会捕获、延寿、主动清除）"
    ],
  },
  'engineering/human-spaceflight': {
    title: "载人航天与空间生命保障",
    books: [
          "Larson, Pranke (eds.), \"Human Spaceflight: Mission Analysis and Design\" (McGraw-Hill, 1999)",
          "Eckart, \"Spaceflight Life Support and Biospherics\" (1996)",
          "NASA, \"NASA-STD-3001 载人航天人因与健康标准\"（公开标准文件）"
    ],
    chapters: [
      "载人航天简史与任务类型（近地轨道/登月/深空）",
      "空间环境对人体的挑战（微重力、辐射、隔离、昼夜节律）",
      "载人飞船系统（神舟/联盟/龙飞船/猎户座构型对比）",
      "发射逃逸与救生（逃逸塔/整罩逃逸/中止模式）",
      "环境控制与生命保障 ECLSS（大气再生、水循环、废物处理）",
      "航天服（舱内服/舱外服、EMU/飞天服、预呼吸与减压病）",
      "出舱活动 EVA（气闸、工效学、安全系绳）",
      "航天医学（骨丢失/肌肉萎缩/心血管失调及对抗措施）",
      "空间辐射防护（银河宇宙线/太阳粒子事件、屏蔽与限值）",
      "空间站工程（ISS/天宫的组装、运营与后勤补给）",
      "交会对接（手动/自动、对接机构、相对导航）",
      "深空载人任务前瞻（月球门户 Gateway、载人登火的生命保障难题）"
    ],
  },
  'engineering/deep-space-exploration': {
    title: "深空探测工程（NASA/JPL 任务体系）",
    books: [
          "Taylor (ed.), \"Deep Space Communications\" (JPL DESCANSO 系列, 2016)",
          "NASA/JPL DESCANSO 深空通信与导航系列专著（公开电子版）",
          "NASA, \"Basics of Space Flight\"（JPL 深空任务公开培训手册）"
    ],
    chapters: [
      "深空任务体系（飞越/环绕/着陆/巡视/采样返回的任务链）",
      "行星际轨道设计（霍曼链、引力弹弓接力、低推力轨道）",
      "深空网 DSN（70m/34m 天线阵、上行下行链路、时延通信）",
      "深空导航（甚长基线干涉 ΔDOR、多普勒/测距、自主导航）",
      "进入下降着陆 EDL（「恐怖七分钟」：降落伞/反推/空中吊车）",
      "行星着陆器与巡视器（好奇/毅力/祝融的构型与移动系统）",
      "采样返回工程（嫦娥五/六号、OSIRIS-REx、隼鸟系列）",
      "外太阳系任务（旅行者/伽利略/卡西尼/新视野的核电源与长寿命设计）",
      "着陆选址与行星保护（前向/后向污染防控、COSPAR 规范）",
      "深空科学载荷（光谱仪/雷达/质谱/地震仪的星载化）",
      "在轨组装与载人深空（SLS/猎户座、地月空间架构）",
      "未来方向（木卫二快船、天王星旗舰任务、星际探测器概念）"
    ],
  },
  'engineering/reusable-launch-commercial-space': {
    title: "可重复使用火箭与商业航天（SpaceX 案例）",
    books: [
          "Berger, \"Liftoff: Elon Musk and the Desperate Early Days That Launched SpaceX\" (2021)",
          "Berger, \"Reentry: SpaceX, Elon Musk, and the Reusable Rockets\" (2024)",
          "SpaceX 官方用户手册（Falcon 9/Starship Payload User's Guide）与 NASA 商业载人项目公开报告"
    ],
    chapters: [
      "商业航天简史（从政府垄断到 NewSpace：SpaceX/蓝源/火箭实验室）",
      "Falcon 9 架构（Merlin 发动机簇、铝锂贮箱、过冷推进剂）",
      "垂直回收技术（再入点火/着陆点火、栅格舵、着陆腿、无人船回收）",
      "复用经济学（翻新成本、复用次数与报价模型、对一次性火箭的颠覆）",
      "Falcon Heavy 与拼车发射（Transporter 小卫星搭载模式）",
      "Starship 系统（全流量分级燃烧 Raptor、不锈钢箭体、筷子臂捕获）",
      "在轨加注与登月/登火架构（HLS 月面着陆器、燃料库概念）",
      "Starlink 星座工程（平板卫星、星间激光链路、相控阵用户终端）",
      "龙飞船与商业载人（NASA 商业乘员计划的采购创新）",
      "快速迭代研发文化（「测试-爆炸-迭代」与传统航天的对比）",
      "发射许可与监管（FAA/环境评估/发射场容量）",
      "全球商业航天格局（中国商业火箭、印度 Skyroot、欧洲阿里安 6 的应对）"
    ],
  },
  'engineering/china-space-program': {
    title: "中国航天工程（长征/载人/探月/空间站）",
    books: [
          "国务院新闻办公室,《2021 中国的航天》白皮书（及历次航天白皮书）",
          "中国航天科技集团,《中国航天科技活动蓝皮书》（年度公开报告）",
          "中国载人航天工程办公室 / 国家航天局公开技术资料与任务公报"
    ],
    chapters: [
      "中国航天发展脉络（「两弹一星」→载人三步走→深空探测）",
      "长征火箭谱系（长征 2/3/5/6/7/8/9 系列的构型与能力定位）",
      "长征五号的跨越（5m 直径芯级、YF-100/YF-77 新动力）",
      "载人航天工程（神舟飞船、空间实验室到天宫空间站）",
      "天宫空间站（三舱构型、机械臂、巡天望远镜共轨飞行）",
      "探月工程（绕落回三步、嫦娥五号采样、嫦娥六号月背采样）",
      "行星探测起步（天问一号一次实现绕落巡、后续小行星/火星采样规划）",
      "北斗导航系统（三步走、混合星座、短报文特色）",
      "高分/风云/资源卫星体系（对地观测的国家基础设施）",
      "测控与发射场（酒泉/太原/西昌/文昌、远望船队、深空站）",
      "商业航天新势力（蓝箭朱雀、星际荣耀、天兵科技、星河动力）",
      "未来规划（载人登月、长征九号、国际月球科研站 ILRS）"
    ],
  },
  'engineering/radar-systems': {
    title: "雷达原理与系统",
    books: [
          "Skolnik, \"Introduction to Radar Systems\" (3rd ed., 2001)（中译《雷达系统导论》）",
          "Skolnik (ed.), \"Radar Handbook\" (3rd ed., 2008)",
          "Richards, \"Fundamentals of Radar Signal Processing\" (2nd ed., 2014)"
    ],
    chapters: [
      "雷达方程与探测距离（功率孔径积、RCS 概念入门）",
      "波形与匹配滤波（脉冲压缩、线性调频 LFM、相位编码）",
      "多普勒处理（MTI/MTD、脉冲多普勒体制）",
      "杂波与干扰环境（地海杂波模型、箔条）",
      "检测理论（恒虚警 CFAR、检测概率与虚警率）",
      "相控阵雷达（波束扫描、T/R 组件、有源阵 AESA）",
      "合成孔径与逆合成孔径（SAR/ISAR 成像雷达）",
      "跟踪雷达（单脉冲、相控阵跟踪、火控雷达）",
      "雷达体制谱系（预警/火控/制导/气象/探地/超视距 OTH）",
      "双/多基地雷达与无源探测（外辐射源雷达）",
      "抗干扰技术 ECCM（频率捷变、低截获概率 LPI 波形）",
      "新体制雷达（MIMO 雷达、认知雷达、量子雷达概念辨析）"
    ],
  },
  'engineering/missile-guidance': {
    title: "导弹与制导控制",
    books: [
          "Zarchan, \"Tactical and Strategic Missile Guidance\" (6th ed., AIAA, 2012)",
          "Siouris, \"Missile Guidance and Control Systems\" (Springer, 2004)",
          "Fleeman, \"Tactical Missile Design\" (2nd ed., AIAA, 2006)"
    ],
    chapters: [
      "导弹分类与总体组成（巡航/弹道/防空导弹的任务剖面）",
      "导弹气动布局（正常式/鸭式/无翼式、大攻角气动）",
      "推进系统选型（固体火箭/冲压/涡喷涡扇/超然冲压）",
      "制导回路基础（制导-控制一体化结构、脱靶量）",
      "经典制导律（追踪法/平行接近/比例导引 PN 及其变型）",
      "现代制导律（最优制导、微分对策、带落角约束制导）",
      "导引头（雷达/红外成像/激光半主动/多模复合制导）",
      "自动驾驶仪与执行机构（舵机/燃气舵/推力矢量）",
      "惯性导航与组合导航（INS/GPS/地形匹配/景象匹配）",
      "弹道导弹力学（主动段/中段/再入段、突防与诱饵）",
      "反导拦截动力学（KKV 动能杀伤、碰撞点预测）",
      "效能评估与仿真（蒙特卡洛打靶、六自由度仿真）"
    ],
  },
  'engineering/combat-aircraft': {
    title: "战斗机与军用飞机工程",
    books: [
          "Raymer, \"Aircraft Design: A Conceptual Approach\" (6th ed., 2018)",
          "Nicolai, Carichner, \"Fundamentals of Aircraft and Airship Design, Vol. I\" (AIAA, 2010)",
          "Huenecke, \"Modern Combat Aircraft Design\" (1987)"
    ],
    chapters: [
      "战斗机代际演进（一代到五代/六代的标志性技术跃迁）",
      "任务需求与总体参数（推重比/翼载/升阻特性权衡）",
      "高机动气动布局（边条翼/鸭翼/静不稳定与增稳控制）",
      "超声速与大攻角气动（激波、涡升力、过失速机动）",
      "隐身外形一体化设计（与《隐身技术与 RCS》衔接）",
      "推进与进气道（变循环发动机概念、超声速进气道设计）",
      "飞行控制（电传飞控、推力矢量、综合飞行/火力/推进控制）",
      "航电与传感器（AESA 雷达、光电分布式孔径 EODAS、头盔显示）",
      "武器系统集成（内埋弹舱、外挂管理、火力控制）",
      "运输机与特种飞机（大运的气动/结构/货舱空投设计、预警机/加油机）",
      "舰载机特殊设计（弹射/拦阻、折叠翼、防腐与甲板适配）",
      "无人作战飞机（忠诚僚机、隐身无人攻击机的总体特点）"
    ],
  },
  'engineering/helicopter-engineering': {
    title: "直升机工程",
    books: [
          "Leishman, \"Principles of Helicopter Aerodynamics\" (2nd ed., 2016)",
          "Johnson, \"Helicopter Theory\" (Dover, 1994)",
          "Seddon, Newman, \"Basic Helicopter Aerodynamics\" (3rd ed., 2011)"
    ],
    chapters: [
      "旋翼气动基础（动量理论、叶素理论、悬停与前飞）",
      "旋翼运动学（挥舞/摆振/变距、铰接式/无铰/无轴承桨毂）",
      "前行桨叶压缩性与后行桨叶失速（速度限制的根源）",
      "尾桨与反扭矩（涵道尾桨、无尾桨 NOTAR）",
      "直升机性能（悬停升限、前飞航程、自转下滑）",
      "操纵与稳定性（周期变距、总距、姿态响应特性）",
      "旋翼振动与噪声（桨-涡干扰 BVI、减振设计）",
      "传动系统（主减速器、离合器、润滑失效后的干运转设计）",
      "构型变体（共轴双旋翼/纵列双旋翼/复合式/倾转旋翼 V-22）",
      "舰载直升机特殊问题（着舰、鱼叉助降、折叠与防腐）",
      "武装直升机设计要点（装甲/抗弹伤冗余、光电转塔、武器短翼）",
      "电动垂直起降 eVTOL（分布式电推进对旋翼设计的重塑）"
    ],
  },
  'engineering/naval-vessels': {
    title: "军用舰艇工程（航母/驱护/两栖）",
    books: [
          "Lewis (ed.), \"Principles of Naval Architecture\" (SNAME, 2nd rev., 1988)",
          "Lamb (ed.), \"Ship Design and Construction\" (SNAME, 2003)",
          "Friedman, \"U.S. Destroyers: An Illustrated Design History\"（舰艇设计史权威系列）"
    ],
    chapters: [
      "军用舰艇类型谱系（航母/驱护/护卫舰/两栖/补给舰的任务定位）",
      "舰艇总体设计流程（任务书→方案→技术设计、重量裕度纪律）",
      "快速性与耐波性（军舰线型、穿浪/深 V/小水线面双体）",
      "舰艇结构（纵强度、局部强度、抗冲击设计）",
      "推进系统（柴燃联合 CODOG/CODAG、全燃 COGAG、综合电力推进 IEP）",
      "隐身与信号特征控制（雷达/红外/声/磁场特征的源头治理）",
      "舰艇作战系统（相控阵雷达 + 垂发的体系、宙斯盾/中华神盾架构）",
      "舰载武器集成（导弹垂发系统、舰炮、近防系统 CIWS）",
      "航母工程专题（弹射/拦阻/滑跃、甲板调度与航空联队运作）",
      "两栖舰艇（坞登/两攻、气垫登陆艇与直升机投送）",
      "舰艇损管与生命力（分舱、抗沉性、消防与三防）",
      "无人舰艇与海上分布式作战（USV 的工程挑战）"
    ],
  },
  'engineering/submarine-engineering': {
    title: "潜艇工程（总体/静音/核动力）",
    books: [
          "Burcher, Rydill, \"Concepts in Submarine Design\" (Cambridge, 1994)",
          "Ross, \"Mechanics of Underwater Noise\" (1976)（水下噪声经典专著）",
          "Friedman, \"U.S. Submarines Since 1945: An Illustrated Design History\""
    ],
    chapters: [
      "潜艇类型与任务（攻击型/战略导弹核潜艇、常规潜艇、特种潜艇）",
      "总体设计（单/双壳体、储备浮力、舱段划分）",
      "耐压结构（耐压壳强度、极限深度、疲劳寿命）",
      "潜浮与操纵（压载水舱、均衡系统、水平舵与操艇）",
      "常规动力（柴电、AIP：斯特林/燃料电池/闭式循环柴油机）",
      "核动力装置（一体化压水堆、自然循环、一回路与屏蔽）",
      "静音工程（噪声源分类：螺旋桨/机械/水动力；减振浮筏、消声瓦）",
      "推进与低噪声（七叶大侧斜螺旋桨、泵喷推进、无轴推进概念）",
      "声呐系统（舷侧阵/拖曳阵/艇首阵，与《水声工程》衔接）",
      "水下通信与导航（长波通信、蓝绿激光、惯导 + 重力/地形辅助）",
      "潜射武器（鱼雷发射管/垂直发射、潜射导弹出水动力学）",
      "救生与损管（深潜救生艇、舱室破损应急、防火防爆）"
    ],
  },
  'engineering/stealth-rcs': {
    title: "隐身技术与雷达散射截面（RCS）",
    books: [
          "Knott, Shaeffer, Tuley, \"Radar Cross Section\" (2nd ed., 2004)",
          "Lynch, \"Introduction to RF Stealth\" (SciTech, 2004)",
          "Jenn, \"Radar and Laser Cross Section Engineering\" (2nd ed., AIAA, 2005)"
    ],
    chapters: [
      "RCS 基础（定义、单位 dBsm、极化与频率/角度依赖）",
      "散射机理（镜面反射/边缘绕射/爬行波/腔体效应）",
      "RCS 预估方法（物理光学 PO、几何绕射 GTD/UTD、矩量法 MoM）",
      "外形隐身设计原则（平行棱边原则、S 形进气道、内埋弹舱）",
      "雷达吸波材料 RAM（ Salisbury 屏、磁性/介电吸波体、超材料吸波）",
      "飞机隐身案例剖析（F-117 的多面体、B-2 的飞翼、F-22/J-20 的综合隐身）",
      "舰艇与车辆的隐身应用（频散设计、上层建筑内倾）",
      "红外隐身（尾焰冷却、二元喷管、热遮蔽）",
      "射频隐身的系统观（低截获概率 LPI 雷达、被动探测协同）",
      "RCS 测量（室外/室内紧缩场、扫频与成像诊断）",
      "反隐身技术（低频雷达、多基地、无源探测、量子雷达辨析）",
      "隐身与反隐身的攻防演化（体系对抗视角）"
    ],
  },
  'engineering/electronic-warfare': {
    title: "电子战与通信对抗",
    books: [
          "Adamy, \"EW 101: A First Course in Electronic Warfare\" (2001)",
          "Adamy, \"EW 102: A Second Course in Electronic Warfare\" (2004)",
          "Schleher, \"Electronic Warfare in the Information Age\" (1999)"
    ],
    chapters: [
      "电子战体系（电子支援 ES/电子攻击 EA/电子防护 EP 三分法）",
      "电子支援接收机（晶体视频/超外差/数字接收机、测向体制）",
      "雷达告警接收机 RWR（威胁识别、告警逻辑）",
      "压制式干扰（噪声干扰、干扰方程、烧穿距离）",
      "欺骗式干扰（距离/速度门拖引、DRFM 数字射频存储）",
      "投掷式对抗（箔条云形成与特性、红外曳光弹）",
      "通信对抗（跳频/直扩信号的侦察与干扰、数据链对抗）",
      "导航战（GPS 干扰与抗干扰、欺骗信号生成）",
      "光电对抗（激光告警、红外定向干扰 DIRCM）",
      "反辐射武器（反辐射导弹导引头、辐射源定位）",
      "ECCM 与抗干扰设计（与《雷达原理与系统》抗干扰章互链）",
      "认知电子战（AI 驱动的实时波形对抗、开放式电磁频谱战）"
    ],
  },
  'engineering/ballistics-ammunition': {
    title: "弹药工程与弹道学",
    books: [
          "Carlucci, Jacobson, \"Ballistics: Theory and Design of Guns and Ammunition\" (3rd ed., 2021)",
          "Cooper, Kurowski, \"Introduction to the Technology of Explosives\" (1996)",
          "Walters, Zukas, \"Fundamentals of Shaped Charges\" (1989)"
    ],
    chapters: [
      "内弹道学（发射药燃烧、膛压曲线、身管设计）",
      "外弹道学（质点弹道、六自由度刚体弹道、气象修正）",
      "终点弹道学（侵彻、贯穿、杀伤破片、冲击波效应）",
      "发射装药与火炸药（单/双/三基药、炸药爆轰基础）",
      "弹药结构（枪弹/炮弹/火箭弹的构造与工艺）",
      "引信技术（触发/时间/近炸/电子安全系统）",
      "穿甲弹药（动能穿甲弹 APFSDS、长杆穿甲机理）",
      "破甲弹药（聚能装药、射流形成与侵彻）",
      "防空与反导弹药（定向破片战斗部、动能拦截）",
      "装甲与防护（均质/复合/爆炸反应装甲、间隙装甲）",
      "制导弹药（末敏弹、制导炮弹、巡飞弹）",
      "弹药安全性与贮存（钝感弹药、寿命评估）"
    ],
  },
  'engineering/fusion-engineering': {
    title: "核聚变工程（托卡马克/惯性约束/ITER）",
    books: [
          "Freidberg, \"Plasma Physics and Fusion Energy\" (Cambridge, 2007)",
          "Wesson, Campbell, \"Tokamaks\" (4th ed., 2011)",
          "ITER 组织与 NIF/LLE 公开技术报告（聚变装置一手工程资料）"
    ],
    chapters: [
      "聚变反应与劳森判据（D-T/D-D/p-B11 反应截面、三重积）",
      "磁约束原理（磁镜、环向约束、托卡马克位形）",
      "等离子体平衡与磁流体不稳定性（扭曲/撕裂模、ELM）",
      "等离子体加热（欧姆/中性束 NBI/射频 ICRH/ECRH）与电流驱动",
      "约束与输运（L 模/H 模、输运垒、约束定标律）",
      "偏滤器与等离子体-壁相互作用（第一壁材料、钨/铍选择）",
      "超导磁体工程（TF/PF 线圈、CICC 导体、失超保护）",
      "包层与氚增殖（锂铅/氦冷包层、氚自持循环）",
      "ITER 装置与计划（设计目标、国际合作架构、EAST/CFETR 对照）",
      "惯性约束聚变（激光间接/直接驱动、NIF 点火成果解读）",
      "仿星器与替代概念（W7-X、球形托卡马克、场反位形、Z 箍缩）",
      "聚变电站工程（能量取出、材料辐照损伤、经济性与时间表辨析）"
    ],
  },
  'engineering/radiation-detection-protection': {
    title: "核辐射探测与防护",
    books: [
          "Knoll, \"Radiation Detection and Measurement\" (4th ed., 2010)",
          "Shultis, Faw, \"Fundamentals of Nuclear Science and Engineering\" (3rd ed., 2017)",
          "ICRP 出版物与 GB 18871《电离辐射防护与辐射源安全基本标准》"
    ],
    chapters: [
      "辐射与物质的相互作用（带电粒子/γ/中子的作用机制）",
      "气体探测器（电离室/正比计数器/GM 管）",
      "闪烁探测器（无机/有机闪烁体、光电倍增与 SiPM）",
      "半导体探测器（高纯锗 HPGe、硅面垒、CdZnTe）",
      "中子探测（BF3/He-3、慢化与反冲法）",
      "γ 谱学与核素识别（全能峰、康普顿平台、效率刻度）",
      "剂量学量（吸收剂量/当量剂量/有效剂量、ICRU 体系）",
      "剂量测量（热释光 TLD、电子个人剂量计、场所监测）",
      "辐射防护原则（正当性/最优化/限值、ALARA）",
      "屏蔽设计与计算（点核方法、蒙特卡洛 MCNP/Geant4 入门）",
      "内照射防护与放射性废物管理基础",
      "天然辐射与氡、核与辐射应急（监测、干预水平）"
    ],
  },
  'engineering/nuclear-fuel-cycle': {
    title: "核燃料循环与核安全",
    books: [
          "Cochran, Tsoulfanidis, \"The Nuclear Fuel Cycle: Analysis and Management\" (2nd ed., 1999)",
          "Wilson (ed.), \"The Nuclear Fuel Cycle: From Ore to Wastes\" (Oxford, 1996)",
          "IAEA 安全标准丛书与《核安全公约》公开文件"
    ],
    chapters: [
      "核燃料循环全景（一次通过 vs 闭式循环的路线之争）",
      "铀矿勘查与采冶（地浸采铀、铀浓缩物黄饼）",
      "铀转化与浓缩（气体扩散/离心法、激光浓缩概念）",
      "燃料元件制造（UO2 芯块、锆合金包壳、组件设计）",
      "堆内辐照行为（燃耗、肿胀、裂变气体释放）",
      "乏燃料贮存（水池/干式贮存、衰变热管理）",
      "后处理（PUREX 流程、铀钚分离、高放废液）",
      "MOX 燃料与快堆闭式循环（嬗变与增殖概念）",
      "放射性废物分类与处置（低中放近地表、高放地质处置库、芬兰 Onkalo）",
      "核设施退役（去污、拆除、场址释放）",
      "核安全文化（纵深防御、三哩岛/切尔诺贝利/福岛事故工程教训）",
      "核保障监督与不扩散（IAEA 保障、材料衡算、两用技术管控）"
    ],
  },
  'engineering/semiconductor-materials': {
    title: "半导体材料（硅/化合物衬底与外延）",
    books: [
          "Shimura, \"Semiconductor Silicon Crystal Technology\" (1989)",
          "Stringfellow, \"Organometallic Vapor-Phase Epitaxy: Theory and Practice\" (2nd ed., 1999)",
          "Holloway, McGuire (eds.), \"Handbook of Compound Semiconductors\" (1995)"
    ],
    chapters: [
      "半导体材料的地位：材料纯度如何决定器件上限（9N-11N 的故事）",
      "多晶硅提纯（西门子法、流化床法、电子级 vs 太阳能级）",
      "单晶硅生长（直拉 CZ 的温场控制、区熔 FZ、大直径化的工程极限）",
      "晶圆加工（切片/研磨/抛光/CMP、平整度与表面缺陷指标）",
      "晶体缺陷工程（位错/点缺陷/氧沉淀、内吸杂）",
      "外延生长（气相外延 VPE、分子束外延 MBE、原子层级控制）",
      "MOCVD 与 III-V 族外延（GaAs/InP/GaN 的晶格匹配问题）",
      "SiC 与 GaN 衬底（PVT 生长、微管缺陷、良率为何难）",
      "SOI 与工程衬底（Smart Cut 键合剥离、应变硅衬底）",
      "锗与 2D 材料衬底化（新型沟道材料的材料学瓶颈）",
      "靶材与湿化学品（溅射靶材纯度、电子级化学品供应链）",
      "材料表征与质量认证（XRD 摇摆曲线、少子寿命、颗粒度检测）"
    ],
  },
  'engineering/superalloys-high-temperature': {
    title: "高温合金与耐热材料（单晶叶片）",
    books: [
          "Reed, \"The Superalloys: Fundamentals and Applications\" (Cambridge, 2006)",
          "Donachie, Donachie, \"Superalloys: A Technical Guide\" (2nd ed., 2002)",
          "Sims, Stoloff, Hagel (eds.), \"Superalloys II\" (1987)（领域经典）"
    ],
    chapters: [
      "高温服役环境的挑战（蠕变/疲劳/氧化/热腐蚀四位一体）",
      "镍基高温合金的相结构（γ 基体 + γ' 强化相的设计哲学）",
      "合金化设计（Al/Ti/Ta 定 γ'、Re/Ru 的代际演进）",
      "铸造工艺演进（等轴晶→定向凝固→单晶，晶界为何是敌人）",
      "单晶叶片制造（选晶器/螺旋选晶、杂晶与雀斑缺陷控制）",
      "粉末冶金高温合金（涡轮盘的制粉/热等静压/锻造路线）",
      "钛合金（Ti-6Al-4V、高温钛合金在压气机上的应用）",
      "热障涂层 TBC（YSZ 陶瓷层 + 粘结层、EB-PVD vs 等离子喷涂）",
      "金属间化合物与难熔合金（TiAl 叶片、Nb-Si 系探索）",
      "蠕变与疲劳寿命预测（Larson-Miller 参数、寿命管理）",
      "氧化与热腐蚀机制（保护性氧化膜、热盐腐蚀）",
      "航空发动机材料图谱（从涡喷到高涵道比的材料代际）"
    ],
  },
  'engineering/advanced-composites': {
    title: "先进复合材料（碳纤维/陶瓷基/树脂基）",
    books: [
          "Chawla, \"Composite Materials: Science and Engineering\" (4th ed., 2019)",
          "Hull, Clyne, \"An Introduction to Composite Materials\" (2nd ed., 1996)",
          "Daniel, Ishai, \"Engineering Mechanics of Composite Materials\" (2nd ed., 2006)"
    ],
    chapters: [
      "复合材料设计哲学（增强体+基体+界面、混合律）",
      "碳纤维制造（PAN 原丝→预氧化→碳化/石墨化、T300 到 T1100 的密码）",
      "树脂基复合材料成型（预浸料/热压罐、RTM、自动铺丝 AFP）",
      "层合板力学（经典层合板理论、铺层设计准则）",
      "界面与损伤（界面结合、分层、冲击后压缩 CAI）",
      "陶瓷基复合材料 CMC（SiC/SiC、航空发动机热端应用）",
      "碳/碳复合材料（刹车盘、火箭喷管喉衬）",
      "金属基复合材料（SiC 增强铝、原位自生）",
      "结构健康监测与无损检测（超声 C 扫、声发射）",
      "航空应用案例（B787/A350 复材机体、风扇叶片）",
      "回收与可持续（碳纤维回收、热塑性复材的兴起）",
      "成本与量产（汽车级碳纤维的成本瓶颈与快速成型）"
    ],
  },
  'engineering/optoelectronic-materials': {
    title: "光电与激光材料",
    books: [
          "Kasap, \"Optoelectronics and Photonics: Principles and Practices\" (2nd ed., 2013)",
          "Koechner, \"Solid-State Laser Engineering\" (6th ed., 2006)",
          "Weber, \"Handbook of Optical Materials\" (2002)"
    ],
    chapters: [
      "光学材料基础（折射率/色散/透过窗口、光学玻璃牌号）",
      "激光增益介质（Nd:YAG/Yb:YAG/钛宝石/光纤增益介质）",
      "非线性光学晶体（KDP/BBO/LBO，频率转换的工程参数）",
      "电光与声光材料（调制器、Q 开关材料）",
      "光纤材料（石英预制棒 MCVD、氟化物光纤）",
      "半导体光电子材料（III-V 发光、InGaN/GaN LED 材料体系）",
      "红外光学材料（锗/硅/硫化锌/硫系玻璃）",
      "光学薄膜（增透/高反/滤光膜系设计、镀膜工艺）",
      "闪烁体与辐射探测材料（NaI/BGO/LYSO）",
      "量子点与钙钛矿光电材料（发光/探测的新兴体系）",
      "光学加工与镀膜装备（超精密抛光、离子束修形）",
      "极端环境光学元件（EUV 多层膜、高功率激光损伤阈值）"
    ],
  },
  'engineering/electronic-ceramics': {
    title: "电子陶瓷与功能陶瓷（MLCC/压电/铁电）",
    books: [
          "Moulson, Herbert, \"Electroceramics: Materials, Properties, Applications\" (2nd ed., 2003)",
          "Jaffe, Cook, Jaffe, \"Piezoelectric Ceramics\" (1971)（压电领域经典）",
          "Buchanan (ed.), \"Ceramic Materials for Electronics\" (3rd ed., 2004)"
    ],
    chapters: [
      "功能陶瓷的物理基础（介电/铁电/压电/热释电的关系网）",
      "介电陶瓷与 MLCC（BaTiO3 体系、贱金属内电极 BME、薄层化极限）",
      "MLCC 制造工艺（流延/叠层/共烧、端电极与可靠性）",
      "压电陶瓷（PZT 相图、掺杂改性、超声换能器应用）",
      "无铅压电陶瓷（KNN/BNT 体系的追赶现状）",
      "铁电存储与铁电薄膜（PZT/HfO2 铁电性的意外发现）",
      "微波介质陶瓷（介电常数/品质因数/温度系数三要素、5G 滤波器）",
      "敏感陶瓷（NTC/PTC 热敏、压敏 ZnO、气敏）",
      "透明陶瓷与激光陶瓷（YAG 透明陶瓷的多晶路线）",
      "结构-功能一体化（陶瓷基板、氮化铝/氮化硅散热基板）",
      "陶瓷成型与烧结科学（干压/注浆/流延、放电等离子烧结 SPS）",
      "可靠性工程（绝缘电阻退化、寿命加速试验）"
    ],
  },
  'engineering/magnetic-materials': {
    title: "磁性材料（永磁/软磁/自旋电子学）",
    books: [
          "Cullity, Graham, \"Introduction to Magnetic Materials\" (2nd ed., 2009)",
          "Coey, \"Magnetism and Magnetic Materials\" (Cambridge, 2010)",
          "Skomski, Coey, \"Permanent Magnetism\" (1999)"
    ],
    chapters: [
      "磁性物理基础（磁矩起源、交换作用、磁畴与技术磁化）",
      "永磁材料性能坐标系（剩磁/矫顽力/最大磁能积）",
      "NdFeB 永磁（速凝片工艺、氢破碎、烧结与晶界扩散）",
      "稀土战略与减镝/无镝化（重稀土晶界渗透的工程权衡）",
      "钐钴与铁氧体（高温与低成本两条路线）",
      "软磁材料（硅钢/铁氧体/非晶纳米晶、铁损的来源）",
      "电工钢制造（取向硅钢的二次再结晶、薄规格趋势）",
      "磁记录材料（垂直记录、HAMR 热辅助的介质挑战）",
      "自旋电子学材料（GMR/TMR、磁性隧道结 MRAM）",
      "磁致伸缩与磁致冷材料（TbDyFe、室温磁制冷探索）",
      "高频磁性元件（功率电感/变压器磁芯的选型逻辑）",
      "磁体应用工程（电机磁钢、MRI 超导磁体对照）"
    ],
  },
  'engineering/energy-storage-materials': {
    title: "电池与储能材料（锂离子/固态/钠电）",
    books: [
          "Julien, Mauger, Vijh, Zaghib (eds.), \"Lithium Batteries: Science and Technology\" (Springer, 2016)",
          "Huggins, \"Energy Storage: Fundamentals, Materials and Applications\" (2nd ed., 2015)",
          "Warner, \"The Handbook of Lithium-Ion Battery Pack Design\" (2nd ed., 2024)"
    ],
    chapters: [
      "电化学储能基础（电压/容量/倍率/循环的物理来源）",
      "正极材料（钴酸锂→三元 NCM/NCA→磷酸铁锂的结构化学）",
      "高镍正极工程（容量-稳定性的矛盾、单晶化与包覆）",
      "负极材料（石墨层间化合物、硅基负极的体积膨胀难题）",
      "电解液与界面膜（SEI 的形成化学、新型锂盐与添加剂）",
      "隔膜与安全（聚烯烃微孔膜、热关闭、涂覆改性）",
      "固态电池（硫化物/氧化物/聚合物电解质、界面阻抗症结）",
      "钠离子电池（层状氧化物/聚阴离子/普鲁士蓝路线）",
      "锂金属负极与锂硫/锂空气（终极体系的现实距离）",
      "电芯制造工艺（匀浆/涂布/辊压/卷绕叠片/注液化成）",
      "电池系统（模组/PACK、BMS 算法、热失控防护）",
      "测评与回收（容量衰减机理诊断、梯次利用、湿法回收）"
    ],
  },
  'engineering/nano-2d-materials': {
    title: "纳米与二维材料（石墨烯/TMD/量子点）",
    books: [
          "Cao, \"Nanostructures & Nanomaterials: Synthesis, Properties, and Applications\" (2nd ed., 2011)",
          "Warner, Schäffel, Bachmatiuk, Rümmeli, \"Graphene: Fundamentals and Emergent Applications\" (2013)",
          "Roduner, \"Nanoscopic Materials: Size-Dependent Phenomena\" (2006)"
    ],
    chapters: [
      "纳米效应的物理（尺寸限制、表面原子比例、量子限域）",
      "纳米材料制备（自下而上 vs 自上而下、气相/液相法）",
      "碳纳米管（结构决定金属/半导体性、阵列生长）",
      "石墨烯（机械剥离→CVD 量产、转移工艺的痛点）",
      "二维过渡金属硫族化合物（MoS2/WS2、直接带隙的机会）",
      "二维材料器件（场效应管、柔性电子、异质结堆叠）",
      "量子点（尺寸可调发光、显示与生物标记应用）",
      "纳米线与纳米棒（VLS 生长、传感器与能源器件）",
      "纳米复合材料（分散难题、增强机制）",
      "纳米材料表征（TEM/AFM/Raman 的专用方法）",
      "安全性与标准化（纳米毒理、表征标准）",
      "从实验室到产业（石墨烯产业的十年复盘）"
    ],
  },
  'engineering/biomaterials': {
    title: "生物医用材料",
    books: [
          "Ratner, Hoffman, Schoen, Lemons (eds.), \"Biomaterials Science: An Introduction to Materials in Medicine\" (4th ed., 2020)",
          "Park, Lakes, \"Biomaterials: An Introduction\" (3rd ed., 2007)",
          "Hench (ed.), \"An Introduction to Bioceramics\" (2nd ed., 2013)"
    ],
    chapters: [
      "生物材料的设计约束（生物相容性、灭菌、法规路径）",
      "金属植入材料（钛合金/钴铬/不锈钢、表面改性）",
      "生物陶瓷（羟基磷灰石、生物玻璃、骨水泥）",
      "医用高分子（PEEK/PLA/硅胶、可吸收聚合物的降解动力学）",
      "组织工程支架（多孔结构、细胞外基质仿生）",
      "药物递送系统（控释载体、脂质体、微球）",
      "血液接触材料（抗凝血表面、人工血管与瓣膜）",
      "牙科与骨科材料（种植体骨结合、关节摩擦副）",
      "水凝胶与软组织修复（隐形眼镜、创面敷料）",
      "可降解金属（镁合金/锌合金血管支架）",
      "生物材料的评价（体外/体内试验、ISO 10993 体系）",
      "3D 打印与再生医学（生物墨水、器官芯片交叉）"
    ],
  },
  'engineering/superconducting-materials': {
    title: "超导材料与应用（NbTi/REBCO/磁体）",
    books: [
          "Seeber (ed.), \"Handbook of Applied Superconductivity\" (1998)",
          "Iwasa, \"Case Studies in Superconducting Magnets\" (2nd ed., 2009)",
          "Rogalla, Kes (eds.), \"100 Years of Superconductivity\" (2012)"
    ],
    chapters: [
      "超导材料坐标系（Tc/Hc/Jc 三临界、实用化的判据）",
      "低温超导 NbTi（合金熔炼-拉丝-绞缆工艺、MRI 的主力）",
      "Nb3Sn 金属间化合物（脆性材料的先绕后反应工艺）",
      "高温超导 REBCO 带材（涂层导体、二代带材的产业化）",
      "Bi-2223/2212 线带材（一代高温超导的存量应用）",
      "MgB2 与铁基超导（中间温度区的新选项）",
      "超导磁体工程（绕制/环氧浸渍/失超保护/低温系统）",
      "MRI 与科研磁体（磁场均匀度、匀场技术）",
      "超导电力（限流器/电缆/储能 SMES 的示范工程）",
      "超导在聚变中的应用（ITER 磁体的超导规模纪录）",
      "悬浮与推进（磁悬浮列车、超导电机）",
      "量子计算用超导器件（约瑟夫森结、SQUID 制备）"
    ],
  },
  'engineering/high-performance-polymers': {
    title: "高性能高分子与特种纤维",
    books: [
          "Fried, \"Polymer Science and Technology\" (3rd ed., 2014)",
          "Fink, \"High Performance Polymers\" (2nd ed., William Andrew, 2014)",
          "Ebewele, \"Polymer Science and Technology\" (2000)"
    ],
    chapters: [
      "高分子的性能阶梯（通用塑料→工程塑料→特种工程塑料）",
      "耐热高分子（PI 聚酰亚胺/PEEK/PPS 的结构-耐热关系）",
      "高强度纤维（芳纶 Kevlar 的液晶纺丝、UHMWPE 冻胶纺丝）",
      "碳纤维前驱体之外的路线（沥青基/粘胶基）",
      "含氟聚合物（PTFE/PVDF 的耐腐蚀与低摩擦）",
      "液晶高分子 LCP（5G 天线基材的介电优势）",
      "分离膜材料（反渗透/气体分离/燃料电池质子膜）",
      "导电与光电高分子（PEDOT/OLED 发光聚合物）",
      "生物基与可降解塑料（PLA/PHA 的性能现实）",
      "高分子加工（注塑/挤出/双向拉伸的结构控制）",
      "老化与寿命（热氧/光氧/水解老化、寿命预测）",
      "回收与循环（机械/化学回收的技术经济性）"
    ],
  },
  'engineering/materials-characterization': {
    title: "材料表征与分析（电镜/XRD/谱学）",
    books: [
          "Brandon, Kaplan, \"Microstructural Characterization of Materials\" (2nd ed., 2008)",
          "Egerton, \"Physical Principles of Electron Microscopy\" (2nd ed., 2016)",
          "Cullity, Stock, \"Elements of X-Ray Diffraction\" (3rd ed., 2001)"
    ],
    chapters: [
      "表征方法论（成分/结构/形貌/性能四维、尺度阶梯）",
      "光学显微与图像分析（金相制样、定量金相学）",
      "扫描电镜 SEM（二次电子/背散射、EDS 能谱）",
      "透射电镜 TEM（衍射/高分辨/STEM、球差校正）",
      "电子显微分析（EELS、原位电镜、冷冻电镜交叉）",
      "X 射线衍射（物相分析、织构、应力测量、Rietveld 精修）",
      "表面分析（XPS/AES/TOF-SIMS 的信息深度差异）",
      "谱学方法（拉曼/红外/紫外可见、核磁在材料中的应用）",
      "热分析（DSC/TG/DMA、相变温度测定）",
      "力学测试（拉伸/硬度/疲劳/断裂韧性的标准方法）",
      "三维表征（FIB 逐层、X 射线 CT、原子探针 APT）",
      "数据与溯源（测量不确定度、实验室间比对）"
    ],
  },
  'engineering/computational-materials': {
    title: "计算材料学与材料基因组（DFT/CALPHAD/ML）",
    books: [
          "Sholl, Steckel, \"Density Functional Theory: A Practical Introduction\" (2009)",
          "Lukas, Fries, Sundman, \"Computational Thermodynamics: The CALPHAD Method\" (2007)",
          "Materials Project / OQMD 开放数据库文档与材料机器学习综述（npj Comput. Mater.）"
    ],
    chapters: [
      "多尺度材料模拟全景（电子→原子→介观→宏观）",
      "密度泛函理论 DFT（交换关联泛函、赝势、能带与相稳定性计算）",
      "分子动力学（势函数、EAM/机器学习势 MLP）",
      "相图计算 CALPHAD（热力学数据库、相平衡预测）",
      "相场模拟（凝固/析出的微结构演化）",
      "晶体塑性有限元（织构与变形的耦合）",
      "材料信息学（描述符设计、性质预测模型）",
      "材料基因组方法（高通量计算+高通量实验的闭环）",
      "机器学习势函数（GAP/NEP、第一性原理精度的分子动力学）",
      "逆向设计（给定性能反推成分与工艺）",
      "开放数据库与工作流（Materials Project、AFLOW、OQMD）",
      "案例复盘（催化材料筛选、高熵合金设计、固态电解质发现）"
    ],
  },
  'engineering/pcb-design-fabrication': {
    title: "PCB 设计与制造（印制电路板）",
    books: [
          "Coombs, Holden (eds.), \"Printed Circuits Handbook\" (8th ed., 2023)",
          "Khandpur, \"Printed Circuit Boards: Design, Fabrication, Assembly and Testing\" (2005)",
          "Bogatin, \"Signal and Power Integrity—Simplified\" (3rd ed., 2018)"
    ],
    chapters: [
      "PCB 的层级世界（单/双/多层/HDI/刚挠结合的类型谱）",
      "基材与铜箔（FR-4 树脂体系、低介电高频板材、铜箔粗糙度）",
      "制造工艺（内层图形转移/压合/钻孔/电镀/外层/阻焊/表面处理）",
      "微孔与 HDI（激光钻孔、任意层互连、类载板 SLP）",
      "原理图与布局布线（约束驱动设计、叠层规划）",
      "信号完整性入门（特性阻抗、回流路径、串扰控制）",
      "电源完整性（去耦电容网络、平面谐振、PDN 阻抗目标）",
      "热设计与机械可靠性（导热路径、CTE 失配、挠曲控制）",
      "可制造性设计 DFM（线宽线距、拼板、工艺边规范）",
      "表面处理的抉择（喷锡/沉金/OSP/沉银的可靠性差异）",
      "检测与认证（飞针/ICT、AOI、阻抗测试、IPC 标准族）",
      "先进封装基板（ABF 载板、与《先进封装与 Chiplet》衔接）"
    ],
  },
  'engineering/smt-electronics-assembly': {
    title: "电子装联与整机制造（SMT/测试/可靠性）",
    books: [
          "Prasad, \"Surface Mount Technology: Principles and Practice\" (1997)",
          "Harper (ed.), \"Electronic Packaging and Interconnection Handbook\" (4th ed., 2004)",
          "IPC 标准族（IPC-A-610 可接受性、J-STD-001 焊接要求等公开标准体系）"
    ],
    chapters: [
      "电子装联全景（芯片封装→PCB 组装→整机的三级互连）",
      "SMT 工艺链（锡膏印刷→贴片→回流焊的温度曲线科学）",
      "锡膏与焊接冶金（SAC 无铅焊料、金属间化合物、空洞控制）",
      "贴装精度与视觉（0201/01005 微元件、BGA 对准）",
      "通孔与混合装联（波峰焊、选择性焊接、压接）",
      "清洗与三防（残留物可靠性、敷形涂覆）",
      "检测体系（SPI/AOI/X-Ray/ICT/FCT 的分工）",
      "返修工艺（BGA 返修台、底部填充 underfill）",
      "可靠性物理（热循环焊点疲劳、跌落、振动寿命模型）",
      "整机集成（结构件/散热/屏蔽/线缆的系统装配）",
      "静电防护与洁净（ESD 体系、MSD 湿敏元件管理）",
      "智能制造在电子厂的落地（MES、追溯、良率大数据）"
    ],
  },
  'engineering/display-technology': {
    title: "显示技术（LCD/OLED/MicroLED）",
    books: [
          "Chen, Cranton, Fihn (eds.), \"Handbook of Visual Display Technology\" (2nd ed., 2016)",
          "Tsujimura, \"OLED Display: Fundamentals and Applications\" (2nd ed., 2017)",
          "den Boer, \"Active Matrix Liquid Crystal Displays\" (2005)"
    ],
    chapters: [
      "显示的评价坐标（分辨率/亮度/色域/对比度/响应/功耗）",
      "液晶物理与显示模式（TN/IPS/VA 的取向控制）",
      "TFT 背板（非晶硅→LTPS→IGZO/氧化物、LTPO 的由来）",
      "LCD 光学系统（背光、导光板、增亮膜、量子点膜 QDEF）",
      "OLED 器件（有机发光堆栈、蒸镀工艺、像素电路补偿）",
      "OLED 蒸镀装备（FMM 精密掩膜、大尺寸化的工艺壁垒）",
      "柔性显示（CPI/UTG 盖板、铰链区的材料疲劳）",
      "MiniLED 背光与 MicroLED（巨量转移的技术路线竞争）",
      "触控集成（in-cell/on-cell、笔迹采样）",
      "投影与近眼显示（LCoS/DLP、光波导、Pancake 光学）",
      "显示驱动 IC 与接口（DDIC、eDP/MIPI 协议）",
      "制造良率经济学（世代线、切割效率、检测修复）"
    ],
  },
  'engineering/power-supply-technology': {
    title: "电源技术（开关电源/VRM/供电网络）",
    books: [
          "Pressman, Billings, Morey, \"Switching Power Supply Design\" (3rd ed., 2009)",
          "Erickson, Maksimović, \"Fundamentals of Power Electronics\" (3rd ed., 2020)",
          "Maniktala, \"Switching Power Supplies A to Z\" (2nd ed., 2012)"
    ],
    chapters: [
      "电源架构总览（AC-DC 整流→PFC→DC-DC 的能量链）",
      "开关变换拓扑（Buck/Boost/Buck-Boost 的工作模态）",
      "隔离拓扑（反激/正激/半桥全桥/LLC 谐振）",
      "磁性元件设计（变压器/电感、磁芯损耗与绕组损耗）",
      "控制环路（电压/电流模式、补偿网络设计）",
      "功率器件选型（MOSFET/GaN 在电源中的权衡，与功率半导体专题互链）",
      "同步整流与多相 VRM（CPU/GPU 供电的瞬态响应挑战）",
      "功率因数校正 PFC（升压 PFC、图腾柱无桥）",
      "EMI 与安规（传导/辐射抑制、绝缘耐压认证）",
      "热设计与效率优化（损耗分解、80 PLUS 体系）",
      "电池充电管理（CC/CV、快充协议、电量计）",
      "数字电源与智能供电（PMBus、服务器 48V 架构）"
    ],
  },
  'engineering/thermal-management-electronics': {
    title: "电子设备热管理（热管/均热板/液冷）",
    books: [
          "Shabany, \"Heat Transfer: Thermal Management of Electronics\" (CRC, 2010)",
          "Azar (ed.), \"Thermal Management of Microelectronic Equipment\" (ASME Press)",
          "JEDEC JESD51 系列热测试与建模标准（公开标准族）"
    ],
    chapters: [
      "热管理的目标函数（结温约束、热阻网络 θjc/θca）",
      "导热界面材料（硅脂/相变片/液态金属、接触热阻）",
      "散热器设计（翅片优化、自然/强迫对流）",
      "热管（毛细芯结构、工作流体、传热极限）",
      "均热板 VC（二维扩展、超薄化的工艺）",
      "风冷系统（风扇 P-Q 曲线、风道设计、噪声权衡）",
      "液冷（冷板/微通道、单相/两相、数据中心的直接液冷）",
      "浸没式冷却（单相/相变浸没、服务器案例）",
      "芯片级热问题（热点、热密度 100W/cm² 时代的应对）",
      "热电制冷与热敏元件（TEC、NTC 温控回路）",
      "热仿真与测量（CFD、红外热像、JEDEC 标准测试）",
      "系统级热设计案例（手机 SoC、GPU 显卡、AI 服务器整机柜）"
    ],
  },
  'engineering/high-speed-interconnect': {
    title: "高速互连与信号完整性（SerDes/PCIe/DDR）",
    books: [
          "Bogatin, \"Signal and Power Integrity—Simplified\" (3rd ed., 2018)",
          "Johnson, Graham, \"High-Speed Digital Design: A Handbook of Black Magic\" (1993)",
          "Hall, Heck, \"Advanced Signal Integrity for High-Speed Digital Designs\" (2009)"
    ],
    chapters: [
      "从并行到串行的历史转折（时钟偏移为何逼出 SerDes）",
      "传输线理论（特性阻抗、反射、端接策略）",
      "S 参数与信道表征（插损/回损/串扰、TDR 测量）",
      "编码与均衡（8b/10b→PAM4、CTLE/DFE/FFE 均衡链）",
      "时钟与抖动（PLL/CDR、抖动分解 RJ/DJ）",
      "PCIe 协议栈（物理层→数据链路→事务层、代际翻倍史）",
      "DDR 存储接口（拓扑/端接/读写训练、信号时序余量）",
      "封装与板级协同（Die-封装-PCB 三级互连的信号接力）",
      "光互连（AOC/光模块、共封装光学趋势）",
      "电源完整性（SSN 同步开关噪声、PDN 设计）",
      "仿真工作流（IBIS-AMI、信道仿真、眼图合规）",
      "标准生态（PCI-SIG/JEDEC/OIF 的规范工程）"
    ],
  },
  'cs/firmware-uefi-boot': {
    title: "固件与启动链（BIOS/UEFI/嵌入式引导）",
    books: [
          "UEFI Forum, \"UEFI Specification\" 与 \"Platform Initialization (PI) Specification\"（公开规范）",
          "Zimmer, Rothman, Marisetty, \"Embedded Firmware Solutions\" (Apress, 2015)",
          "Intel 开源固件文档（coreboot/EDK II 官方文档）"
    ],
    chapters: [
      "固件的位置（硬件与 OS 之间的隐形层、从上电复位开始）",
      "x86 启动链（RESET→SEC→PEI→DXE→BDS→OS 的接力）",
      "UEFI 体系（驱动模型、Protocol、UEFI Shell 与变量服务）",
      "传统 BIOS 与 legacy 兼容（CSM、实模式遗产）",
      "安全启动（Secure Boot 信任链、TPM 度量启动）",
      "内存初始化（MRC 内存参考代码、SPD 读取与训练）",
      "外设枚举（PCIe 枚举、ACPI 表的生成）",
      "嵌入式引导（ARM 的 BootROM→TF-A→U-Boot→内核链）",
      "开源固件（coreboot/LinuxBoot、固件供应链透明化）",
      "固件更新机制（ capsules 更新、防回滚、A/B 分区）",
      "固件安全（BIOS rootkit、Intel ME/PSP 的争议与边界）",
      "调试手段（串口日志、POST code、JTAG/SWD）"
    ],
  },
  'engineering/vacuum-cryogenic-engineering': {
    title: "真空与低温工程（半导体/超导支撑技术）",
    books: [
          "O'Hanlon, \"A User's Guide to Vacuum Technology\" (3rd ed., 2003)",
          "Jousten (ed.), \"Handbook of Vacuum Technology\" (2nd ed., 2016)",
          "Flynn, \"Cryogenic Engineering\" (2nd ed., 2005)"
    ],
    chapters: [
      "真空的分级（粗真空→高真空→超高真空的物理差异）",
      "抽气机组（机械泵/分子泵/离子泵/低温泵的组合逻辑）",
      "真空测量（皮拉尼/电离规、残余气体分析 RGA）",
      "密封与材料（金属密封、出气率、真空烘烤）",
      "超高真空系统（EUV 光刻/同步辐射的 UHV 工程）",
      "真空工艺应用（镀膜/刻蚀/注入的真空环境设计）",
      "低温温区与制冷循环（焦汤/斯特林/GM/稀释制冷机）",
      "液氮/液氦系统（杜瓦、传输线、零挥发磁体）",
      "超导磁体的低温集成（与《超导材料与应用》互链）",
      "量子计算的毫开尔文工程（稀释制冷机、布线热锚定）",
      "低温材料学（低温强度/韧性、绝热设计）",
      "空间低温与红外探测（空间制冷机、低温光学）"
    ],
  },
  'engineering/electromagnetic-aircraft-launch': {
    title: "电磁弹射与拦阻装置（EMALS/AAG/综合电力）",
    books: [
          "Boldea, \"Linear Electric Machines, Drives, and MAGLEVs Handbook\" (2013)",
          "Gieras, Piech & Tomczuk, \"Linear Synchronous Motors\" (2nd ed., 2012)",
          "Doyle et al., \"Electromagnetic Aircraft Launch System — EMALS\" (IEEE Trans. Magnetics, 1995)",
          "Patel, \"Shipboard Electrical Power Systems\" (2012)"
    ],
    chapters: [
      "航母舰载机起降的工程约束（起飞重量/甲板长度/出动架次率）",
      "蒸汽弹射的物理极限（效率、淡水消耗、末速度不可调）",
      "直线电机原理（直线感应/直线同步的推力与法向力）",
      "分段供电与位置反馈（长定子分段切换、无槽设计）",
      "飞轮储能与盘式发电机（动能缓冲、充电/放电循环）",
      "脉冲功率变换（IGCT/IGBT 变流器、四象限运行）",
      "中压直流综合电力系统（MVDC 电网、推进与弹射共用能量池）",
      "电磁兼容与甲板环境（盐雾/冲击/EMI 抑制）",
      "先进拦阻装置 AAG（水涡轮+感应电机、能量回收）",
      "福特级工程实践（EMALS/AAG 的研制教训与可靠性爬坡）",
      "电磁弹射的战术收益（无人机轻载弹射、能量精确匹配）",
      "电磁发射的延伸（电磁炮/轨道发射/航天电磁助推）"
    ],
  },
  'engineering/marine-nuclear-propulsion': {
    title: "舰船核动力装置（压水堆舰船化/A1B/潜艇堆）",
    books: [
          "LaMarsh & Baratta, \"Introduction to Nuclear Engineering\" (4th ed., 2017)",
          "Todreas & Kazimi, \"Nuclear Systems I: Thermal Hydraulic Fundamentals\" (2nd ed., 2012)",
          "Ragheb, \"Nuclear Naval Propulsion\" (InTechOpen, 2013)"
    ],
    chapters: [
      "舰船核动力的战术价值（无限续航/高功率/静音的取舍）",
      "压水堆舰船化改造（紧凑化、抗摇摆/抗冲击设计）",
      "一回路系统（反应堆冷却剂泵、稳压器、自然循环能力）",
      "蒸汽发生器与二回路（蒸汽动力循环、凝给水系统）",
      "全寿期堆芯（A4W/A1B 的 40-50 年不换料设计）",
      "潜艇反应堆（S6W/S9G 的自然循环静音运行）",
      "辐射屏蔽与舱室布置（一次/二次屏蔽、重量代价）",
      "反应堆安全（纵深防御、失水事故、弹棒事故的舰船场景）",
      "核动力与综合电力（堆—汽轮机—电网的能量链）",
      "换料与大修（ROH 换料大修、反应堆舱切割工艺）",
      "退役与处置（反应堆舱封存、放射性废物管理）",
      "民用核动力船舶（破冰船/浮动核电站的经验与教训）"
    ],
  },
  'engineering/ai-server-rack-engineering': {
    title: "AI 服务器整机柜工程（NVL72/NVLink/液冷/供电母排）",
    books: [
          "NVIDIA, \"GB200 NVL72 System Architecture\" 官方技术文档 (2024)",
          "Barroso, Hölzle & Ranganathan, \"The Datacenter as a Computer\" (3rd ed., 2018)",
          "ASHRAE, \"Liquid Cooling Guidelines for Datacom Equipment Centers\" (2nd ed., 2021)",
          "OCP, \"Open Rack V3 (ORV3) 供电与机柜规范\" 官方规范"
    ],
    chapters: [
      "从单机到整机柜（Scale-Up 与 Scale-Out 的架构分界）",
      "NVLink 域设计（72 GPU 全互连、NVLink Switch 托盘拓扑）",
      "铜互连背板（ACC/AEC 有源电缆、背板布线的信号完整性）",
      "计算托盘结构（Bianca 板：Grace CPU + Blackwell GPU 的 1U 形态）",
      "供电母排（Busbar 大电流传输、48V/±400V 高压直流演进）",
      "机柜级液冷（冷板/manifold 分液器/CDU 冷量分配单元）",
      "120kW+ 机柜的热设计（热密度、进出水温、漏液检测）",
      "机柜管理（RMC 机柜管理控制器、遥测与固件带外管理）",
      "可靠性工程（RAS 特性、故障域隔离、GPU 热插拔与降频降级）",
      "OCP 开放计算（ORV3/DC-MHS 规范、供应链开放生态）",
      "交付形态（L10→L11→L12 集成级别、数据中心部署约束）",
      "演进路线（GB200→GB300→Rubin Ultra、600kW 机柜与 Kyber 架构）"
    ],
  },
  'cs/post-quantum-cryptography': {
    title: "后量子密码（格密码/Kyber/Dilithium/PQC 迁移）",
    books: [
          "Bernstein, Buchmann & Dahmen (eds.), \"Post-Quantum Cryptography\" (2009)",
          "NIST, \"FIPS 203/204/205 — ML-KEM/ML-DSA/SLH-DSA 标准\" (2024)",
          "Hoffstein, Pipher & Silverman, \"An Introduction to Mathematical Cryptography\" (2nd ed., 2014)"
    ],
    chapters: [
      "量子威胁（Shor 算法对 RSA/ECC 的毁灭性打击、先存后解攻击）",
      "格密码基础（LWE/RLWE/Module-LWE 困难问题、格基约化）",
      "密钥封装 ML-KEM（CRYSTALS-Kyber 的构造与实现）",
      "数字签名 ML-DSA（CRYSTALS-Dilithium 的 Fiat-Shamir 变换）",
      "哈希签名 SLH-DSA（SPHINCS+ 的无状态设计）",
      "其他路线（编码密码 McEliece、多变量、同源密码 SIKE 的兴衰）",
      "侧信道与实现安全（NTT 实现的时序攻击、掩码防护）",
      "标准化进程（NIST PQC 竞赛五轮评审、CNSA 2.0 时间表）",
      "迁移工程（混合模式、证书双签、密码敏捷性 crypto-agility）",
      "性能与部署（嵌入式/TLS/固件签名中的 PQC 开销）",
      "量子密码的边界（QKD 与 PQC 的互补与争论）",
      "全同态加密的格基础（与《隐私计算》互链）"
    ],
  },
  'engineering/mobile-communication-5g-6g': {
    title: "移动通信与 5G/6G（NR 空口/Massive MIMO/网络切片）",
    books: [
          "Dahlman, Parkvall & Sköld, \"5G NR: The Next Generation Wireless Access Technology\" (2nd ed., 2020)",
          "Tse & Viswanath, \"Fundamentals of Wireless Communication\" (2005)",
          "Marzetta et al., \"Fundamentals of Massive MIMO\" (2016)"
    ],
    chapters: [
      "蜂窝演进史（1G→6G 的代际逻辑、ITU IMT-2030 愿景）",
      "5G NR 空口（OFDM 参数集、帧结构、参考信号）",
      "Massive MIMO（波束赋形、信道互易性、导频污染）",
      "毫米波与 Sub-6G（传播特性、波束管理、覆盖补偿）",
      "网络架构（SA/NSA、核心网 SBA 服务化、UPF 下沉）",
      "网络切片与边缘计算（eMBB/URLLC/mMTC 场景、MEC）",
      "RedCap 与物联网（轻量 5G、NB-IoT/LTE-M 演进）",
      "6G 候选技术（太赫兹通信、通感一体 ISAC、智能超表面 RIS）",
      "星地融合（NTN 非地面网络、手机直连卫星）",
      "AI 原生空口（信道估计/波束预测的神经网络化）",
      "安全与隐私（5G 鉴权 AKA、SUPI 加密、伪基站防护）",
      "产业与标准（3GPP Release 15-19、频谱政策、Open RAN）"
    ],
  },
  'engineering/quantum-communication-networking': {
    title: "量子通信与量子网络（QKD/量子中继/量子互联网）",
    books: [
          "Bouwmeester, Ekert & Zeilinger (eds.), \"The Physics of Quantum Information\" (2000)",
          "Nielsen & Chuang, \"Quantum Computation and Quantum Information\" (10th anniv. ed., 2010)",
          "Van Meter, \"Quantum Networking\" (2014)"
    ],
    chapters: [
      "量子通信的物理基础（叠加/纠缠/不可克隆定理）",
      "QKD 协议（BB84/E91/诱骗态、测量设备无关 MDI-QKD）",
      "成码率与距离极限（PLOB 界、信道损耗的指数诅咒）",
      "单光子源与探测（SPDC 纠缠源、SNSPD 超导探测器）",
      "量子中继（纠缠交换/纠缠纯化、量子存储器）",
      "卫星量子通信（墨子号工程、星地链路的对准与损耗）",
      "城域与骨干网（京沪干线、可信中继的工程妥协）",
      "量子互联网（端节点纠缠、量子协议栈的层次化设计）",
      "与 PQC 的关系（物理层安全 vs 数学层安全的争论与互补）",
      "量子隐形传态网络实验（城域纠缠分发的最新进展）",
      "工程化挑战（与光纤网络共纤传输、经典-量子串扰）",
      "标准化与产业（ETSI/ITU-T QKD 标准、商用 QKD 设备）"
    ],
  },
  'engineering/nuclear-power-plant': {
    title: "核电站工程（三代堆/华龙一号/常规岛/安全壳）",
    books: [
          "Todreas & Kazimi, \"Nuclear Systems II: Elements of Thermal Hydraulic Design\" (2001)",
          "IAEA, \"Design of the Reactor Coolant System and Associated Systems for Nuclear Power Plants\" (SSG-34, 2014)",
          "Cacuci (ed.), \"Handbook of Nuclear Engineering\" (2010)"
    ],
    chapters: [
      "核电站总貌（核岛/常规岛/BOP 辅助设施的划分）",
      "二代到三代堆型（AP1000/EPR/VVER 的非能动安全理念）",
      "华龙一号（自主三代：177 堆芯、双层安全壳、能动+非能动）",
      "四代堆候选（高温气冷堆/钠冷快堆/熔盐堆的技术路线）",
      "一回路与反应堆冷却剂系统（主泵/稳压器/蒸汽发生器）",
      "常规岛（饱和蒸汽汽轮机、汽水分离再热器、发电机）",
      "安全壳工程（预应力混凝土、钢衬里、氢复合器）",
      "仪控与保护系统（反应堆保护系统 RPS、数字化 DCS）",
      "事故序列与应对（失水 LOCA、全厂断电 SBO、严重事故管理）",
      "乏燃料与放射性废物（水池贮存、干式贮存、后处理）",
      "建造与运维（模块化建造、在役检查、换料大修）",
      "核能经济性与政策（造价/电价、监管体系、小型模块化堆 SMR）"
    ],
  },
  'engineering/datacenter-cluster-engineering': {
    title: "算力集群与数据中心工程（SuperPOD/InfiniBand/万卡组网）",
    books: [
          "Barroso, Hölzle & Ranganathan, \"The Datacenter as a Computer\" (3rd ed., 2018)",
          "NVIDIA, \"DGX SuperPOD Reference Architecture\" 官方文档 (2024)",
          "Faisal et al., \"The Datacenter as a Networked Computer: RDMA 与 RoCE 实践\" (IEEE HotI, 2015)"
    ],
    chapters: [
      "集群架构层次（整机柜→SuperPOD→多园区算力中心）",
      "GPU 互联网络（NVLink 域内 + InfiniBand/RoCE 域外的两层设计）",
      "无损以太网（RoCEv2、PFC/ECN 拥塞控制、DCQCN）",
      "网络拓扑（Fat-Tree/Dragonfly+/轨式优化 Rail-Optimized）",
      "万卡集群的集合通信（NCCL/拓扑感知 AllReduce、网络拥塞实测）",
      "作业调度（Slurm/Kubernetes 拓扑感知调度、Gang Scheduling）",
      "训练容错（Checkpoint 策略、故障预测、弹性训练）",
      "存储系统（并行文件系统 Lustre/GPFS、检查点带宽墙）",
      "供电基础设施（市电→UPS→母线→机柜的配电链、柴发与储能）",
      "制冷基础设施（风冷/液冷混合、冷却塔、PUE/WUE 指标）",
      "数据中心等级与可靠性（Tier I-IV、2N 冗余、可用性数学）",
      "绿色算力（余热回收、碳足迹、东数西算与算力网络）"
    ],
  },
}
