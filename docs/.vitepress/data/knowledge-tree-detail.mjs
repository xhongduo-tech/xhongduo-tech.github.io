// 待建专题详细主题 · 依据权威书籍章节
// 共 329 个待建专题，2397 个章节级子主题
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
      "同余、剩余类与中国剩余定理 (Niven-Zuckerman §2.3-2.6)",
      "费马小定理、欧拉定理与 Wilson 定理 (Hardy & Wright §6.1-6.5)",
      "二次剩余与 Gauss 互反律 (Hardy & Wright §6.7-6.13)",
      "原根与指标 (Niven-Zuckerman §4.2-4.4)",
      "连分数与最佳逼近 (Hardy & Wright §10.1-10.9)",
      "数论函数与 Möbius 反演 (Hardy & Wright §16-18)",
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
      "正交曲线坐标与度量系数 (Arfken §2.1-2.5)",
      "张量定义与指标记号 (Arfken §2.6-2.7)",
      "协变、逆变张量与商律 (Arfken §2.6-2.8)",
      "二阶张量的本征值与本征轴 (Arfken §2.9)",
      "张量在连续介质力学中的应用 (Boas §10.4)",
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
      "矩阵范数与扰动理论 (Horn & Johnson Ch. 5)",
      "正定矩阵与平方根 (Horn & Johnson Ch. 7)",
      "非负矩阵与 Perron-Frobenius 理论 (Horn & Johnson Ch. 8)",
      "Jordan 标准型与广义特征向量 (Horn §3.1-3.2)",
      "Cayley-Hamilton 定理与最小多项式 (Horn §3.3)",
      "矩阵函数（矩阵指数/对数） (Horn §6.1)",
      "Kronecker 积与张量积 (Horn §4.3)",
      "线性矩阵方程（Sylvester/Lyapunov） (Horn §4.4)",
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
      "态射与有理映射 (Hartshorne Ch. I §4-5)",
      "层与概形 (Hartshorne Ch. II §1-2)",
      "概形的态射与纤维积 (Hartshorne Ch. II §3)",
      "凝聚层与蛇引理 (Hartshorne Ch. II §5)",
      "除子、线性系与微分形式 (Hartshorne Ch. II §6-8)",
      "Riemann-Roch 定理 (Hartshorne Ch. IV §1)",
      "层上同调与 Čech 上同调 (Hartshorne §III.2-III.4)",
      "Serre 对偶定理 (Hartshorne §III.7)",
      "曲线的 Riemann-Roch 定理 (Hartshorne §IV.1)",
      "双有理几何（blow-up/有理映射） (Hartshorne §V)",
      "GAGA 原理 (Hartshorne §III.5)",
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
      "半单李代数与 Killing 型 (Humphreys §5)",
      "根系与 Weyl 群 (Humphreys §9-10)",
      "Cartan 子代数与分类定理 (Humphreys §11)",
      "最高权表示与 Verma 模 (Humphreys §20-21)",
      "SU(2) 与 SU(3) 的表示 (Fulton & Harris Part I-II)",
      "李群-李代数对应与指数映射 (Hall Ch. 5-9)",
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
      "Fourier 级数与收敛性 (Stein & Shakarchi Ch. 1-2)",
      "极大函数与 Hardy-Littlewood 定理 (Stein & Shakarchi Ch. 4)",
      "Calderón-Zygmund 分解 (Stein & Shakarchi Ch. 4)",
      "Marcinkiewicz 插值定理 (Stein & Shakarchi Ch. 4)",
      "Fourier 变换与反演公式 (Stein & Shakarchi Ch. 5-6)",
      "Poisson 求和公式与采样定理 (Stein & Shakarchi Ch. 5)",
      "Paley-Wiener 定理 (Stein & Shakarchi Ch. 6)",
      "局部紧群上的 Haar 测量与调和分析 (Rudin Ch. 1)",
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
      "谱序列 (Weibel Ch. 5)",
      "双复形与 Künneth 公式 (Weibel Ch. 2.7, 5.6)",
      "群同调与 Lie 代数同调 (Weibel Ch. 6-7)",
      "导出范畴简介 (Weibel Ch. 10)",
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
      "局部化与分数理想 (Atiyah-Macdonald Ch. 3-4)",
      "准素分解 (Atiyah-Macdonald Ch. 4)",
      "链条件与 Noether 环、Artin 环 (Atiyah-Macdonald Ch. 6-8)",
      "Hilbert 零点定理与 Zariski 拓扑 (Atiyah-Macdonald Ch. 7)",
      "离散赋值环与 Dedekind 整环 (Atiyah-Macdonald Ch. 9)",
      "完备化与 Hensel 引理 (Atiyah-Macdonald Ch. 10)",
      "维数理论与正则环 (Matsumura Ch. 5-6)",
      "张量积与平坦模 (Atiyah §2-3)",
      "整扩张（Going-up/Going-down） (Atiyah §5)",
      "深度与正则序列 (Atiyah §6)",
      "Cohen-Macaulay 模与 Gorenstein 环 (Atiyah §7)",
      "相伴素与支集 (Atiyah §4)",
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
      "完备性与 Hopf-Rinow 定理 (do Carmo Ch. 7)",
      "Cartan-Hadamard 定理与共形几何 (do Carmo Ch. 7-8)",
      "变分公式与 Bonnet-Myers 定理 (do Carmo Ch. 9-10)",
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
      "直接方法与下半连续性 (Evans Ch. 8.2)",
      "Hamilton-Jacobi 方程 (Evans Ch. 10)",
      "极小曲面问题与平均曲率 (Evans Ch. 8.5)",
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
      "Weierstrass 方程与椭圆曲线群律 (Silverman Ch. III)",
      "椭圆曲线上的几何加法 (Silverman Ch. III §3)",
      "有限域上的椭圆曲线 (Silverman Ch. V)",
      "挠点与 Mordell-Weil 定理 (Silverman Ch. VIII)",
      "复乘法与模函数 (Silverman Ch. VI)",
      "模形式与 Eisenstein 级数 (Serre Ch. VII)",
      "j-不变量与模曲线 (Serre Ch. VII-VIII)",
      "Hecke 算子与 L-函数 (Koblitz Ch. III)",
      "同源与对偶同源 (Silverman §III.4)",
      "椭圆曲线上的复结构（格与一致化） (Silverman §VI)",
      "BSD 猜想 (Silverman §X.6)",
      "模性定理（Taniyama-Shimura-Weil） (Silverman §C.16)",
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
      "Riemann 假设与零点分布 (Montgomery & Vaughan Ch. 13-14)",
      "特征函数与均值估计 (Tenenbaum Ch. II)",
      "大筛法与 Bombieri-Vinogradov 定理 (Montgomery & Vaughan Ch. 27)",
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
      "Dedekind 整环与理想唯一分解 (Neukirch Ch. I §6-8)",
      "类数与理想类群 (Neukirch Ch. I §6)",
      "赋值、完备化与 p-adic 数 (Neukirch Ch. II §1-5)",
      "Minkowski 几何与有限性定理 (Neukirch Ch. III §1-2)",
      "差积与判别式 (Neukirch Ch. III §3-4)",
      "Artin 互反律 (Neukirch Ch. VI)",
      "Dedekind Zeta 函数与类数公式 (Neukirch Ch. VII)",
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
      "整数规划与分支定界 (Hillier & Lieberman Ch. 11-12)",
      "网络优化与最短路、最大流 (Hillier & Lieberman Ch. 9)",
      "动态规划 (Hillier & Lieberman Ch. 10)",
      "非线性规划与 KKT 条件 (Hillier & Lieberman Ch. 13)",
      "排队论 (Hillier & Lieberman Ch. 17)",
      "决策分析与博弈论 (Hillier & Lieberman Ch. 16)",
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
      "Lax-Richtmyer 等价定理 (Strikwerda Ch. 9)",
      "有限元变分形式 (C. Johnson Ch. 1-2)",
      "Sobolev 空间与误差估计 (C. Johnson Ch. 4-5)",
      "多维问题与边界条件处理 (LeVeque Ch. 13)",
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
      "19 世纪抽象代数与分析严密化 (Katz Ch. 21)",
      "20 世纪数学与公理化方法 (Katz Ch. 25)",
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
      "单子与代数 (Mac Lane Ch. VI)",
      "加性范畴与 Abel 范畴 (Mac Lane Ch. VIII)",
      "Yoneda 引理与可表函子 (Riehl Ch. 2-3)",
      "Kan 扩张 (Mac Lane Ch. X)",
      "拓扑斯与层简介 (Mac Lane Ch. I, Riehl Ch. 3)",
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
      "贝叶斯计算与 Stan (Gelman Ch. 14)",
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
      "光电效应与波粒二象性 (Tipler Ch. 3-4)",
      "氢原子玻尔模型 (Tipler Ch. 4)",
      "薛定谔方程与一维势阱 (Tipler Ch. 6)",
      "角动量与自旋 (Tipler Ch. 7)",
      "多电子原子与元素周期表 (Tipler Ch. 8)",
      "核物理基础 (Tipler Ch. 11)",
      "粒子物理标准模型引论 (Tipler Ch. 13)",
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
      "粘性流动与边界层 (Batchelor Ch. 5)",
      "不可压缩无粘流动 (Batchelor Ch. 6)",
      "涡量、环量与 Kelvin 定理 (Batchelor Ch. 7)",
      "湍流简介与 Reynolds 应力 (Landau & Lifshitz §31-34)",
      "表面波、重力波与声波 (Landau & Lifshitz §9-12)",
      "可压缩流动与气体动力学（激波/Mach 数） (Landau §VI)",
      "势流理论与复势 (Landau §III)",
      "流体稳定性（Rayleigh-Taylor/Kelvin-Helmholtz） (Landau §VIII)",
      "低雷诺数流动（Stokes 流） (Landau §IV)",
      "计算流体力学（CFD）方法 (Anderson §8)",
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
      "Hamilton 系统与 KAM 定理 (Guckenheimer & Holmes Ch. 7)",
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
      "塞曼效应与斯塔克效应 (Foot Ch. 5-6)",
      "跃迁概率与选择定则 (Foot Ch. 6)",
      "分子轨道与振动-转动光谱 (Bransden & Joachain Ch. 10)",
      "激光原理与光泵浦 (Foot Ch. 7-8)",
      "激光冷却与磁光阱 (Foot Ch. 10)",
      "精密光谱与原子钟 (Foot Ch. 9)",
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
      "核子性质与核力 (Krane Ch. 3-4)",
      "壳模型与集体模型 (Krane Ch. 5)",
      "放射性衰变 (Krane Ch. 6)",
      "α、β、γ 衰变动力学 (Krane Ch. 7)",
      "核反应与裂变 (Krane Ch. 11-13)",
      "核聚变与恒星核合成 (Krane Ch. 14)",
      "核谱学与激发态 (Krane Ch. 10)",
      "中子物理与散射 (Krane Ch. 4)",
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
      "等离子体振荡与色散关系 (Chen Ch. 4)",
      "磁流体力学方程 (Chen Ch. 5)",
      "磁约束与托卡马克 (Chen Ch. 5, 9)",
      "输运过程与碰撞 (Chen Ch. 5)",
      "等离子体不稳定性 (Chen Ch. 6)",
      "惯性约束聚变 (Chen Ch. 9)",
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
      "常微分方程数值解 (Newman Ch. 8)",
      "偏微分方程数值解 (Newman Ch. 9)",
      "快速傅里叶变换 (Newman Ch. 7)",
      "蒙特卡洛方法与随机数 (Newman Ch. 10)",
      "分子动力学模拟 (Giordano & Nakanishi Ch. 8-9)",
      "有限元方法 (Giordano & Nakanishi Ch. 10)",
      "随机游走与马尔可夫链蒙特卡洛 (Newman Ch. 10)",
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
      "载流子统计与 Fermi-Dirac 分布 (Sze Ch. 2)",
      "漂移、扩散与复合 (Sze Ch. 2-3)",
      "p-n 结二极管 (Sze Ch. 4)",
      "双极型晶体管 (Sze Ch. 5)",
      "MOS 场效应晶体管 (Sze Ch. 6)",
      "光电器件：LED 与太阳能电池 (Sze Ch. 12-13)",
      "异质结与量子阱结构 (Grundmann Ch. 12)",
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
      "Landauer-Büttiker 公式 (Nazarov & Blanter Ch. 1)",
      "量子霍尔效应 (Nazarov & Blanter Ch. 1)",
      "Aharonov-Bohm 效应与 Berry 相位 (Nazarov & Blanter Ch. 1)",
      "散射矩阵与电导量子化 (Nazarov & Blanter Ch. 2)",
      "库仑阻塞与单电子隧穿 (Nazarov & Blanter Ch. 3)",
      "量子点能级与输运谱 (Nazarov & Blanter Ch. 3)",
      "退相干与量子噪声 (Nazarov & Blanter Ch. 6)",
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
      "量子信息熵与 Schumacher 压缩 (Nielsen & Chuang Ch. 11)",
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
      "气体探测器：电离室与正比计数管 (Knoll Ch. 4-5)",
      "半导体探测器 (Knoll Ch. 11-12)",
      "闪烁探测器 (Knoll Ch. 8-9)",
      "光电倍增管与读出电子学 (Knoll Ch. 9, Grupen Ch. 16)",
      "粒子鉴别方法 (Grupen & Shwartz Ch. 14)",
      "触发与数据获取系统 (Grupen & Shwartz Ch. 17)",
      "加速器原理与对撞机 (Leo Ch. 3)",
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
      "弦量子化与谱 (Zwiebach Ch. 12-13)",
      "弦的约束与临界维数 (Polchinski Vol. 1 Ch. 1-2)",
      "开弦与 D-膜 (Polchinski Vol. 1 Ch. 8)",
      "T-对偶性 (Polchinski Vol. 1 Ch. 8)",
      "弦场论与规范对称性 (Polchinski Vol. 1 Ch. 6)",
      "量子引力与黑洞熵 (Becker-Becker-Schwarz Ch. 14)",
      "超弦与 GSO 投影 (Polchinski §10-12)",
      "Calabi-Yau 紧致化 (Polchinski §15)",
      "M 理论与 S 对偶 (Polchinski §14)",
      "AdS/CFT 对应 (Polchinski §23)",
      "D 膜世界体积规范理论 (Polchinski §13)",
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
      "热学与统计力学的发展 (Segrè Part II)",
      "电磁学的建立：从奥斯特到麦克斯韦 (Segrè Part III)",
      "相对论的诞生 (Pais Part II)",
      "量子力学的建立：普朗克到海森堡 (Jammer Ch. 3-5)",
      "原子核物理与粒子物理的兴起 (Jammer Ch. 8)",
      "凝聚态物理的兴起 (Jammer Ch. 9)",
      "20 世纪物理学的统一 (Pais Part VII)",
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
      "Doppler 与 Sisyphus 激光冷却 (Metcalf & van der Straten Ch. 3-6)",
      "磁光阱与磁阱 (Metcalf & van der Straten Ch. 7-11)",
      "Feshbach 共振 (Pethick & Smith Ch. 5)",
      "光晶格与 Bose-Hubbard 模型 (Bloch-Dalibard-Zwerger §III)",
      "旋量凝聚体 (Pethick & Smith Ch. 12)",
      "超冷费米气体与 BCS-BEC 穿越 (Pethick & Smith Ch. 16)",
      "量子模拟：磁性相变与 Hubbard 模型 (Bloch-Dalibard-Zwerger §V)",
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
      "致密天体作为引力波源 (Maggiore Ch. 4)",
      "双星并合波形建模 (Creighton & Anderson Ch. 5)",
      "LIGO 与干涉仪原理 (Maggiore Ch. 7)",
      "噪声分析与灵敏度曲线 (Creighton & Anderson Ch. 6)",
      "匹配滤波与信号检测 (Creighton & Anderson Ch. 7)",
      "多信使天文学与并合事件 (Creighton & Anderson Ch. 9)",
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
      "p 区元素 (Housecroft Ch. 13-15)",
      "d 区元素与配位化学 (Housecroft Ch. 7-8)",
      "过渡金属的氧化态与颜色 (Cotton Ch. 17-18)",
      "f 区元素：镧系与锕系 (Housecroft Ch. 25)",
      "主族氢化物与卤化物 (Cotton Ch. 5-7)",
      "金属簇化合物 (Cotton Ch. 18)",
      "固态结构与离子固体 (Housecroft Ch. 6)",
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
      "热力学第一定律与焓 (Atkins Ch. 2)",
      "热力学第二定律与熵 (Atkins Ch. 3)",
      "热力学第三定律与统计熵 (Atkins Ch. 3)",
      "化学势与相平衡 (Atkins Ch. 5)",
      "理想与实际溶液：Raoult 与 Henry 定律 (Atkins Ch. 5)",
      "化学平衡与平衡常数 (Atkins Ch. 6)",
      "电化学与 Nernst 方程 (Atkins Ch. 6)",
      "表面热力学与 Kelvin 方程 (Atkins Ch. 5)",
      "统计热力学（配分函数/Boltzmann 分布） (Atkins §22-23)",
      "非平衡热力学（Onsager 倒易关系） (Atkins §24)",
      "相图（二元/三元体系） (Atkins §20)",
      "热化学（Hess 定律/键焓） (Atkins §2)",
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
      "反应速率与速率方程 (Atkins Ch. 7)",
      "一级、二级反应动力学 (Atkins Ch. 7)",
      "Arrhenius 方程与活化能 (Atkins Ch. 7)",
      "复合反应与稳态近似 (Atkins Ch. 7)",
      "反应机理与中间体 (Espenson Ch. 4)",
      "催化：均相与多相 (Atkins Ch. 7)",
      "光化学反应动力学 (Laidler Ch. 11)",
      "链反应与爆炸机理 (Laidler Ch. 5)",
      "过渡态理论（Eyring 方程） (Atkins §27.6)",
      "单分子反应理论（Lindemann/RRKM） (Atkins §27.7)",
      "酶动力学（Michaelis-Menten） (Atkins §27.8)",
      "振荡反应（Belousov-Zhabotinsky） (Atkins §27.9)",
      "飞秒化学与实时观测 (Atkins §27.10)",
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
      "表面张力与毛细现象 (Butt-Graf-Kappl Ch. 3)",
      "表面能与润湿性 (Butt-Graf-Kappl Ch. 4)",
      "双电层与 Zeta 电势 (Butt-Graf-Kappl Ch. 5)",
      "表面光谱技术 (Somorjai Ch. 6)",
      "多相催化原理 (Somorjai Ch. 8-9)",
      "自组装单层 SAM (Butt-Graf-Kappl Ch. 10)",
      "Langmuir 与 BET 吸附等温线 (Atkins §25.4-25.5)",
      "表面活性剂与胶束 (Atkins §25.7)",
      "纳米颗粒表面化学 (Atkins §25.8)",
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
      "烯醇化学与 aldol 反应 (Clayden Ch. 21-27)",
      "立体选择性反应 (Clayden Ch. 16-17)",
      "周环反应：Diels-Alder 与电环化 (Clayden Ch. 35)",
      "逆合成分析与保护基 (Clayden Ch. 28-29)",
      "芳香化学与亲电取代 (Clayden Ch. 22-24)",
      "杂环化学 (Clayden Ch. 43)",
      "全合成实例：天然产物 (Nicolaou & Sorensen Ch. 1-12)",
      "氧化与还原反应（Swern/PCC/NaBH4/LiAlH4） (Clayden §34-35)",
      "过渡金属催化 C-C 偶联（Suzuki/Heck/Sonogashira） (Clayden §47-48)",
      "不对称合成（手性辅助/Sharpless 环氧化） (Clayden §41)",
      "保护基策略与正交性 (Clayden §33)",
      "有机催化（Organocatalysis） (Clayden §42)",
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
      "对映体与非对映体 (Eliel Ch. 4-5)",
      "构型命名与 CIP 规则 (Eliel Ch. 5)",
      "立体选择性与立体专一性 (Eliel Ch. 6-7)",
      "前手性与立体异构化 (Eliel Ch. 9)",
      "不对称合成方法 (Eliel Ch. 14)",
      "立体异构体的分离 (Eliel Ch. 7)",
      "手性的光谱与 X 射线测定 (Eliel Ch. 11)",
      "手性识别与对映体过量测定 (Eliel §6)",
      "不对称诱导与立体选择性控制 (Eliel §7)",
      "动态立体化学 (Eliel §8)",
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
      "红外光谱与官能团鉴定 (Silverstein Ch. 1-2)",
      "质谱与分子量测定 (Silverstein Ch. 1)",
      "核磁共振：1H 与 13C NMR (Silverstein Ch. 3-5)",
      "自旋-自旋耦合与化学位移 (Williams & Fleming Ch. 3)",
      "二维 NMR：COSY、HSQC、NOESY (Williams & Fleming Ch. 4)",
      "电子顺磁共振 EPR (Atkins Ch. 12)",
      "结构鉴定综合应用 (Silverstein Ch. 7)",
      "X 射线衍射与晶体结构 (Silverstein §7)",
      "拉曼光谱与 SERS (Silverstein §8)",
      "荧光光谱 (Silverstein §9)",
      "圆二色光谱（CD） (Silverstein §10)",
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
      "单重态与三重态动力学 (Turro Ch. 7)",
      "光致电子转移与 Marcus 理论 (Turro Ch. 9)",
      "光化学反应：周环与异构化 (Turro Ch. 12)",
      "光敏化与猝灭 (Turro Ch. 8)",
      "激光化学与光催化 (Turro Ch. 13)",
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
      "胶体分类与稳定性 (Hiemenz Ch. 1-4)",
      "DLVO 理论 (Hiemenz Ch. 13)",
      "双电层与 Zeta 电势 (Hiemenz Ch. 12)",
      "表面活性剂与胶束 (Hiemenz Ch. 18)",
      "乳液与微乳液 (Hunter Ch. 5)",
      "高分子溶液 (Hunter Ch. 8)",
      "流变学性质 (Hunter Ch. 14)",
      "凝胶与网络结构 (Hiemenz Ch. 14)",
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
      "分子描述符与拓扑指标 (Gasteiger & Engel Ch. 4)",
      "化学数据库与信息检索 (Leach & Gillet Ch. 3)",
      "构象生成与 3D 结构预测 (Gasteiger & Engel Ch. 5)",
      "定量构效关系 QSAR (Leach & Gillet Ch. 8)",
      "药效团识别 (Leach & Gillet Ch. 9)",
      "分子对接与虚拟筛选 (Leach & Gillet Ch. 10)",
      "机器学习与化学数据挖掘 (Bajorath Ch. 4)",
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
      "核稳定性与放射性衰变 (Choppin Ch. 3-4)",
      "α、β、γ 衰变模式 (Choppin Ch. 4)",
      "放射性活度与半衰期 (Choppin Ch. 4)",
      "核反应与中子活化 (Choppin Ch. 10)",
      "放射性核素的化学分离 (Choppin Ch. 7)",
      "放射性测量技术 (Choppin Ch. 8)",
      "同位素示踪剂应用 (Loveland Ch. 17)",
      "核废料管理与环境影响 (Choppin Ch. 21)",
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
      "离子交换色谱 (Poole Ch. 8)",
      "尺寸排阻色谱 (Poole Ch. 9)",
      "薄层色谱 TLC (Poole Ch. 5)",
      "毛细管电泳 (Skoog Ch. 30)",
      "超临界流体色谱 (Skoog Ch. 29)",
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
      "绿色能源与工业应用 (Lancaster Ch. 9)",
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
      "环糊精与包合物 (Steed & Atwood Ch. 5)",
      "分子自组装与自组织 (Steed & Atwood Ch. 10)",
      "超分子聚合物 (Steed & Atwood Ch. 10)",
      "机械互锁分子：索烃与轮烷 (Steed & Atwood Ch. 10)",
      "超分子催化与酶模型 (Lehn Ch. 9)",
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
      "燃素说与拉瓦锡革命 (Brock Ch. 7-8)",
      "原子论与分子学说 (Brock Ch. 10)",
      "元素周期表的发现 (Brock Ch. 12)",
      "有机化学的兴起 (Brock Ch. 13-14)",
      "物理化学的建立 (Brock Ch. 15)",
      "20 世纪化学键理论 (Brock Ch. 16)",
      "现代仪器分析与化学工业 (Brock Ch. 18)",
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
      "单分子化学动力学 (Dekker Ch. 8)",
      "单分子生物大分子与折叠 (Bustamante surveys)",
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
      "无溶剂有机合成 (Modern Mechanochemistry Ch. 6)",
      "机械合金化 (Michalchuk Ch. 6)",
      "机械化学合成 MOF 与共价有机框架 (Modern Mechanochemistry Ch. 9)",
      "工业应用与放大 (Michalchuk Ch. 8)",
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
      "控制流与跳转 (CS:APP §3.6)",
      "x86-64 指令编码 (Hyde Ch.4)",
      "汇编与 C 互操作 (CS:APP §3.11)",
      "中断与系统调用机制 (Patterson §5.9)",
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
      "锁的实现与 CAS (Herlihy §7)",
      "条件变量与读者-写者 (Herlihy §8)",
      "死锁检测与预防 (CS:APP §12.7)",
      "并行算法与工作窃取 (Herlihy §15)",
      "内存一致性与顺序 (Herlihy §3)",
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
      "反汇编与控制流恢复 (Dang Ch.2)",
      "ELF/PE 文件格式解析 (Dang Ch.1)",
      "反调试与反虚拟机技术 (Sikorski Ch.16)",
      "漏洞挖掘与模糊测试 (Dang Ch.6)",
      "恶意代码分析实战 (Sikorski Ch.12)",
      "二进制插桩与符号执行 (Dang Ch.7)",
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
      "函数式语言范式 (Sebesta Ch.15)",
      "并发语言构造 (Scott §12.4)",
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
      "符号模型检测与 BDD (Clarke §4)",
      "定理证明与 Coq/Isabelle (Nipkow Ch.5)",
      "程序验证与霍尔逻辑 (Nipkow Ch.7)",
      "抽象解释与不变量 (Huth §3.6)",
      "实时与混成系统验证 (Clarke §9)",
      "SAT/SMT 求解（CDCL/DPLL(T)） (Huth §6)",
      "精化演算与程序推导 (Huth §8)",
      "进程代数（CSP/CCS/π-演算） (Hoare Ch.4)",
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
      "磁盘调度与 RAID (Silberschatz §12)",
      "虚拟文件系统 VFS (Tanenbaum §4.5)",
      "分布式文件系统 NFS (Tanenbaum §10.4)",
      "块存储与对象存储 (OSTEP §44)",
      "闪存与 FTL (Tanenbaum §4.9)",
      "分布式文件系统（GFS/HDFS/Ceph） (Tanenbaum §11)",
      "一致性模型（CAP/线性一致性） (Tanenbaum §7)",
      "对象存储（S3/Ceph RGW） (Tanenbaum §12)",
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
      "Hypervisor 类型与架构 (Tanenbaum §7.7)",
      "CPU 虚拟化与陷入模拟 (Smith §3.3)",
      "内存虚拟化与影子页表 (Smith §3.4)",
      "I/O 虚拟化与设备模拟 (Smith §3.5)",
      "硬件辅助虚拟化 Intel VT-x (Tanenbaum §7.7.2)",
      "容器与操作系统级虚拟化 (Tanenbaum §7.8)",
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
      "网络编程深入 (CS:APP §11)",
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
      "缓冲区溢出与栈破坏 (CS:APP §3.12)",
      "恶意代码与 Rootkit (Stallings §6)",
      "漏洞利用与缓解技术 (Du §11)",
      "操作系统安全加固 (Gollmann §4)",
      "网络安全与防火墙 (Stallings §9)",
      "密码学基础与 PKI (Stallings §2)",
      "可信计算与硬件安全 (Gollmann §12)",
      "Web 安全（SQLi/XSS/CSRF） (OWASP §Top10)",
      "认证与授权（Kerberos/OAuth/SAML） (Anderson §5)",
      "内存保护机制（ASLR/DEP/CFI） (Anderson §6)",
      "模糊测试（Fuzzing） (Anderson §10)",
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
      "嵌入式存储与外设 (Wolf §3)",
      "实时操作系统与调度 (Wolf §6)",
      "嵌入式 Linux 与驱动 (Heath Ch.7)",
      "功耗管理与低功耗设计 (Wolf §5)",
      "嵌入式通信总线 (Wolf §4)",
      "硬件/软件协同设计 (Wolf §7)",
      "嵌入式系统验证 (Ganssle Ch.10)",
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
      "边缘智能与雾计算 (Hassanein §6)",
      "IoT 数据管理与云平台 (Zuhair Ch.5)",
      "物联网安全与隐私 (Vermesan Ch.6)",
      "智能城市与工业 IoT 应用 (Hassanein §8)",
      "IoT 互操作性与标准化 (Zuhair Ch.7)",
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
      "Docker 镜像与运行时 (Burns Ch.3)",
      "Kubernetes 对象模型 (Burns Ch.6)",
      "Pod 与服务发现 (Burns Ch.7)",
      "控制器与声明式 API (Burns Ch.9)",
      "服务网格 Istio (Atchison §8)",
      "云原生 12 因素应用 (Atchison §3)",
      "分布式系统设计模式 (Burns \"Designing\" Ch.2)",
      "持久化存储与 CSI (Burns §7)",
      "网络模型（CNI/Service Mesh） (Burns §8)",
      "安全（RBAC/NetworkPolicy/Secrets） (Burns §9)",
      "可观测性（Prometheus/OpenTelemetry） (Burns §10)",
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
      "API 网关模式 (Richardson Ch.3)",
      "服务发现与注册 (Richardson Ch.4)",
      "Saga 分布式事务模式 (Richardson Ch.4)",
      "事件驱动架构与 CQRS (Richardson Ch.8)",
      "CAP 与最终一致性 (Newman Ch.11)",
      "微服务部署与监控 (Newman Ch.12)",
      "弹性模式（熔断/舱壁/重试） (Newman §11)",
      "每服务数据库与数据一致性 (Newman §4)",
      "契约测试与 API 版本管理 (Newman §8)",
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
      "Spark RDD 与 DAG (Zaharia Ch.2)",
      "Spark SQL 与 DataFrame (Zaharia Ch.9)",
      "流处理与 Structured Streaming (Zaharia Ch.20)",
      "Lambda 架构与批流一体 (Marz Ch.1)",
      "数据湖与湖仓一体 (White Ch.17)",
      "消息队列（Kafka） (White §6)",
      "流处理（Flink） (White §19)",
      "NoSQL（Cassandra/MongoDB/HBase） (White §14)",
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
      "消除工作量与 Toil (SRE Book Ch.5)",
      "事件管理与事后复盘 (SRE Book Ch.12)",
      "容量规划与负载测试 (SRE Book Ch.18)",
      "变更管理与发布工程 (SRE Book Ch.8)",
      "DevOps 三步工作法 (DevOps Handbook §1)",
      "混沌工程与弹性测试 (SRE Workbook Ch.9)",
      "CI/CD 流水线（Jenkins/GitLab/ArgoCD） (Beyer §16)",
      "基础设施即代码（Terraform/Ansible） (Beyer §14)",
      "监控与可观测性（Prometheus/Grafana/OpenTelemetry） (Beyer §6)",
      "SLO/SLI/SLO 错误预算 (Beyer §5)",
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
      "边缘 AI 与模型部署 (Shi Ch.7)",
      "计算卸载与任务调度 (Osama §3)",
      "边缘存储与缓存 (Shi Ch.5)",
      "边缘安全与隐私 (Yang Ch.6)",
      "车联网与工业边缘应用 (Shi Ch.8)",
      "云-边-端协同 (Shi Ch.9)",
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
      "虚实映射与同步机制 (Tao Ch.3)",
      "数字孪生驱动的智能制造 (Tao Ch.4)",
      "实时数据采集与互联 (Tao Ch.5)",
      "仿真验证与预测维护 (Tao Ch.7)",
      "工业 4.0 与数字孪生应用 (Tao Ch.8)",
      "数字孪生平台架构 (Kritzinger §4)",
      "开放挑战与未来方向 (Dutta §5)",
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
      "正则化与 Lasso/Ridge (ESL §3.4)",
      "基展开与样条 (ESL §5)",
      "核方法与 SVM (ESL §12)",
      "决策树与随机森林 (ESL §9)",
      "神经网络基础 (ESL §11)",
      "集成学习与 Boosting (ESL §10)",
      "朴素贝叶斯与贝叶斯网络 (Hastie §6.6, §17.1)",
      "EM 算法与高斯混合模型 (Hastie §8.5)",
      "主成分分析（PCA）与降维 (Hastie §14.5)",
      "聚类方法（K-means/层次聚类） (Hastie §14.3)",
      "隐马尔可夫模型（HMM） (Hastie §17.2)",
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
      "共轭梯度法 (Nocedal §5)",
      "随机梯度下降与方差减小 (Bertsekas §2)",
      "对偶理论与 KKT (Boyd §5)",
      "次梯度与近端方法 (Bertsekas §6)",
      "Adam 与自适应优化器 (Goodfellow §8.5)",
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
      "自编码器与稀疏编码 (Goodfellow §14)",
      "变分自编码器 VAE (Goodfellow §20.10)",
      "对比学习与 InfoNCE (Bengio Review §5)",
      "流形学习假设 (Goodfellow §5.11)",
      "生成对抗表示 (Goodfellow §20.4)",
      "表征评估与线性探测 (Bengio Review §7)",
      "深度聚类表示 (Goodfellow §15.5)",
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
      "自监督预训练任务 (Liu §2)",
      "对比学习框架 SimCLR/MoCo (Liu §3)",
      "掩码图像建模 MAE (Liu §4)",
      "对比损失与 InfoNCE (Liu §3.2)",
      "BYOL 与负样本免方法 (Jing §5)",
      "视觉自监督评估协议 (Jing §7)",
      "语言自监督 MLM (Goodfellow §12.4)",
      "多模态自监督 CLIP (Liu §6)",
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
      "迁移学习分类与定义 (Pan Survey §2)",
      "域适应与域对抗 (Pan Survey §4)",
      "微调与特征迁移 (Goodfellow §5.4)",
      "元学习框架 (Hospedales §3)",
      "基于度量的 ProtoNet/MatchingNet (Hospedales §4)",
      "基于优化的 MAML (Hospedales §5)",
      "基于模型的元学习 (Hospedales §6)",
      "少样本学习应用 (Hospedales §8)",
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
      "联邦学习框架与分类 (Yang Ch.1)",
      "联邦平均 FedAvg (McMahan §3)",
      "异构数据与非 IID (Kairouz §2)",
      "差分隐私保护 (Kairouz §4)",
      "安全聚合协议 (Kairouz §5)",
      "客户端选择与激励 (Yang Ch.4)",
      "纵向联邦与联邦迁移 (Yang Ch.5)",
      "联邦学习系统部署 (Kairouz §8)",
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
      "因果图与有向无环图 (Pearl §1)",
      "do 演算与干预 (Pearl §3)",
      "潜在结果框架 (Hernán §1)",
      "混杂控制与倾向评分 (Hernán §12)",
      "工具变量与 IV 估计 (Hernán §16)",
      "反事实推理 (Pearl §7)",
      "因果发现算法 (Peters Ch.4)",
      "结构因果模型 SCM (Pearl §2)",
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
      "ROC 与 AUC 评估 (ESL §9.2)",
      "模型选择准则 AIC/BIC (Murphy §5.3)",
      "超参数调优方法 (Goodfellow §5.3)",
      "代价敏感学习与混淆矩阵 (ESL §9.3)",
      "集成模型评估 (Murphy §6)",
      "学习曲线与容量分析 (Goodfellow §5.2)",
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
      "MDP 与序列建模对比 (Sutton §3)",
      "Return-to-go 表示 (Chen §3)",
      "因果自注意力 Transformer (Chen §3)",
      "离线 RL 与数据分布 (Levine §3)",
      "轨迹采样策略 (Chen §4)",
      "与 Q-learning 对比 (Chen §5)",
      "Online Decision Transformer 扩展 (Levine §5)",
      "序列建模 RL 局限性 (Levine §6)",
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
      "隐马尔可夫与词性标注 (Manning §10)",
      "上下文无关文法与 CFG (Jurafsky §13)",
      "依存句法分析 (Jurafsky §18)",
      "语义角色标注 (Jurafsky §18.6)",
      "词义消歧 WSD (Manning §7)",
      "分布语义与词嵌入 (Eisenstein §14)",
      "命名实体识别（NER） (Jurafsky §8)",
      "共指消解 (Jurafsky §21)",
      "依存句法分析（CKY/Shift-Reduce） (Jurafsky §13)",
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
      "注意力机制与 Transformer (Jurafsky §13.4)",
      "子词 BPE 与词汇表 (Koehn NMT Ch.7)",
      "BLEU 与翻译评估 (Jurafsky §13.5)",
      "反向翻译与数据增强 (Koehn NMT Ch.9)",
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
      "意图识别与槽位填充 (Gao §2)",
      "对话状态跟踪 (Tur Ch.5)",
      "对话策略学习 (Tur Ch.6)",
      "自然语言生成 NLG (Jurafsky §24.2)",
      "检索式问答系统 (Jurafsky §23)",
      "生成式闲聊模型 (Gao §4)",
      "端到端神经对话系统 (Gao §5)",
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
      "滑动窗口与候选区域 (Szeliski §5.1)",
      "HOG 特征与 DPM (Forsyth §16)",
      "R-CNN 系列 (Szeliski §5.2)",
      "YOLO 单阶段检测 (Yang §6)",
      "FCN 语义分割 (Szeliski §5.4)",
      "U-Net 编码解码结构 (Yang §8)",
      "Mask R-CNN 实例分割 (Szeliski §5.5)",
      "全景分割 Panoptic (Yang §9)",
      "无锚框检测器（CenterNet/FCOS） (Szeliski §5.4)",
      "DETR 与 Transformer 检测 (Szeliski §5.5)",
      "3D 检测与 BEV 感知 (Szeliski §5.6)",
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
      "声学特征 MFCC/Filterbank (Rabiner §5)",
      "隐马尔可夫模型 HMM (Huang §8)",
      "CTC 损失与端到端 (Yu §7)",
      "声学模型 DNN/TDNN (Yu §6)",
      "语言模型与解码 (Huang §13)",
      "Tacotron 端到端 TTS (Yu §9)",
      "声码器 WaveNet/HifiGAN (Huang §15)",
      "端到端 ASR Conformer (Rabiner §15)",
      "RNN-T（RNN Transducer） (Huang §13)",
      "大规模 ASR（Whisper） (Radford §3)",
      "语音增强与分离 (Huang §15)",
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
      "知识图谱嵌入 TransE/RotatE (Ji §5)",
      "知识推理与规则 (Hogan §7)",
      "知识问答 QA (Kejriwal Ch.6)",
      "本体与本体构建 (Hogan §8)",
      "知识图谱应用 (Ji §7)",
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
      "GARCH 波动率建模 (Hamilton §21)",
      "状态空间与卡尔曼滤波 (Box §6)",
      "指数平滑与 ETS (Hyndman §8)",
      "深度学习时序 LSTM/Transformer (Hyndman §12)",
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
      "Transformer 架构 (Goodfellow §12.4)",
      "自回归预训练目标 (Radford §2)",
      "Scaling Laws (Touvron §2.1)",
      "预训练数据构造 (Touvron §3)",
      "位置编码 RoPE (Touvron §2.2)",
      "RMSNorm 与 SwiGLU (Touvron §2.2)",
      "高效训练与并行 (Touvron §3.2)",
      "评估基准与 zero-shot (Radford §3)",
      "分词算法（BPE/SentencePiece/Unigram） (Sennrich §3)",
      "模型初始化与正则化（Xavier/He init） (Goodfellow §8.4)",
      "优化器选择与学习率调度（AdamW/Cosine） (Goodfellow §8.5)",
      "检查点保存与断点续训 (Megatron-LM §4)",
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
      "SFT 监督微调 (Ouyang §3)",
      "奖励模型训练 (Ouyang §4)",
      "PPO 强化学习优化 (Ouyang §5)",
      "RLHF 框架流程 (Goodfellow §15.3)",
      "DPO 直接偏好优化 (Rafailov §3)",
      "宪法 AI 与 RLAIF (Rafailov §5)",
      "对齐税与平衡 (Ouyang §6)",
      "安全对齐与红队 (Ouyang §7)",
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
      "思维链 Chain-of-Thought (Wei §3)",
      "思维树 Tree-of-Thought (Wei §5)",
      "ReAct 推理与行动 (Yao §3)",
      "自洽性 Self-Consistency (Wei §4)",
      "提示模板与角色扮演 (Jurafsky §13.5)",
      "复杂推理分解 Least-to-Most (Yao §4)",
      "提示安全与注入防御 (Jurafsky §13.7)",
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
      "稀疏门控 MoE 原理 (Shazeer §2)",
      "Top-k 路由机制 (Fedus §3)",
      "负载均衡损失 (Lepikhin §3.2)",
      "专家容量因子 (Fedus §3.2)",
      "专家并行 Expert Parallelism (Lepikhin §4)",
      "路由策略对比 (Shazeer §4)",
      "DeepSeek MoE 细粒度专家 (Fedus §5)",
      "MoE 训练稳定性 (Fedus §6)",
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
      "注意力计算复杂度 (Goodfellow §12.4)",
      "稀疏注意力 Longformer (Beltagy §3)",
      "线性注意力机制 (Dao §5)",
      "FlashAttention IO 优化 (Dao §3)",
      "分块注意力 Blockwise (Dao §4)",
      "滑动窗口注意力 (Beltagy §4)",
      "位置外推与 NTK-aware (Beltagy §6)",
      "长上下文评估 LongBench (Beltagy §7)",
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
      "MMLU 多任务知识评测 (Hendrycks §3)",
      "HELM 全方位评估 (Liang §3)",
      "BigBench 任务基准 (Liang §4)",
      "人类对齐评估 (Zheng §3)",
      "LLM-as-a-Judge (Zheng §4)",
      "代码评测 HumanEval (Liang §5)",
      "数学评测 GSM8K (Hendrycks §4)",
      "安全与偏见评估 (HELM §6)",
      "幻觉评测（TruthfulQA/HaluEval） (Lin §4)",
      "多语言基准（C-Eval/CMMLU） (Huang §5)",
      "智能体/工具使用评测 (Yao §6)",
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
      "RAG 框架与架构 (Lewis §2)",
      "向量检索与 ANN (Karpukhin §2)",
      "稠密检索 DPR (Karpukhin §3)",
      "重排序 Cross-Encoder (Karpukhin §4)",
      "RAG-Sequence 与 RAG-Token (Lewis §3)",
      "检索增强多跳推理 (Asai §4)",
      "检索增强自反 Self-RAG (Asai §5)",
      "RAG 评估与基准 (Asai §6)",
      "文档分块策略（固定/语义/递归分块） (Lewis §3.2)",
      "嵌入模型选择与评估 (Reimers §4)",
      "向量数据库（FAISS/Milvus/Pinecone） (Johnson §4)",
      "生成融合与 RAG-Fusion (Lewis §3.5)",
      "查询重写与扩展 (Ma §4)",
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
      "智能体架构概览 (Xi Survey §2)",
      "规划与任务分解 (Yao ReAct §3)",
      "工具调用与 API 使用 (Yao ReAct §4)",
      "记忆与反思机制 (Shinn §3)",
      "多智能体协作 (Xi Survey §4)",
      "环境交互与具身智能 (Xi Survey §5)",
      "智能体评估基准 (Xi Survey §6)",
      "AutoGPT/MetaGPT 框架 (Xi Survey §7)",
    ],
  },
  'advanced/reinforcement-learning/decision-transformer': {
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
      "扩展工作 Online DT/Boosting (Levine §5)",
    ],
  },
  'advanced/llm-principles/rwkv': {
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
      "与 Transformer 的实验对比 (Peng §8)",
    ],
  },
  'advanced/llm-principles/gshard': {
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
      "多语言翻译应用 (Lepikhin §7)",
    ],
  },
  'advanced/computer-vision/dit-sora': {
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
      "世界模拟器与世界模型 (OpenAI §5)",
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
      "GPU 架构与 Tensor Core (Hennessy §4)",
      "TPU 脉动阵列 (Hennessy §7)",
      "AI 加速器数据流 (Sze §3)",
      "量化与稀疏计算 (Sze §5)",
      "内存层次与带宽 (Sze §4)",
      "近存计算与存内计算 (Karl §3)",
      "能效评估与 Roofline (Hennessy §4.6)",
      "边缘 AI 芯片设计 (Sze §7)",
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
      "脉冲神经网络 SNN (Indiveri §2)",
      "LIF 神经元模型 (Indiveri §3)",
      "神经形态硬件架构 (Schuman §3)",
      "事件驱动计算 (Davies §3)",
      "Loihi 架构与可编程学习 (Davies §4)",
      "TrueNorth 与 SpiNNaker (Schuman §4)",
      "STDP 在线学习规则 (Indiveri §5)",
      "神经形态应用与基准 (Schuman §5)",
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
      "突触传递与可塑性 (Kandel §67)",
      "感知系统与信息处理 (Kandel §22)",
      "学习与记忆的神经基础 (Kandel §65)",
      "脑认知与意识 (Kandel §61)",
      "神经编码与放电模型 (Dayan §1-2)",
      "计算神经科学与类脑算法 (Dayan §7)",
      "人工神经网络与类脑计算 (Dayan §8)",
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
      "演化与数字生命 Tierra (Adami §5)",
      "复杂系统与涌现 (Langton §4)",
      "群体动力学 (Adami §7)",
      "合成生物学交叉 (Boden §5)",
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
      "脑电信号 EEG 采集 (Rao §2)",
      "皮层脑电 ECoG 与侵入式 (Nicolelis §3)",
      "特征提取与信号处理 (Wolpaw §4)",
      "解码算法与机器学习 (Rao §5)",
      "运动想象与神经反馈 (Wolpaw §5)",
      "感觉恢复与神经假体 (Nicolelis §6)",
      "BCI 伦理与安全 (Wolpaw §12)",
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
      "粒子群优化 PSO (Kennedy §3)",
      "蜂群算法 ABC (Yang §6)",
      "群体机器人系统 (Bonabeau §7)",
      "群体分工与协同 (Kennedy §5)",
      "群体智能应用 (Yang §9)",
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
      "事后解释方法 (Molnar §8)",
      "LIME 局部解释 (Arrieta §3.2)",
      "SHAP 与博弈论归因 (Molnar §9)",
      "反事实解释 (Molnar §10)",
      "深度学习可解释性 (Arrieta §4)",
      "XAI 评估与人类研究 (Adadi §4)",
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
      "量子比特与态空间 (Nielsen §1)",
      "量子纠缠与 Bell 态 (Nielsen §2)",
      "量子门与量子电路 (Nielsen §4)",
      "量子算法 Shor/Grover (Nielsen §5-6)",
      "量子测量与不可克隆 (Wilde §3)",
      "量子纠错码 (Nielsen §10)",
      "量子密钥分发 QKD (Preskill §4)",
      "量子隐形传态与超密编码 (Nielsen §4.4)",
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
      "可持续发展定义与演进 (Kates §2)",
      "地球系统与人类世 (Clark §3)",
      "社会-生态系统耦合 (Biermann §4)",
      "韧性与阈值 (Kates §5)",
      "可持续性转型 (Biermann §6)",
      "可持续指标 SDGs (Clark §4)",
      "地球系统治理 (Biermann §5)",
      "可持续性科学与政策 (Kates §6)",
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
      "技术责任与问责 (Floridi §5)",
      "技术美德伦理 (Vallor §3)",
      "自动化伦理与自主性 (Vallor §7)",
      "新兴技术治理 (van den Hoven §10)",
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
      "太空探索历史与动机 (Sagan §13)",
      "空间环境与轨道力学 (NRC §3)",
      "深空探测与行星科学 (Sagan §10)",
      "载人航天与生命保障 (NRC §4)",
      "空间资源与小行星采矿 (NRC §5)",
      "空间站与近地轨道利用 (Logsdon §6)",
      "火星与外太阳系探测 (Sagan §7)",
      "空间政策与国际合作 (NRC §6)",
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
      "海洋云层增亮 (NRC Reflecting §4)",
      "碳移除 CDR 技术 (NRC CDR §2)",
      "直接空气捕集 DAC (NRC CDR §3)",
      "海洋施肥与碱化 (NRC CDR §4)",
      "气候工程伦理与治理 (Keith §6)",
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
      "全球公共物品 (Biermann §3)",
      "多层治理与网络 (Rosenau §4)",
      "地球系统治理 (Biermann §4)",
      "非国家行为体参与 (Weiss §8)",
      "全球治理改革 (Biermann §8)",
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
      "力控制与视觉伺服 (Craig §6-9)",
      "移动机器人与运动规划 (Spong §10)",
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
      "数控技术与编程 (Groover §6)",
      "工业机器人应用 (Groover §7)",
      "物料搬运与自动识别 (Groover §10)",
      "柔性制造系统 (Groover §16)",
      "计算机集成制造系统 (Groover §23)",
      "制造执行系统与工业物联网 (Tay §4)",
      "数字孪生与智能工厂 (Tay §7)",
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
      "光固化成型工艺 (Gibson §4)",
      "熔融沉积成型 (Gibson §6)",
      "选择性激光烧结与熔化 (Gibson §5)",
      "三维打印与黏结剂喷射 (Gibson §8)",
      "直接金属激光烧结 (Gibson §5)",
      "增材制造后处理与质量 (Chua §11)",
      "增材制造设计与生物医学应用 (Gibson §12)",
    ],
  },
  'engineering/engineering-mechanics': {
    title: "工程力学（理论/材料/结构力学）",
    books: [
          "Hibbeler, \"Mechanics of Materials\" (11th ed., 2019)",
          "Hibbeler, \"Engineering Mechanics: Statics\" (14th ed., 2019)"
    ],
    chapters: [
      "质点与刚体静力学 (Hibbeler Statics §3-5)",
      "应力与应变 (Hibbeler Mechanics §1-2)",
      "轴向载荷与扭转 (Hibbeler Mechanics §3-5)",
      "梁的弯曲 (Hibbeler Mechanics §6)",
      "组合载荷下的应力 (Hibbeler Mechanics §8)",
      "压杆稳定 (Hibbeler Mechanics §13)",
      "运动学与动力学（速度/加速度分析） (Hibbeler §Dynamics Ch.12-16)",
      "功-能原理与冲量-动量 (Hibbeler §Dynamics Ch.17-19)",
      "机械振动（自由/受迫振动） (Hibbeler §Dynamics Ch.22)",
      "Mohr 应力圆与应变转换 (Hibbeler §MechMat §7)",
      "失效理论（von Mises/Tresca/Mohr） (Hibbeler §MechMat §10)",
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
      "疲劳设计 (Shigley §6)",
      "轴的设计 (Shigley §12)",
      "齿轮传动设计 (Shigley §13-14)",
      "螺纹联接与紧固件 (Shigley §9)",
      "离合器与制动器 (Norton §16)",
      "机构运动学（连杆/凸轮/齿轮系） (Norton §Mechanism Ch.2-4)",
      "轴承设计（滚动/滑动接触） (Norton §Ch.10)",
      "弹簧设计与带链传动 (Norton §Ch.13-14)",
      "摩擦学与润滑 (Norton §Ch.11)",
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
      "精密磨削工艺 (Kalpakjian §26)",
      "抛光与化学机械抛光 (Kalpakjian §27)",
      "表面粗糙度与测量 (Whitehouse §3-4)",
      "纳米尺度加工 (Kalpakjian §29)",
      "超精密机床结构与静压轴承 (Kalpakjian §27)",
      "在线测量与误差补偿技术 (Whitehouse §6)",
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
      "高级量测体系AMI (Borlase §4)",
      "配电自动化系统 (Borlase §6)",
      "分布式能源接入 (Borlase §8)",
      "需求响应与负荷管理 (Borlase §9)",
      "电网自愈与可靠性 (Momoh §7)",
      "广域测量系统（WAMS/PMU） (Momoh §10)",
      "微电网与分布式能源 (Momoh §8)",
      "EV/V2G 集成 (Momoh §11)",
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
      "直流电机 (Fitzgerald §7)",
      "同步电机 (Fitzgerald §5)",
      "感应电机 (Fitzgerald §6)",
      "永磁与特种电机 (Chapman §9)",
      "电机损耗、发热与冷却 (Fitzgerald §3)",
      "电力电子驱动与变频控制 (Chapman §10)",
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
      "HVDC 输电 (Grainger §17)",
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
      "直流-直流变换器 (Mohan §8)",
      "电压源逆变器与PWM技术 (Mohan §9)",
      "交流传动控制 (Mohan §11)",
      "谐振变换器 (Rashid §8)",
      "多电平变换器 (Mohan §11)",
      "功率因数校正（PFC） (Mohan §8)",
      "HVDC 与柔性交流输电（FACTS） (Mohan §12)",
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
      "拉普拉斯变换 (Oppenheim §9)",
      "离散时间信号与Z变换 (Oppenheim §10)",
      "采样定理 (Oppenheim §7)",
      "DFT/FFT 与频谱分析 (Oppenheim §9-10)",
      "数字滤波器（FIR/IIR） (Oppenheim §7-8)",
      "状态空间表示 (Oppenheim §11)",
      "随机信号与相关 (Oppenheim §12)",
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
      "纳米生物医学应用 (Poole §7)",
      "自组装与分子机器 (Rogers §5)",
      "纳米环境安全与毒性 (Poole §10)",
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
      "热力学性质计算 (Smith §6)",
      "相平衡与活度系数 (Smith §10-12)",
      "化学反应平衡 (Smith §13)",
      "过程热力学分析 (Smith §15)",
      "溶液热力学与超额性质 (Smith §11)",
      "分子热力学与状态方程 (Prausnitz §4-5)",
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
      "多组分传递与耦合 (Bird §22-24)",
      "辐射传热 (Bird §14-15)",
      "边界层理论 (Bird §4)",
      "非牛顿流体与流变学 (Bird §8)",
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
      "精馏过程 (Seader §5-7)",
      "吸收与汽提 (Seader §6)",
      "液-液萃取 (Seader §8)",
      "膜分离过程 (Seader §11)",
      "吸附与离子交换 (Seader §15)",
      "结晶过程与设计 (Seader §17)",
      "干燥过程与设备 (Geankoplis §9)",
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
      "理想反应器设计 (Fogler §4-5)",
      "非理想反应器与停留时间分布 (Fogler §13-14)",
      "催化反应与催化剂 (Fogler §10)",
      "多相催化反应器 (Fogler §12)",
      "聚合反应工程 (Levenspiel §9)",
      "非等温反应器设计与能量平衡 (Fogler §9)",
      "生化反应器与酶动力学 (Fogler §12)",
      "停留时间分布（RTD）建模 (Fogler §13-14)",
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
      "混凝土梁的弯曲设计 (Nilson §2-3)",
      "混凝土柱设计 (Nilson §8)",
      "钢筋混凝土板 (Nilson §12)",
      "钢结构设计（梁/柱/连接） (Hibbeler §Structural Analysis Ch.6)",
      "抗震设计原理与反应谱 (Hibbeler §Seismic §3)",
      "有限元分析 (Hibbeler §FEA Ch.1-3)",
      "屈曲与稳定性分析 (Hibbeler §Structural §7)",
      "预应力混凝土设计 (Nilson §12)",
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
      "土中应力与有效应力原理 (Das §6-7)",
      "土的压缩性与固结 (Das §8-9)",
      "土的抗剪强度 (Das §10-11)",
      "浅基础承载力 (Das Foundation §3)",
      "深基础桩基 (Das Foundation §11)",
      "边坡稳定性分析 (Das §11-12)",
      "挡土墙与土压力 (Das §13)",
      "原位测试（SPT/CPT/十字板） (Das §7)",
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
      "隧道围岩力学 (Hoek §2-4)",
      "隧道支护与衬砌设计 (Hoek §8-9)",
      "隧道施工方法 (Hoek §11)",
      "桥梁施工方法与架设 (Troitsky §7)",
      "隧道通风与防灾救援 (Hoek §12)",
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
      "太阳能光伏与热利用 (Tester §7-8)",
      "风能系统 (Tester §9)",
      "生物质能与生物燃料 (Tester §10)",
      "地热能 (Tester §11)",
      "海洋能与水电 (Sørensen §4)",
      "分布式能源与微电网 (Tester §13)",
      "氢能与燃料电池 (Sørensen §5)",
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
      "超级电容器 (Huggins §9)",
      "飞轮与机械储能 (Huggins §12)",
      "储能系统集成与应用 (Doughty §8)",
      "压缩空气与抽水蓄能 (Huggins §13)",
      "相变热储能技术 (Huggins §11)",
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
      "质子交换膜燃料电池 (O'Hayre §8)",
      "氢气制备方法 (Gupta §1-3)",
      "氢气储存与运输 (Gupta §6-8)",
      "氢能系统集成 (Gupta §12)",
      "固体氧化物与熔融碳酸盐燃料电池 (O'Hayre §9)",
      "氢安全与规范标准 (Gupta §14)",
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
      "汽油机燃烧过程 (Heywood §4-9)",
      "柴油机燃烧过程 (Heywood §10)",
      "燃油喷射与雾化 (Heywood §7)",
      "排放污染物与控制 (Heywood §11)",
      "增压与进气系统 (Ferguson §5)",
      "发动机性能与循环分析 (Heywood §5)",
      "替代燃料与混合动力总成 (Heywood §13)",
    ],
  },
  'engineering/refrigeration-air-conditioning': {
    title: "制冷与空调",
    books: [
          "Stoecker, Jones, \"Refrigeration and Air Conditioning\" (2nd ed., 1982)",
          "ASHRAE, \"ASHRAE Handbook: Fundamentals\" (2021 ed., 2021)"
    ],
    chapters: [
      "制冷热力学基础 (Stoecker §2-3)",
      "蒸气压缩制冷循环 (Stoecker §4-5)",
      "制冷剂性质与选择 (ASHRAE §29)",
      "吸收式制冷系统 (Stoecker §15)",
      "空气调节原理 (Stoecker §16-17)",
      "制冷系统部件设计 (Stoecker §7)",
      "通风与室内空气品质 (ASHRAE §12)",
      "制冷空调自动控制 (Stoecker §18)",
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
      "物理处理工艺 (Metcalf §5-7)",
      "生物处理原理 (Metcalf §8-9)",
      "活性污泥法 (Metcalf §10-11)",
      "生物膜法与厌氧处理 (Metcalf §13)",
      "深度处理与消毒 (Metcalf §15)",
      "污泥处理与处置（浓缩/脱水/消化） (Metcalf&Eddy §14-15)",
      "生物脱氮除磷（BNR） (Metcalf&Eddy §8)",
      "膜生物反应器（MBR） (Metcalf&Eddy §10)",
      "污水再生回用 (Metcalf&Eddy §13)",
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
      "旋风与静电除尘器 (Cooper §6-9)",
      "硫氧化物控制技术 (Cooper §11)",
      "氮氧化物控制技术 (Cooper §12)",
      "挥发性有机物控制 (Cooper §13)",
      "大气扩散模型（高斯烟羽） (Cooper§Alley §4)",
      "重金属与汞污染控制 (Cooper§Alley §9)",
      "室内空气质量 (Cooper§Alley §11)",
      "CEMS 连续排放监测 (Cooper§Alley §12)",
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
      "卫生填埋设计 (Tchobanoglous §11)",
      "焚烧与热解处理 (Tchobanoglous §12-13)",
      "资源回收与循环利用 (Tchobanoglous §8-9)",
      "生物转化处理 (Vesilind §9)",
      "危险废物处理与处置 (Tchobanoglous §15)",
      "电子废物回收处理 (Vesilind §10)",
    ],
  },
  'engineering/soil-remediation': {
    title: "土壤污染与修复",
    books: [
          "Sharma, Reddy, \"Geoenvironmental Engineering: Site Remediation, Waste Containment, and Emerging Waste Management Technologies\" (1st ed., 2004)",
          "Hester, Harrison, \"Soil Remediation and Rehabilitation\" (1st ed., 2014)"
    ],
    chapters: [
      "土壤污染物类型与来源 (Sharma §5-6)",
      "土壤污染调查与监测 (Sharma §7)",
      "土壤物理修复技术 (Sharma §8)",
      "土壤化学修复与稳定化 (Sharma §9)",
      "土壤生物修复 (Sharma §10)",
      "污染场地风险评估 (Hester §3)",
      "地下水污染与修复 (Sharma §11)",
      "原位化学氧化与还原修复 (Hester §5)",
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
      "植被恢复与重建 (Clewell §5)",
      "土壤生态修复 (Hobbs §8)",
      "湿地生态系统恢复 (Hobbs §12)",
      "矿区生态重建 (Hobbs §15)",
      "恢复效果监测与评价 (Clewell §7)",
      "河流与水生生态系统恢复 (Hobbs §10)",
      "城市与受损土地生态修复 (Hobbs §18)",
    ],
  },
  'engineering/environmental-planning-management': {
    title: "环境规划与管理",
    books: [
          "Goodstein, Polasky, \"Economics and the Environment\" (8th ed., 2019)",
          "Clements, \"The Encyclopedia of Environmental Management\" (1st ed., 2013)"
    ],
    chapters: [
      "环境规划原理与方法 (Goodstein §1-3)",
      "环境影响评价 (Goodstein §6)",
      "环境管理体系ISO14001 (Clements §4)",
      "环境监测与质量评价 (Clements §7)",
      "环境法规与政策工具 (Goodstein §8-9)",
      "可持续发展管理 (Goodstein §14)",
      "环境风险评价与管理 (Clements §10)",
      "企业环境责任与生命周期管理 (Goodstein §16)",
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
      "环境政策工具与设计 (Tietenberg §10-11)",
      "排污权交易制度 (Tietenberg §13)",
      "环境法规与监管 (Kraft §3-4)",
      "国际环境治理 (Kraft §8)",
      "环境正义与公平 (Kraft §11)",
      "多层级环境治理体系 (Kraft §6)",
      "全球气候治理与碳中和政策 (Tietenberg §14)",
    ],
  },
  'engineering/circular-economy-carbon-neutrality': {
    title: "循环经济与碳中和",
    books: [
          "Lyle, \"Regenerative Design for Sustainable Development\" (1st ed., 1994)",
          "MacDowell, \"Carbon Capture and Storage\" (2nd ed., 2020)"
    ],
    chapters: [
      "循环经济原理与模式 (Lyle §1-3)",
      "生命周期评价方法 (Lyle §7)",
      "碳足迹核算与碳计量 (MacDowell §2)",
      "碳捕集与封存技术 (MacDowell §5-7)",
      "零碳能源系统 (MacDowell §11)",
      "工业共生与资源循环 (Lyle §9)",
      "碳捕集利用与封存（CCUS） (MacDowell §8)",
      "负排放技术与碳汇 (MacDowell §13)",
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
      "仿生机器人与运动 (Bar-Cohen §8)",
      "仿生表面与界面 (Bar-Cohen §6)",
      "仿生功能系统与自修复材料 (Vincent §10)",
      "仿生医学器件与应用 (Bar-Cohen §10)",
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
      "发酵培养基设计 (Stanbury §3)",
      "发酵罐设计与传质 (Stanbury §7-8)",
      "发酵过程控制 (Stanbury §9)",
      "灭菌与无菌技术 (Stanbury §5)",
      "下游处理与产物回收 (Stanbury §10)",
      "动植物细胞培养与生物反应器 (Shuler §9)",
      "发酵产物分离纯化与制剂 (Stanbury §11)",
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
      "晶体学基础（Miller 指数/点群/空间群） (Klein §5)",
    ],
  },
  'foundations/petrology': {
    title: "岩石学",
    books: [
          "Winter, \"Principles of Igneous and Metamorphic Petrology\" (2nd ed., 2013)",
          "Boggs, \"Principles of Sedimentology and Stratigraphy\" (5th ed., 2012)"
    ],
    chapters: [
      "岩浆成因与演化 (Winter §5-7)",
      "火成岩分类与岩相学 (Winter §3-4)",
      "变质作用与变质相 (Winter §21-25)",
      "沉积岩形成与成岩作用 (Boggs §1-2)",
      "沉积岩的分类 (Boggs §4-6)",
      "岩石构造与组构 (Winter §9)",
      "火成岩地球化学与同位素 (Winter §14)",
      "岩石大地构造与构造环境 (Winter §16)",
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
      "微体古生物与化石记录 (Prothero §4)",
      "无脊椎动物演化 (Clarkson §2-5)",
      "脊椎动物演化 (Prothero §15-17)",
      "古植物与孢粉学 (Prothero §19)",
      "演化模式与灭绝事件 (Prothero §10)",
      "古生态学与群落分析 (Prothero §11)",
      "生物地层学 (Prothero §12)",
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
      "微量元素地球化学 (White §7)",
      "有机地球化学 (White §12)",
      "地球化学循环 (White §12)",
      "水-岩相互作用 (White §6)",
      "放射性同位素与年代学 (Faure §11-12)",
      "地球化学热力学与相平衡 (White §5)",
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
      "地热学与地球内部温度 (Fowler §7)",
      "地球内部结构与板块构造 (Fowler §2)",
      "地球物理探测方法 (Lowrie §9)",
      "地电学与电磁方法 (Lowrie §6)",
      "板块运动与地幔动力学 (Fowler §8)",
    ],
  },
  'intermediate/sedimentology-stratigraphy': {
    title: "沉积学与地层学",
    books: [
          "Boggs, \"Principles of Sedimentology and Stratigraphy\" (5th ed., 2012)",
          "Catuneanu, \"Principles of Sequence Stratigraphy\" (1st ed., 2006)"
    ],
    chapters: [
      "沉积岩的成因与类型 (Boggs §1-3)",
      "沉积构造与沉积结构 (Boggs §4-5)",
      "沉积环境与沉积相 (Boggs §12-13)",
      "地层学原理与地层对比 (Boggs §14)",
      "层序地层学 (Catuneanu §4-6)",
      "沉积盆地分析 (Boggs §16)",
      "碳酸盐沉积学 (Boggs §6)",
      "成岩作用 (Boggs §7)",
      "层序地层学模型 (Boggs §12)",
    ],
  },
  'intermediate/structural-geology': {
    title: "构造地质学",
    books: [
          "Fossen, \"Structural Geology\" (2nd ed., 2016)",
          "Davis, Reynolds, Kluth, \"Structural Geology of Rocks and Regions\" (3rd ed., 2011)"
    ],
    chapters: [
      "应力与应变分析 (Fossen §1-3)",
      "褶皱构造与机制 (Fossen §8-11)",
      "断层与剪切带 (Fossen §5-9)",
      "节理与裂隙分析 (Fossen §7)",
      "构造分析方法 (Davis §10)",
      "区域构造与造山带 (Davis §20)",
      "韧性剪切带与糜棱岩 (Fossen §15)",
      "盐构造与底辟作用 (Davis §18)",
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
      "地震观测与地震仪 (Shearer §5)",
      "地震定位方法 (Shearer §6)",
      "地震危险性与预测 (Stein §10)",
      "地震层析成像 (Stein §7)",
      "强震动地震学与工程地震 (Shearer §10)",
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
      "火山喷发产物 (Sigurdsson §15-17)",
      "火山监测与预警 (Francis §7-8)",
      "火山灾害评估 (Sigurdsson §60)",
      "火山与板块构造 (Sigurdsson §2)",
      "火山碎屑沉积与相模式 (Sigurdsson §25)",
      "火山喷发与气候影响 (Sigurdsson §65)",
    ],
  },
  'intermediate/meteorology': {
    title: "气象学",
    books: [
          "Wallace, Hobbs, \"Atmospheric Science: An Introductory Survey\" (2nd ed., 2006)",
          "Ahrens, Henson, \"Meteorology Today: An Introduction to Weather, Climate, and the Environment\" (12th ed., 2019)"
    ],
    chapters: [
      "大气组成与结构 (Wallace §1)",
      "大气热力学与辐射 (Wallace §3-4)",
      "云与降水物理 (Wallace §5)",
      "天气系统与气旋 (Ahrens §9-12)",
      "大气边界层 (Wallace §9)",
      "气象观测与预报 (Ahrens §13)",
      "中尺度气象与对流系统 (Wallace §10)",
      "雷达与卫星气象观测 (Ahrens §6)",
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
      "水循环与水文气候 (Hartmann §5-6)",
      "气候分类与气候带 (Ruddiman §7)",
      "古气候与气候历史 (Ruddiman §8-12)",
      "气候模拟与气候模型 (Hartmann §11)",
      "海洋-大气耦合与 ENSO (Hartmann §10)",
      "气候变率与年际-年代际振荡 (Ruddiman §11)",
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
      "大气环流与环流定理 (Holton §3)",
      "涡度与位涡度 (Holton §4)",
      "大气波动与罗斯贝波 (Holton §5-7)",
      "大气不稳定性与斜压发展 (Holton §8)",
      "天气尺度动力学 (Holton §9)",
      "热带动力学与 Kelvin/Rossby 波 (Holton §11)",
      "数值天气预报与模式 (Holton §13)",
    ],
  },
  'intermediate/physical-oceanography': {
    title: "物理海洋学",
    books: [
          "Knauss, Garfield, \"Introduction to Physical Oceanography\" (3rd ed., 2017)",
          "Talley, Pickard, Emery, Swift, \"Descriptive Physical Oceanography: An Introduction\" (6th ed., 2011)"
    ],
    chapters: [
      "海水物理性质 (Knauss §3-4)",
      "海洋热量与水量平衡 (Talley §5)",
      "风驱大洋环流 (Knauss §8-9)",
      "海洋波浪与海浪谱 (Knauss §10)",
      "潮汐与潮流 (Knauss §11)",
      "海洋-大气相互作用 (Talley §7)",
      "温盐环流与全球传送带 (Knauss §8)",
      "中尺度涡旋 (Knauss §9)",
      "海岸海洋学与河口动力学 (Knauss §11)",
    ],
  },
  'intermediate/marine-chemistry': {
    title: "海洋化学",
    books: [
          "Millero, \"Chemical Oceanography\" (4th ed., 2013)",
          "Libes, \"An Introduction to Marine Biogeochemistry\" (2nd ed., 2009)"
    ],
    chapters: [
      "海水化学组成与盐度 (Millero §1-4)",
      "海洋碳循环与碳酸盐体系 (Libes §10-11)",
      "海洋营养盐循环 (Libes §14)",
      "海洋同位素化学 (Millero §8)",
      "海洋有机地球化学 (Libes §20)",
      "海洋污染化学 (Millero §10)",
      "海-气气体交换与 CO₂ (Millero §9)",
      "化学示踪与古海洋学 (Libes §15)",
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
      "底栖生物群落 (Levinton §5-6)",
      "海洋鱼类与脊椎动物 (Castro §7-9)",
      "深海生物与热液生态 (Levinton §7)",
      "海洋生态系统与保护 (Levinton §11)",
      "珊瑚礁生态系统 (Levinton §10)",
      "海洋食物网与能流 (Levinton §9)",
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
      "磁层物理与磁暴 (Bothmer §5)",
      "电离层扰动与影响 (Hanslmeier §6)",
      "空间天气对技术系统的影响 (Bothmer §9)",
      "空间天气监测与预报 (Hanslmeier §9)",
      "地球辐射带与高能粒子 (Bothmer §6)",
      "高层大气与热层耦合 (Hanslmeier §5)",
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
      "地球形状与重力场 (Hofmann-Wellenhof §2-3)",
      "重力测量与重力位理论 (Hofmann-Wellenhof §4-6)",
      "大地水准面确定 (Hofmann-Wellenhof §7)",
      "卫星大地测量方法 (Torge §9)",
      "地壳形变监测 (Torge §12)",
      "几何大地测量与三角/导线 (Torge §5)",
      "参考系与时间系统 (Torge §3)",
    ],
  },
  'intermediate/gnss-positioning': {
    title: "GNSS 定位与导航",
    books: [
          "Hofmann-Wellenhof, Lichtenegger, Wasle, \"GNSS: Global Navigation Satellite Systems\" (2nd ed., 2008)",
          "Misra, Enge, \"Global Positioning System: Signals, Measurements, and Performance\" (2nd ed., 2011)"
    ],
    chapters: [
      "GNSS系统组成与星座 (Hofmann-Wellenhof §4-8)",
      "信号结构与测距码 (Misra §3-4)",
      "观测量与定位原理 (Misra §7-8)",
      "误差源与修正模型 (Misra §5)",
      "差分GPS与RTK定位 (Hofmann-Wellenhof §9)",
      "精密单点定位技术 (Hofmann-Wellenhof §10)",
      "多星座与多频定位 (Hofmann-Wellenhof §11)",
      "GNSS 遥感与气象应用 (Misra §13)",
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
      "空间分析方法 (Longley §14-15)",
      "空间数据库设计 (Longley §9)",
      "地理可视化与制图 (Bolstad §7)",
      "GIS应用与项目管理 (Longley §16-17)",
      "坐标系统与地图投影 (Bolstad §4)",
      "空间数据质量与不确定性 (Bolstad §9)",
      "遥感与 GIS 集成 (Bolstad §13)",
      "空间统计 (Bolstad §14)",
    ],
  },
  'advanced/deep-space-exploration': {
    title: "深空探测",
    books: [
          "Brown, \"Elements of Spacecraft Design\" (1st ed., 2002)",
          "Wertz, Everett, Puschell, \"Space Mission Engineering: The New SMAD\" (1st ed., 2011)"
    ],
    chapters: [
      "轨道力学与星际轨道设计 (Brown §3-4)",
      "航天器结构与机构 (Brown §6-7)",
      "推进系统设计 (Brown §8)",
      "深空通信与测控 (Wertz §19)",
      "电源系统设计 (Brown §9)",
      "行星着陆与表面探测 (Wertz §21)",
      "热控系统设计 (Brown §10)",
      "自主导航与制导 (Wertz §15)",
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
      "温室效应与辐射强迫 (IPCC §7)",
      "气候系统反馈机制 (IPCC §7)",
      "气候模型与情景预测 (IPCC §1-4)",
      "碳循环与生物地球化学 (IPCC §5)",
      "气候变化影响与极端事件 (IPCC §11-12)",
      "气候变化的检测与归因 (IPCC §3)",
      "区域气候变化与极地系统 (IPCC §9-12)",
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
      "行星表面过程与地貌 (Melosh §5-7)",
      "撞击坑形成与作用 (Melosh §9)",
      "火星表面地质 (Carr §3-5)",
      "行星大气与气候演化 (Carr §10)",
      "冰卫星与海洋世界 (Melosh §12)",
      "行星火山活动 (Carr §7)",
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
      "太阳大气与日球层 (Kivelson §3-4)",
      "行星际介质与太阳风 (Kivelson §5)",
      "地球磁层物理 (Kivelson §8-9)",
      "电离层物理 (Kivelson §11)",
      "空间等离子体波动 (Gurnett §9)",
      "磁重联与爆发过程 (Gurnett §7)",
      "地球辐射带动力学 (Kivelson §10)",
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
      "酶催化机制 (Lehninger §6)",
      "糖酵解与糖异生 (Lehninger §14)",
      "三羧酸循环 (Lehninger §16)",
      "氧化磷酸化与光合磷酸化 (Lehninger §19)",
      "脂质代谢 (Lehninger §17, §21)",
      "氨基酸代谢与尿素循环 (Lehninger §18)",
      "DNA复制与修复 (Lehninger §25)",
      "转录与 RNA 加工（Lehninger §26）",
      "翻译与蛋白质合成（Lehninger §27）",
      "核苷酸代谢（Lehninger §22）",
      "信号转导（Lehninger §12）",
      "维生素与辅酶（Lehninger §13）",
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
      "物种概念与界定 (Mayr §2)",
      "特征与同源性 (Schuh §3)",
      "支序分析原理 (Schuh §4)",
      "分支图构建与检验 (Schuh §5)",
      "系统发育与分类体系 (Schuh §6)",
      "分子系统学方法 (Schuh §7)",
      "生物命名法与法规 (Mayr §5)",
    ],
  },
  'intermediate/comparative-anatomy': {
    title: "生物解剖与比较解剖",
    books: [
          "Kardong, \"Vertebrates: Comparative Anatomy, Function, Evolution\" (8th, 2018)",
          "Hildebrand & Goslow, \"Analysis of Vertebrate Structure\" (6th, 2001)"
    ],
    chapters: [
      "脊椎动物起源与演化 (Kardong §1)",
      "脊椎动物胚胎学 (Kardong §2)",
      "皮肤及其衍生物 (Kardong §6)",
      "骨骼系统：颅骨与中轴骨骼 (Kardong §7)",
      "肌肉系统 (Kardong §10)",
      "消化与呼吸系统 (Kardong §13)",
      "循环系统 (Kardong §14)",
      "神经系统与感觉器官 (Kardong §16)",
    ],
  },
  'intermediate/developmental-biology': {
    title: "发育生物学",
    books: [
          "Gilbert, \"Developmental Biology\" (12th, 2019)",
          "Wolpert & Tickle, \"Principles of Development\" (5th, 2015)"
    ],
    chapters: [
      "发育的模式与原理 (Gilbert §1-2)",
      "受精与早期卵裂 (Gilbert §7)",
      "囊胚与原肠胚形成 (Gilbert §8)",
      "神经胚形成与体轴建立 (Gilbert §9)",
      "同源异型框基因与体节发育 (Gilbert §11)",
      "器官发生与四肢发育 (Gilbert §13)",
      "干细胞与再生 (Gilbert §23)",
      "环境对发育的调节 (Gilbert §18)",
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
      "核酸结构（DNA/RNA）（Lodish §5）",
      "蛋白质-蛋白质复合物（Lodish §3）",
      "蛋白质结构预测（AlphaFold/CASP）（Lodish §4）",
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
      "固有免疫 (Janeway §3)",
      "抗原识别受体 (Janeway §4)",
      "T细胞发育与活化 (Janeway §8)",
      "B细胞发育与抗体应答 (Janeway §9)",
      "体液与细胞免疫效应机制 (Janeway §10)",
      "免疫耐受与自身免疫 (Janeway §15)",
      "过敏与超敏反应 (Janeway §14)",
      "补体系统与激活途径（Janeway §2）",
      "移植免疫与组织相容性（Janeway §15）",
      "肿瘤免疫与免疫治疗（Janeway §17）",
      "免疫缺陷病（Janeway §16）",
      "黏膜免疫（Janeway §12）",
    ],
  },
  'intermediate/virology': {
    title: "病毒学",
    books: [
          "Flint et al., \"Principles of Virology\" (5th, 2020)",
          "Knipe & Howley, \"Fields Virology\" (7th, 2021)"
    ],
    chapters: [
      "病毒学基础与分类 (Flint §1)",
      "病毒颗粒结构与组装 (Flint §3)",
      "病毒基因组复制 (Flint §6)",
      "病毒基因表达调控 (Flint §7)",
      "病毒致病机制 (Flint §10)",
      "宿主免疫与抗病毒应答 (Flint §11)",
      "病毒与肿瘤 (Flint §12)",
      "病毒疫苗与抗病毒药物 (Flint §13)",
      "HIV 与艾滋病（Fields §41）",
      "流感病毒（Fields §41）",
      "疱疹病毒（Fields §35）",
      "肝炎病毒（Fields §28）",
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
      "觅食行为与最优化 (Davies §3)",
      "捕食者与猎物策略 (Davies §4)",
      "配偶选择与性选择 (Davies §7)",
      "交配系统 (Davies §8)",
      "亲本照顾与繁殖策略 (Davies §9)",
      "社会行为与利他主义 (Davies §11)",
      "通讯行为 (Davies §12)",
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
      "水分与转运 (Taiz §3)",
      "矿质营养 (Taiz §5)",
      "光合作用：光反应 (Taiz §7)",
      "光合作用：碳同化 (Taiz §8)",
      "韧皮部运输 (Taiz §10)",
      "植物激素与信号 (Taiz §19)",
      "植物生长发育与光形态建成 (Taiz §16)",
      "植物呼吸作用（Taiz §11）",
      "次生代谢物（Taiz §24）",
      "逆境生理（干旱/盐/低温）（Taiz §25-26）",
      "开花与生殖生理（Taiz §21）",
    ],
  },
  'intermediate/animal-physiology': {
    title: "动物生理学",
    books: [
          "Hill, Wyse & Anderson, \"Animal Physiology\" (4th, 2016)",
          "Schmidt-Nielsen, \"Animal Physiology: Adaptation and Environment\" (5th, 1997)"
    ],
    chapters: [
      "稳态与体温调节 (Hill §4)",
      "神经元与神经生理 (Hill §6)",
      "感觉系统 (Hill §7)",
      "肌肉与运动 (Hill §8)",
      "呼吸生理 (Hill §11)",
      "循环与血液 (Hill §12)",
      "渗透调节与排泄 (Hill §14)",
      "消化与能量代谢 (Hill §15)",
      "内分泌生理（Guyton §9-10）",
      "生殖生理（Guyton §16）",
      "免疫生理（Guyton §19）",
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
      "生物多样性价值 (Primack §3)",
      "物种灭绝机制 (Primack §7)",
      "生境破坏与破碎化 (Primack §5)",
      "外来入侵种 (Primack §8)",
      "种群与生活史分析 (Primack §11)",
      "保护地与保护区设计 (Primack §15)",
      "恢复生态学 (Primack §19)",
      "气候变化与生物多样性（Primack §10）",
      "保护遗传与小种群遗传学（Primack §5）",
      "保护政策与立法（Primack §16）",
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
      "米氏动力学与抑制 (Cornish-Bowden §2)",
      "别构酶与协同性 (Cornish-Bowden §7)",
      "酶反应机制 (Palmer §4)",
      "辅酶与辅因子 (Palmer §6)",
      "酶活性调节 (Palmer §7)",
      "酶工程与定向进化 (Palmer §9)",
      "酶的应用与临床 (Palmer §10)",
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
      "N-连接与O-连接糖基化 (Varki §9)",
      "糖蛋白与蛋白聚糖 (Varki §10)",
      "糖基转移酶 (Varki §5)",
      "凝集素与糖识别 (Varki §27)",
      "糖基化与疾病 (Varki §23)",
      "糖组学技术 (Varki §50)",
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
      "明场与相差显微术 (Murphy §4)",
      "荧光显微术原理 (Murphy §9)",
      "共聚焦与多光子显微术 (Murphy §12)",
      "超分辨显微技术 (Murphy §13)",
      "电子显微术基础 (Pawley §1-2)",
      "活细胞成像 (Pawley §19)",
      "图像处理与分析 (Murphy §15)",
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
      "遗传变异与种群遗传 (Futuyma §9)",
      "自然选择与适应 (Futuyma §11)",
      "分子进化与中性理论 (Futuyma §10)",
      "物种形成 (Futuyma §18)",
      "系统发育与比较方法 (Futuyma §19)",
      "进化与创新 (Futuyma §22)",
      "人类进化 (Futuyma §25)",
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
      "转录网络结构与基序 (Alon §3)",
      "负反馈与前馈调控 (Alon §5)",
      "生物振荡器 (Alon §9)",
      "代谢网络建模 (Klipp §7)",
      "信号转导通路建模 (Klipp §9)",
      "基因调控网络推断 (Klipp §11)",
      "系统生物学与医学 (Klipp §12)",
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
      "scRNA-seq数据分析流程 (Kolodziejczyk §7)",
      "细胞类型鉴定与聚类 (Kolodziejczyk §8)",
      "细胞轨迹与发育推断 (Tamir §4)",
      "空间转录组学方法 (Tamir §12)",
      "多组学单细胞整合 (Tamir §15)",
      "疾病与临床应用 (Tamir §18)",
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
      "sgRNA设计与靶点选择 (Doudna §4)",
      "基因敲除与敲入技术 (Doudna §5)",
      "碱基编辑与先导编辑 (Bollmann §8)",
      "CRISPR筛选 (Bollmann §10)",
      "CRISPR在疾病模型中的应用 (Bollmann §14)",
      "基因治疗与伦理 (Bollmann §18)",
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
      "酶活性的定向改造 (Park §3)",
      "酶稳定性与底物特异性 (Park §4)",
      "计算蛋白质设计 (Park §7)",
      "从头设计蛋白质 (Park §9)",
    ],
  },
  'life/anatomy': {
    title: "系统解剖学",
    books: [
          "Netter, \"Atlas of Human Anatomy\" (8th, 2022)",
          "Moore, Dalley & Agur, \"Clinically Oriented Anatomy\" (8th, 2017)"
    ],
    chapters: [
      "骨学与关节学 (Moore §1)",
      "骨骼肌系统 (Moore §2)",
      "头颈部解剖 (Netter §1)",
      "胸部与纵隔 (Netter §3)",
      "腹部与盆部 (Netter §4)",
      "上肢与下肢 (Netter §6)",
      "神经系统解剖 (Netter §7)",
      "心血管系统解剖 (Moore §4)",
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
      "受精与早期胚胎发育 (Moore §2)",
      "胚层分化与器官形成 (Moore §5)",
      "心血管系统发生 (Moore §13)",
      "消化与呼吸系统发生 (Moore §11)",
      "血液与免疫系统组织学（Ross §14）",
      "生殖系统组织学（Ross §22）",
      "内分泌系统组织学（Ross §18）",
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
      "肿瘤总论 (Robbins §7)",
      "心血管系统疾病 (Robbins §11)",
      "呼吸系统疾病 (Robbins §15)",
      "消化系统疾病 (Robbins §17)",
      "肾脏疾病 (Robbins §20)",
      "免疫性疾病病理（Robbins §6）",
      "遗传性疾病病理（Robbins §7）",
      "感染性疾病病理（Robbins §8）",
      "神经系统病理（Robbins §28）",
      "肝胆系统病理（Robbins §18）",
      "内分泌系统病理（Robbins §24）",
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
      "炎症与免疫紊乱 (McCance §6)",
      "肿瘤生物学 (McCance §10)",
      "水电解质与酸碱平衡 (McCance §5)",
      "心血管病理生理 (McCance §23)",
      "呼吸病理生理 (McCance §27)",
      "肾脏病理生理 (McCance §31)",
      "神经病理生理 (McCance §36)",
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
      "胸部影像学 (Webb §2)",
      "腹部与盆腔CT (Webb §4)",
      "骨骼系统影像 (Brant §8)",
      "神经影像学 (Brant §10)",
      "介入放射学 (Brant §12)",
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
      "气道管理与插管 (Morgan §19)",
      "全身麻醉的实施 (Morgan §25)",
      "局部麻醉与神经阻滞 (Morgan §23)",
      "围术期监测 (Morgan §6)",
      "麻醉并发症与处理 (Morgan §35)",
      "疼痛医学 (Morgan §42)",
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
      "跌倒与谵妄 (Hazzard §66)",
      "老年心血管疾病 (Hazzard §39)",
      "老年痴呆与认知障碍 (Hazzard §63)",
      "多重用药与药物管理 (Hazzard §18)",
      "临终关怀与缓和医疗 (Hazzard §25)",
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
      "银屑病 (Habif §8)",
      "皮肤感染 (Habif §9-10)",
      "皮肤肿瘤 (Habif §20-21)",
      "自身免疫性大疱病 (Fitzpatrick §37)",
      "性传播疾病 (Habif §11)",
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
      "外耳与中耳疾病 (Cummings §160)",
      "鼻与鼻窦疾病 (Cummings §42)",
      "咽喉与吞咽障碍 (Cummings §109)",
      "头颈肿瘤 (Cummings §87)",
      "甲状腺与甲状旁腺外科 (Cummings §91)",
      "睡眠呼吸障碍 (Cummings §101)",
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
      "眼睑与泪器疾病 (Yanoff §1.8)",
      "角膜与外眼疾病 (Yanoff §4.1)",
      "青光眼 (Yanoff §10)",
      "白内障 (Yanoff §8)",
      "视网膜疾病 (Yanoff §6.1)",
      "神经眼科学 (Yanoff §12)",
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
      "运动中急性创伤 (Brukner §47)",
      "过度使用与慢性损伤 (Brukner §14)",
      "运动康复与训练 (Brukner §11)",
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
      "窒息死亡 (Dolinak §6)",
      "猝死与心血管死亡 (Dolinak §8)",
      "中毒法医学 (Dolinak §11)",
      "法医DNA分型 (Spitz §28)",
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
      "动脉粥样硬化发病机制 (Braunwald §44)",
      "冠心病与急性冠脉综合征 (Braunwald §58)",
      "心力衰竭 (Braunwald §25)",
      "心律失常与电生理 (Braunwald §35)",
      "心脏瓣膜病 (Braunwald §66)",
      "心肌病与心肌炎 (Braunwald §79)",
      "高血压与肺动脉高压 (Braunwald §47)",
      "先天性心脏病（Braunwald §14）",
      "心脏药理学（ACEI/ARB/β受体阻滞剂）（Braunwald §36）",
      "心包疾病（Braunwald §71）",
      "感染性心内膜炎（Braunwald §73）",
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
      "哮喘 (Murray §57)",
      "慢性阻塞性肺疾病 (Murray §55)",
      "肺部感染与肺炎 (Murray §33)",
      "肺栓塞与肺动脉高压 (Murray §70)",
      "肺癌（West §Respiratory §16）",
      "间质性肺疾病（West §Respiratory §14）",
      "胸膜疾病（West §Respiratory §18）",
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
      "炎症性肠病 (Sleisenger §22)",
      "胰腺疾病 (Sleisenger §32)",
      "病毒性肝炎 (Sleisenger §80)",
      "肝硬化与门脉高压 (Sleisenger §86)",
      "消化道肿瘤（Sleisenger §52）",
      "功能性胃肠病（Sleisenger §48）",
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
      "肾上腺皮质与髓质疾病 (Williams §15)",
      "糖尿病与代谢综合征 (Williams §34)",
      "男性与女性生殖内分泌 (Williams §19-22)",
      "钙磷代谢与骨病 (Williams §27)",
      "内分泌肿瘤综合征 (Williams §42)",
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
      "急性肾损伤 (Brenner §33)",
      "慢性肾脏病与透析 (Brenner §54)",
      "高血压与肾脏 (Brenner §45)",
      "遗传性肾病 (Brenner §45)",
      "肾小管间质疾病（Brenner §34）",
      "糖尿病肾病（Brenner §38）",
      "肾移植（Brenner §45）",
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
      "白血病 (Hoffman §65)",
      "淋巴瘤 (Hoffman §76)",
      "多发性骨髓瘤 (Hoffman §83)",
      "出血与血栓性疾病 (Hoffman §111)",
      "造血干细胞移植 (Hoffman §92)",
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
      "脑血管疾病 (Adams §34)",
      "癫痫与发作性疾病 (Adams §16)",
      "运动障碍疾病 (Adams §39)",
      "痴呆与认知障碍 (Adams §21)",
      "周围神经病 (Bradley §85)",
      "中枢神经系统感染 (Bradley §75)",
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
      "物质相关障碍 (Synopsis §12)",
      "人格障碍 (Synopsis §22)",
      "儿童青少年精神障碍 (Synopsis §31)",
      "认知障碍 (Synopsis §21)",
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
      "化疗与靶向治疗 (DeVita §14)",
      "放射治疗 (DeVita §15)",
      "免疫治疗 (DeVita §16)",
      "常见实体瘤诊治 (DeVita §49-65)",
      "血液肿瘤与姑息治疗 (DeVita §146)",
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
      "革兰阳性菌感染 (Mandell §31)",
      "革兰阴性菌感染 (Mandell §35)",
      "结核与非结核分枝杆菌 (Mandell §41)",
      "病毒性感染 (Mandell §76)",
      "寄生虫感染 (Mandell §88)",
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
      "脊柱关节炎 (Kelley §68)",
      "干燥综合征 (Kelley §76)",
      "系统性硬化症 (Kelley §74)",
      "血管炎 (Kelley §84)",
      "晶体性关节炎 (Kelley §90)",
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
      "预防保健与健康促进 (Rakel §7)",
      "常见呼吸道感染 (Taylor §22)",
      "高血压与心血管管理 (Taylor §35)",
      "糖尿病管理 (Taylor §45)",
      "心理健康与基层干预 (Taylor §55)",
      "老年与缓和医疗 (Rakel §21)",
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
      "肿瘤PET显像 (Mettler §10)",
      "甲状腺与甲状旁腺显像 (Mettler §12)",
      "放射性药物治疗 (Saha §14)",
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
      "自主原则与知情同意 (Beauchamp §4)",
      "不伤害与有利原则 (Beauchamp §6)",
      "公正原则与资源分配 (Beauchamp §7)",
      "医患关系与共同决策 (Lo §4)",
      "临终决策与安乐死 (Lo §17)",
      "人类受试者研究伦理 (Beauchamp §8)",
      "器官移植伦理 (Lo §20)",
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
      "混杂与偏倚 (Rothman §10)",
      "因果推断 (Rothman §12)",
      "筛检试验评价 (Gordis §11)",
      "传染病流行病学 (Gordis §15)",
      "实验性研究/RCT 设计（Gordis §9）",
      "临床流行病学（Gordis §12）",
      "分子流行病学（Gordis §15）",
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
      "回归与相关 (Rosner §11)",
      "分类数据分析 (Rosner §10)",
      "生存分析 (Rosner §14)",
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
      "营养与慢性病 (Ross §114)",
      "食品卫生与食品安全 (Ross §104)",
      "食源性疾病 (Ross §105)",
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
      "生殖与发育毒理 (Casarett §10)",
      "遗传毒理与致癌 (Casarett §8-9)",
      "环境毒理与风险评估 (Casarett §25)",
      "毒理学试验方法 (Hayes §3)",
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
      "组织工程原理 (Atala §15)",
      "生物材料与支架 (Atala §17)",
      "器官再生与移植 (Atala §32)",
      "临床转化与伦理 (Atala §85)",
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
      "转化医学伦理与监管 (Wehling §19)",
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
      "基因编辑治疗 (Sibbald §9)",
      "单基因遗传病治疗 (Mátrai §10)",
      "肿瘤基因治疗 (Sibbald §12)",
      "免疫与伦理监管 (Sibbald §15)",
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
      "农业气候资源与区划 (Hay §10)",
      "农业气象灾害 (Hay §12)",
      "气候变化与农业 (Hay §14)",
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
      "分子标记辅助选择 (Acquaah §22)",
      "转基因与基因编辑育种 (Acquaah §25)",
      "GWAS 与全基因组关联分析（Acquaah §14）",
      "突变育种与多倍体育种（Acquaah §12）",
      "分子标记辅助选择（MAS）（Acquaah §15）",
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
      "杀虫剂与杀菌剂应用 (Pedigo §11)",
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
      "果园与蔬菜栽培管理 (Adams §10)",
      "设施园艺与气候调控 (Adams §15)",
      "采后生理与贮藏 (Adams §17)",
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
      "种子生物技术 (Bewley §9)",
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
      "温室温度与湿度调控 (Hanan §8)",
      "温室通风与CO2施肥 (Hanan §11)",
      "无土栽培原理 (Resh §1)",
      "营养液配方与管理 (Resh §4)",
      "水培与基质栽培系统 (Resh §7)",
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
      "人工智能与机器学习应用 (Oliver §16)",
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
      "茶树栽培与修剪 (Sharma §3)",
      "茶叶加工工艺 (Sharma §7)",
      "茶叶化学成分 (Wilson §2)",
      "茶多酚与儿茶素 (Wilson §4)",
      "茶叶香气与滋味 (Wilson §6)",
      "茶叶保健功能 (Wilson §10)",
      "茶叶品质评价与检验 (Wilson §12)",
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
      "蜜蜂生物学与社会 (Crane §3)",
      "蜂群管理与蜂具 (Crane §6)",
      "蜜蜂产品：蜂蜜与蜂王浆 (Crane §11)",
      "蜜蜂授粉与保护 (Crane §13)",
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
      "能量代谢与饲料能值 (McDonald §9)",
      "饲料分类与营养价值 (Pond §11)",
      "饲料加工与添加剂 (Pond §13)",
      "不同动物营养需求 (McDonald §14)",
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
      "动物繁殖生理 (Senger §2)",
      "发情周期与人工授精 (Senger §6)",
      "妊娠与分娩 (Senger §11)",
      "繁殖生物技术 (Senger §14)",
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
      "营养与饲料 (Stickney §7)",
      "池塘与网箱养殖 (Pillay §7)",
      "鱼病防治 (Stickney §10)",
      "虾蟹与贝类养殖 (Pillay §10)",
      "水产养殖可持续发展 (Pillay §14)",
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
      "海洋生态系统与渔业 (Jennings §2)",
      "捕捞对群落的影响 (Jennings §14)",
      "渔业可持续发展 (Jennings §17)",
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
      "森林立地与生态学 (Smith §3)",
      "天然林更新 (Smith §9)",
      "人工造林与苗木培育 (Smith §11)",
      "抚育间伐 (Smith §15)",
      "森林主伐与再生 (Smith §17)",
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
      "森林火灾与防火 (Davis §10)",
      "森林调查与资源清查 (Davis §4)",
      "森林经营规划 (Davis §7)",
      "收获调度与可持续收获 (Davis §9)",
      "森林可持续管理 (Davis §13)",
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
      "木材化学成分 (Bowyer §3)",
      "木材物理性质 (Hoadley §5)",
      "木材力学性质 (Bowyer §7)",
      "木材水分关系与干燥 (Hoadley §6)",
      "木材耐久性与防腐 (Bowyer §11)",
      "木质复合材料与人造板 (Bowyer §13)",
      "木材产品加工工艺 (Bowyer §15)",
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
      "食品加工对营养的影响 (Belitz §9)",
      "功能性食品与生物活性成分 (Ross §115)",
      "营养与慢性病预防 (Ross §114)",
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
      "HACCP体系原理 (Forsythe §11)",
      "风险评估与风险管理 (Forsythe §13)",
      "食品质量管理体系 (ISO) (Forsythe §14)",
      "食品安全检测技术 (Forsythe §15)",
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
      "水资源与农业利用 (Lal §5)",
      "农业生态系统养分循环 (Matson §4)",
      "农业面源污染 (Lal §9)",
      "农业温室气体与气候变化 (Lal §11)",
      "废弃物资源化利用 (Lal §13)",
      "可持续农业管理 (Lal §16)",
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
      "种群与群落生态 (Gliessman §6)",
      "生物多样性与农业 (Altieri §4)",
      "土壤生态与养分管理 (Gliessman §9)",
      "病虫害生态管理 (Altieri §8)",
      "农业可持续性评价 (Gliessman §17)",
      "可持续食物系统转型 (Gliessman §22)",
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
      "土壤物理性质 (Brady §4)",
      "土壤化学与胶体 (Brady §8)",
      "土壤生物学与有机质 (Brady §11)",
      "土壤养分与肥力 (Brady §14)",
      "植物必需营养元素 (Marschner §2)",
      "养分吸收与转运机制 (Marschner §3)",
      "养分胁迫与植物适应 (Marschner §9)",
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
      "农场经营决策 (Barnard §3)",
      "风险与不确定性 (Barnard §7)",
      "农业政策与贸易 (Colman §12)",
      "农业合作与组织 (Colman §14)",
      "农业可持续发展经济 (Colman §15)",
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
      "贫困与不平等 (Ellis §5)",
      "农村资源管理与可持续 (Ellis §8)",
      "农村产业与乡村振兴 (Chambers §8)",
      "农村基础设施与服务 (Chambers §10)",
      "农村治理与制度 (Ellis §12)",
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
      "播种与施肥机械 (Kepner §6)",
      "植保机械 (Kepner §9)",
      "收获机械 (Kepner §11)",
      "精准农业装备 (Hunt §10)",
      "农业机械化经营管理 (Hunt §12)",
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
      "土地经济评价 (Dent §10)",
      "土地利用规划 (FAO §5)",
      "土地资源可持续管理 (FAO §7)",
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
      "异方差性与OLS渐近性 (Wooldridge §5, §8)",
      "面板数据方法 (Wooldridge §13-14)",
      "工具变量与两阶段最小二乘 (Wooldridge §15)",
      "设定与数据问题 (Greene §5)",
      "联立方程模型 (Wooldridge §16)",
      "时间序列（ARMA/ARIMA/单位根/协整） (Wooldridge §18)",
      "限值因变量模型（Probit/Logit/Tobit） (Wooldridge §17)",
      "GMM 广义矩估计 (Wooldridge §19)",
      "面板数据深化（固定/随机效应） (Wooldridge §14)",
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
      "工资结构与补偿性工资差异 (Borjas §5)",
      "劳动力流动与移民 (Borjas §9)",
      "工会与集体谈判 (Borjas §10)",
      "劳动市场歧视 (Borjas §9)",
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
      "金融机构与银行业 (Mishkin §9-11)",
      "中央银行与联邦储备体系 (Mishkin §13)",
      "货币供给过程 (Mishkin §14)",
      "货币政策工具与策略 (Mishkin §15-16)",
      "外汇市场与国际金融体系 (Mishkin §17-18)",
      "货币理论与货币政策传导机制 (Mishkin §22-23)",
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
      "税收与归宿理论 (Rosen §14)",
      "社会保障与收入再分配 (Rosen §11-12)",
      "财政联邦主义 (Stiglitz §25)",
      "成本收益分析与公共支出评价 (Rosen §8)",
    ],
  },
  'social/international-economics': {
    title: "国际经济学",
    books: [
          "Krugman, Obstfeld & Melitz, \"International Economics: Theory and Policy\" (11th, 2018)"
    ],
    chapters: [
      "比较优势与李嘉图模型 (Krugman §3)",
      "要素禀赋与赫克歇尔-俄林模型 (Krugman §5)",
      "新贸易理论与规模经济 (Krugman §7-8)",
      "贸易政策工具 (Krugman §9-10)",
      "汇率与开放经济的宏观经济学 (Krugman §14-16)",
      "国际货币体系 (Krugman §19-20)",
      "国际要素流动与跨国公司 (Krugman §8)",
      "汇率决定理论与购买力平价 (Krugman §15)",
    ],
  },
  'social/economic-history': {
    title: "经济史",
    books: [
          "Allen, \"Global Economic History: A Very Short Introduction\" (2011)",
          "North, \"Structure and Change in Economic History\" (1981)"
    ],
    chapters: [
      "工业革命与西方起飞 (Allen §3)",
      "经济增长的制度基础 (North §1-2)",
      "产权、国家与交易成本 (North §8)",
      "大分流与全球不平等 (Allen §7)",
      "殖民主义与世界经济 (Allen §5)",
      "制度变迁的经济史分析 (North §13)",
      "亚洲经济史与东亚复兴 (Allen §6)",
      "金融与货币制度演化 (North §9)",
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
      "制度与交易成本 (North §2)",
      "制度变迁理论 (North §8-9)",
      "路径依赖与锁入效应 (North §11)",
      "产权理论 (North §3)",
      "包容性与榨取性制度 (Acemoglu §1)",
      "国家理论 (North §7)",
      "意识形态与制度绩效 (North §12)",
    ],
  },
  'social/financial-engineering': {
    title: "金融工程",
    books: [
          "Hull, \"Options, Futures, and Other Derivatives\" (11th, 2021)"
    ],
    chapters: [
      "期货市场机制与套期保值 (Hull §2-3)",
      "远期与期货定价 (Hull §5)",
      "互换市场 (Hull §7)",
      "期权性质与交易策略 (Hull §10-11)",
      "二叉树模型与Black-Scholes-Merton模型 (Hull §12-14)",
      "希腊字母与风险管理 (Hull §18)",
      "利率衍生品 (Hull §28-29)",
      "信用衍生品与风险价值 VaR (Hull §22-23)",
    ],
  },
  'social/insurance': {
    title: "保险学",
    books: [
          "Vaughan & Vaughan, \"Fundamentals of Risk and Insurance\" (10th, 2008)",
          "Rejda & McNamara, \"Principles of Risk Management and Insurance\" (13th, 2017)"
    ],
    chapters: [
      "风险与风险管理 (Rejda §2)",
      "保险合同原理 (Rejda §3)",
      "人寿保险 (Vaughan §13-14)",
      "财产与责任保险 (Rejda §22)",
      "健康保险与社会保障 (Rejda §15-16)",
      "再保险 (Vaughan §21)",
      "保险监管 (Rejda §3)",
      "精算原理与风险定价 (Rejda §4)",
    ],
  },
  'social/comparative-politics': {
    title: "比较政治学",
    books: [
          "Lijphart, \"Patterns of Democracy\" (2nd, 2012)",
          "Hague, Harrop & McCormick, \"Comparative Government and Politics\" (10th, 2016)"
    ],
    chapters: [
      "民主与威权政体 (Hague §3)",
      "多数民主与共识民主 (Lijphart §2)",
      "选举制度与政党制度 (Hague §8)",
      "总统制与议会制 (Lijphart §7)",
      "政治文化与政治社会化 (Hague §5)",
      "国家建构与制度转型 (Hague §4)",
      "民主化理论（转型学/现代化） (Almond §7)",
      "民族冲突与革命 (Almond §9)",
      "联邦主义 (Almond §6)",
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
      "国际货币体系与汇率政治 (Oatley §8)",
      "全球化与跨国投资 (Frieden §11)",
      "发展与不平等 (Oatley §11)",
      "全球治理与金融危机 (Oatley §10)",
      "国际发展援助与援助政治 (Frieden §17)",
      "全球化反弹与保护主义 (Oatley §12)",
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
      "公民基本权利与义务 (许崇德 §3)",
      "国家机构 (许崇德 §5)",
      "行政法基础理论 (应松年 §1)",
      "行政行为 (应松年 §4)",
      "行政程序与行政救济 (应松年 §6-7)",
      "行政复议与行政诉讼 (应松年 §8)",
      "宪法实施与违宪审查 (许崇德 §7)",
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
      "合同法 (王利明 §5)",
      "侵权责任法 (王利明 §7)",
      "公司法 (范健 §3)",
      "票据法与证券法 (范健 §6-7)",
      "破产法 (范健 §9)",
      "婚姻家庭法与继承法 (王利明 §6)",
      "知识产权法（著作权/专利/商标） (王利明 §7)",
      "海商法 (王利明 §8)",
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
      "正当防卫与紧急避险 (高铭暄 §8)",
      "故意犯罪形态 (张明楷 §10)",
      "共同犯罪 (高铭暄 §11)",
      "刑罚体系与裁量 (高铭暄 §13)",
      "刑罚执行与消灭 (高铭暄 §15)",
      "罪数理论（想象竞合/牵连犯/连续犯） (高铭暄 §10)",
      "侵犯公民人身权利罪 (高铭暄 §14)",
      "侵犯财产罪 (高铭暄 §15)",
      "危害公共安全罪 (高铭暄 §12)",
      "贪污贿赂罪与渎职罪 (高铭暄 §18)",
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
      "国际法主体 (邵津 §3)",
      "国家领土法 (邵津 §4)",
      "海洋法 (邵津 §5)",
      "条约法 (Brownlie §13)",
      "国际责任 (邵津 §8)",
      "国际争端解决 (Brownlie §30)",
      "国际组织与国际责任法 (Brownlie §14)",
    ],
  },
  'social/social-work': {
    title: "社会工作",
    books: [
          "王思斌,《社会工作导论》(第3版, 2021)",
          "Kirst-Ashman & Hull, \"Understanding Generalist Practice\" (8th, 2015)"
    ],
    chapters: [
      "社会工作理论框架 (王思斌 §2)",
      "社会工作价值观与伦理 (Kirst-Ashman §3)",
      "个案工作方法 (王思斌 §5)",
      "小组工作 (Kirst-Ashman §8)",
      "社区工作 (王思斌 §7)",
      "社会政策与社会福利 (王思斌 §9)",
      "儿童与家庭社会工作 (Kirst-Ashman §11)",
      "医务与精神卫生社会工作 (Kirst-Ashman §13)",
    ],
  },
  'social/social-psychology': {
    title: "社会心理学",
    books: [
          "Myers & Twenge, \"Social Psychology\" (13th, 2018)",
          "Aronson, \"The Social Animal\" (12th, 2018)"
    ],
    chapters: [
      "社会心理学导论与方法 (Myers §1-2)",
      "社会信念与判断 (Myers §3)",
      "态度与说服 (Aronson §7)",
      "从众与服从 (Myers §6)",
      "群体影响 (Myers §8)",
      "偏见与刻板印象 (Myers §9)",
      "攻击、吸引与亲密 (Aronson §8-9)",
      "自我与同一性 (Myers §3)",
      "利他行为与亲社会行为 (Myers §11)",
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
      "差序格局 (费孝通 §4)",
      "礼治秩序 (费孝通 §8)",
      "血缘与地缘 (费孝通 §11)",
      "从传统到现代的农村变迁 (Fei §9)",
      "城乡关系与小城镇问题 (Fei 附录)",
      "农村社会结构与家族 (费孝通 §6)",
      "农民工与城乡流动 (Fei §10)",
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
      "生物与心理犯罪理论 (Vold §6-7)",
      "社会结构与犯罪 (Siegel §7)",
      "犯罪类型:暴力、财产与白领犯罪 (Siegel §10-13)",
      "犯罪预防与环境犯罪学 (Siegel §15)",
      "受害者学 (Vold §15)",
      "社会过程理论（标签/社会学习/社会控制） (Siegel §8)",
      "社会冲突理论（马克思主义/左翼现实主义） (Siegel §9)",
      "刑罚学与矫正 (Siegel §14)",
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
      "议程设置与政策制定 (Anderson §4-5)",
      "政策执行 (Anderson §6)",
      "政策评估 (Anderson §7)",
      "政策工具与设计 (Dye §3)",
      "经济政策与管制政策 (Dye §9-10)",
      "比较公共政策 (Dye §16)",
      "社会福利政策 (Dye §15)",
    ],
  },
  'social/public-administration': {
    title: "行政管理",
    books: [
          "Rosenbloom, Kravchuk & Clerkin, \"Public Administration: Understanding Management, Politics, and Law in the Public Sector\" (8th, 2015)"
    ],
    chapters: [
      "公共行政的三大途径 (Rosenbloom §1)",
      "组织理论 (Rosenbloom §4)",
      "人事行政与公务员制度 (Rosenbloom §5)",
      "预算与财务行政 (Rosenbloom §7)",
      "行政法与公共利益 (Rosenbloom §10)",
      "行政伦理与问责 (Rosenbloom §12)",
      "新公共管理（NPM） (Rosenbloom §6)",
      "电子政务 (Rosenbloom §10)",
      "绩效管理 (Rosenbloom §9)",
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
      "人力资源规划与工作分析 (Dessler §3-4)",
      "招聘与甄选 (Dessler §6-7)",
      "培训与开发 (Noe §7)",
      "绩效管理与评估 (Dessler §10)",
      "薪酬管理 (Dessler §11)",
      "员工关系与职业安全 (Noe §15)",
      "国际人力资源管理 (Dessler §15)",
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
      "旅游动机与旅游者行为 (Cooper §5)",
      "旅游目的地开发 (Cooper §9)",
      "旅游营销 (Goeldner §14)",
      "旅游的经济、社会文化与环境影响 (Cooper §13)",
      "旅游业的未来趋势 (Goeldner §17)",
      "酒店与接待业管理 (Cooper §11)",
      "可持续旅游与生态旅游 (Cooper §15)",
    ],
  },
  'social/developmental-psychology': {
    title: "发展心理学",
    books: [
          "Berk, \"Development Through the Lifespan\" (7th, 2017)",
          "Santrock, \"Life-Span Development\" (17th, 2019)"
    ],
    chapters: [
      "发展理论与研究方法 (Berk §1-2)",
      "婴儿期的生理与认知发展 (Berk §4)",
      "儿童期认知与语言发展 (Berk §5-6)",
      "青少年期的同一性与社会化 (Berk §11)",
      "成年早期与中期的心理发展 (Berk §13-15)",
      "老年期与生命终结 (Berk §17)",
      "道德发展（Piaget/Kohlberg） (Berk §12)",
      "性别发展 (Berk §13)",
      "非典型发展与心理障碍 (Berk §14)",
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
      "心境障碍与自杀 (Barlow §7)",
      "精神分裂症与其他精神病性障碍 (Kring §11)",
      "人格障碍 (Barlow §12)",
      "心理治疗与干预 (Kring §15)",
      "物质使用障碍 (Barlow §11)",
      "进食障碍 (Barlow §10)",
      "认知行为治疗（CBT）技术 (Barlow §15)",
    ],
  },
  'social/educational-psychology': {
    title: "教育心理学",
    books: [
          "Woolfolk, \"Educational Psychology\" (14th, 2019)",
          "Slavin, \"Educational Psychology: Theory and Practice\" (12th, 2018)"
    ],
    chapters: [
      "认知发展与语言发展 (Woolfolk §2-3)",
      "学习理论:行为主义与社会认知 (Slavin §6)",
      "认知与建构主义学习观 (Woolfolk §8-9)",
      "学习动机 (Slavin §10)",
      "课堂管理与教学策略 (Woolfolk §13)",
      "教学评估与标准化测试 (Slavin §14)",
      "学习者差异与特殊教育 (Woolfolk §4)",
      "教师心理与专业发展 (Slavin §13)",
    ],
  },
  'social/industrial-organizational-psychology': {
    title: "工业与组织心理学",
    books: [
          "Muchinsky & Culbertson, \"Psychology Applied to Work\" (11th, 2017)",
          "Spector, \"Industrial and Organizational Psychology: Research and Practice\" (7th, 2017)"
    ],
    chapters: [
      "工作分析与胜任力 (Muchinsky §3)",
      "人员选拔与效度验证 (Spector §4)",
      "绩效评估 (Muchinsky §5)",
      "组织理论与组织行为 (Spector §10)",
      "工作动机与工作设计 (Muchinsky §8)",
      "领导力与管理 (Muchinsky §10)",
      "工作压力与职业健康 (Spector §12)",
      "组织发展与变革管理 (Muchinsky §12)",
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
      "世界文学的流通与翻译 (Damrosch §2-3)",
      "跨文化与跨学科研究 (Damrosch §10)",
      "比较文学的中国学派 (Damrosch §11)",
      "翻译研究的转向 (Bassnett §8)",
      "后殖民文学与世界文学 (Damrosch §6)",
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
      "结构主义与符号学 (Eagleton §3)",
      "后结构主义与解构 (Eagleton §4)",
      "精神分析批评 (Abrams §Psychoanalytic)",
      "女性主义批评 (Eagleton §6)",
      "新历史主义与文化批评 (Eagleton §7)",
      "文学批评的基本概念 (Abrams §1)",
      "后殖民主义批评（Said/Spivak） (Eagleton §6)",
      "接受美学与读者反应批评 (Eagleton §7)",
      "马克思主义文学批评 (Eagleton §4)",
      "后现代主义批评 (Eagleton §8)",
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
      "古籍数字化与电子文献 (黄永年 §6)",
    ],
  },
  'humanities/historical-geography': {
    title: "历史地理学",
    books: [
          "侯仁之,《历史地理学四论》(2007)",
          "谭其骧,《中国历史地图集》(1996)"
    ],
    chapters: [
      "历史地理学理论与方法 (侯仁之 §1)",
      "历史气候变迁 (侯仁之 §2)",
      "历史人口地理 (侯仁之 §3)",
      "历史交通地理 (侯仁之 §4)",
      "政区沿革与疆域变迁 (谭其骧 图集说明)",
      "历史城市地理 (侯仁之 §附录)",
      "历史经济地理 (侯仁之 §5)",
      "历史军事地理 (谭其骧 图集附录)",
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
      "类书、丛书与工具书 (张舜徽 §7)",
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
      "纸质与纺织品文物保护 (王蕙贞 §6)",
      "金属文物保护 (王蕙贞 §3)",
      "壁画与彩塑保护 (王蕙贞 §8)",
      "木质与漆器文物保护 (王蕙贞 §7)",
      "考古现场文物保护 (李晓东 §8)",
    ],
  },
  'humanities/archaeometry': {
    title: "科技考古",
    books: [
          "陈铁梅,《科技考古》(2008)",
          "Renfrew & Bahn, \"Archaeology: Theories, Methods, and Practice\" (8th, 2020)"
    ],
    chapters: [
      "考古测年:碳十四与释光测年 (陈铁梅 §3)",
      "文物成分分析与产地溯源 (陈铁梅 §5)",
      "稳定同位素与食谱分析 (陈铁梅 §7)",
      "动物考古 (Renfrew §10)",
      "植物考古与农业起源 (Renfrew §9)",
      "遥感与GIS考古 (Renfrew §3)",
      "古DNA与分子考古 (Renfrew §11)",
      "冶金考古与金属器物研究 (陈铁梅 §9)",
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
      "甲骨文 (高明 §2)",
      "金文 (高明 §3)",
      "战国文字 (裘锡圭 §4)",
      "简帛文字 (裘锡圭 §5)",
      "古文字的考释方法 (裘锡圭 §8)",
      "秦汉文字与小篆 (裘锡圭 §6)",
      "隶变与汉字演变 (高明 §5)",
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
      "气候危机与社会崩溃 (Brooke §14)",
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
      "海外移民与华人华侨 (葛剑雄 移民史 §7)",
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
      "东北边疆与满洲 (Lattimore §7)",
      "西南边疆与改土归流 (马大正 §6)",
      "海疆与近代边疆危机 (马大正 §9)",
      "北疆与中俄边界 (马大正 §7)",
      "边疆民族政策与朝贡体系 (Lattimore §9)",
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
      "稻作农业与江南开发 (李伯重 §3)",
      "水利与农业地理格局 (Chi §4)",
      "农业技术变迁与人口压力 (李伯重 §5)",
      "农业商品化与市镇经济 (李伯重 §7)",
      "经济作物与商品农业 (李伯重 §6)",
      "灾荒与农业波动 (李伯重 §8)",
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
      "历史的叙述与话语 (Jenkins §4)",
      "中国史学的发展历程 (白寿彝 §1)",
      "《史记》与传统史学体例 (白寿彝 §3)",
      "近代新史学的兴起 (白寿彝 §6)",
      "西方史学流派演变 (Jenkins §5)",
      "年鉴学派（Annales） (Jenkins §4)",
      "后现代史学（怀特/安克斯密特） (Jenkins §6)",
      "马克思主义史学 (Jenkins §3)",
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
      "宋明理学 (葛兆光 §4)",
      "明清之际与近代思想转型 (侯外庐 §5)",
      "清代考据学与实学 (侯外庐 §6)",
      "近现代思想转型 (葛兆光 §6)",
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
      "20世纪思想与现代性批判 (Skinner §6)",
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
      "中国美术史 (Gombrich §10)",
      "摄影与新媒体艺术 (Kleiner §29)",
    ],
  },
  'humanities/sculpture': {
    title: "雕塑",
    books: [
          "Boardman, \"Greek Sculpture: The Classical Period\" (1985)",
          "孙振华,《中国雕塑史》(2014)"
    ],
    chapters: [
      "雕塑的语言与基本原理 (孙振华 §1)",
      "古希腊雕塑 (Boardman §2)",
      "中国古代雕塑:陵墓与宗教 (孙振华 §3)",
      "文艺复兴雕塑 (孙振华 §5)",
      "现代雕塑与公共艺术 (孙振华 §7)",
      "当代雕塑的媒介拓展 (孙振华 §9)",
      "中世纪雕塑与教堂艺术 (Boardman §5)",
      "雕塑材料与铸造工艺 (孙振华 §10)",
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
      "动画原理与叙事 (Wells §3)",
      "数字媒体的本体论 (Manovich §1)",
      "界面与交互美学 (Manovich §2)",
      "数字叙事与数据库逻辑 (Manovich §5)",
      "动画与新媒介融合 (Wells §8)",
      "3D 动画与计算机图形学 (Wells §6)",
      "虚拟现实与沉浸式媒介 (Manovich §8)",
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
      "艺术批评与诠释 (Carroll §6)",
      "艺术、情感与表现 (Carroll §5)",
      "艺术与社会功能 (Danto §7)",
      "艺术体制理论与历史 (Danto §5)",
      "非西方艺术理论 (Carroll §10)",
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
      "认知电影理论 (Bordwell §7)",
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
      "藏品管理与编目 (Macdonald §5)",
      "博物馆教育与公共阐释 (Macdonald §11)",
      "展览策划与诠释 (Macdonald §8)",
      "物质文化遗产保护原则 (ICOMOS Venice Charter)",
      "社区参与与遗产伦理 (Macdonald §17)",
      "数字博物馆与虚拟展览 (Macdonald §20)",
      "遗产地管理与旅游 (Macdonald §15)",
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
      "口头传统与表现形式 (UNESCO §2.1)",
      "表演艺术 (UNESCO §2.2)",
      "社会实践、仪式与节庆 (UNESCO §2.3)",
      "传统手工艺 (UNESCO §2.5)",
      "非遗的保护、传承与伦理 (Kurin §4)",
      "有关自然和宇宙的知识与实践 (UNESCO §2.4)",
      "非遗档案化与数字化保护 (Kurin §6)",
    ],
  },
  'humanities/philosophy-overview': {
    title: "哲学通史",
    books: [
          "Russell, \"A History of Western Philosophy\" (1945)",
          "Copleston, \"A History of Philosophy\" (9 vols, 1946-1975)"
    ],
    chapters: [
      "古希腊哲学:苏格拉底、柏拉图、亚里士多德 (Russell §古代)",
      "中世纪哲学:教父与经院哲学 (Copleston 卷2)",
      "近代哲学:理性主义与经验主义 (Russell §近代)",
      "启蒙运动与德国古典哲学 (Copleston 卷6)",
      "19世纪哲学:黑格尔、马克思、尼采 (Russell §近代)",
      "20世纪哲学:分析哲学与大陆哲学 (Copleston 卷8-9)",
      "中国与印度哲学概览 (Copleston 卷1东方)",
      "伊斯兰哲学黄金时代 (Copleston 卷2)",
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
      "共相与殊相 (Loux §2)",
      "实体与属性 (Aristotle 卷Z)",
      "因果性与自然律 (Loux §7)",
      "自由意志与决定论 (Loux §9)",
      "必然性与可能性 (Loux §5)",
      "时间与持存的同一性 (Loux §4)",
      "空间与时间哲学 (Loux §8)",
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
      "信念与真理理论 (Audi §7)",
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
      "道德心理学与道德情感 (Frankena §5)",
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
      "艺术的定义 (Carroll §1)",
      "艺术本体论与艺术作品 (Carroll §3)",
      "艺术批评与诠释 (Beardsley §9)",
      "悲剧理论与崇高 (Beardsley §7)",
      "艺术与道德 (Carroll §7)",
      "康德判断力批判 (Scruton §3)",
      "环境美学 (Scruton §8)",
      "后现代美学 (Scruton §9)",
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
      "宋明理学:程朱陆王 (冯友兰 §8)",
      "中国佛学 (冯友兰 §7)",
      "法家思想（韩非子） (冯友兰 §法家)",
      "道教哲学（道家宗教化） (冯友兰 §道教)",
      "现代新儒家（牟宗三/唐君毅） (冯友兰 §近现代)",
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
      "笛卡尔与大陆理性主义 (Russell §近代卷1)",
      "洛克、休谟与英国经验主义 (Russell §近代卷2)",
      "康德与德国唯心论 (Copleston 卷6)",
      "19世纪：叔本华、尼采与克尔凯郭尔 (Russell §近代卷3)",
      "现象学与存在主义 (Copleston 卷8)",
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
      "当代印度哲学（甘地/奥罗宾多） (Radhakrishnan 卷2 §附录)",
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
      "伊本·鲁世德与亚里士多德主义 (Fakhry §6)",
      "照明学派与苏菲主义 (Nasr §3)",
      "伊斯兰哲学的现代发展 (Fakhry §11)",
      "伊斯玛仪派哲学 (Fakhry §5)",
      "伊斯兰伦理与政治哲学 (Nasr §4)",
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
      "他心问题 (Chalmers §7)",
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
      "比较伦理学与跨文化价值 (Larson §6)",
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
      "数学实践与数学哲学 (Shapiro §10)",
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
      "量子力学的解释 (Sklar §5)",
      "决定论与混沌 (Sklar §4)",
      "统计力学与时间箭头 (Albert §2)",
      "量子非定域性与纠缠 (Albert §4)",
      "物理定律与自然类 (Sklar §7)",
      "热力学第二定律与时间 (Albert §3)",
      "量子测量与诠释比较 (Sklar §6)",
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
      "技术决定论与社会建构论（SCOT） (Ihde §6)",
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
      "法律推理与判决 (Dworkin §1)",
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
      "教育认识论 (Noddings §6)",
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
      "后现代历史哲学（怀特） (Dray §9)",
    ],
  },
  'humanities/applied-ethics': {
    title: "应用伦理学（科技/生命/环境）",
    books: [
          "Beauchamp & Childress, \"Principles of Biomedical Ethics\" (8th, 2019)",
          "Singer, \"Practical Ethics\" (3rd, 2011)"
    ],
    chapters: [
      "生命伦理四原则 (Beauchamp §1)",
      "生命起始与生殖伦理 (Beauchamp §6)",
      "死亡与临终决策 (Beauchamp §7)",
      "环境伦理与动物权利 (Singer §3-5)",
      "科技伦理与责任 (Singer §10)",
      "人工智能伦理 (Singer 附录)",
      "商业与企业伦理 (Singer §6)",
      "工程与计算机伦理 (Singer §11)",
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
      "道德心理学与道德判断 (Smith §7)",
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
      "此在与在世存在 (Heidegger §1-2)",
      "本真性与向死存在 (Heidegger §6)",
      "自在与自为 (Sartre §1)",
      "他者与凝视 (Sartre §3)",
      "时间意识与内时间意识 (Husserl §5)",
      "身体现象学：梅洛-庞蒂 (Heidegger §8)",
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
      "解构主义与德里达 (Lyotard §10)",
      "权力与知识 (Foucault §4)",
      "话语分析与陈述形成 (Foucault §2)",
      "系谱学与权力谱系 (Foucault §附录)",
      "差异与延异 (Lyotard §附录)",
      "德勒兹差异哲学 (Copleston §当代)",
      "鲍德里亚拟像理论 (Copleston §当代)",
      "哈贝马斯现代性批判 (Copleston §当代)",
    ],
  },
  'humanities/analytic-continental-philosophy': {
    title: "分析哲学与大陆哲学",
    books: [
          "Soames, \"Philosophical Analysis in the Twentieth Century\" (2 vols, 2003)",
          "Critchley & Schroeder, \"A Companion to Continental Philosophy\" (1998)"
    ],
    chapters: [
      "分析哲学的起源 (Soames 卷1 §1)",
      "逻辑实证主义与语言转向 (Soames 卷1 §4)",
      "日常语言哲学 (Soames 卷2 §3)",
      "现象学传统 (Critchley §2)",
      "存在主义与解释学 (Critchley §3)",
      "两大传统的对话与融合 (Critchley §导论)",
      "维特根斯坦前后期哲学 (Russell §晚期)",
      "弗雷格与罗素（分析哲学源头） (Russell §逻辑)",
      "后结构主义（Derrida/Deleuze） (Copleston §后现代)",
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
      "行动知识与道义学 (Anscombe §7)",
    ],
  },
}
