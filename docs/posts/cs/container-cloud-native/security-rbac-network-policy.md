---
title: 安全（RBAC/NetworkPolicy/Secrets）
date: 2026-08-07
---

# 安全（RBAC/NetworkPolicy/Secrets）

<div class="epigraph">
<p>安全不是一堆独立开关，而是一层层递进的「不信任」——从认证，到授权，到网络，到数据本身。</p>
<footer>—— 意译自 Brendan Burns（《Kubernetes: Up &amp; Running》）</footer>
</div>

<div class="article-byline">
<p>第三级 · 容器与云原生 ｜ Burns Ch.14,19 ｜ 2026-08-07</p>
</div>

## 为什么安全是一条「纵深」而非一道墙

回顾前面的课程，Kubernetes 的默认姿态其实是「宽松」的：扁平网络默认全通、容器默认能访问 API Server、Secret 默认只是 base64。安全课的任务，就是把这套宽松默认**逐层收紧**——这就是**纵深防御（defense in depth）**：不指望任何单点拦住攻击，而是让每一层都设卡。我们从「谁能用集群」问到「数据怎么保存」，正好走完一整条信任链。

## 1 认证与授权：API 的两道门

任何请求到 API Server，先过两道门：

- **认证（Authentication）**：你是谁？——校验客户端证书（X.509）、token（ServiceAccount）、OIDC 等。
- **授权（Authorization）**：你允许干什么？——默认机制是 **RBAC（Role-Based Access Control，基于角色的访问控制）**。

**RBAC 的四个对象**：

- **Role**：命名空间内的权限集合（如「能 get/list Pods」）。
- **ClusterRole**：集群级权限（可授权任何资源，包括节点、PV）。
- **RoleBinding**：把 Role/ClusterRole 绑给用户/组/ServiceAccount（限定在命名空间内）。
- **ClusterRoleBinding**：集群级绑定。

**核心概念：最小权限原则（least privilege）**：每个身份只拿完成工作所需的最小权限。<span class="marginnote">RBAC 的哲学是「默认拒绝，显式放行」——一个全新的 ServiceAccount 什么都做不了，直到你给它绑定 Role。这与默认全通的网络模型恰好相反，也提醒我们：安全必须显式构建。</span>

一个最小权限的 ServiceAccount（只让它的 Pod 读本命名空间 ConfigMap）：

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata: { namespace: app, name: config-reader }
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list"]
---
kind: RoleBinding
metadata: { namespace: app, name: read-config }
roleRef: { kind: Role, name: config-reader, apiGroup: rbac.authorization.k8s.io }
subjects:
  - { kind: ServiceAccount, name: app-sa, namespace: app }
```

## 2 NetworkPolicy：给扁平网络装「门禁」

上一节讲过，网络模型默认「所有 Pod 互通」。**NetworkPolicy** 是叠加其上的**策略层**：按标签选择一组 Pod，声明「允许谁访问它们的哪些端口」。默认策略是「没有策略 = 全放行」，一旦定义策略，未匹配的流量被拒绝（常见的全拒收底策略是显式声明空 ingress）。<span class="marginnote">NetworkPolicy 是对网络模型的刻意收紧：模型保证「可达」，策略定义「谁和谁可达」。Calico、Cilium 等 CNI 都支持它；而纯 flannel 默认不支持——选 CNI 时就要确认 NetworkPolicy 支持度。</span>

```yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: db-allow-api
  namespace: app
spec:
  podSelector:
    matchLabels: { app: database }
  policyTypes: [Ingress]
  ingress:
    - from:
        - podSelector:
            matchLabels: { app: api }
      ports:
        - protocol: TCP
          port: 5432
```

这段策略说：只有带 `app: api` 标签的 Pod，才能访问带 `app: database` 标签的 Pod 的 5432 端口。**纵深防御在这里落地：即使一个 Pod 被攻破，它也够不到其他命名空间/服务。**

## 3 Secrets：数据保护的第一道也是最后一道

**核心概念：Secret**：保存敏感数据的对象（密码、token、TLS 私钥），以 volume 或环境变量形式注入容器。

**辨析｜易错点（安全课最重要的辨析）**：**Secret 的默认存储只是 base64 编码，不是加密**。

- `apiVersion: v1 / kind: Secret` 里 `data` 的值是 base64——任何能读集群的人 `kubectl get secret -o yaml` 就能解码拿到明文。
- base64 的唯一作用是**让非文本数据可嵌入 YAML**，与保密无关。
- 生产必须做到：(1) 启用 **etcd 加密**（`--encryption-provider-config`，AES-CBC 等）；(2) 启用 **KMS 外部密钥管理**；(3) 对 Secret 做 **RBAC 收紧**——「谁 read Secret」是最敏感的权限；(4) 用外部 Secret 管理工具（Vault、External Secrets Operator）注入，让明文不落 etcd。

**第二个易错点**：把敏感信息放进 **ConfigMap** 或**环境变量明文**——两者都不加密。**ConfigMap 只能装非敏感配置，Secret 才是敏感数据的容器**（详见《Helm 与配置管理》）。

## 4 公式解析：权限决策

RBAC 的授权判定是一套交集逻辑，可以写成：

$$
\text{Allow}(u, r, v) \iff \exists\ \text{Binding}(u, R), \ \text{Rule}(R, r, v)
$$

- $u$：请求者（user / ServiceAccount）。
- $r$：资源类型（如 `pods`）。
- $v$：操作动词（如 `get`、`create`）。
- $\text{Binding}(u, R)$：存在一个把角色 $R$ 绑给 $u$ 的 Binding。
- $\text{Rule}(R, r, v)$：角色 $R$ 里有一条对资源 $r$ 放行动词 $v$ 的规则。

三步拆解：

- **第一步，权限沿「角色」间接授予**：$u$ 不直接持有权限，而是「被绑定了某个角色」。这层间接让权限管理可以**批量化**（按角色管理，而非按人管理）——团队换人只需改绑定。
- **第二步，规则是「资源 × 动词」的格子**：Rule 声明「哪些资源上的哪些动作」。最小权限原则就是尽量只填必要的格子。
- **第三步，判定是存在性检查**：只要存在一条路径能证明放行，就放行；找不到就拒绝（默认拒绝）。**这保证「少配了权限 = 无权限」是安全的失败方向。**

## 5 纵深防御的完整拼图

把本专题讲过的安全点串成一条链：

1. **认证/授权**：谁能用 API（RBAC、ServiceAccount）。
2. **准入**：创建资源时的准入控制（如 **Pod Security Standards**，限制特权容器、hostPath、宿主命名空间）。
3. **网络**：NetworkPolicy 限制东西向流量。
4. **入口**：Ingress 上的 TLS、认证（前面 Ingress 课的 TLS 终结）。
5. **容器运行时**：容器能力裁剪、只读根文件系统、非 root 运行（《容器原理》一节）。
6. **数据**：Secret 加密、KMS、外部密钥管理。

**安全不是一个开关，而是一组同时成立的默认值的改写**——每层都关掉一个「默认宽松」。

## 6 小结

- 请求先过**认证**（你是谁），再过**授权**（你能干什么）；RBAC 是授权的默认机制。
- RBAC 四件套：**Role / ClusterRole / RoleBinding / ClusterRoleBinding**，遵循最小权限原则，默认拒绝。
- **NetworkPolicy** 给扁平网络加「门禁」，按标签声明谁可访问谁；没有策略 = 全放行。
- **Secret 默认只是 base64，不是加密**——必须 etcd 加密 + KMS + RBAC 收紧。
- 纵深防御：认证 → 授权 → 准入 → 网络 → 入口 → 运行时 → 数据，层层设卡。

在下一节，我们处理「配置与发布」这个安全课的相邻问题——进入 **Helm 与配置管理（ConfigMap/Secret）**。
