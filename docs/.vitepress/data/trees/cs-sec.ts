import type { Outline } from './schema'

/** 网络安全：假设与密码 → 系统/网络攻击面。不写 exploit PoC。 */
export const csSec: Outline = [
  '网络安全',
  [
    [
      '密码与认证',
      [
        [
          '假设',
          [
            'CIA 三元组|cia-triad',
            '威胁模型|threat-model',
            'STRIDE|stride',
            'Dolev–Yao|dolev-yao',
            '对称加密够用的那一层|symmetric-crypto',
            '分组模式 CBC / GCM|block-modes',
            '公钥与信封|pubkey-envelope',
            'DH 与 RSA 分工|dh-vs-rsa',
            '哈希与碰撞|hash-collision',
            'MAC 与签名|mac-signature',
            '证书与 PKI|cert-pki',
            '吊销与 OCSP|cert-revoke',
            '口令与加盐|password-salt',
            'KDF 与慢哈希|kdf-slow-hash',
            '认证协议|auth-protocol',
            'Kerberos 直觉|kerberos',
            'OAuth / OIDC 入口|oauth-oidc',
          ],
        ],
      ],
    ],
    [
      '系统与网络攻击面',
      [
        [
          '机制',
          [
            '访问控制与最小特权|acl-least-priv',
            'DAC / MAC / RBAC|dac-mac-rbac',
            '能力对 ACL|capability-vs-acl',
            'TOCTOU|toctou',
            '缓冲区溢出直觉|buffer-overflow',
            'ASLR 与 NX|aslr-nx',
            'CFI 直觉|cfi',
            '隔离与沙箱|isolation-sandbox',
            '侧信道直觉|side-channel',
            '瞬态执行对照|transient-exec',
            'TLS 握手|tls-handshake',
            'TLS 1.3 与 0-RTT|tls13-0rtt',
            '防火墙与状态|firewall-state',
            '中间人|mitm',
            'HSTS 与钉扎对照|hsts',
            'DoS 与放大|dos-amplify',
            '同源与 CSRF|same-origin-csrf',
            '注入作为输入边界|injection-boundary',
            '计算栈到此为止|to-systems-boundary',
          ],
        ],
      ],
    ],
  ],
]
