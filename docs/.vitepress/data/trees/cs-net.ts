import type { Outline } from './schema'

/** 计算机网络：链路 → 网络 → 传输 → 应用。 */
export const csNet: Outline = [
  '计算机网络',
  [
    [
      '链路与互联',
      [
        [
          '从共享介质到转发',
          [
            '分层与端到端|layering-e2e',
            '五层与四层对照|five-vs-four-layer',
            '帧与 MAC|frame-mac',
            'CRC 在帧上|frame-crc',
            'CSMA 与交换|csma-switch',
            'CSMA/CA 无线对照|csma-ca',
            'VLAN|vlan',
            '生成树|spanning-tree',
            'ARP|arp',
            'ARP 缓存与代理|arp-cache',
            'IP 编址与子网|ip-subnet',
            'CIDR 与聚合|cidr-agg',
            '最长前缀匹配|lpm',
            'IP 分片|ip-fragment',
            'ICMP|icmp',
            'DHCP 与 NAT|dhcp-nat',
            'NAPT 与端口|napt',
            'IPv6 对照|ipv6-contrast',
            'IPv6 地址与 NDP|ipv6-ndp',
          ],
        ],
      ],
    ],
    [
      '路由与传输',
      [
        [
          '路径与可靠',
          [
            '距离向量|distance-vector',
            '计数到无穷|count-to-infinity',
            '链路状态|link-state',
            'Dijkstra 在 OSPF|ospf-dijkstra',
            'BGP 直觉|bgp-intuition',
            'AS 路径与策略|bgp-policy',
            'UDP|udp',
            'UDP 校验与端口|udp-checksum',
            'TCP 三次握手|tcp-handshake',
            '四次挥手与 TIME_WAIT|tcp-time-wait',
            '序号与重传|tcp-seq-rexmit',
            'RTO 与 Karn|tcp-rto',
            'SACK|tcp-sack',
            '流控与窗口|tcp-window',
            'Nagle 与延迟 ACK|nagle-delayed-ack',
            '拥塞控制|tcp-congestion',
            'Reno 与 Cubic|reno-cubic',
            'BBR 对照|bbr',
            'QUIC 对照|quic-contrast',
          ],
        ],
      ],
    ],
    [
      '应用',
      [
        [
          '名字与内容',
          [
            'DNS|dns',
            '递归与迭代|dns-recursive',
            'DNS 记录类型|dns-rr',
            'HTTP|http',
            '方法、幂等与缓存头|http-semantics',
            'HTTP/2 多路|http2',
            'TLS 入口|http-tls-entry',
            'CDN 直觉|cdn-intuition',
            '任播与边缘|anycast-edge',
            '套接字 API|socket-api',
            '阻塞与非阻塞套接字|socket-nonblock',
          ],
        ],
      ],
    ],
  ],
]
