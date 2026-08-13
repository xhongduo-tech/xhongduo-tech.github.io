---
pageClass: plain-doc
---

# 家用 NAS 与私有云搭建

从选购硬件到落地私有云，系统掌握 NAS 机型的选型逻辑、硬盘阵列与存储规划、系统部署与基础服务、远程访问与网络安全，最终建立一套可靠的数据备份与灾难恢复方案。学完这些章节，就能独立搭建并长期运维自己的家用数据中心。

## 对标教材

- 《NAS 搭建与私有云》
- 群晖/威联通官方文档
- 《家庭数据中心搭建》

## 主题规划

<ProgressGrid cat="skills/jia-yong-nas" />

### 第1篇 NAS 选购与硬件基础

- [x] [NAS 与私有云的定位与价值](./nas-position-value)
- [x] [NAS 机型选购：成品 vs DIY](./nas-off-the-shelf-vs-diy)
- [x] [核心硬件：CPU、内存、网口与盘位](./nas-core-hardware)
- [x] [硬盘选购：机械硬盘 vs SSD、家用盘 vs 企业盘](./nas-hdd-selection)
- [x] [扩展设备：阵列卡、UPS 与网络基础设施](./nas-extension-devices)

### 第2篇 硬盘阵列与存储规划

- [x] [RAID 基础概念与单盘直通](./raid-basics-and-single-disk)
- [x] [RAID 0/1/5/6/10 对比与选型](./raid-0-1-5-6-10-comparison)
- [x] [Synology Hybrid RAID 与卷组](./synology-hybrid-raid)
- [x] [存储池、卷与文件系统规划](./storage-pool-volume-filesystem)
- [x] [硬盘健康监控与 SMART](./disk-health-monitoring-smart)
- [x] [坏道检测与硬盘更换流程](./bad-sector-detection-disk-replacement)

### 第3篇 系统部署与基础服务

- [x] [操作系统选择：DSM、QTS 与 TrueNAS](./nas-os-selection)
- [x] [系统安装与初始化配置](./nas-system-installation)
- [x] [用户、权限与共享文件夹](./nas-users-permissions-shares)
- [x] [存储池与卷的创建管理（威联通）](./qnap-storage-pool-volume-management)
- [x] [文件服务：SMB/NFS/AFP 与挂载](./file-services-smb-nfs-afp)
- [x] [Docker 基础与容器化应用部署](./docker-basics-container-deployment)

### 第4篇 远程访问与网络安全

- [x] [内网穿透与 DDNS 基础](./nat-traversal-ddns-basics)
- [x] [QuickConnect 与零配置远程访问](./synology-quickconnect)
- [x] [路由器端口转发与 UPnP](./router-port-forwarding-upnp)
- [x] [公网 IP、IPv6 与异地组网](./public-ip-ipv6-remote-networking)
- [x] [SSL 证书与 HTTPS 加密访问](./ssl-certificates-https)
- [x] [网络安全基线：防火墙与防暴力破解](./network-security-baseline)

### 第5篇 私有云与数据备份

- [x] [私有云应用：相册、影音、笔记与网盘](./private-cloud-apps)
- [x] [多媒体管理：Plex/Jellyfin/Emby 影音库](./media-management-plex-jellyfin-emby)
- [x] [数据备份策略：3-2-1 原则](./backup-strategy-3-2-1)
- [x] [本地备份：Hyper Backup 与 Rsync](./local-backup-hyper-backup-rsync)
- [x] [云备份与异地备份方案](./cloud-offsite-backup)
- [x] [快照与版本回滚](./snapshots-version-rollback)
- [x] [灾难恢复与应急预案](./disaster-recovery-emergency-plan)
