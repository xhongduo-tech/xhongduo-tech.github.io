---
pageClass: plain-doc
---

# 存储与文件系统

对标权威教材体系，按章节逐节写成博文。学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- Silberschatz, Galvin & Gagne, "Operating System Concepts" (10th, 2018)
- Andrew S. Tanenbaum & Herbert Bos, "Modern Operating Systems" (4th, 2014)
- Remzi H. Arpaci-Dusseau & Andrea C. Arpaci-Dusseau, "Operating Systems: Three Easy Pieces" (2018)

## 主题规划

<ProgressGrid cat="cs/storage-file-systems" />

### 第1篇

- [ ] 文件系统接口与目录 (Silberschatz §13)
- [ ] 文件系统实现与 inode (Silberschatz §14)
- [ ] 日志结构与日志文件系统 (OSTEP §40)
- [ ] 崩溃一致性与日志文件系统 journaling (OSTEP §42)
- [ ] 数据完整性保护（校验和/静默损坏） (OSTEP §45)
- [ ] 磁盘调度与 RAID (Silberschatz §12)
- [ ] 虚拟文件系统 VFS (Tanenbaum §4.5)
- [ ] 分布式文件系统 NFS (Tanenbaum §10.4)

### 第2篇

- [ ] 块存储与闪存 SSD (OSTEP §44)
- [ ] 闪存与 FTL (Tanenbaum §4.9)
- [ ] NVMe 与新型存储介质 (Silberschatz §12)
- [ ] 分布式文件系统（GFS/HDFS/Ceph） (Tanenbaum §11)
- [ ] 一致性模型（CAP/线性一致性） (Tanenbaum §7)
- [ ] 对象存储（S3/Ceph RGW） (Tanenbaum §12)
