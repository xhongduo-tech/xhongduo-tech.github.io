---
pageClass: plain-doc
---

# Linux 命令行与 Shell 脚本

Linux 命令行是驾驭服务器、开发与运维的必修基本功，Shell 脚本则是把零散命令沉淀为可复用工具的关键能力。对标《鸟哥的Linux私房菜 基础学习篇》与《Linux 命令行与 Shell 脚本编程大全》，按「命令行基础 → 系统管理命令 → Shell 脚本基础 → 高级脚本与文本处理」的路径逐节写成博文，学完一个学科 = 写完该学科权威教材对应的全部博文。

## 对标教材

- 《鸟哥的Linux私房菜 基础学习篇（第四版）》鸟哥（郦伟鹏）著，人民邮电出版社
- 《Linux 命令行与 Shell 脚本编程大全（第3版）》（Linux Command Line and Shell Scripting Bible）Richard Blum、Christine Bresnahan 著，人民邮电出版社

## 主题规划

<ProgressGrid cat="cs/linux-shell" />

### 第1篇 命令行基础

- [x] [文件与目录管理：ls/cd/cp/mv/rm](./file-directory-management)（鸟哥《Linux私房菜》第6章）
- [x] [文件权限与目录配置：权限位、umask 与默认权限](./file-permissions-umask)（鸟哥《Linux私房菜》第5章）
- [x] [查找与过滤：find、grep、sort、uniq 与管道](./find-filter-pipeline)（鸟哥《Linux私房菜》第10章）
- [x] [vim 程序编辑器与文本处理](./vim-editor)（鸟哥《Linux私房菜》第9章）
- [x] [认识 BASH：Shell 类型、命令别名与历史命令](./bash-shell-intro)（鸟哥《Linux私房菜》第10章）
- [x] [正则表达式与文件格式化处理](./regex-file-formatting)（鸟哥《Linux私房菜》第11章）

### 第2篇 系统管理命令

- [x] [磁盘与文件系统管理：分区、挂载与 df/du](./disk-filesystem-management)（鸟哥《Linux私房菜》第7章）
- [x] [压缩、打包与归档：tar/gzip/bzip2/xz](./archive-compression-tools)（鸟哥《Linux私房菜》第8章）
- [x] [账号管理与 ACL 权限设定](./user-account-management-acl)（鸟哥《Linux私房菜》第13章）
- [x] [程序管理与进程信号：ps/top/kill 与后台作业](./process-management-signals)（鸟哥《Linux私房菜》第16章）
- [x] [例行性工作排程：at 与 crontab](./scheduled-tasks-at-crontab)（鸟哥《Linux私房菜》第15章）
- [x] [认识系统服务与 systemd 管理](./systemd-service-management)（鸟哥《Linux私房菜》第17章）
- [x] [日志文件管理与日志轮替](./log-management-rotation)（鸟哥《Linux私房菜》第18章）

### 第3篇 Shell 脚本基础

- [x] [构建基础脚本：变量、命令替换与数学运算](./shell-basic-scripting)（Blum《Shell 脚本编程大全》第11章）
- [x] [使用结构化命令：if-then、test 与 case](./shell-structured-commands)（Blum《Shell 脚本编程大全》第12章）
- [x] [更多结构化命令：for、while 与 until 循环](./shell-loops)（Blum《Shell 脚本编程大全》第13章）
- [x] [处理用户输入：read、位置参数与命令行选项](./shell-user-input)（Blum《Shell 脚本编程大全》第14章）
- [x] [呈现数据：echo、重定向与文件描述符](./shell-redirection-fd)（Blum《Shell 脚本编程大全》第15章）
- [x] [创建函数：函数定义、返回值与作用域](./shell-functions)（Blum《Shell 脚本编程大全》第17章）

### 第4篇 高级脚本与文本处理

- [x] [使用 Linux 环境变量与 Bash 启动文件](./linux-env-variables-bashrc)（Blum《Shell 脚本编程大全》第6章）
- [x] [脚本控制：信号捕捉、后台运行与运行控制](./script-control-signals)（Blum《Shell 脚本编程大全》第16章）
- [x] [初识 sed 与 gawk](./sed-gawk-intro)（Blum《Shell 脚本编程大全》第19章）
- [x] [正则表达式进阶：BRE 与 ERE](./regex-bre-ere)（Blum《Shell 脚本编程大全》第20章）
- [x] [sed 进阶：多命令、多行处理与文本替换](./sed-advanced)（Blum《Shell 脚本编程大全》第21章）
- [x] [gawk 进阶：字段、数组与自定义函数](./gawk-advanced)（Blum《Shell 脚本编程大全》第22章）
- [x] [编写实用脚本：脚本排错与日常运维脚本实战](./practical-scripting-debugging)（Blum《Shell 脚本编程大全》第24章）
