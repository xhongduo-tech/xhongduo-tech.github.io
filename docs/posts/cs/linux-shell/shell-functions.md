---
title: 创建函数：函数定义、返回值与作用域
date: 2026-08-07
---

# 创建函数：函数定义、返回值与作用域

<div class="epigraph">
<p>一个函数是把一段逻辑命名并收编，让它从此只被调用、不被复制。</p>
<footer>—— 程序设计的复用思想（Don't Repeat Yourself）</footer>
</div>

<div class="article-byline">
<p>第三级 · Linux 命令行与 Shell 脚本 ｜ Blum《Shell 脚本编程大全》 第17章 ｜ 2026-08-07</p>
</div>

## 为什么从函数开始

脚本写到 200 行时，你会发现自己把「检查磁盘」这段逻辑复制粘贴了三遍——而任何一次修改都要同步改三处，这就是复制粘贴的代价。**函数（function）** 把一段逻辑打包成可命名的单元，让脚本从「一串命令」进化为「模块化程序」。本章讲四个关键点：怎么定义与调用、**怎么把结果传出来**（return 与 echo 两条路）、**局部变量的作用域**、以及**把函数组织成库**供多个脚本共用。函数是 shell 脚本从「能跑」走向「能维护」的分界线。

## 1 定义与调用：两种写法

bash 函数的定义有两种等价写法：

```bash
function greet {
    echo "Hello, $1"
}

greet() {
    echo "Hello, $1"
}
```

`greet()` 写法更通用、兼容性更好。**调用就像执行命令一样**：`greet alice`——函数内部用 `$1` 拿到调用时传的参数。<span class="marginnote">注意函数<strong>必须在调用之前定义</strong>——bash 自上而下执行，先定义后调用，顺序反了会得到 `command not found`。这与 C 语言头文件声明或 Python 的运行时查找都不相同。</span>

定义与调用放在一起看：

```bash
#!/bin/bash
check_disk() {
    local usage=$(df -h / | tail -1 | awk '{print $5}' | tr -d '%')
    if [ "$usage" -gt 90 ]; then
        echo "磁盘告警: 已用 $usage%"
        return 1
    fi
    echo "磁盘正常: $usage%"
}

check_disk
echo "退出状态: $?"
```

函数体里 `$1`、`$2` 是**调用时传入的参数**，与脚本的位置参数相互独立——这一点常被忽视，是函数参数混淆的根源。

## 2 公式解析：return 与 echo——两条返回通道

shell 函数「返回值」有两条完全不同的通道，混用是新手第一大坑：

$$
\underbrace{\text{return } N}_{\text{退出状态：只存 0-255 的整数}} \qquad
\underbrace{\text{echo "..."}}_{\text{标准输出：任意字符串}}
$$

拆解：

- **第一步，`return N`**：设置函数返回后的 `$?`，表达「成功（0）/失败（非 0）」。它**只能返回 0–255 的整数**，且**不输出到屏幕**。
- **第二步，`echo`**：把字符串写到标准输出——这就是「函数的值」。
- **第三步**：调用方拿 echo 的值用命令替换：`result=$(my_func)`。
- **第四步**：于是分工明确——**`return` 报状态、`echo` 传数据**。

```bash
get_name() {
    echo "alice"          # 这是函数的值
    return 0              # 这是状态，顺便返回
}

name=$(get_name)          # 拿到 alice
echo "名字: $name"        # 名字: alice
```

**易错点**：把「函数返回值」理解成 `return` 的返回值是最大误区。`result=$(func)` 拿到的永远只有 echo 的输出，`return 42` 的 42 只存在于 `$?`。<span class="marginnote">`return` 不写数字默认返回上一条命令的状态。函数里 `exit` 会直接结束<strong>整个脚本</strong>，而 `return` 只结束当前函数——在函数里误用 `exit`，是脚本提前退出的隐藏原因。</span>

## 3 作用域：local 与全局变量

函数内定义的变量**默认是全局的**——脚本顶层与函数共享命名空间，这正是「函数改了脚本的变量」这类神秘 Bug 的来源：

```bash
counter=0
bump() {
    counter=$(( counter + 1 ))     # 直接改全局 counter
}
bump
echo "$counter"                    # 1
```

想要函数内部私有的变量，用 **`local`** 声明：

```bash
bump() {
    local tmp="临时值"             # 只在本函数内有效
    counter=$(( counter + 1 ))
}
```

`local` 变量在函数结束时自动销毁，且**不会污染全局命名空间**。规范是：**函数内部用到的一切中间变量都加 `local`**，只有需要对外输出的才用全局。<span class="marginnote">`local` 只在函数内合法，脚本顶层声明会报错。函数的全局变量还能在函数间传值（「隐式返回值」），但耦合度高、难排查——能显式 `echo` + 命令替换就不用全局传值。</span>

**易错点**：`local` 与赋值写在一起时，变量名与值之间**不能有空格**：`local x=1` 正确，`local x = 1` 会把 `= 1` 当成命令参数。忘了 `local` 的中间变量会悄悄成为全局，两个函数用同名临时变量时互相踩——这是脚本「偶尔出错」的经典来源。

## 4 数组与递归：函数的高级形态

函数可以返回数组（用 echo 逐行输出，调用方 `readarray` 接收）：

```bash
get_errors() {
    grep -c "error" /var/log/app.log
    grep -c "warn" /var/log/app.log
}
readarray counts < <(get_errors)   # 每行一个元素
```

**递归函数**同样可行——阶乘是经典例子：

```bash
fact() {
    if [ "$1" -le 1 ]; then
        echo 1
    else
        local prev=$(( $1 - 1 ))
        local pv=$(fact "$prev")
        echo $(( $1 * pv ))
    fi
}
echo "5! = $(fact 5)"
```

递归里 `local` 尤其关键：每次调用都有独立的 `prev`、`pv`，互不覆盖——这正是递归需要局部变量的原因。<span class="marginnote">递归函数要<strong>收敛</strong>：必须有终止条件（上面的 `-le 1`）。bash 递归层数有限（约千层），写太深会 `Segmentation fault`。工程上能用循环就用循环，递归只用于语义清晰的小函数。</span>

## 5 函数库：用 source 复用

函数写好后，把它存进一个**库文件**，多个脚本 `source` 就能共用：

```bash
# lib.sh —— 函数库
check_disk() { ... }
get_date() { echo "$(date +%F)"; }
```

```bash
#!/bin/bash
source ./lib.sh          # 把库文件读入当前 shell
check_disk               # 直接调用库里的函数
```

`source`（等价 `.`）**在当前位置执行库文件**，于是库里的函数定义就进入了当前脚本的命名空间——比 `bash lib.sh`（子 shell 执行，函数不保留）正确得多。<span class="marginnote">`source` 与 `bash file` 的本质差别：前者在当前 shell 执行、共享变量与函数；后者开子 shell、结束后一切消失。`. ~/.bashrc` 里那个点号就是 `source` 的简写——改完配置重载，用的正是这个机制。</span>

**易错点**：库文件路径写相对路径 `./lib.sh` 依赖当前工作目录。脚本被从别的目录调用时（`/opt/bin/myscript.sh`），`source ./lib.sh` 会找错文件。稳妥做法是 `source "$(dirname "$0")/lib.sh"`，让路径相对**脚本自身**。

## 6 小结

- **函数先定义后调用**：`name() { }` 两种写法，`$1` 是函数自己的参数。
- **return 报状态、echo 传数据**：`result=$(func)` 拿 echo，`$?` 拿 return，整数 0–255。
- **`local` 私有变量**：函数中间变量一律 `local`，避免污染全局、避免互相踩。
- **递归与数组可行**：递归要收敛 + local，数组用 `readarray` 接收逐行 echo。
- **`source` 复用函数库**：`source lib.sh` 在当前 shell 加载；路径用 `$(dirname "$