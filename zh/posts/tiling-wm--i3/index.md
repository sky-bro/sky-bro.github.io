
## 什么是窗口管理器 {#什么是窗口管理器}

窗口管理器 (Window Manager) 就是用来管理窗口的. 主要分为两大类:


### Stacking WM: {#stacking-wm}

每个窗口都可以拖拽, 改变大小, 窗口之间可以有重叠, 主要靠鼠标控制.

比如下图, Stacking WM的屏幕利用率比较低, 而且操作严重依赖鼠标, 比较慢.

{{< figure src="/images/posts/Tiling WM/stacking-wm-example.jpeg" >}}


### Tiling WM: {#tiling-wm}

窗口不能拖拽, 窗口之间没有重叠, 像瓷砖一样铺满屏幕, 配合很多快捷键使用, 基本不或
者少依赖鼠标进行控制.

-   自动放置窗口, 不重叠, 空间利用率高
-   多种布局方式切换, 适应各种需求 (基本所有tiling wm都不止Tiling一种布局, 比如i3wm还有Tabbed, Stacked, 还可以选择让某窗口全屏或是悬浮)
-   基本全靠键盘控制, 快速方便

{{< figure src="/images/posts/Tiling WM/tiling-wm-example.jpeg" >}}


## container与tree {#container与tree}

{{< alert theme="warning" >}}

我这里并没有严格去看源码这个树到底是怎么构建的(尤其是很多特殊情况), 不过就简单理解下树和窗口之间的关系是可以的.

{{< /alert >}}

大多窗口管理器采用树结构来保存窗口, i3也是如此.

每个workspace的所有窗口保存在一个tree数据结构里面, 这个tree的每个节点都是一个 container (window算特殊的container, 是没有孩子的叶子节点), 一个container里面又可以包含多个container.


### container属性 {#container属性}

每个container有两个比较重要的属性:


#### 布局方式 {#布局方式}

split (分splith, splitv), tabbed 或者 stacked

<a id="figure--fig:tiling-wm-ambiguous-layout"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-ambiguous-layout.svg" caption="<span class=\"figure-number\">Figure 1: </span>tree of ambiguous layout" >}}

图 [1](#figure--fig:tiling-wm-ambiguous-layout) 代表的窗口可能是图 [2](#figure--fig:possible-window-status-01), 也可能是图 [3](#figure--fig:possible-window-status-02)或者其他的, 因为我们不知道每个container内部的窗口布局方式. 所以窗口管理器会为每个节点保存布局方式, 表示内部的孩子节点应该按照什么方式放置.

<a id="figure--fig:possible-window-status-01"></a>

{{< figure src="/images/posts/Tiling WM/possible-window-status-01.svg" caption="<span class=\"figure-number\">Figure 2: </span>possible window status 01" >}}

<a id="figure--fig:possible-window-status-02"></a>

{{< figure src="/images/posts/Tiling WM/possible-window-status-02.svg" caption="<span class=\"figure-number\">Figure 3: </span>possible window status 02" >}}

所以如果想只代表图 [2](#figure--fig:possible-window-status-01) 中的窗口, 那么应该为container增加布局方式属性, 如图 [1](#figure--fig:tiling-wm-ambiguous-layout) 所示.

<a id="figure--fig:tiling-wm-unambiguous-layout"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-unambiguous-layout.svg" caption="<span class=\"figure-number\">Figure 4: </span>tree of unambiguous layout" >}}


#### split方向对应占多少百分比 {#split方向对应占多少百分比}

因为我们通常还会控制每个窗口/容器的宽度/高度, 所以还应该为每个容器增加split方向的空间占比, 如图 [5](#figure--fig:tiling-wm-percentage) 所示.

<a id="figure--fig:tiling-wm-percentage"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-percentage.svg" caption="<span class=\"figure-number\">Figure 5: </span>tree with container percentage" >}}


### 练习与理解 {#练习与理解}

为了理解我们在操作窗口时对应tree的构造, 我们将窗口/容器都放在另一个设为tabbed或
者stacked布局的容器内 -- **因为这两种布局才会显示标题**.

1.  切换到一个没有任何窗口的workspace: 如 `$Mod+3`.
2.  用 `$Mod+e` 设置默认的布局为split(水平/垂直), (一般不会设为tabbed或stacked).
3.  `$Mod+ENTER` 打开一个terminal (我这里是st, 或者任何别的窗口也行).
4.  `$Mod+w/s` 设置所在container布局为tabbed或stacked
5.  `$Mod+v/V` 新建一个垂直/水平split布局的container包裹当前的st窗口
    -   在节点(container或window)所在container只有一个窗口, 且container布局方式为水平/垂直split时, `$Mod+v/V` 只会切换split方向
    -   当节点所在container不止一个窗口, 或者container布局方式为tabbed或者stacked时, `$Mod+v/V` 会创建一个新的垂直/水平split布局的container包裹当前节点
6.  后续在这个stabbed或者stacked container下操作就可以看到标题了

当前看到标题应该为 `V[st]`, 再 `$Mod+ENTER` 之后显示为 `V[st st]`.

{{< alert theme="info" >}}

Tips:
使用 `$Mod+c` 或者 `$Mod+p` 来选择孩子或父亲节点窗口, 然后对该节点进行操作:

-   创建一个兄弟节点
-   删除/关闭节点窗口 (节点下所有窗口都会被关闭)
-   移动节点窗口
-   修改节点窗口大小

{{< /alert >}}


## 快捷键设置 {#快捷键设置}

这里仅列出了比较重要的快捷键, 我详细的配置放在了github的[.dotfiles](https://github.com/sky-bro/.dotfiles/blob/master/.config/i3/config)仓库.

```sh
# some configs from my ~/.config/i3/config
set $mod Mod4

set $up k
set $down j
set $left h
set $right l

# change focus
bindsym $mod+$left focus left
bindsym $mod+$down focus down
bindsym $mod+$up focus up
bindsym $mod+$right focus right

# move focused window
bindsym $mod+Shift+$left move left
bindsym $mod+Shift+$down move down
bindsym $mod+Shift+$up move up
bindsym $mod+Shift+$right move right

# split in horizontal orientation
bindsym $mod+Shift+v split h

# split in vertical orientation
bindsym $mod+v split v

# enter fullscreen mode for the focused container
bindsym $mod+f fullscreen toggle

# change container layout (stacked, tabbed, toggle split)
bindsym $mod+s layout stacking
bindsym $mod+w layout tabbed
bindsym $mod+e layout toggle split

# toggle tiling / floating
bindsym $mod+Shift+space floating toggle

# change focus between tiling / floating windows
bindsym $mod+space focus mode_toggle

# focus the parent container
bindsym $mod+p focus parent

# focus the child container
bindsym $mod+c focus child

# resize window (you can also use the mouse for that)
set $resize_step 5

bindsym $mod+y resize shrink width $resize_step px or $resize_step ppt
bindsym $mod+i resize grow height $resize_step px or $resize_step ppt
bindsym $mod+u resize shrink height $resize_step px or $resize_step ppt
bindsym $mod+o resize grow width $resize_step px or $resize_step ppt
```

_**Mod-h/j/k/l**_
: 切换到左/上/下/右边窗口

_**Mod-S-h/j/k/l**_
: 移动窗口/容器

_**Mod-y/u/i/o**_
: 调整窗口/容器大小

_**Mod-v**_
: **增加一个container** 存放当前focused window(或者container), 容器内采用垂直split布局

_**Mod-S-v**_
: 同上, 不过容器内采用水平split布局

_**Mod-e/w/s**_
: 设置 **所在container** 的布局为Split(会在splith, splitv间循环), Tabbed, Stacked

_**Mod-p**_
: Focus parent

_**Mod-c**_
: Focus child


## 参考 {#参考}

-   [i3wm 用户手册 &gt;&gt; tree](https://i3wm.org/docs/userguide.html#_tree)
-   [youtube: TheAlternative.ch - LinuxDays FS16 - Linux for Experts course](https://www.youtube.com/watch?v=Api6dFMlxAA)
-   [wiki: window manager types](https://en.wikipedia.org/wiki/Window_manager#Types)