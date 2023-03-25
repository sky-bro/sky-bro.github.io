
## What is a window manager {#what-is-a-window-manager}

i3 is a tiling window manager, which


### Stacking WM {#stacking-wm}

-   mostly mouse driven, slow
-   windows overlap with each other, space inefficient

{{< figure src="/images/posts/Tiling WM/stacking-wm-example.jpeg" >}}


### Tiling WM {#tiling-wm}

-   automatically layout windows like tiles, high space efficiency
-   mostly keyboard driven, fast
-   multiple layout mode: tabbed, stacked, horizontal/vertical split, float

{{< figure src="/images/posts/Tiling WM/tiling-wm-example.jpeg" >}}


## Understand Container &amp; Tree {#understand-container-and-tree}

{{< alert theme="warning" >}}

The actual implementation of window managers may differ. But most tiling window managers use tree structure to manager their windows.

{{< /alert >}}

In i3,  windows are stored as leaf nodes of a tree, each workspace has a tree. Non leaf nodes of the tree are called containers, whose children are other containers or windows.


### Attributes of a container {#attributes-of-a-container}


#### layout mode {#layout-mode}

i3 has layout mode like:

-   split (horizontally or vertically)
-   tabbed
-   stacked

Why we need a layout mode attribute, to give you an example, imagine we have a tree structure like this, managing 3 windows.

<a id="figure--fig:tiling-wm-ambiguous-layout"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-ambiguous-layout.svg" caption="<span class=\"figure-number\">Figure 1: </span>tree of ambiguous layout" >}}

Since we do not know how to place windows inside a container, figure [1](#figure--fig:tiling-wm-ambiguous-layout) may represents windows like figure [2](#figure--fig:possible-window-status-01), or figure [3](#figure--fig:possible-window-status-02) or something else. Therefore, containers must have a layout mode.

<a id="figure--fig:possible-window-status-01"></a>

{{< figure src="/images/posts/Tiling WM/possible-window-status-01.svg" caption="<span class=\"figure-number\">Figure 2: </span>possible window status 01" >}}

<a id="figure--fig:possible-window-status-02"></a>

{{< figure src="/images/posts/Tiling WM/possible-window-status-02.svg" caption="<span class=\"figure-number\">Figure 3: </span>possible window status 02" >}}

To only represents windows in figure [2](#figure--fig:possible-window-status-01), we add layout attribute to each container, as in figure [4](#figure--fig:tiling-wm-unambiguous-layout).

<a id="figure--fig:tiling-wm-unambiguous-layout"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-unambiguous-layout.svg" caption="<span class=\"figure-number\">Figure 4: </span>tree of unambiguous layout" >}}


#### percentage of width or height in split direction {#percentage-of-width-or-height-in-split-direction}

To control the width or height of windows, we need to add percentage of width or height in split direction. as in figure [5](#figure--fig:tiling-wm-percentage)

<a id="figure--fig:tiling-wm-percentage"></a>

{{< figure src="/images/posts/Tiling WM/tiling-wm-percentage.svg" caption="<span class=\"figure-number\">Figure 5: </span>tree with container percentage" >}}


### Practice {#practice}

<a id="figure--understand tree structure through container title"></a>

{{< figure src="/images/posts/tiling-wm--i3/tree-structure-through-container-title.png" >}}

To better understand the tree structure, we can put our windows or containers inside a tabbed/stacked container, and get the tree structure through the container title.

Here's how you can get the container title:

1.  switch to an empty workspace: `$Mod+3`.
2.  toggle root container layout mode as split(horizontal or vertical): `$Mod+e` (usually we do not set root container layout mode as stacked or tabbed)
3.  open the first window in this workspace: `$Mod+ENTER` (opens my st terminal)
4.  set container (of this window) layout as tabbed or stacked: `$Mod+w/s`
5.  create a new vertical/horizontal split container: `$Mod+v/V`
    -   when container of the focused node(container or window) contains only a single node, and the container layout is set to split, `$Mod+v/V` only changes the split direction
    -   when container of the focused node contains more than one node, or container layout is set to tabbed/stacked, `$Mod+v/V` will create a new vertical/horizontal split container encapsulating current focused node
6.  now you can see a container title showing `V[st]` (V means horizontal split layout), create more containers or change container layout to see the title change: open one more st window with `$Mod+ENTER` changes the title to `V[st st]`

{{< alert theme="info" >}}

Tips:
use `$Mod+c` or `$Mod+p` to select child or parent of a node, then operate on that node:

-   create a sibling window
-   close window of node (all window under that node will be closed)
-   move  window of node
-   resize window of node

{{< /alert >}}


## some of my i3 shortcuts {#some-of-my-i3-shortcuts}

Here's a snippet of my i3 configuration. Complete configuration is stored at my  [.dotfiles](https://github.com/sky-bro/.dotfiles/blob/master/.config/i3/config) repository.

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
: change focus to the left/upper/lower/right window

_**Mod-S-h/j/k/l**_
: move focused window/container

_**Mod-y/u/i/o**_
: change size of focused window/container

_**Mod-v/V**_
: add a container for current window, set layout to vertical/horizontal split

_**Mod-e/w/s**_
: set layout of container of focused window to split(toggles between splith, splitv)/tabbed/stacked

_**Mod-p**_
: focus parent

_**Mod-c**_
: focus child


## References {#references}

-   [i3wm user's guide &gt;&gt; tree](https://i3wm.org/docs/userguide.html#_tree)
-   [youtube: TheAlternative.ch - LinuxDays FS16 - Linux for Experts course](https://www.youtube.com/watch?v=Api6dFMlxAA)
-   [wiki: window manager types](https://en.wikipedia.org/wiki/Window_manager#Types)
