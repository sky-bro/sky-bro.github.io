
## 分体键盘介绍/如何选择分体键盘 {#分体键盘介绍-如何选择分体键盘}


### 选择键盘固件 {#选择键盘固件}

zmk官网有对常见键盘固件功能的对比: [zmk features](https://zmk.dev/docs), 我自己没有接触过除zmk的其它固件, 冲分体和蓝牙多设备我直接选择了zmk, 而且感觉zmk社区也比较活跃.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/keyboard-firmware-features.png" caption="<span class=\"figure-number\">Figure 1: </span>固件功能对比" >}}


### 选择多少键位的分体 {#选择多少键位的分体}

[zmk: supported hardware &gt; composite keyboards](https://zmk.dev/docs/hardware#composite)中列举了很多分体键盘方案. 不过比较出名的主要有下面这些


#### [corne](https://github.com/foostan/crkbd/) {#corne}

3&times;6或3&times;5的配列, 3个拇指键, 个人觉得拇指键两个就够了, 多一个太靠近掌心很难按, 支持屏幕.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-corne.jpg" caption="<span class=\"figure-number\">Figure 2: </span>corne" >}}

也有一些corne的变体, 比如[corne-ish zen](https://www.reddit.com/r/mechmarket/comments/jyfrv2/ic_the_corneish_zen_a_low_profile_wireless_split/), 支持了墨水屏.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-corne-ish-zen.jpg" caption="<span class=\"figure-number\">Figure 3: </span>corne-ish zen" >}}


#### [sweep](https://github.com/davidphilipbarr/Sweep) {#sweep}

sweep采用3&times;5的配列, 两个拇指键, 不支持屏幕, 不需要二极管, 焊接简单. 最重要的是我觉得官方文档写得很好.

3&times;5配列我也觉得是最合适的配列, 充分利用切层, 让所有手指移动距离保持在一个键以内.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-sweep-bling-lp.jpg" caption="<span class=\"figure-number\">Figure 4: </span>my sweep bling lp" >}}


#### [Lily58](https://github.com/kata0510/Lily58) {#lily58}

Lily58从名字就看出一共58个按键, 4&times;6配列, 个人觉得按键比较冗余了, 尤其最下面一排不知道是交给拇指按(太靠近掌心不好按)还是中间三根手指按(列不对齐有点奇怪).

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-lily58.jpg" caption="<span class=\"figure-number\">Figure 5: </span>lily58" >}}


#### [Sofle](https://github.com/josefadamcik/SofleKeyboard) {#sofle}

和Lily58类似, 一共58个按键, 4&times;6配列, 多了两个编码器键(encoder).

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-sofle.jpg" caption="<span class=\"figure-number\">Figure 6: </span>sofle" >}}


#### [Redox](https://github.com/mattdibi/redox-keyboard) {#redox}

多少键已经数不清楚了, 而且这个拇指键就有点疯狂了, 以前两个拇指按一个空格太清闲, 现在充分利用起来是吧...

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-redox.jpg" caption="<span class=\"figure-number\">Figure 7: </span>Redox" >}}


## 硬件部分(sweep bling lp) {#硬件部分--sweep-bling-lp}

我最终选择了sweep键盘, 因为他有比较好的文档, 而且焊接简单(不需要二极管), 外观小巧精致.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-sweep-bling-lp-components.png" caption="<span class=\"figure-number\">Figure 8: </span>sweep bling lp物料表" >}}


### PCB {#pcb}

从[github: Sweep Bling LP gerbers](https://github.com/davidphilipbarr/Sweep/tree/main/Sweep%20Bling%20LP/gerbers)下载gerbers文件, 然后用它到[嘉立创PCB下单](https://www.jlc.com/newOrder/#/pcb/newOnlinePlaceOrder).

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/jlc-pcb-order.png" caption="<span class=\"figure-number\">Figure 9: </span>嘉立创PCB下单" >}}


### 主控 {#主控}

按照官网所说需要两块兼容promicro的或nice!nano主控, 淘宝搜索promicro nrf52840或nice!nano, 睫毛外设店和无名科技Nologo应该比较火.


#### 排针 &amp; 排母 {#排针-and-排母}

通过排针排母可以让我们的主控变成可插拔的, 未来如果想换别的键盘非常方便.

选择2.54mm间距12 pin的排针排母, 选择12 pin这样就不用回来再掰了, 排针比较好掰不选12还好, 但是排母最好选12 pin的.


#### 电池 {#电池}

主控可以靠usb或者电池供电, 想要无线使用那么电池必须有.

排针排母把主控架起来, 和PCB板间有个空间可以放电池, 淘宝搜索601235有300mah的电池, 这基本是能放下的最大的电池了.


#### 开关 {#开关}

需要一个开关来控制电池对主控供电. 根据官方所说的型号MSK 12C02搜索.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-battery-and-switch.jpg" caption="<span class=\"figure-number\">Figure 10: </span>电源和开关" >}}


### 按键 {#按键}

按键分为三个部分, 从下到上依次是: 轴座, 键轴, 键帽. 兼容性需要根据选择的PCB来.

sweep bling lp需要使用凯华的矮轴轴座 + 键轴 + 键帽. 矮轴个人用其来觉得比较舒服, 但小众, 价格比MX轴贵, 选择范围小.


#### 轴座 {#轴座}

轴座可以让我们的键轴变为可插拔的, 未来想要尝试不同的键轴很方便.

淘宝搜索凯华 轴座 1350.


#### 键轴 {#键轴}

我目前只尝试过白轴和粉轴, 白轴听起来比较清脆, 但可能吵到别人; 粉轴特别轻, 也比较安静, 适合在人多的环境使用.

淘宝搜索凯华 矮轴 1350.


#### 键帽 {#键帽}

键帽可选择的太少了, 我目前只搜到淘宝哈狐外设企业店一个, 别的店也有矮轴, 但是sweep bling lp按键特别紧凑, 键帽安上去上下两排几乎没有空隙. 键帽是1.65cm&times;1.65cm的, 别的店我看都比这个大, 会安不下.


### 外壳 {#外壳}

外壳可以网上搜就行, 然后淘宝或别的地方找个3D打印的店帮忙打印下即可.

比如我用的是这个[printables: Ferris Sweep Bling LP](https://www.printables.com/model/782368-ferris-sweep-bling-lp), 把stl文件下载下来发给店家.


#### 螺丝 {#螺丝}

外壳和PCB板一般要靠螺丝固定, 上面这个外壳我用的是4颗4mm长的m2螺丝, 左右手一共8颗螺丝.


#### 橡胶脚垫 {#橡胶脚垫}

外壳放到桌面上一般都会比较滑, 所以还需要用橡胶垫来防滑. 贴的时候注意尽可能贴到角落, 防止按的时候翘起来.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/kbd-rubber-feet.jpg" caption="<span class=\"figure-number\">Figure 11: </span>键盘底部用橡胶垫防滑" >}}


## 软件部分: ZMK固件 {#软件部分-zmk固件}

参考我的[zmk-config](https://github.com/sky-bro/zmk-config)


### 修改设备名字 {#修改设备名字}

给自己的键盘起个独特的名字, 名字限制16个字符长.


### 电量查看 {#电量查看}

有两种方式, 第一种是官方固件已经支持配置分体键盘电量上报, 可以看到左右手的电量. 只需要修改下配置文件;

第二种是使用自定义的behavior, 绑定一个快捷键直接把电量输出出来.

{{< figure src="/images/posts/sweep-bling-lp-矮轴分体键盘/custom-battery-report-behavior.gif" >}}


### 鼠标模拟 {#鼠标模拟}

目前还没有合入主干, 可以自己合下代码用一下: [pr 2027](https://github.com/zmkfirmware/zmk/pull/2027).

这个功能主要是针对少量的鼠标操作, 大量的操作还是直接用鼠标方便.


## 打个广告 {#打个广告}

兄弟姐妹们如果不想自己组装也可以移步咸鱼, 搜索 `k4i_top` 找我直接购买, 备注博客或者b站: [分体键盘视频合集](https://space.bilibili.com/356650397/channel/collectiondetail?sid=2323692&ctype=0)来的可以优惠 :)
