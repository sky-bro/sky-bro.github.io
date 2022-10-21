
## Introduction {#introduction}

Magisk[^fn:1] is a suite of open source software for customizing Android, supporting devices higher than Android 5.0.

Some highlight features:

-   MagiskSU: Provide root access for applications
-   Magisk Modules: Modify read-only partitions by installing modules
-   MagiskBoot: The most complete tool for unpacking and repacking Android boot images
-   Zygisk: Run code in every Android applications' processes

This is my notes on installing it on my op6 following the [official installation guide](https://topjohnwu.github.io/Magisk/install.html).


## Download and install Magisk app {#download-and-install-magisk-app}

Download latest Magisk apk from [github release](https://github.com/topjohnwu/Magisk/releases/latest). Install it:

```shell
adb insatll Magisk.apk
```

After launching the app, notice the Ramdisk value (mine is Yes), it means whether or not your device has boot ramdisk.

-   if Yes, patch boot partition (I'll choose this)
-   if No, patch recovery partition

{{< figure src="/images/posts/root-android-with-magisk/magisk-first-installed.png" width="30%" >}}


## backup images {#backup-images}

This will need root access, so reboot to twrp recovery first.

```shell
# go to twrp recovery to get root access to your image partition
# adb reboot recovery
adb shell # commands below are executed in the android shell
# get current slot (A/B)
/bin/getprop ro.boot.slot_suffix
# _a, so I will backup /dev/block/by-name/boot_a
dd if=/dev/block/by-name/boot_a of=/sdcard/boot.img
# optional, if you have vbmeta partition
# dd if=/dev/block/by-name/vbmeta_a of=/sdcard/vbmeta.img
# adb pull /sdcard/vbmeta.img
```


## patch image and install {#patch-image-and-install}

patch image inside magisk app.

-   press install button in the magisk card
-   select the image just extracted

pull the patched image to your computer, reboot to bootloader, flash the new patched image to your android device.

```shell
adb pull /sdcard/Download/magisk_patched-xxx.img
adb reboot bootloader
fastboot flash boot magisk_patched-xxx.img
# if you patched the recovery partition
# fastboot flash recovery magisk_patched-xxx.img
# optional, patch and install the vbmeta partition
# fastboot flash vbmeta --disable-verity --disable-verification vbmeta.img
fastboot reboot
```

Now open Magisk app again, you can see it's installed.

{{< figure src="/images/posts/root-android-with-magisk/magisk-patch-installed.png" width="30%" >}}

[^fn:1]: [Magisk](https://github.com/topjohnwu/Magisk) -- The Magic Mask for Android