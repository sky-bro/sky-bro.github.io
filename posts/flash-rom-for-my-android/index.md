
## flash twrp recovery {#flash-twrp-recovery}

Download TWRP[^fn:1] recovery for your android device, and probably follow the installation guide there.

Generally, there are two ways of installing the recovery: from a  `.img` file or a `.zip` file.


### backup your boot.img {#backup-your-boot-dot-img}

There's no recovery partition, recovery is now part of the boot partition.

So in case we break our boot partition, backup it first.

```shell
# === On your android shell (adb shell) ===
# cd `find /dev/block/platform -type d -name by-name` # mine is /dev/block/platform/soc/1d84000.ufshc/by-name
# or just
cd /dev/block/bootdevice/by-name/boot
# store the boot partition to /sdcard/boot.img file
dd if=boot of=/sdcard/boot.img

# === On you computer shell ===
# copy the boot.img to your computer
adb pull /sdcard/boot.img
```

You can restore the boot partition with `fastboot`.

```shell
# === on your computer shell ===
fastboot flash boot boot.img
```


### Install with the recovery.img file {#install-with-the-recovery-dot-img-file}

First temporarily boot into the new recovery.

```shell
adb reboot bootloader
fastboot boot recovery.img
```

Once booted, make this recovery permanent:

-   navigate to Advanced &gt; Flash Current TWRP option (preferably), or
-   navigate to Advanced &gt; Install Recovery Ramdisk &gt; select the `recovery.img` file from your phone storage, or
-   as in the next section: install with the `recovery.zip` file.


### Install with the recovery.zip file {#install-with-the-recovery-dot-zip-file}

If you already have a working recovery, you only need to have this file on you phone storage (no computer needed). And flash this zip file from your recovery.

Navigate to Install &gt; select the `recovery.zip` file.


## flash rom {#flash-rom}

You can get many useful resources for OnePlus from 大侠阿木云盘[^fn:2].


### Wipes {#wipes}

-   Dalvik Cache
-   Cache


### Install rom.zip {#install-rom-dot-zip}

put the `rom.zip` file on your phone storage.

boot to recovery, navigate to Install &gt; select the `rom.zip` file.


### Trouble Shooting {#trouble-shooting}

-   After flashing a offcial rom for my oneplus 6, my device keeps boots to recovery instead of the system.
    -   Solution: go to recovery, Wipe &gt; Format Data.

[^fn:1]: [TWRP](https://twrp.me/) is the leading custom recovery for Android phones
[^fn:2]: I downloaded official ROM (many history versions) for my op6 from [大侠阿木云盘](https://yun.daxiaamu.com/)
