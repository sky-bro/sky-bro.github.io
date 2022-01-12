
## Introduction {#introduction}

Just the other day I updated all my packages through `yay -Syu` (like `pacman -Syu` but also updates aur packages).

And after a reboot, it entered a boot loop...

I believe this had happened to most arch users, and most of the time its just because we broke the dependencies of some packages after the upgrade.

So here's how I saved boot failure after `yay -Syu`.

{{< alert theme="info" dir="ltr" >}}

You'll need a bootable usb stick (preferable the one you use for installing the system).

{{< /alert >}}


## Manually boot from grub (optional) {#manually-boot-from-grub--optional}

follow this guide[^fn:1] to boot your linux from grub (generates log).

The grub command line can also be entered from your bootable usb drive.

```shell
ls # list partitions
ls (hd1,gpt2)/ # see files in a partition
set root=(hd1,gpt2) # your linux root partition
linux /boot/vmlinuz-5.13-x86_64 ro root=/dev/nvme0n1p1
initrd /boot/initramfs-5.13-x86_64.img
boot
```


## chroot to your system {#chroot-to-your-system}

manjaro-chroot is provide in `manjaro-tools-base` package, and is already installed in your live system.

```shell
# mount root
mount /dev/nvme0n1p2 /mnt
# mount boot
# mount /dev/xxx /mnt/boot
# mount efi
mount /dev/nvme0n1p1 /mnt/boot/efi/

manjaro-chroot /mnt
```


## check your boot log {#check-your-boot-log}

```shell
# -b: show boot log
# -1: offset, last boot
journalctl -b -1
```


## fix any problems {#fix-any-problems}

find any suspicious errors in the boot log, and search it on the web, see how to fix them.

for me, a package from aur was causing the problem, and I tried to fix it, but no luck.

So I just uninstalled it!


## reboot {#reboot}

success!

[^fn:1]: [How to manually boot up a Linux](https://forums.justlinux.com/showthread.php?150600-How-to-manually-boot-up-a-Linux)
