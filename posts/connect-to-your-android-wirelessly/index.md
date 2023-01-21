
## Introduction {#introduction}

This artical will show you how to connect to your android, so you can play with your android via adb[^fn:1].


## Setup your environment {#setup-your-environment}


### get Android SDK Platform Tools {#get-android-sdk-platform-tools}

download [platform-tools](https://developer.android.com/studio/releases/platform-tools) directly or install it with your package manager:

```shell
# arch
sudo pacman -Sy android-tools
```


### enable debugging {#enable-debugging}

first enable developer mode, for my op6, clicking the build number a bunch of times enables developer mode

{{< figure src="/images/posts/connect-to-your-android-wirelessly/android-enable-developer-mode.png" caption="<span class=\"figure-number\">Figure 1: </span>enable developer mode for my op6" width="30%" >}}

after that, go to "System &gt; Developer Options", enable USB debugging and Wireless ADB debugging

{{< figure src="/images/posts/connect-to-your-android-wirelessly/android-enable-USB-debugging.png" caption="<span class=\"figure-number\">Figure 2: </span>enable debugging" width="30%" >}}


## Connect to your phone with USB {#connect-to-your-phone-with-usb}

now if you have a USB cable, you can directly connect your android to your computer with that.

You can tell if device is connected via `adb devices`

{{< figure src="/images/posts/connect-to-your-phone-with-usb/adb-devices.png" caption="<span class=\"figure-number\">Figure 3: </span>adb list connected devices" >}}


## Connect Wirelessly (no root required) {#connect-wirelessly--no-root-required}

After connecting with usb, you can enable tcpip on your phone, then connect with that wirelessly:

```shell
# enable tcpip
adb -s <device-serial> tcpip 5555
# get device ipv4 address
adb -s <device-serial> shell ip -f inet addr
# connect to it
adb connect <device-ip>:5555
```

{{< figure src="/images/posts/connect-to-your-android-wirelessly/enable-tcpip-on-computer.png" caption="<span class=\"figure-number\">Figure 4: </span>enable tcpip on the computer" >}}


## Connect Wirelessly (require root, no usb) {#connect-wirelessly--require-root-no-usb}

If you have root access to you android, you can enable tcpip directly on your phone, then connect to it on the computer.
I use termux[^fn:2] to do these:

```shell
su
setprop service.adb.tcp.port 5555
stop adbd && start adbd
ip -f inet addr show wlan0
```

{{< figure src="/images/posts/connect-to-your-android-wirelessly/enable-tcpip-on-android.png" caption="<span class=\"figure-number\">Figure 5: </span>enable tcpip on the phone" >}}

[^fn:1]: [Android Debug Bridge](https://developer.android.com/studio/command-line/adb) (adb) is a versatile command-line tool that lets you communicate with a device
[^fn:2]: [Termux](https://termux.dev/en/) is an Android terminal emulator and Linux environment app