
## Introduction {#introduction}

KVM[^fn:1] is part of linux kernel, and QEMU[^fn:2] (Quick EMUlator) is an emulator. KVM by itself cannot provide the complete virtualization solution, it needs QEMU to provide full hypervisor functionality. QEMU can emulate cpu on its own, but with KVM, QEMU can achieve near native performance by executing the guest code directly on the host CPU.

So it's best for them to work together.

{{< figure src="/images/posts/linux-hypervisor-setup/architecture.svg" caption="<span class=\"figure-number\">Figure 1: </span>architecture" >}}


## tools you need {#tools-you-need}

Use your own package manager to install these tools:

```shell
# arch users
sudo pacman -Sy --needed \
        qemu \
        virt-viewer \
        libvirt \
        dnsmasq \
        ebtables \
        virt-install \
        virt-manager \
```

-   **kvm** (Kernel-based Virtual Machine): Kernel module that handles CPU and memory communication
-   **qemu** (Quick EMUlator): emulates many hardware resources -- dick, network, usb...
-   **libvirt**: an open-source API, daemon and management tool for managing platform virtualization. It can be used to manage KVM, Xen, VMware ESXi, QEMU and other virtualization technologies.
    -   **virsh**: comes with libvirt, command-line tools for communicating with libvirt
-   **virt-manager**: GUI alternative to virsh, albeit less capable.
-   **virt-install**: part of virt-manager project, create new VM guests
-   **virt-viewer**: part of virt-manager project, UI for interacting with VMs via VNC/SPICE
-   **dnsmasq**: light-weight DNS/DHCP server. Primarily used for allocating IPs to VMs.
-   **ebtables**: used for setting up NAT networking the host


## some setup {#some-setup}

two problems

1.  by default, virt-manager talks to `qemu:///system`, and virsh talks to `qemu:///session` (unless run as sudo).
2.  when talking to qemu:///system, we need to input password every time, especially unpleasant experience when a cli tool like virsh.

for the first problem, we can tell virsh to use `qemu:///system` by default

```shell
cp /etc/libvirt/libvirt.conf ~/.config/libvirt/libvirt.conf
vim ~/.config/libvirt/libvirt.conf # uncomment or add: uri_default = "qemu:///system"
```

To solve the second problem, we can add a rule to [polkit](https://wiki.archlinux.org/index.php/Polkit) to allow our group (`wheel` -- administrator group) to use virt-manager or vish without being asked for password.

edit `/etc/polkit-1/rules.d/xxx.rules`, your path may be different, put this in.

```js
/* Allow users in wheel group to manage the libvirt daemon without authentication */
polkit.addRule(function (action, subject) {
  if (action.id == "org.libvirt.unix.manage" && subject.isInGroup("wheel")) {
    return polkit.Result.YES;
  }
});
```


## start services {#start-services}

```shell
systemctl enable libvirtd # start on boot
systemctl enable virtstoraged # start on boot
systemctl enable virtnetworkd # start on boot
systemctl start libvirtd  # start libvirtd
systemctl start virtstoraged # start on boot
systemctl start virtnetworkd # start on boot
virsh net-autostart --network default
virsh net-start --network default # start the default network
```


## add shrarefolders {#add-shrarefolders}

Inside virtual machine manager, double click on one of your machine, then select `view->details->Add Hardware`, set something like below:

{{< figure src="/images/posts/linux-hypervisor-setup/add-sharefolder.png" caption="<span class=\"figure-number\">Figure 2: </span>add sharefolder" >}}

The above setting will add a new device `/ctf` in the virtual machine


### if in ubuntu {#if-in-ubuntu}

`sudo vim /etc/rc.local`

```shell
#!/bin/sh -e
#
# rc.local
#
# This script is executed at the end of each multiuser runlevel.
# Make sure that the script will "exit 0" on success or any other
# value on error.
#
# In order to enable or disable this script just change the execution
# bits.
#
# By default this script does nothing.

mount -t 9p -o trans=virtio,version=9p2000.L /ctf /home/sky/ctf

exit 0
```

After restarting this server, `/ctf` will automatically be mounted on `/home/sky/ctf`

To make the user (actually kvm) writing the share folder same as the user at host (vm host):
`sudo vim /etc/libvirt/qemu.conf`, find two lines with `user`"xxx"= and `group`"xxx"=, change them to yourself (by default, xxx should be \`root\`), then uncomment the two lines. for me, they are:

```conf
user = "sky"
group = "sky"
```

You may need to restart the libvirtd.service for this to take effect.

Also, you need to `chown` the disk to the above `user:group`: `sudo chown sky:sky /var/lib/libvirt/images/ubt16-server.qcow2`


## create a VM {#create-a-vm}

-   gui: virt-manager
-   cli: virt-install

examples:

-   Windows 11 VM with QEMU/KVM


## clone a VM {#clone-a-vm}

-   gui: virt-manager
-   cli: virt-clone


## useful virsh commands {#useful-virsh-commands}

```shell
virsh list --all
virsh start ubuntu-server
virsh shutdown ubuntu-server
virsh reboot ubuntu-server
```


## Resources {#resources}

-   [Linux Hypervisor Setup (libvirt/qemu/kvm)](https://octetz.com/docs/2020/2020-05-06-linux-hypervisor-setup/)
-   [Sharing folder with VM through libvirt, 9p, permission denied](https://askubuntu.com/questions/548208/sharing-folder-with-vm-through-libvirt-9p-permission-denied)
-   [macOS-Simple-KVM](https://github.com/foxlet/macOS-Simple-KVM)

[^fn:1]: [KVM](https://wiki.archlinux.org/title/KVM) is a hypervisor built into the linux kernel.
[^fn:2]: [QEMU](https://wiki.qemu.org/) is a generic and open source machine emulator and virtualizer.
